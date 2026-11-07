"""Shared fixtures: a journal on disk, seeded through the gate, and scripted workers."""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, Mapping

from commit_gate.gate import CommitGate
from commit_gate.ops import AddEdge, UpsertNode
from commit_gate.proposal import Proposal
from commit_gate.state import MemoryView
from commit_gate.store import JournalStore
from mathproof.dispatch import ScriptedDispatcher
from mathproof.formal_atp import FakeFormalATP

from coordinator.langgraph.workspace import ProofWorkspace, project
from coordinator.langgraph.mechanisms import MECHANISM_FIELDS, mechanism_moves


def mechanism_answer():
    """Complete scripted research answers, never formal evidence."""
    answers = []
    for tag in ("induction", "extremal", "algebraic", "analytic", "probabilistic"):
        value = {name: f"{name} via {tag}" for name in MECHANISM_FIELDS}
        value.update(technique_tags=[tag], prerequisites=[], promise=.5, cost=.5,
                     novelty=.5, risk=.5, stance="proves")
        answers.append(value)
    return answers


def verified_target(proof_id):
    """Synthetic bound replay records for gate tests; no real Lean was run."""
    ids = {name: f"{proof_id}/{name}" for name in
           ("proof", "c-verified", "fd-verified", "env-v", "fr-v", "cert-v", "replay-v", "al-v")}
    p, c, d, env, run, cert, replay, al = ids.values()
    env_hash, artifact = "sha256:" + "e" * 64, "sha256:" + "a" * 64
    ops = [UpsertNode("Proof", p, {}), UpsertNode("Claim", c, {"status": "lean-verified"}),
        UpsertNode("FormalDeclaration", d, {"status": "replay-accepted"}),
        UpsertNode("Environment", env, {"environment_hash": env_hash}),
        UpsertNode("FormalRun", run, {"status": "proved-pending-replay", "environment_hash": env_hash}),
        UpsertNode("Certificate", cert, {"actor": "scripted-producer", "status": "replay-accepted",
            "artifact_hash": artifact, "environment_hash": env_hash, "producer_run_id": run}),
        UpsertNode("LeanReplay", replay, {"actor": "scripted-replayer", "status": "verified",
            "sorry_detected": False, "environment_hash": env_hash,
            "certificate_id": cert, "artifact_hash": artifact}),
        UpsertNode("Alignment", al, {"lifecycle": "reviewed", "verdict": "aligned"})]
    for index, (rel, src, dst) in enumerate((
        ("HAS_TARGET", p, c), ("PROVED_BY", c, cert), ("REPLAYED_BY", cert, replay),
        ("CERTIFIES", cert, d), ("PRODUCED_CERTIFICATE", run, cert), ("SEARCHES", run, d),
        ("PINNED_ENVIRONMENT", d, env), ("CERTIFICATE_ENVIRONMENT", cert, env),
        ("RAN_UNDER", run, env), ("REPLAY_ENVIRONMENT", replay, env),
        ("ALIGNS_CLAIM", al, c), ("ALIGNS_DECLARATION", al, d))):
        ops.append(AddEdge(rel, src, dst, f"{proof_id}/v-edge-{index}"))
    return ops


def declaration(proof_id: str, serial: int = 1, status: str = "aligned") -> UpsertNode:
    return UpsertNode(
        "FormalDeclaration",
        f"{proof_id}/fd-{serial}",
        {
            "status": status,
            "lean_name": f"t{serial}",
            "lean_type": "True",
            "lean_value": f"t{serial} : True := trivial",
        },
    )


def aligned_claim(proof_id: str, serial: int = 1) -> list:
    """A provisional claim with the reviewed alignment the critic gate requires."""
    claim_id = f"{proof_id}/c-{serial}"
    alignment_id = f"{proof_id}/al-{serial}"
    return [
        UpsertNode("Claim", claim_id, {"status": "provisional", "claim_text": "x"}),
        UpsertNode("Alignment", alignment_id, {"lifecycle": "reviewed", "verdict": "aligned"}),
        AddEdge("ALIGNS_CLAIM", alignment_id, claim_id, f"{alignment_id}-aligns"),
    ]


class JournalCase(unittest.TestCase):
    """A fresh journal file per test, opened once per caller like production."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.db_path = os.path.join(self._dir.name, "journal.db")

    def open_store(self) -> JournalStore:
        return JournalStore(self.db_path)

    def seed(self, proof_id: str, *ops) -> None:
        """Commit `ops` as maintenance, so the journal holds the starting state."""
        store = self.open_store()
        lease_id = f"{proof_id}/ls-seed-{store.head(proof_id)[0]}"
        token = store.acquire_lease(proof_id, lease_id, actor="seed", worker_class="maintenance")
        commit = CommitGate(self.view_of(proof_id, store), store).commit(
            Proposal(
                proof_id,
                "seed",
                "maintenance",
                ops=tuple(ops),
                base_revision=store.head(proof_id)[0],
                lease_id=lease_id,
                fencing_token=token,
            )
        )
        store.release_lease(proof_id, lease_id, fencing_token=token)
        self.assertTrue(commit.accepted, commit.rejections)

    def view_of(self, proof_id: str, store: JournalStore | None = None) -> MemoryView:
        """The proof rebuilt from the journal alone."""
        view = MemoryView()
        project(view, store or self.open_store(), proof_id)
        return view

    def workspace(
        self,
        proof_id: str,
        *,
        handlers: Mapping[str, Any] | None = None,
        plan: Mapping[str, Any] | None = None,
    ) -> ProofWorkspace:
        return ProofWorkspace(
            proof_id,
            open_store=self.open_store,
            dispatcher=ScriptedDispatcher(dict(handlers or {})),
            adapters={"formal-atp": FakeFormalATP(plan or {})},
        )

    def coordinator_events(self, proof_id: str) -> list[dict[str, Any]]:
        return [
            event
            for event in self.open_store().read_events(proof_id)
            if event["worker_class"] == "coordinator"
        ]

    def moves(self, proof_id: str, escalation: str | None = None) -> list:
        view = self.view_of(proof_id)
        found = [
            record
            for record in view.nodes.values()
            if record.label == "ResearchMove" and record.node_id.startswith(f"{proof_id}/")
        ]
        if escalation is not None:
            found = [record for record in found if record.fields.get("escalation") == escalation]
        return found


def explorer(answers: Mapping[str, Any] | None = None, worked: list | None = None):
    """A research worker that marks each move it takes as worked.

    ``answers`` maps an escalation name to the ``moves`` it proposes in reply.
    ``worked`` collects the move ids it was given.
    """

    def handler(lease, packet):
        move = packet["move"]["fields"]
        if worked is not None:
            worked.append(lease.selected_move_id)
        return {
            "parent_state_id": packet["parent_state"]["id"],
            "moves": list((answers or {}).get(move.get("escalation"),
                mechanism_moves(mechanism_answer()) if move.get("escalation") == "five-mechanisms" else [])),
            "move_status": "exhausted",
        }

    return handler


def critic(verdict: str = "critic-accepted", **fields):
    def handler(lease, packet):
        return {"verdict": verdict, "actor": lease.actor, **fields}

    return handler
