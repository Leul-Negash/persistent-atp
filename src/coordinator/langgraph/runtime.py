"""Additive trusted calling boundary over the existing coordinator functions.

This synchronous facade owns no operation database or remote retry policy.
Workspaces and workers are supplied by the trusted host at registration.
"""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict
from types import MappingProxyType

from commit_gate.state import NodeRecord
from mathproof.scheduler import GlobalScheduler

from .escalate import commit_escalations
from .policy import Escalation, RULES, _not_on_record
from .portfolio import PORTFOLIO, run_portfolio
from .progress import coordinator_moves, read_proof
from .rounds import work, assess
from .workspace import ProofWorkspace, project
from .contracts import CoordinationState, StaleAssessment


class CoordinatorRuntime:
    """One registry entry and lock per proof, with no caller-selected authority.

    Slot calls are intentional new executions. An exception report is not a
    rollback acknowledgement and must never trigger an automatic replacement.
    """

    def __init__(self, workspaces: Sequence[ProofWorkspace]):
        registry = {}
        for workspace in workspaces:
            if not workspace.proof_id or workspace.proof_id in registry:
                raise ValueError("register each nonempty proof ID exactly once")
            registry[workspace.proof_id] = workspace
        self.workspaces = MappingProxyType(registry)

    @property
    def proof_ids(self) -> tuple[str, ...]:
        return tuple(self.workspaces)

    def _workspace(self, proof_id):
        try:
            return self.workspaces[proof_id]
        except KeyError:
            raise KeyError(f"proof {proof_id!r} is not registered") from None

    @staticmethod
    def _capture(workspace, store):
        """Bounded retries for reads only; mutations are never retried here."""
        proof_id = workspace.proof_id
        for _ in range(3):
            project(workspace.view, store, proof_id)
            head = store.head(proof_id)
            revision, snapshot = workspace.view.snapshot(proof_id)
            if head[0] == 0:
                raise KeyError("registered proof has no committed records")
            if revision != head[0]:
                continue
            reading = read_proof(snapshot, proof_id, revision)
            frontier = GlobalScheduler(snapshot, store).build_frontier(proof_id)
            if store.head(proof_id) != head:
                continue
            prefix = proof_id + "/"
            nodes = tuple(n for _, n in sorted(snapshot.nodes.items()) if n.node_id.startswith(prefix))
            edges = tuple(e for _, e in sorted(snapshot.edges.items()) if e.edge_id.startswith(prefix))
            state = CoordinationState(proof_id, revision, head[1], reading, nodes, edges,
                tuple(frontier),
                tuple(n for n in nodes if n.node_id in {
                    move.node_id for move in coordinator_moves(snapshot, proof_id)}))
            return snapshot, state
        raise StaleAssessment("journal changed during three coordination reads")

    @staticmethod
    def _expect(state, expected_revision):
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        if state.revision != expected_revision:
            raise StaleAssessment(f"expected revision {expected_revision}, observed {state.revision}")

    def read_coordination(self, proof_id: str) -> CoordinationState:
        workspace = self._workspace(proof_id)
        with workspace.lock, closing(workspace.open_store()) as store:
            return self._capture(workspace, store)[1]

    def read_escalations(self, proof_id: str) -> tuple[NodeRecord, ...]:
        return self.read_coordination(proof_id).escalations

    def run_slot(self, proof_id: str, category: str, *, expected_revision: int) -> dict:
        slot = next((slot for slot in PORTFOLIO if slot.category == category), None)
        if slot is None:
            raise ValueError("unknown frontier category")
        workspace = self._workspace(proof_id)
        with workspace.lock, closing(workspace.open_store()) as store:
            _, state = self._capture(workspace, store)
            self._expect(state, expected_revision)
            reports = run_portfolio(workspace, store, slots=(slot,), expected_revision=expected_revision)
            report = reports[0] if reports else {"slot": slot.role, "worker_class": slot.worker_class,
                "move_id": None, "accepted": False, "revision": None, "rejections": [], "outcome": "idle"}
            return {"proof_id": proof_id, "base_revision": state.revision,
                "head_revision": store.head(proof_id)[0], **report}

    def queue_escalations(self, proof_id: str, escalations: Sequence[Escalation], *,
                          expected_revision: int, round_number: int) -> dict:
        if type(round_number) is not int or round_number < 0:
            raise ValueError("round_number must be a nonnegative integer")
        if not isinstance(escalations, (list, tuple)) or len(escalations) > 100:
            raise ValueError("submit a batch of at most 100 escalations")
        workspace = self._workspace(proof_id)
        with workspace.lock, closing(workspace.open_store()) as store:
            snapshot, state = self._capture(workspace, store)
            self._expect(state, expected_revision)
            for item in escalations:
                if not isinstance(item, Escalation) or item.rule not in RULES:
                    raise ValueError("unknown escalation rule")
                if not isinstance(item.detail, str) or not item.detail.strip() or len(item.detail.encode()) > 4096:
                    raise ValueError("escalation detail must be nonempty and at most 4096 bytes")
                if item.target_id != proof_id and (not isinstance(item.target_id, str)
                        or not item.target_id.startswith(proof_id + "/") or snapshot.node(item.target_id) is None):
                    raise ValueError("escalation target is not in this proof")
                if item.claim_id is not None:
                    claim = snapshot.node(item.claim_id)
                    if not item.claim_id.startswith(proof_id + "/") or claim is None or claim.label != "Claim":
                        raise ValueError("escalation claim is not in this proof")
            fresh = _not_on_record(snapshot, proof_id, state.progress, list(escalations))
            prior_events = store.read_lease_events(proof_id)
            after = prior_events[-1]["seq"] if prior_events else 0
            result = commit_escalations(store, workspace.view, proof_id, fresh,
                reading=state.progress, round_number=round_number)
            project(workspace.view, store, proof_id)
            outcome = "duplicate" if result is None else "accepted" if result.accepted else "rejected"
            return {"proof_id": proof_id, "outcome": outcome,
                "commit": result.to_dict() if result else None,
                "revision": store.head(proof_id)[0],
                "escalations": [asdict(item) for item in fresh] if result and result.accepted else [],
                "lease_events": [{key: event[key] for key in
                    ("lease_id", "event", "selected_move_id", "base_revision")}
                    for event in store.read_lease_events(proof_id) if event["seq"] > after]}

    def work(self, proof_id: str, round_number: int, *, slots=PORTFOLIO):
        return work(self._workspace(proof_id), round_number, slots=slots)

    def assess(self, proof_id: str, track, report, **options):
        if report["proof_id"] != proof_id:
            raise ValueError("round report belongs to another proof")
        return assess(self._workspace(proof_id), track, report, **options)

    def run_round(self, proof_id: str, round_number: int, track, **options):
        report = self.work(proof_id, round_number)
        track, raised, milestones = self.assess(proof_id, track, report, **options)
        return {"report": report, "track": track, "escalations": raised, "milestones": milestones}
