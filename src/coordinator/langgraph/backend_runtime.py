"""Backend-independent rounds; all execution and writes remain on ProofBackend."""
from __future__ import annotations

from .assessment import finish_assessment, plan_assessment
from .contracts import ProofBackend, StaleAssessment
from .policy import Escalation, StagnationPolicy
from .portfolio import PORTFOLIO
from .progress import read_proof
from .snapshot import SnapshotView


class BackendRoundRuntime:
    """Adapt the three backend calls to the graph's round interface.

    This class owns neither stores nor workers. It performs no automatic
    mutation retry. Durable remote admission/reconciliation belongs to a
    backend adapter; an observation timeout never authorizes replacement work.
    """

    def __init__(self, backend: ProofBackend):
        self.backend = backend
        self.proof_ids = tuple(backend.proof_ids)
        if (any(not isinstance(p, str) or not p for p in self.proof_ids)
                or len(set(self.proof_ids)) != len(self.proof_ids)):
            raise ValueError("register each nonempty proof ID exactly once")

    def _read(self, proof_id):
        if proof_id not in self.proof_ids:
            raise KeyError(f"proof {proof_id!r} is not registered")
        state = self.backend.read_coordination(proof_id)
        if state.proof_id != proof_id:
            raise ValueError("coordination snapshot belongs to another proof")
        view = SnapshotView(state)
        if read_proof(view, proof_id, state.revision) != state.progress:
            raise ValueError("coordination progress disagrees with snapshot records")
        if any(not c.move_id.startswith(proof_id + "/") for c in state.frontier):
            raise ValueError("coordination frontier belongs to another proof")
        return state, view

    def work(self, proof_id, round_number, *, slots=PORTFOLIO):
        if type(round_number) is not int or round_number < 0:
            raise ValueError("round_number must be a nonnegative integer")
        # The backend selects trusted workers. Custom authority cannot be
        # smuggled through a Slot whose category happens to be supported.
        if any(slot not in PORTFOLIO for slot in slots):
            raise ValueError("backend rounds support only registered portfolio slots")
        state, _ = self._read(proof_id)
        before = list(state.progress.progress)
        reports = []
        for slot in slots:
            if not any(c.category == slot.category for c in state.frontier):
                continue
            try:
                result = self.backend.run_slot(proof_id, slot.category,
                                               expected_revision=state.revision)
            except StaleAssessment:
                result = {"proof_id": proof_id, "slot": slot.role,
                          "worker_class": slot.worker_class, "move_id": None,
                          "accepted": False, "revision": None, "rejections": [],
                          "outcome": "stale-assessment"}
            if result["proof_id"] != proof_id:
                raise ValueError("slot result belongs to another proof")
            if result["outcome"] != "idle":
                reports.append(result)
            if result["outcome"] in {"execution-unknown", "stale-assessment", "invalid-worker-result"}:
                break
            state, _ = self._read(proof_id)
        return {"proof_id": proof_id, "round": round_number, "before": before, "slots": reports}

    def assess(self, proof_id, track, report, *, policy=StagnationPolicy()):
        if report["proof_id"] != proof_id:
            raise ValueError("round report belongs to another proof")
        state, view = self._read(proof_id)
        plan = plan_assessment(view, proof_id, state.progress, track, report,
                               frontier_empty=not state.frontier, policy=policy)
        if plan.deferred_status is not None:
            return finish_assessment(proof_id, plan, frontier_empty=not state.frontier)
        outcome, committed, reasons = "duplicate", (), ()
        expected_head = state.revision
        if plan.escalations:
            try:
                result = self.backend.queue_escalations(proof_id, plan.escalations,
                    expected_revision=state.revision, round_number=report["round"])
            except StaleAssessment:
                return finish_assessment(proof_id, plan, frontier_empty=False,
                                         outcome="stale-assessment")
            if result["proof_id"] != proof_id:
                raise ValueError("escalation result belongs to another proof")
            outcome = result["outcome"]
            if outcome == "accepted":
                committed = tuple(Escalation(**item) for item in result["escalations"])
                if committed != plan.escalations:
                    raise ValueError("backend acknowledgement differs from the complete escalation batch")
            commit = result.get("commit")
            if commit:
                reasons = [r["reason"] for r in commit.get("rejections", ())]
            if outcome == "accepted":
                if (not commit or commit.get("accepted") is not True
                        or type(commit.get("revision")) is not int
                        or commit["revision"] != state.revision + 1
                        or result.get("revision") != commit["revision"]):
                    raise ValueError("escalation receipt does not identify the expected atomic commit")
                expected_head = commit["revision"]
        if outcome in {"execution-unknown", "stale-assessment"}:
            return finish_assessment(proof_id, plan, frontier_empty=False, outcome=outcome)
        latest, _ = self._read(proof_id)
        if latest.revision == state.revision and latest.journal_hash != state.journal_hash:
            raise ValueError("journal identity changed within one assessment revision")
        if outcome == "accepted" and latest.revision == expected_head:
            if commit.get("event_hash") != latest.journal_hash:
                raise ValueError("escalation receipt journal hash differs from the committed snapshot")
        # A known commit can still be reported, but a later concurrent
        # mutation means this assessment cannot park or solve the proof.
        return finish_assessment(
            proof_id, plan, frontier_empty=not latest.frontier, outcome=outcome,
            committed=committed, rejection_reasons=reasons,
            refresh_required=latest.revision != expected_head)
