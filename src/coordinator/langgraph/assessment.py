"""Round policy decisions over committed reads, without execution or persistence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .policy import Escalation, StagnationPolicy, classify
from .progress import PolicyView, ProofReading

ACTIVE = "active"
SOLVED = "solved"
PARKED = "parked"
STALLED = "stalled"
WAITING = "waiting-on-worker"


@dataclass(frozen=True)
class AssessmentPlan:
    reading: ProofReading
    stalled_rounds: int
    escalations: tuple[Escalation, ...]
    milestones: tuple[str, ...]
    deferred_status: str | None = None


def plan_assessment(view: PolicyView, proof_id: str, reading: ProofReading,
                    track: dict[str, Any], report: dict[str, Any], *,
                    frontier_empty: bool,
                    policy: StagnationPolicy = StagnationPolicy()) -> AssessmentPlan:
    """Choose policy actions; unknown or stale execution never triggers new work."""
    if report["proof_id"] != proof_id:
        raise ValueError("round report belongs to another proof")
    settled, closed, replayed = report["before"]
    earlier = ProofReading(reading.revision, settled, closed, replayed)
    milestones = []
    for name, now, then in (
        ("claims settled", reading.settled_claims, earlier.settled_claims),
        ("formal states closed", reading.closed_states, earlier.closed_states),
        ("declarations replayed", reading.replayed_declarations, earlier.replayed_declarations),
    ):
        if now > then:
            milestones.append(f"{proof_id}: {now - then} {name}")
    outcomes = {slot.get("outcome") for slot in report["slots"]}
    if "execution-unknown" in outcomes:
        milestones.append(f"{proof_id}: worker outcome unknown; reconcile before resuming")
        return AssessmentPlan(reading, track["stalled_rounds"], (), tuple(milestones), WAITING)
    if "stale-assessment" in outcomes:
        return AssessmentPlan(reading, track["stalled_rounds"], (),
                              (f"{proof_id}: assessment became stale; refresh before deciding",), ACTIVE)
    stalled_rounds = 0 if reading.advanced_from(earlier) else track["stalled_rounds"] + 1
    if reading.awaiting_replay and not track["awaiting_replay"]:
        milestones.append(f"{proof_id}: a certificate awaits replay; formal search pauses")
    escalations = classify(view, proof_id, reading, stalled_rounds=stalled_rounds,
                           frontier_empty=frontier_empty, policy=policy)
    return AssessmentPlan(reading, stalled_rounds, tuple(escalations), tuple(milestones))


def finish_assessment(proof_id: str, plan: AssessmentPlan, *, frontier_empty: bool,
                      outcome: str = "duplicate", committed: Sequence[Escalation] = (),
                      rejection_reasons: Sequence[str] = (), refresh_required: bool = False
                      ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Describe the decision and acknowledged commits; completion is not proof."""
    reading = plan.reading
    milestones = list(plan.milestones)
    raised = []
    status = plan.deferred_status
    if status is None:
        if outcome == "accepted":
            for escalation in committed:
                raised.append({"rule": escalation.rule, "target_id": escalation.target_id,
                               "detail": escalation.detail})
                milestones.append(f"{proof_id}: {escalation.rule} -- {escalation.detail}")
        elif outcome == "rejected":
            milestones.append(f"{proof_id}: escalation proposal rejected ({', '.join(rejection_reasons)})")
        elif outcome == "stale-assessment":
            milestones.append(f"{proof_id}: assessment became stale; refresh before deciding")
        elif outcome == "execution-unknown":
            milestones.append(f"{proof_id}: escalation outcome unknown; reconcile before resuming")
        elif outcome != "duplicate":
            raise ValueError("unknown escalation outcome")
        status = WAITING if outcome == "execution-unknown" else ACTIVE
        if outcome not in {"execution-unknown", "stale-assessment"} and not refresh_required:
            if reading.solved:
                status = SOLVED
                milestones.append(f"{proof_id}: solved; every registered target is Lean verified")
            elif outcome != "rejected" and frontier_empty:
                if reading.awaiting_replay:
                    status = PARKED
                else:
                    status = STALLED
                    milestones.append(f"{proof_id}: nothing left to lease and every escalation is on record; "
                                      "needs a human decision")
        if refresh_required:
            milestones.append(f"{proof_id}: assessment became stale; refresh before deciding")
    return ({"status": status, "progress": list(reading.progress),
             "stalled_rounds": plan.stalled_rounds, "awaiting_replay": reading.awaiting_replay},
            raised, milestones)
