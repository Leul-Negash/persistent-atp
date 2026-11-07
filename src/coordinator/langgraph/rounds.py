"""One coordinator round for one proof, without the graph around it.

`work` is the expensive half -- the portfolio's cycles -- and is what runs in
parallel across proofs. `assess` is the cheap half: read the committed
outcome, update the stall count, raise and commit escalations, and say what
happened in milestone messages (Section 10.7). Both exchange plain data only,
so whatever carries them between steps (a LangGraph checkpoint, say) holds a
cache of the journal and never a second copy of the proof.
"""

from __future__ import annotations

from typing import Any
from contextlib import closing

from .escalate import commit_escalations
from .policy import StagnationPolicy
from .portfolio import PORTFOLIO, Slot, frontier_by_category, run_portfolio
from .progress import read_proof
from .assessment import ACTIVE, SOLVED, PARKED, STALLED, WAITING, plan_assessment, finish_assessment
from .workspace import ProofWorkspace, project

__all__ = ["ACTIVE", "SOLVED", "PARKED", "STALLED", "WAITING", "new_track", "work", "assess"]


def new_track() -> dict[str, Any]:
    """What the coordinator remembers about a proof between rounds."""
    return {"status": ACTIVE, "progress": None, "stalled_rounds": 0, "awaiting_replay": False}


def work(
    workspace: ProofWorkspace, round_number: int, *, slots: tuple[Slot, ...] = PORTFOLIO
) -> dict[str, Any]:
    """Run this round's portfolio on one proof and report it."""
    with workspace.lock, closing(workspace.open_store()) as store:
        revision = project(workspace.view, store, workspace.proof_id)
        _, snapshot = workspace.view.snapshot(workspace.proof_id)
        before = read_proof(snapshot, workspace.proof_id, revision)
        return {
            "proof_id": workspace.proof_id,
            "round": round_number,
            "before": list(before.progress),
            "slots": run_portfolio(workspace, store, slots=slots),
        }


def assess(
    workspace: ProofWorkspace,
    track: dict[str, Any],
    report: dict[str, Any],
    *,
    policy: StagnationPolicy = StagnationPolicy(),
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Judge the round from committed state; commit what the policy calls for.

    Returns the updated track, the escalations committed, and the milestones.
    """
    proof_id = workspace.proof_id
    with workspace.lock, closing(workspace.open_store()) as store:
        revision = project(workspace.view, store, proof_id)
        _, snapshot = workspace.view.snapshot(proof_id)
        reading = read_proof(snapshot, proof_id, revision)
        frontier = frontier_by_category(workspace, store)
        if store.head(proof_id)[0] != revision:
            return ({**track, "status": ACTIVE}, [],
                    [f"{proof_id}: assessment became stale; refresh before deciding"])
        plan = plan_assessment(snapshot, proof_id, reading, track, report,
                               frontier_empty=not frontier, policy=policy)
        if plan.deferred_status is not None:
            return finish_assessment(proof_id, plan, frontier_empty=not frontier)
        commit = commit_escalations(store, workspace.view, proof_id, plan.escalations,
                                    reading=reading, round_number=report["round"])
        if commit is not None and commit.accepted:
            project(workspace.view, store, proof_id)
        return finish_assessment(
            proof_id, plan, frontier_empty=not frontier_by_category(workspace, store),
            outcome="duplicate" if commit is None else "accepted" if commit.accepted else "rejected",
            committed=plan.escalations if commit is not None and commit.accepted else (),
            rejection_reasons=[r.reason.value for r in commit.rejections] if commit is not None else (),
            refresh_required=store.head(proof_id)[0] != (
                commit.revision if commit is not None and commit.accepted else revision),
        )
