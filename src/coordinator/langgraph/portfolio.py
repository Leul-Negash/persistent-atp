"""A round's portfolio of cycles (Section 8.6): exploit, explore, critique.

`GlobalScheduler` ranks one frontier for every worker class, so a formal
worker can be handed a critic task. A portfolio slot pins a worker class to
its frontier category: the exploitation slot continues formal search, the
exploration slot works research moves (including the coordinator's own
escalations) and the critic slot reviews provisional claims. Each slot is one
`run_cycle`; within a proof the slots run in turn, because a proposal must be
based on the head it commits onto and two cycles on one proof would race.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from commit_gate.gate import CommitGate
from commit_gate.state import ReadView
from commit_gate.store import JournalStore, ConcurrencyError
from commit_gate.reasons import Reason
from mathproof.cycle import CycleDigest, run_cycle
from mathproof.scheduler import Candidate, GlobalScheduler, SchedulerStatistics

from .workspace import ProofWorkspace, project
from .mechanisms import MechanismDispatcher, InvalidMechanisms

__all__ = ["Slot", "PORTFOLIO", "SlotScheduler", "frontier_by_category", "run_portfolio"]


@dataclass(frozen=True, slots=True)
class Slot:
    role: str
    category: str
    worker_class: str


PORTFOLIO = (
    Slot("exploit", "formal-run", "formal-atp"),
    Slot("explore", "research-move", "llm-research"),
    Slot("critic", "critic-task", "critic"),
)


class SlotScheduler(GlobalScheduler):
    """The global scheduler, restricted to one frontier category."""

    def __init__(self, view, store, *, category: str, statistics: SchedulerStatistics,
                 expected_revision: int | None = None):
        super().__init__(view, store, statistics=statistics)
        self.category = category
        self.last_lease = None
        self.expected_revision = expected_revision

    def _snapshot_frontier(self, proof_id):
        revision, frontier = super()._snapshot_frontier(proof_id)
        if self.expected_revision is not None and revision != self.expected_revision:
            raise ConcurrencyError(Reason.STALE_BASE_REVISION, "slot assessment revision changed")
        return revision, frontier

    def lease_next(self, *args, **kwargs):
        self.last_lease = super().lease_next(*args, **kwargs)
        return self.last_lease

    def _build_frontier(
        self, proof_id: str, view: ReadView, node_ids: Mapping[str, None]
    ) -> list[Candidate]:
        return [
            candidate
            for candidate in super()._build_frontier(proof_id, view, node_ids)
            if candidate.category == self.category
        ]


def frontier_by_category(workspace: ProofWorkspace, store: JournalStore) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in GlobalScheduler(workspace.view, store).build_frontier(workspace.proof_id):
        counts[candidate.category] = counts.get(candidate.category, 0) + 1
    return counts


def run_portfolio(
    workspace: ProofWorkspace,
    store: JournalStore,
    *,
    slots: tuple[Slot, ...] = PORTFOLIO,
    expected_revision: int | None = None,
) -> list[dict[str, Any]]:
    """Run one cycle per slot that has work; report each as plain data."""
    proof_id = workspace.proof_id
    gate = CommitGate(workspace.view, store)
    reports = []
    for slot in slots:
        revision = project(workspace.view, store, proof_id)
        if expected_revision is not None and revision != expected_revision:
            reports.append({"slot": slot.role, "worker_class": slot.worker_class,
                "move_id": None, "accepted": False, "revision": None,
                "rejections": [Reason.STALE_BASE_REVISION.value], "outcome": "stale-assessment"})
            break
        if not frontier_by_category(workspace, store).get(slot.category):
            continue
        scheduler = SlotScheduler(
            workspace.view, store, category=slot.category, statistics=workspace.statistics,
            expected_revision=expected_revision,
        )
        try:
            digest = run_cycle(
                proof_id,
                view=workspace.view,
                gate=gate,
                scheduler=scheduler,
                dispatcher=MechanismDispatcher(workspace.dispatcher, workspace.view),
                adapters=workspace.adapters,
                worker_class=slot.worker_class,
                maintenance=lambda proposal, commit: project(workspace.view, store, proof_id),
            )
        except Exception as exc:
            lease = scheduler.last_lease
            known_invalid = isinstance(exc, InvalidMechanisms)
            stale = isinstance(exc, ConcurrencyError) and lease is None
            if known_invalid and lease is not None:
                scheduler.release(proof_id, lease.lease_id, fencing_token=lease.fencing_token)
            reports.append({"slot": slot.role, "worker_class": slot.worker_class,
                "move_id": lease.selected_move_id if lease else None, "accepted": False,
                "revision": None, "rejections": [exc.reason.value] if stale else [],
                "outcome": "invalid-worker-result" if known_invalid else
                           "stale-assessment" if stale else "execution-unknown",
                "error_code": type(exc).__name__, "detail": str(exc)[:500]})
            # Unknown execution may still be in flight. Keep its fencing lease
            # alive, and never dispatch a replacement inside this call.
            break
        reports.append(_slot_report(slot, digest))
    project(workspace.view, store, proof_id)
    return reports


def _slot_report(slot: Slot, digest: CycleDigest) -> dict[str, Any]:
    return {
        "slot": slot.role,
        "worker_class": slot.worker_class,
        "move_id": digest.selected_move_id,
        "accepted": digest.accepted,
        "revision": digest.revision,
        "rejections": [rejection.reason.value for rejection in digest.rejections],
        "outcome": ("accepted" if digest.accepted else
                    "rejected" if digest.rejections else "waiting-on-lease"),
    }
