"""The coordinator's outer loop as a LangGraph StateGraph (Sections 10.2, 12.1).

    plan_round --Send per active proof--> work --> assess --> plan_round | END

`plan_round` opens a round and fans the active proofs out with `Send`; each
`work` branch delegates one proof's portfolio to the runtime, so the
expensive cycles of different proofs overlap. `assess` joins the round: it
judges every proof from committed state, commits the escalations the
stagnation policy calls for and decides whether another round is worth
running.

The graph state is a cache. It holds round counters, stall counts and
messages, never proof state: every decision is re-read from the journal, and
the only writes are proposals through the commit gate. A checkpointer may be
passed to resume a run, but losing its checkpoints loses no proof work -- a
fresh coordinator on the same journal sees the same escalations on record and
does not raise them again.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .policy import StagnationPolicy
from .portfolio import PORTFOLIO, Slot
from .contracts import RoundRuntime
from .rounds import ACTIVE, SOLVED, new_track
from .workspace import ProofWorkspace
from .runtime import CoordinatorRuntime

__all__ = ["CoordinatorState", "Coordinator", "HALTED", "BUDGET_EXHAUSTED"]

HALTED = "halted"
"""No proof can move without a replay or a human; each track says which."""
BUDGET_EXHAUSTED = "budget-exhausted"


class CoordinatorState(TypedDict, total=False):
    round: int
    tracks: dict[str, dict[str, Any]]
    reports: Annotated[list[dict[str, Any]], operator.add]
    escalations: Annotated[list[dict[str, Any]], operator.add]
    milestones: Annotated[list[str], operator.add]
    outcome: str | None


class WorkOrder(TypedDict):
    proof_id: str
    round: int


class Coordinator:
    """Runs rounds over several proofs until each is solved or the budget ends.

    Supply local ``workspaces`` or a ``runtime`` implementing :class:`RoundRuntime`.
    An injected runtime needs no workspace mapping. Its registered proof IDs
    are captured at construction; a run does not discover new proofs mid-round.
    """

    def __init__(
        self,
        workspaces: Sequence[ProofWorkspace] | None = None,
        *,
        runtime: RoundRuntime | None = None,
        max_rounds: int = 8,
        policy: StagnationPolicy = StagnationPolicy(),
        slots: tuple[Slot, ...] = PORTFOLIO,
    ):
        if max_rounds < 1:
            raise ValueError("a coordinator run needs at least one round")
        if (workspaces is None) == (runtime is None):
            raise ValueError("supply either workspaces or runtime")
        self.runtime = CoordinatorRuntime(workspaces) if runtime is None else runtime
        if not isinstance(self.runtime, RoundRuntime):
            raise TypeError("runtime must implement proof_ids, work and assess")
        self.proof_ids = tuple(self.runtime.proof_ids)
        if any(not isinstance(p, str) or not p for p in self.proof_ids):
            raise ValueError("runtime proof IDs must be nonempty strings")
        if len(set(self.proof_ids)) != len(self.proof_ids):
            raise ValueError("runtime proof IDs must be unique")
        self.max_rounds = max_rounds
        self.policy = policy
        self.slots = slots

    @property
    def workspaces(self):
        """Original local registry; injected runtimes need not expose workspaces."""
        if isinstance(self.runtime, CoordinatorRuntime):
            return self.runtime.workspaces
        raise AttributeError("workspace access requires a local CoordinatorRuntime")

    def build(self, checkpointer=None):
        graph = StateGraph(CoordinatorState)
        graph.add_node("plan_round", self._plan_round)
        graph.add_node("work", self._work)
        graph.add_node("assess", self._assess)
        graph.add_edge(START, "plan_round")
        graph.add_conditional_edges("plan_round", self._fan_out, ["work"])
        graph.add_edge("work", "assess")
        graph.add_conditional_edges("assess", self._route, ["plan_round", END])
        return graph.compile(checkpointer=checkpointer)

    def initial_state(self) -> CoordinatorState:
        return {
            "round": 0,
            "tracks": {proof_id: new_track() for proof_id in self.proof_ids},
            "reports": [],
            "escalations": [],
            "milestones": [],
            "outcome": None,
        }

    def run(self, *, checkpointer=None, thread_id: str = "coordinator") -> CoordinatorState:
        app = self.build(checkpointer)
        config = {
            "configurable": {"thread_id": thread_id},
            # Three steps a round, plus the start and a margin.
            "recursion_limit": 3 * self.max_rounds + 5,
        }
        return app.invoke(self.initial_state(), config)

    # -- nodes ---------------------------------------------------------------

    def _plan_round(self, state: CoordinatorState) -> dict[str, Any]:
        return {"round": state["round"] + 1}

    def _fan_out(self, state: CoordinatorState) -> list[Send]:
        return [
            Send("work", {"proof_id": proof_id, "round": state["round"]})
            for proof_id, track in state["tracks"].items()
            if track["status"] == ACTIVE
        ]

    def _work(self, order: WorkOrder) -> dict[str, Any]:
        report = self.runtime.work(order["proof_id"], order["round"], slots=self.slots)
        if report["proof_id"] != order["proof_id"] or report["round"] != order["round"]:
            raise ValueError("runtime report belongs to another proof or round")
        return {"reports": [report]}

    def _assess(self, state: CoordinatorState) -> dict[str, Any]:
        tracks = dict(state["tracks"])
        escalations: list[dict[str, Any]] = []
        milestones: list[str] = []
        for report in state["reports"]:
            if report["round"] != state["round"]:
                continue
            proof_id = report["proof_id"]
            track, raised, messages = self.runtime.assess(
                proof_id, tracks[proof_id], report, policy=self.policy
            )
            tracks[proof_id] = track
            escalations.extend({"proof_id": proof_id, "round": report["round"], **e} for e in raised)
            milestones.extend(messages)

        statuses = {track["status"] for track in tracks.values()}
        outcome = None
        if statuses == {SOLVED}:
            outcome = SOLVED
        elif ACTIVE not in statuses:
            outcome = HALTED
        elif state["round"] >= self.max_rounds:
            outcome = BUDGET_EXHAUSTED
        if outcome is not None:
            milestones.append(f"coordinator: {outcome} after {state['round']} rounds")
        return {
            "tracks": tracks,
            "escalations": escalations,
            "milestones": milestones,
            "outcome": outcome,
        }

    def _route(self, state: CoordinatorState) -> str:
        return END if state.get("outcome") else "plan_round"
