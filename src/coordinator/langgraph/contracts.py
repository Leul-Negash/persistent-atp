"""Coordinator-facing execution contract, independent of transport and LangGraph.

Backends supply committed, proof-scoped snapshots. Workers, leases, proposal
construction and commits belong to the trusted execution host. This interface
does not add remote admission, cancellation or exactly-once execution semantics.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from commit_gate.state import EdgeRecord, NodeRecord
from mathproof.scheduler import Candidate

from .policy import Escalation
from .progress import ProofReading

__all__ = ["CoordinationState", "ProofBackend", "StaleAssessment"]


class StaleAssessment(ValueError):
    """Refresh committed state and reassess; never silently rebase a decision."""


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class CoordinationState:
    """One committed revision; mutable input records are copied and frozen.

    A backend must return all scoped nodes, edges and frontier entries, rather
    than a partial page. Mathematical status remains evidence read from the
    trusted service, never a conclusion drawn from operation completion.
    """

    proof_id: str
    revision: int
    journal_hash: str
    progress: ProofReading
    nodes: tuple[NodeRecord, ...]
    edges: tuple[EdgeRecord, ...]
    frontier: tuple[Candidate, ...]
    escalations: tuple[NodeRecord, ...]

    def __post_init__(self):
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("coordination revision must be a nonnegative integer")
        if self.progress.revision != self.revision:
            raise ValueError("progress belongs to another coordination revision")
        object.__setattr__(self, "nodes", tuple(
            NodeRecord(n.node_id, n.label, _freeze(n.fields)) for n in self.nodes))
        object.__setattr__(self, "edges", tuple(
            EdgeRecord(e.edge_id, e.rel_type, e.src_id, e.dst_id, _freeze(e.fields))
            for e in self.edges))
        object.__setattr__(self, "frontier", tuple(
            Candidate(c.move_id, c.category, c.detail, _freeze(c.features),
                      tuple(c.conflict_ids)) for c in self.frontier))
        object.__setattr__(self, "escalations", tuple(
            NodeRecord(n.node_id, n.label, _freeze(n.fields)) for n in self.escalations))

    def to_dict(self) -> dict[str, Any]:
        return {"proof_id": self.proof_id, "revision": self.revision,
            "journal_hash": self.journal_hash, "progress": asdict(self.progress),
            "nodes": [{"id": n.node_id, "label": n.label, "fields": _plain(n.fields)} for n in self.nodes],
            "edges": [{"id": e.edge_id, "rel_type": e.rel_type, "src_id": e.src_id,
                "dst_id": e.dst_id, "fields": _plain(e.fields)} for e in self.edges],
            "frontier": [{"move_id": c.move_id, "category": c.category, "detail": c.detail,
                "features": _plain(c.features), "conflict_ids": list(c.conflict_ids)} for c in self.frontier],
            "escalation_ids": [n.node_id for n in self.escalations]}


@runtime_checkable
class ProofBackend(Protocol):
    """Execution boundary shared by local and future receiver-backed callers.

    Registration belongs to the trusted host. Calls cannot select credentials,
    worker implementations, actors or fencing tokens. Stale mutation decisions
    fail before dispatch or commit; callers refresh and reassess explicitly.
    """

    @property
    def proof_ids(self) -> tuple[str, ...]:
        """Registered proofs in stable order, without opening a proof store."""
        ...

    def read_coordination(self, proof_id: str) -> CoordinationState:
        """Read one complete immutable revision; bounded read retries are allowed."""
        ...

    def run_slot(self, proof_id: str, category: str, *, expected_revision: int) -> dict:
        """Run one selected category; an unknown outcome is not a rollback."""
        ...

    def queue_escalations(self, proof_id: str, escalations: Sequence[Escalation], *,
                          expected_revision: int, round_number: int) -> dict:
        """Queue at most 100 complete escalations atomically, preserving policy identity."""
        ...
