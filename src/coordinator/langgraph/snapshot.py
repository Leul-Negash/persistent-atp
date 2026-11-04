"""Read-only policy adapter for a complete committed coordination snapshot."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from commit_gate.state import EdgeRecord, NodeRecord

from .contracts import CoordinationState


@dataclass(frozen=True, slots=True)
class SnapshotView:
    """A proof-scoped read cache, never a projection or a commit authority."""

    state: CoordinationState
    nodes: Mapping[str, NodeRecord] = field(init=False)
    edges: Mapping[str, EdgeRecord] = field(init=False)
    _out: Mapping[tuple[str, str], tuple[EdgeRecord, ...]] = field(init=False, repr=False)
    _in: Mapping[tuple[str, str], tuple[EdgeRecord, ...]] = field(init=False, repr=False)

    def __post_init__(self):
        prefix = self.state.proof_id + "/"
        nodes = {n.node_id: n for n in self.state.nodes}
        edges = {e.edge_id: e for e in self.state.edges}
        if len(nodes) != len(self.state.nodes) or len(edges) != len(self.state.edges):
            raise ValueError("coordination snapshot contains duplicate record IDs")
        if (any(not n.startswith(prefix) for n in nodes)
                or any(not all(i.startswith(prefix) for i in (e.edge_id, e.src_id, e.dst_id))
                       for e in edges.values())):
            raise ValueError("coordination snapshot contains records from another proof")
        outgoing, incoming = {}, {}
        for e in sorted(edges.values(), key=lambda e: e.edge_id):
            outgoing.setdefault((e.src_id, e.rel_type), []).append(e)
            incoming.setdefault((e.dst_id, e.rel_type), []).append(e)
        object.__setattr__(self, "nodes", MappingProxyType(dict(sorted(nodes.items()))))
        object.__setattr__(self, "edges", MappingProxyType(dict(sorted(edges.items()))))
        object.__setattr__(self, "_out", MappingProxyType({k: tuple(v) for k, v in outgoing.items()}))
        object.__setattr__(self, "_in", MappingProxyType({k: tuple(v) for k, v in incoming.items()}))

    def snapshot(self, proof_id: str):
        if proof_id != self.state.proof_id:
            raise KeyError("snapshot belongs to another proof")
        return self.state.revision, self

    def node(self, node_id: str) -> NodeRecord | None:
        return self.nodes.get(node_id)

    def edge(self, edge_id: str) -> EdgeRecord | None:
        return self.edges.get(edge_id)

    def edges_from(self, node_id: str, rel_type: str) -> tuple[EdgeRecord, ...]:
        return self._out.get((node_id, rel_type), ())

    def edges_to(self, node_id: str, rel_type: str) -> tuple[EdgeRecord, ...]:
        return self._in.get((node_id, rel_type), ())
