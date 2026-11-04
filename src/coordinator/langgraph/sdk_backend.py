"""Optional SDK binding for complete coordination reads.

Mutation calls fail closed until durable delivery and the corresponding
receiver guarantees are supported. Client credentials and lifecycle belong to
the caller. No SDK or transport dependency is imported by the core package.
"""
from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Sequence

from commit_gate.state import EdgeRecord, NodeRecord
from mathproof.scheduler import Candidate

from .contracts import CoordinationState, StaleAssessment
from .portfolio import PORTFOLIO
from .progress import ProofReading, coordinator_moves, read_proof
from .snapshot import SnapshotView

if TYPE_CHECKING:
    from persistent_atp_client import AsyncClient


class CoordinationReadLimitExceeded(ValueError):
    """The configured read bounds cannot hold a complete snapshot."""


def _sdk():
    try:
        from persistent_atp_client import AsyncClient
        from persistent_atp_client.errors import (
            CursorExpired, ReceiverProtocolError, StalePrecondition, UnsupportedCapability,
        )
    except ImportError as exc:
        raise ImportError("SDKBackend requires persistent-atp-client 0.2; install the SDK "
                          "distribution or persistent-atp[sdk]") from exc
    return SimpleNamespace(AsyncClient=AsyncClient, CursorExpired=CursorExpired,
        ProtocolError=ReceiverProtocolError, StalePrecondition=StalePrecondition,
        UnsupportedCapability=UnsupportedCapability)


class SDKBackend:
    """Read registered proofs through the actual AsyncClient, with bounded retries.

    ``read_coordination`` is synchronous for graph worker threads;
    ``aread_coordination`` is for hosts with an active event loop. The supplied
    client/port must support separate event loops for synchronous parallel reads
    (the SDK HTTP adapter does). This object never closes the caller's client.
    """

    def __init__(self, client: AsyncClient, *, workspace_id: str, proof_ids: Sequence[str],
                 page_size: int = 100, max_pages: int = 100, max_items: int = 10_000,
                 max_bytes: int = 16 * 1024 * 1024, read_timeout: float = 10,
                 read_attempts: int = 3):
        self._sdk = _sdk()
        if not isinstance(client, self._sdk.AsyncClient):
            raise TypeError("client must be persistent_atp_client.AsyncClient")
        # Match the SDK workspace alias contract without selecting any authority.
        import re
        if not isinstance(workspace_id, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", workspace_id) is None:
            raise ValueError("workspace_id must be a registered alias")
        if isinstance(proof_ids, (str, bytes)):
            raise ValueError("proof_ids must be a sequence of registered proof IDs")
        registered = tuple(proof_ids)
        if (any(not isinstance(p, str) or not 1 <= len(p) <= 128
                or any(ord(c) < 32 for c in p) for p in registered)
                or len(set(registered)) != len(registered)):
            raise ValueError("register each bounded nonempty proof ID exactly once")
        for name, value in (("page_size", page_size), ("max_pages", max_pages),
                            ("max_items", max_items), ("max_bytes", max_bytes),
                            ("read_attempts", read_attempts)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if page_size > 200:
            raise ValueError("page_size must be within 1..200")
        if type(read_timeout) not in (int, float) or not math.isfinite(read_timeout) or read_timeout <= 0:
            raise ValueError("read_timeout must be finite and positive")
        self.client = client
        self.workspace_id = workspace_id
        self.proof_ids = registered
        self.page_size, self.max_pages, self.max_items = page_size, max_pages, max_items
        self.max_bytes, self.read_timeout, self.read_attempts = max_bytes, read_timeout, read_attempts

    def read_coordination(self, proof_id: str) -> CoordinationState:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aread_coordination(proof_id))
        raise RuntimeError("use await aread_coordination inside an active event loop")

    async def aread_coordination(self, proof_id: str) -> CoordinationState:
        if proof_id not in self.proof_ids:
            raise KeyError(f"proof {proof_id!r} is not registered")
        async with asyncio.timeout(self.read_timeout):
            for attempt in range(self.read_attempts):
                try:
                    return await self._collect(proof_id)
                except (self._sdk.StalePrecondition, self._sdk.CursorExpired, StaleAssessment) as exc:
                    if attempt + 1 == self.read_attempts:
                        raise StaleAssessment("coordination read remained stale within the retry bound") from exc
        raise AssertionError("unreachable read attempt")

    async def _collect(self, proof_id):
        cursor, identity = None, None
        cursors, keys, records = set(), set(), []
        used_bytes = 0
        for _ in range(self.max_pages):
            page = await self.client.coordination_page(proof_id, workspace_id=self.workspace_id,
                                                       cursor=cursor, page_size=self.page_size)
            current = (page.revision, page.journal_hash, page.total_items, page.component_mode)
            if identity is None:
                identity = current
                if page.total_items > self.max_items:
                    raise CoordinationReadLimitExceeded("snapshot exceeds max_items")
            elif current[:2] != identity[:2]:
                if current[0] == identity[0]:
                    raise self._sdk.ProtocolError("journal hash changed within one revision")
                raise StaleAssessment("revision changed during coordination pagination")
            elif current != identity:
                raise self._sdk.ProtocolError("snapshot count or component mode changed during pagination")
            if page.projection_watermark != page.revision:
                raise StaleAssessment("projection has not reached the coordination revision")
            if len(page.items) > self.page_size:
                raise self._sdk.ProtocolError("coordination page exceeds requested page_size")
            for item in page.items:
                key = (item.item_type, item.record_id)
                if key in keys:
                    raise self._sdk.ProtocolError("duplicate coordination record")
                keys.add(key)
                try:
                    encoded = json.dumps({"type": item.item_type, "id": item.record_id,
                        "label": item.label, "fields": item.fields}, ensure_ascii=False,
                        allow_nan=False, separators=(",", ":")).encode("utf-8")
                    used_bytes += len(encoded)
                    if used_bytes > self.max_bytes:
                        raise CoordinationReadLimitExceeded("snapshot exceeds max_bytes")
                    records.append(json.loads(encoded))
                except (TypeError, ValueError, RecursionError) as exc:
                    if isinstance(exc, CoordinationReadLimitExceeded):
                        raise
                    raise self._sdk.ProtocolError("coordination fields must be finite JSON values") from exc
            if len(records) > page.total_items:
                raise self._sdk.ProtocolError("coordination records exceed total_items")
            if page.next_cursor is None:
                if len(records) != page.total_items:
                    raise self._sdk.ProtocolError("incomplete coordination snapshot")
                return self._convert(proof_id, identity, records)
            if not page.items or len(records) == page.total_items or page.next_cursor in cursors:
                raise self._sdk.ProtocolError("coordination pagination made no progress")
            cursors.add(page.next_cursor)
            cursor = page.next_cursor
        raise CoordinationReadLimitExceeded("snapshot exceeds max_pages")

    def _convert(self, proof_id, identity, records):
        nodes, edges, frontier = [], [], []
        try:
            for item in records:
                record_id, label, fields = item["id"], item["label"], item["fields"]
                if not record_id.startswith(proof_id + "/"):
                    raise ValueError("record belongs to another proof")
                if item["type"] == "node":
                    nodes.append(NodeRecord(record_id, label, fields))
                elif item["type"] == "edge":
                    fields = dict(fields)
                    src, dst = fields.pop("src_id"), fields.pop("dst_id")
                    edges.append(EdgeRecord(record_id, label, src, dst, fields))
                else:
                    if label not in {s.category for s in PORTFOLIO}:
                        raise self._sdk.UnsupportedCapability("unsupported coordination frontier category")
                    if set(fields) != {"detail", "features", "conflict_ids"}:
                        raise ValueError("invalid frontier fields")
                    if (not isinstance(fields["detail"], str) or not isinstance(fields["features"], dict)
                            or any(type(v) not in (int, float) for v in fields["features"].values())
                            or not isinstance(fields["conflict_ids"], list)
                            or any(not isinstance(i, str) or not i.startswith(proof_id + "/")
                                   for i in fields["conflict_ids"])):
                        raise ValueError("invalid frontier values")
                    frontier.append(Candidate(record_id, label, fields["detail"], fields["features"],
                                               tuple(fields["conflict_ids"])))
            ids = {n.node_id for n in nodes}
            if (any(e.src_id not in ids or e.dst_id not in ids for e in edges)
                    or any(c.move_id not in ids for c in frontier)):
                raise ValueError("coordination snapshot has unresolved graph references")
            revision, journal_hash, _, mode = identity
            provisional = CoordinationState(proof_id, revision, journal_hash, ProofReading(revision),
                tuple(nodes), tuple(edges), tuple(frontier), (), mode)
            view = SnapshotView(provisional)
            return CoordinationState(proof_id, revision, journal_hash, read_proof(view, proof_id, revision),
                provisional.nodes, provisional.edges, provisional.frontier,
                tuple(coordinator_moves(view, proof_id)), mode)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise self._sdk.ProtocolError("invalid complete coordination snapshot") from exc

    def run_slot(self, proof_id: str, category: str, *, expected_revision: int) -> dict:
        raise self._sdk.UnsupportedCapability(
            "SDK slot execution requires durable admission and execution-time revision enforcement")

    def queue_escalations(self, proof_id: str, escalations, *, expected_revision: int, round_number: int) -> dict:
        raise self._sdk.UnsupportedCapability(
            "SDK escalation submission requires the structured atomic escalation contract")
