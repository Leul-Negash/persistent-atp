"""Actual SDK client with explicitly scripted coordination pages; no ATP/replay."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest

try:
    from persistent_atp_client import AsyncClient, CapabilitySet, CoordinationItem, CoordinationPage
    from persistent_atp_client.errors import ReceiverProtocolError, StalePrecondition, UnsupportedCapability
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

from coordinator.langgraph import ProofBackend, StaleAssessment
from coordinator.langgraph.sdk_backend import CoordinationReadLimitExceeded, SDKBackend


def records(proof="p1"):
    def item(kind, local, label, fields):
        return CoordinationItem(kind, proof + "/" + local, label, fields)
    return (
        item("node", "proof", "Proof", {}),
        item("node", "c-1", "Claim", {"status": "provisional"}),
        item("node", "rs-1", "ResearchState", {"role": "coordinator", "status": "open"}),
        item("node", "rm-1", "ResearchMove", {"status": "queued", "escalation": "five-mechanisms",
            "target_id": proof, "basis": "0/0/0", "round": 1}),
        item("node", "fs-1", "FormalState", {"status": "open", "exact_state_hash": "same"}),
        item("node", "fs-2", "FormalState", {"status": "open", "exact_state_hash": "same"}),
        item("edge", "target", "HAS_TARGET", {"src_id": proof + "/proof", "dst_id": proof + "/c-1"}),
        item("edge", "proposed", "PROPOSES", {"src_id": proof + "/rs-1", "dst_id": proof + "/rm-1",
                                                 "nested": {"values": [1, 2]}}),
        item("frontier", "rm-1", "research-move", {"detail": "investigate", "features": {"novelty": .5},
                                                   "conflict_ids": [proof + "/rm-1"]}),
        item("frontier", "c-1", "critic-task", {"detail": "review", "features": {}, "conflict_ids": []}),
    )


class ScriptedPages:
    """Read-only ReceiverPort fixture, explicitly labeled scripted-test."""
    def __init__(self):
        self.calls, self.submissions = [], []
        self.items = records() if SDK_AVAILABLE else ()
        self.read_hook = None
        self.allow_read = True

    async def capabilities(self, *, contract_version="2.0"):
        return CapabilitySet(contract_version, frozenset(),
                             frozenset({"coordination-read"}) if self.allow_read else frozenset(),
                             "scripted-test", False, "scripted-test")

    def page(self, workspace_id, proof_id, cursor, page_size, *, revision=1):
        offset = int(cursor or 0)
        selected = self.items[offset:offset + page_size]
        end = offset + len(selected)
        return CoordinationPage(workspace_id, proof_id, revision, "sha256:" + str(revision) * 64,
            revision, selected, len(self.items), str(end) if end < len(self.items) else None, "scripted-test")

    async def read_coordination(self, *, workspace_id, proof_id, cursor=None, page_size=100):
        self.calls.append((workspace_id, proof_id, cursor, page_size))
        if self.read_hook:
            return await self.read_hook(workspace_id, proof_id, cursor, page_size)
        return self.page(workspace_id, proof_id, cursor, page_size)

    async def submit(self, request):
        self.submissions.append(request)
        raise AssertionError("read adapter must never submit an operation")


@unittest.skipUnless(SDK_AVAILABLE, "optional persistent-atp-client distribution is unavailable")
class TestSDKReads(unittest.TestCase):
    def setUp(self):
        self.port = ScriptedPages()
        self.client = AsyncClient(self.port)

    def backend(self, **options):
        return SDKBackend(self.client, workspace_id="lab", proof_ids=("p1",), page_size=2, **options)

    def test_complete_snapshot_preserves_records_progress_metadata_and_input_ownership(self):
        backend = self.backend()
        self.assertIsInstance(backend, ProofBackend)
        state = backend.read_coordination("p1")
        self.assertEqual((len(state.nodes), len(state.edges), len(state.frontier)), (6, 2, 2))
        self.assertEqual(state.progress.target_claims, 1)
        self.assertFalse(state.progress.solved)
        self.assertEqual(state.escalations[0].node_id, "p1/rm-1")
        self.assertEqual(state.component_mode, "scripted-test")
        self.assertEqual(state.to_dict()["component_mode"], "scripted-test")
        self.assertEqual(len([n for n in state.nodes if n.label == "FormalState"]), 2)
        self.assertEqual([c[2] for c in self.port.calls], [None, "2", "4", "6", "8"])
        self.port.items[7].fields["nested"]["values"].append(99)
        self.assertEqual(state.edges[1].fields["nested"]["values"], (1, 2))
        with self.assertRaises(TypeError):
            state.nodes[0].fields["extra"] = True
        json.dumps(state.to_dict())
        self.assertEqual(self.port.submissions, [])

    def test_async_read_and_sync_call_inside_loop_are_explicit(self):
        async def scenario():
            backend = self.backend()
            with self.assertRaisesRegex(RuntimeError, "aread_coordination"):
                backend.read_coordination("p1")
            return await backend.aread_coordination("p1")
        self.assertEqual(asyncio.run(scenario()).proof_id, "p1")

    def test_parallel_proofs_use_isolated_reads_on_one_client(self):
        barrier = threading.Barrier(2, timeout=5)
        class ParallelPages(ScriptedPages):
            async def read_coordination(self, *, workspace_id, proof_id, cursor=None, page_size=100):
                if cursor is None:
                    barrier.wait()
                source = ScriptedPages()
                source.items = records(proof_id)
                return source.page(workspace_id, proof_id, cursor, page_size)
        backend = SDKBackend(AsyncClient(ParallelPages()), workspace_id="lab", proof_ids=("p1", "p2"), page_size=2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            states = list(pool.map(backend.read_coordination, ("p1", "p2")))
        self.assertEqual([s.proof_id for s in states], ["p1", "p2"])
        for state in states:
            self.assertTrue(all(n.node_id.startswith(state.proof_id + "/") for n in state.nodes))

    def test_read_capability_and_registration_fail_before_pages_or_mutations(self):
        self.port.allow_read = False
        with self.assertRaises(UnsupportedCapability):
            self.backend().read_coordination("p1")
        with self.assertRaises(KeyError):
            self.backend().read_coordination("other")
        self.assertEqual(self.port.calls, [])

    def test_changed_revision_discards_partial_pages_and_restarts_from_zero(self):
        async def moving(workspace, proof, cursor, size):
            revision = 1 if len(self.port.calls) == 1 else 2
            return self.port.page(workspace, proof, cursor, size, revision=revision)
        self.port.read_hook = moving
        state = self.backend().read_coordination("p1")
        self.assertEqual(state.revision, 2)
        self.assertEqual(sum(c[2] is None for c in self.port.calls), 2)
        self.assertEqual(len(state.nodes) + len(state.edges) + len(state.frontier), 10)

    def test_stale_cursor_retries_are_bounded(self):
        async def stale(*args):
            raise StalePrecondition("scripted cursor expiry")
        self.port.read_hook = stale
        with self.assertRaisesRegex(StaleAssessment, "retry bound"):
            self.backend(read_attempts=3).read_coordination("p1")
        self.assertEqual(len(self.port.calls), 3)
        self.assertTrue(all(c[2] is None for c in self.port.calls))

    def test_projection_lag_never_publishes_a_partial_snapshot(self):
        async def lagging(workspace, proof, cursor, size):
            return replace(self.port.page(workspace, proof, cursor, size), projection_watermark=0)
        self.port.read_hook = lagging
        with self.assertRaises(StaleAssessment):
            self.backend(read_attempts=2).read_coordination("p1")
        self.assertEqual(len(self.port.calls), 2)

    def test_scope_hash_count_and_mode_changes_fail_closed(self):
        for changes in ({"proof_id": "other"}, {"workspace_id": "other"},
                        {"journal_hash": "sha256:" + "f" * 64},
                        {"total_items": 11}, {"component_mode": "different-mode"}):
            with self.subTest(changes=changes):
                port = ScriptedPages()
                async def corrupt(workspace, proof, cursor, size):
                    page = port.page(workspace, proof, cursor, size)
                    return replace(page, **changes) if cursor else page
                port.read_hook = corrupt
                backend = SDKBackend(AsyncClient(port), workspace_id="lab", proof_ids=("p1",), page_size=2)
                with self.assertRaises(ReceiverProtocolError):
                    backend.read_coordination("p1")
                self.assertEqual(len(port.calls), 2)

    def test_duplicate_records_empty_pages_and_repeated_cursors_cannot_loop(self):
        for fault in ("duplicate", "empty", "cursor", "incomplete"):
            with self.subTest(fault=fault):
                port = ScriptedPages()
                async def broken(workspace, proof, cursor, size):
                    page = port.page(workspace, proof, cursor, size)
                    if not cursor:
                        return page
                    if fault == "duplicate":
                        return replace(page, items=port.items[:2])
                    if fault == "empty":
                        return replace(page, items=())
                    if fault == "cursor":
                        return replace(page, next_cursor=cursor)
                    return replace(page, next_cursor=None)
                port.read_hook = broken
                backend = SDKBackend(AsyncClient(port), workspace_id="lab", proof_ids=("p1",), page_size=2)
                with self.assertRaises(ReceiverProtocolError):
                    backend.read_coordination("p1")
                self.assertEqual(len(port.calls), 2)

    def test_snapshot_size_page_and_byte_limits_reject_instead_of_truncating(self):
        for limits in ({"max_items": 9}, {"max_pages": 2}, {"max_bytes": 10}):
            with self.subTest(limits=limits), self.assertRaises(CoordinationReadLimitExceeded):
                self.backend(**limits).read_coordination("p1")

    def test_deadline_cancels_only_the_read_and_does_not_leave_a_running_read(self):
        cancelled = []
        async def hanging(*args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)
        self.port.read_hook = hanging
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            self.backend(read_timeout=.03).read_coordination("p1")
        self.assertLess(time.monotonic() - start, 1)
        self.assertEqual(cancelled, [True])
        self.assertEqual(self.port.submissions, [])

    def test_malformed_frontier_foreign_records_and_dangling_edges_are_rejected(self):
        variants = (
            replace(self.port.items[0], record_id="other/proof"),
            replace(self.port.items[6], fields={"src_id": "p1/missing", "dst_id": "p1/c-1"}),
            replace(self.port.items[-1], fields={"detail": "x", "features": {"x": "invalid"}, "conflict_ids": []}),
        )
        for bad in variants:
            with self.subTest(bad=bad.record_id):
                port = ScriptedPages()
                port.items = (bad,)
                backend = SDKBackend(AsyncClient(port), workspace_id="lab", proof_ids=("p1",))
                with self.assertRaises(ReceiverProtocolError):
                    backend.read_coordination("p1")

    def test_nonfinite_json_is_not_accepted_as_graph_data(self):
        self.port.items[0].fields["bad_score"] = float("nan")
        with self.assertRaisesRegex(ReceiverProtocolError, "finite JSON"):
            self.backend().read_coordination("p1")

    def test_mutations_fail_closed_without_submitting_generic_guidance_or_a_cycle(self):
        backend = self.backend()
        with self.assertRaisesRegex(UnsupportedCapability, "execution-time revision"):
            backend.run_slot("p1", "critic-task", expected_revision=1)
        with self.assertRaisesRegex(UnsupportedCapability, "structured atomic"):
            backend.queue_escalations("p1", (), expected_revision=1, round_number=1)
        self.assertEqual(self.port.calls, [])
        self.assertEqual(self.port.submissions, [])

    def test_invalid_configuration_is_rejected_before_network_access(self):
        for options in ({"max_pages": 0}, {"max_items": 0}, {"max_bytes": 0},
                        {"read_attempts": 0}, {"read_timeout": float("inf")}, {"read_timeout": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.backend(**options)
        with self.assertRaises(ValueError):
            SDKBackend(self.client, workspace_id="lab", proof_ids="p1")
        with self.assertRaises(ValueError):
            SDKBackend(self.client, workspace_id="lab", proof_ids=("p1", "p1"))
        self.assertEqual(self.port.calls, [])


class TestOptionalSDKDependency(unittest.TestCase):
    def test_core_and_binding_module_import_without_the_sdk_and_report_missing_dependency(self):
        source = str(Path(__file__).resolve().parents[3])
        code = """
import importlib.abc
import sys
class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'persistent_atp_client':
            raise ModuleNotFoundError('SDK deliberately unavailable')
sys.meta_path.insert(0, BlockSDK())
sys.path.insert(0, sys.argv[1])
from coordinator.langgraph import CoordinatorRuntime
from coordinator.langgraph.sdk_backend import SDKBackend
from coordinator.langgraph.sdk_operations import DurableSDKOperations
try:
    SDKBackend(None, workspace_id='lab', proof_ids=('p1',))
except ImportError as exc:
    assert 'requires persistent-atp-client 0.2' in str(exc)
else:
    raise AssertionError('missing dependency did not block adapter construction')
try:
    DurableSDKOperations(None, '/unused/cache.sqlite3', receiver_id='test', workspace_id='lab', proof_ids=('p1',))
except ImportError as exc:
    assert 'requires persistent-atp-client 0.2' in str(exc)
else:
    raise AssertionError('missing dependency did not block durable operations')
"""
        result = subprocess.run([sys.executable, "-I", "-c", code, source],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
