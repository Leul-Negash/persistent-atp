"""Durable client identity tests with the actual SDK and scripted-test receiver.

SQLite here is a client intent cache; it is not the trusted service journal
or operation ledger. These tests execute no ATP or Lean replay.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

try:
    from persistent_atp_client import AsyncClient, Admission, OperationRef, Preconditions, ProofAudit
    from persistent_atp_client.errors import AdmissionRejected, ReceiverProtocolError
    from persistent_atp_client.testing import ScriptedReceiverPort
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    ScriptedReceiverPort = object

from coordinator.langgraph.sdk_operations import DurableSDKOperations, IntentConflict, OperationPending


class AutoFinishPort(ScriptedReceiverPort):
    def __init__(self):
        super().__init__()
        self.submits, self.observes = [], []
        self.auto_finish = True

    async def submit(self, request):
        self.submits.append(request.to_dict())
        receipt = await super().submit(request)
        if self.auto_finish and self._operations[receipt.operation_ref].lifecycle == "queued":
            self.finish(receipt.operation_ref, mathematical_disposition="not-verified")
        return receipt

    async def get_operation(self, ref):
        self.observes.append(ref)
        return await super().get_operation(ref)


@unittest.skipUnless(SDK_AVAILABLE, "optional persistent-atp-client distribution is unavailable")
class TestDurableSDKOperations(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "client-intents.sqlite3"
        self.port = AutoFinishPort()
        self.client = AsyncClient(self.port)

    def controller(self, **options):
        return DurableSDKOperations(self.client, self.path, receiver_id="receiver-one", workspace_id="lab",
                                     proof_ids=("p1",), poll_interval=.005, **options)

    def prepared(self, controller=None):
        return (controller or self.controller()).prepare("audit-round-one", ProofAudit("p1"), request_id="audit-one")

    def test_prepared_request_is_durable_before_submission_and_reused_on_restart(self):
        first = self.controller()
        deadline = datetime.now(timezone.utc) + timedelta(minutes=1)
        intent = first.prepare("audit", ProofAudit("p1"), preconditions=Preconditions(expected_revision=7), deadline=deadline)
        reopened = self.controller().prepare("audit", ProofAudit("p1"),
            preconditions=Preconditions(expected_revision=7), deadline=deadline)
        self.assertEqual(intent, reopened)
        self.assertEqual(intent.state, "prepared")
        self.assertEqual(self.port.submits, [])
        checkpoint = json.loads(json.dumps(intent.checkpoint_ref()))
        self.assertEqual(self.controller().restore(checkpoint).request_id, intent.request_id)
        self.assertNotIn("request_json", checkpoint)

    def test_changed_intent_or_duplicate_request_identity_cannot_replace_original_bytes(self):
        controller = self.controller()
        original = self.prepared(controller)
        with self.assertRaises(IntentConflict):
            controller.prepare(original.intent_id, ProofAudit("p1"), preconditions=Preconditions(expected_revision=2))
        with self.assertRaises(IntentConflict):
            controller.prepare("another-intent", ProofAudit("p1"), request_id=original.request_id)
        self.assertEqual(controller.get(original.intent_id), original)

    def test_terminal_operation_reattaches_after_restart_without_resubmission(self):
        controller = self.controller()
        intent = self.prepared(controller)
        result = asyncio.run(controller.advance(intent.intent_id, timeout=1))
        self.assertEqual(result.lifecycle, "succeeded")
        self.assertEqual(result.mathematical_disposition, "not-verified")
        cached = controller.get(intent.intent_id)
        self.assertEqual(cached.state, "terminal")
        reopened = self.controller()
        self.assertEqual(reopened.restore(cached.checkpoint_ref()).fingerprint, intent.fingerprint)
        again = asyncio.run(reopened.advance(intent.intent_id, timeout=1))
        self.assertEqual(again, result)
        self.assertEqual(len(self.port.submits), 1)
        self.assertEqual(len(self.port.observes), 2)

    def test_lost_acknowledgement_recovers_by_original_request_lookup_without_resend(self):
        original_submit = self.port.submit
        original_lookup = self.port.lookup_request
        async def lost_ack(request):
            await original_submit(request)
            raise TimeoutError("scripted acknowledgement loss")
        async def unavailable(*args):
            raise ConnectionError("scripted lookup outage")
        self.port.submit, self.port.lookup_request = lost_ack, unavailable
        controller = self.controller()
        intent = self.prepared(controller)
        with self.assertRaises(OperationPending) as error:
            asyncio.run(controller.advance(intent.intent_id, timeout=1))
        self.assertEqual(controller.get(intent.intent_id).state, "uncertain")
        self.port.lookup_request = original_lookup
        recovered = self.controller()
        recovered.restore(error.exception.checkpoint_ref)
        outcome = asyncio.run(recovered.advance(intent.intent_id, timeout=1))
        self.assertEqual(outcome.request_id, intent.request_id)
        self.assertEqual(len(self.port.submits), 1)

    def test_not_found_admission_does_not_silently_resubmit_but_explicit_retry_preserves_body(self):
        original_submit = self.port.submit
        sent = []
        async def lost_before_delivery(request):
            sent.append(request.to_dict())
            raise TimeoutError("scripted failure before admission")
        self.port.submit = lost_before_delivery
        controller = self.controller()
        intent = self.prepared(controller)
        with self.assertRaises(OperationPending):
            asyncio.run(controller.advance(intent.intent_id, timeout=1))
        with self.assertRaises(OperationPending):
            asyncio.run(self.controller().advance(intent.intent_id, timeout=1))
        self.assertEqual(len(sent), 1)
        self.port.submit = original_submit
        result = asyncio.run(self.controller().advance(intent.intent_id, timeout=1, retry_same_request=True))
        self.assertEqual(result.request_id, intent.request_id)
        self.assertEqual(sent[0], self.port.submits[0])

    def test_observation_timeout_keeps_reference_and_running_work_for_reattachment(self):
        self.port.auto_finish = False
        controller = self.controller()
        intent = self.prepared(controller)
        with self.assertRaises(OperationPending):
            asyncio.run(controller.advance(intent.intent_id, timeout=.02))
        saved = controller.get(intent.intent_id)
        self.assertEqual(saved.state, "admitted")
        ref = OperationRef.from_dict(json.loads(saved.operation_json))
        self.assertEqual(self.port._operations[ref].lifecycle, "queued")
        self.port.finish(ref, mathematical_disposition="not-verified")
        result = asyncio.run(self.controller().advance(intent.intent_id, timeout=1))
        self.assertEqual(result.ref, ref)
        self.assertEqual(len(self.port.submits), 1)

    def test_process_interruption_after_admission_does_not_lose_reference(self):
        original = self.port.get_operation
        async def interrupted(ref):
            raise asyncio.CancelledError("scripted client process interruption")
        self.port.get_operation = interrupted
        controller = self.controller()
        intent = self.prepared(controller)
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(controller.advance(intent.intent_id, timeout=1))
        self.assertEqual(controller.get(intent.intent_id).state, "admitted")
        self.port.get_operation = original
        self.assertEqual(asyncio.run(self.controller().advance(intent.intent_id, timeout=1)).lifecycle, "succeeded")
        self.assertEqual(len(self.port.submits), 1)

    def test_conflicting_lookup_fingerprint_never_attaches_unrelated_work(self):
        controller = self.controller()
        intent = self.prepared(controller)
        controller._claim_first_send(intent.intent_id)
        async def wrong(*args):
            return Admission(intent.request_id, "sha256:" + "f" * 64, True, OperationRef("lab", "unrelated"))
        self.port.lookup_request = wrong
        with self.assertRaises(ReceiverProtocolError):
            asyncio.run(controller.advance(intent.intent_id, timeout=1))
        self.assertEqual(self.port.observes, [])
        self.assertEqual(self.port.submits, [])

    def test_confirmed_rejection_is_remembered_and_never_refreshes_preconditions(self):
        controller = self.controller()
        intent = self.prepared(controller)
        async def rejected(request):
            return Admission(request.request_id, intent.fingerprint, False, error_code="STALE_PRECONDITION")
        self.port.submit = rejected
        for _ in range(2):
            with self.assertRaises(AdmissionRejected):
                asyncio.run(self.controller().advance(intent.intent_id, timeout=1))
        self.assertEqual(self.controller().get(intent.intent_id).state, "rejected")
        with self.assertRaises(IntentConflict):
            controller.prepare(intent.intent_id, ProofAudit("p1"), preconditions=Preconditions(expected_revision=2))
        fresh = controller.prepare("new-assessed-intent", ProofAudit("p1"), preconditions=Preconditions(expected_revision=2))
        self.assertNotEqual(fresh.request_id, intent.request_id)

    def test_cached_observation_versions_and_terminal_results_cannot_regress(self):
        controller = self.controller()
        intent = self.prepared(controller)
        final = asyncio.run(controller.advance(intent.intent_id, timeout=1))
        for invalid in (replace(final, observation_version=0, lifecycle="queued"),
                        replace(final, result={"different": True}),
                        replace(final, observation_version=2, lifecycle="running")):
            with self.subTest(version=invalid.observation_version, lifecycle=invalid.lifecycle):
                self.port._operations[final.ref] = invalid
                with self.assertRaises(ReceiverProtocolError):
                    asyncio.run(self.controller().advance(intent.intent_id, timeout=1))
        self.assertEqual(controller.get(intent.intent_id).state, "terminal")

    def test_operation_request_and_proof_scope_are_checked_after_reattachment(self):
        controller = self.controller()
        intent = self.prepared(controller)
        final = asyncio.run(controller.advance(intent.intent_id, timeout=1))
        for changes in ({"request_id": "other-request"}, {"proof_id": "other-proof"}):
            with self.subTest(changes=changes):
                self.port._operations[final.ref] = replace(final, **changes)
                with self.assertRaises(ReceiverProtocolError):
                    asyncio.run(self.controller().advance(intent.intent_id, timeout=1))

    def test_concurrent_first_send_claims_are_atomic_between_cache_instances(self):
        controllers = self.controller(), self.controller()
        intent = self.prepared(controllers[0])
        barrier = threading.Barrier(2, timeout=5)
        def claim(controller):
            barrier.wait()
            return controller._claim_first_send(intent.intent_id)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(claim, controllers))
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(self.port.submits, [])

    def test_corrupted_cached_request_fails_closed_instead_of_being_replaced(self):
        controller = self.controller()
        intent = self.prepared(controller)
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE coordinator_sdk_intents SET request_json=?", (b"{}",))
        with self.assertRaises(IntentConflict):
            controller.get(intent.intent_id)
        self.assertEqual(self.port.submits, [])

    def test_checkpoint_scope_and_missing_cache_cannot_create_new_work(self):
        controller = self.controller()
        checkpoint = self.prepared(controller).checkpoint_ref()
        with self.assertRaises(IntentConflict):
            controller.restore({**checkpoint, "receiver_id": "other-receiver"})
        with self.assertRaises(KeyError):
            controller.restore({**checkpoint, "intent_id": "unknown"})
        self.assertEqual(self.port.submits, [])

    def test_observation_size_bound_does_not_discard_durable_request_identity(self):
        controller = self.controller(max_observation_bytes=10)
        intent = self.prepared(controller)
        with self.assertRaisesRegex(ValueError, "byte bound"):
            asyncio.run(controller.advance(intent.intent_id, timeout=1))
        self.assertEqual(controller.get(intent.intent_id).state, "admitted")
        self.assertEqual(asyncio.run(self.controller().advance(intent.intent_id, timeout=1)).request_id, intent.request_id)
        self.assertEqual(len(self.port.submits), 1)

    def test_invalid_retry_flag_never_turns_a_string_into_resend_permission(self):
        controller = self.controller()
        self.prepared(controller)
        with self.assertRaises(ValueError):
            asyncio.run(controller.advance("audit-round-one", retry_same_request="false"))
        self.assertEqual(self.port.submits, [])

    def test_memory_cache_and_unknown_schema_cannot_silently_replace_durable_state(self):
        with self.assertRaisesRegex(ValueError, "persistent file"):
            DurableSDKOperations(self.client, ":memory:", receiver_id="receiver-one", workspace_id="lab", proof_ids=("p1",))
        self.prepared()
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE coordinator_sdk_cache_meta SET version=99")
        with self.assertRaisesRegex(ValueError, "unsupported SDK client cache schema"):
            self.controller()

    def test_journal_identity_is_checked_across_nonterminal_observation_restart(self):
        controller = self.controller()
        intent = self.prepared(controller)
        self.port.auto_finish = False
        with self.assertRaises(OperationPending):
            asyncio.run(controller.advance(intent.intent_id, timeout=.02))
        ref = OperationRef.from_dict(json.loads(controller.get(intent.intent_id).operation_json))
        queued = self.port._operations[ref]
        first = replace(queued, observation_version=1, project_revision=7, journal_hash="sha256:" + "a" * 64)
        controller._observe(intent.intent_id, first)
        self.port._operations[ref] = replace(first, observation_version=2, journal_hash="sha256:" + "b" * 64)
        with self.assertRaisesRegex(ReceiverProtocolError, "journal revision or identity"):
            asyncio.run(self.controller().advance(intent.intent_id, timeout=1))

    def test_terminal_operation_accepts_live_project_progress_without_changing_its_result(self):
        controller = self.controller()
        intent = self.prepared(controller)
        terminal = asyncio.run(controller.advance(intent.intent_id, timeout=1))
        first = replace(terminal, project_revision=1, journal_hash="sha256:" + "a" * 64, projection_watermark=1)
        self.port._operations[terminal.ref] = first
        asyncio.run(self.controller().advance(intent.intent_id, timeout=1))
        later = replace(first, project_revision=2, journal_hash="sha256:" + "b" * 64, projection_watermark=2)
        self.port._operations[terminal.ref] = later
        observed = asyncio.run(self.controller().advance(intent.intent_id, timeout=1))
        self.assertEqual(observed.result, terminal.result)
        self.assertEqual(observed.project_revision, 2)
        self.assertEqual(len(self.port.submits), 1)
