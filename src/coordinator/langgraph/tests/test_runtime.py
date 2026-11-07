"""Calling-boundary tests with a real gate/journal and explicitly scripted workers."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from commit_gate.ops import UpsertNode, AddEdge
from coordinator.langgraph import CoordinatorRuntime, StaleAssessment
from coordinator.langgraph.policy import Escalation, GLOBAL_AUDIT, REPAIR
from coordinator.langgraph.rounds import work, assess, new_track
from coordinator.langgraph.portfolio import run_portfolio
from mathproof.scheduler import GlobalScheduler
from .fixtures import JournalCase, declaration, aligned_claim, critic, explorer


class TestRuntime(JournalCase):
    def runtime(self, proof_id="p1", **options):
        return CoordinatorRuntime([self.workspace(proof_id, **options)])

    def test_registered_proofs_and_categories_are_validated(self):
        workspace = self.workspace("p1")
        with self.assertRaises(ValueError):
            CoordinatorRuntime([workspace, self.workspace("p1")])
        runtime = CoordinatorRuntime([workspace])
        with self.assertRaises(KeyError):
            runtime.read_coordination("p2")
        with self.assertRaises(KeyError):
            runtime.read_coordination("p1")
        with self.assertRaises(ValueError):
            runtime.run_slot("p1", "shell", expected_revision=0)

    def test_read_is_complete_scoped_serializable_and_deeply_immutable(self):
        self.seed("p1", declaration("p1"), *aligned_claim("p1"),
            UpsertNode("ResearchState", "p1/rs-1", {"status": "open", "notes": {"values": [1, 2]}}))
        self.seed("p2", declaration("p2"))
        runtime = self.runtime()
        state = runtime.read_coordination("p1")
        self.assertEqual(state.revision, 1)
        self.assertEqual(state.journal_hash, self.open_store().head("p1")[1])
        self.assertTrue(all(node.node_id.startswith("p1/") for node in state.nodes))
        self.assertEqual({item.category for item in state.frontier}, {"formal-run", "critic-task"})
        node = next(node for node in state.nodes if node.node_id == "p1/rs-1")
        with self.assertRaises(TypeError):
            node.fields["notes"]["values"][0] = 99
        json.dumps(state.to_dict())
        self.assertEqual(self.open_store().head("p1")[0], 1)

    def test_read_retries_a_race_and_never_returns_a_mixed_revision(self):
        self.seed("p1", declaration("p1"))
        runtime = self.runtime()
        original = GlobalScheduler.build_frontier
        calls = []
        def moving(scheduler, proof_id):
            result = original(scheduler, proof_id)
            calls.append(1)
            if len(calls) == 1:
                self.seed("p1", UpsertNode("ResearchState", "p1/rs-1", {"status": "open"}))
            return result
        with patch.object(GlobalScheduler, "build_frontier", moving):
            state = runtime.read_coordination("p1")
        self.assertEqual(len(calls), 2)
        self.assertEqual(state.revision, 2)
        self.assertIn("p1/rs-1", {node.node_id for node in state.nodes})

    def test_read_has_a_bounded_retry_limit(self):
        self.seed("p1", declaration("p1"))
        runtime = self.runtime()
        original = GlobalScheduler.build_frontier
        calls = []
        def moving(scheduler, proof_id):
            result = original(scheduler, proof_id)
            calls.append(1)
            self.seed("p1", UpsertNode("ResearchState", f"p1/rs-{len(calls)}", {"status": "open"}))
            return result
        with patch.object(GlobalScheduler, "build_frontier", moving):
            with self.assertRaises(StaleAssessment):
                runtime.read_coordination("p1")
        self.assertEqual(len(calls), 3)

    def test_selected_slot_is_category_bound_and_reports_idle(self):
        self.seed("p1", declaration("p1"), *aligned_claim("p1"))
        runtime = self.runtime(handlers={"critic": critic()})
        result = runtime.run_slot("p1", "critic-task", expected_revision=1)
        self.assertEqual((result["outcome"], result["move_id"], result["accepted"]),
                         ("accepted", "p1/c-1", True))
        self.assertEqual(self.view_of("p1").node("p1/fd-1").fields["status"], "aligned")
        idle = runtime.run_slot("p1", "critic-task", expected_revision=2)
        self.assertEqual(idle["outcome"], "idle")
        with self.assertRaises(StaleAssessment):
            runtime.run_slot("p1", "formal-run", expected_revision=1)

    def test_slot_race_before_selection_cannot_silently_rebase(self):
        self.seed("p1", *aligned_claim("p1"))
        runtime = self.runtime(handlers={"critic": critic()})
        def racing(*args, **kwargs):
            self.seed("p1", UpsertNode("ResearchState", "p1/rs-1", {"status": "open"}))
            return run_portfolio(*args, **kwargs)
        with patch("coordinator.langgraph.runtime.run_portfolio", racing):
            result = runtime.run_slot("p1", "critic-task", expected_revision=1)
        self.assertEqual(result["outcome"], "stale-assessment")
        self.assertIsNone(result["move_id"])
        self.assertEqual(self.view_of("p1").node("p1/c-1").fields["status"], "provisional")

    def test_unknown_execution_is_not_retried_or_reported_as_rollback(self):
        self.seed("p1", *aligned_claim("p1"))
        calls = []
        def timeout(lease, packet):
            calls.append(lease)
            raise TimeoutError("worker acknowledgement lost")
        runtime = self.runtime(handlers={"critic": timeout})
        result = runtime.run_slot("p1", "critic-task", expected_revision=1)
        self.assertEqual(result["outcome"], "execution-unknown")
        self.assertFalse(result["accepted"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.open_store().head("p1")[0], 1)
        self.assertEqual(len(self.open_store().active_leases("p1")), 1)

    def test_batch_preserves_claim_links_and_restarts_deduplicate_from_journal(self):
        self.seed("p1", *aligned_claim("p1"),
            UpsertNode("Attempt", "p1/at-1", {"worker_class": "critic", "status": "refuted"}),
            AddEdge("REVIEWS_CLAIM", "p1/at-1", "p1/c-1", "p1/reviews"))
        items = [Escalation(REPAIR, "p1/at-1", "repair the base case", "p1/c-1"),
                 Escalation(GLOBAL_AUDIT, "p1", "audit")]
        result = self.runtime().queue_escalations("p1", items, expected_revision=1, round_number=7)
        self.assertEqual(result["outcome"], "accepted")
        self.assertEqual(self.open_store().head("p1")[0], 2)
        repair = next(move for move in self.moves("p1") if move.fields["escalation"] == REPAIR)
        self.assertEqual((repair.fields["round"], repair.fields["basis"], repair.fields["target_id"]),
                         (7, "0/0/0", "p1/at-1"))
        self.assertEqual(self.view_of("p1").edges_from(repair.node_id, "MOVE_TARGETS")[0].dst_id, "p1/c-1")
        restarted = self.runtime()
        duplicate = restarted.queue_escalations("p1", items, expected_revision=2, round_number=8)
        self.assertEqual(duplicate["outcome"], "duplicate")
        self.assertEqual(len(restarted.read_escalations("p1")), 2)
        self.assertEqual(self.open_store().verify_chain("p1"), 2)

    def test_invalid_batch_and_stale_revision_cannot_partially_commit(self):
        self.seed("p1", declaration("p1"))
        runtime = self.runtime()
        with self.assertRaises(ValueError):
            runtime.queue_escalations("p1", [Escalation(GLOBAL_AUDIT, "p1", "audit"),
                Escalation(REPAIR, "p2/at-1", "wrong proof")], expected_revision=1, round_number=1)
        self.assertEqual(self.open_store().verify_chain("p1"), 1)
        with self.assertRaises(StaleAssessment):
            runtime.queue_escalations("p1", [Escalation(GLOBAL_AUDIT, "p1", "audit")],
                expected_revision=0, round_number=1)
        self.assertEqual(self.moves("p1"), [])

    def test_lease_supersession_is_visible_in_the_call_result(self):
        self.seed("p1", *aligned_claim("p1"))
        store = self.open_store()
        store.issue_lease("p1", "p1/inflight", actor="critic", worker_class="critic",
                          selected_move_id="p1/c-1", base_revision=1)
        result = self.runtime().queue_escalations("p1", [Escalation(GLOBAL_AUDIT, "p1", "audit")],
            expected_revision=1, round_number=1)
        superseded = [event["lease_id"] for event in result["lease_events"] if event["event"] == "superseded"]
        self.assertEqual(superseded, ["p1/inflight"])
        self.assertEqual(result["outcome"], "accepted")

    def test_facade_round_matches_existing_direct_calls(self):
        for proof_id in ("p1", "p2"):
            self.seed(proof_id, declaration(proof_id))
        runtime = self.runtime(handlers={"llm-research": explorer()}, plan={"p1/fr-1": ["stagnated"]})
        via_runtime = runtime.run_round("p1", 1, new_track())
        workspace = self.workspace("p2", handlers={"llm-research": explorer()}, plan={"p2/fr-1": ["stagnated"]})
        report = work(workspace, 1)
        track, raised, messages = assess(workspace, new_track(), report)
        self.assertEqual(via_runtime["track"], track)
        self.assertEqual([item["rule"] for item in via_runtime["escalations"]], [item["rule"] for item in raised])
        self.assertEqual(len(self.coordinator_events("p1")), len(self.coordinator_events("p2")))
        json.dumps(via_runtime)

    def test_independent_proofs_execute_concurrently(self):
        barrier = threading.Barrier(2, timeout=5)
        def together(lease, packet):
            barrier.wait()
            return critic()(lease, packet)
        for proof_id in ("p1", "p2"):
            self.seed(proof_id, *aligned_claim(proof_id))
        runtime = CoordinatorRuntime([self.workspace(p, handlers={"critic": together}) for p in ("p1", "p2")])
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda p: runtime.run_slot(p, "critic-task", expected_revision=1), ("p1", "p2")))
        self.assertTrue(all(result["accepted"] for result in results))

    def test_same_proof_calls_share_one_lock_and_reject_the_stale_second_call(self):
        self.seed("p1", *aligned_claim("p1"))
        calls = []
        start = threading.Barrier(2, timeout=5)
        def handler(lease, packet):
            calls.append(lease)
            return critic()(lease, packet)
        runtime = self.runtime(handlers={"critic": handler})
        def invoke(_):
            start.wait()
            try:
                return runtime.run_slot("p1", "critic-task", expected_revision=1)["outcome"]
            except StaleAssessment:
                return "stale"
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(invoke, range(2)))
        self.assertEqual(sorted(outcomes), ["accepted", "stale"])
        self.assertEqual(len(calls), 1)

    def test_unknown_round_waits_without_escalating_or_preempting_worker(self):
        self.seed("p1", *aligned_claim("p1"))
        def timeout(lease, packet):
            raise TimeoutError("lost acknowledgement")
        runtime = self.runtime(handlers={"critic": timeout})
        result = runtime.run_round("p1", 1, {**new_track(), "stalled_rounds": 10})
        self.assertEqual(result["track"]["status"], "waiting-on-worker")
        self.assertEqual(result["escalations"], [])
        self.assertEqual(self.open_store().head("p1")[0], 1)
        self.assertEqual(len(self.open_store().active_leases("p1")), 1)
