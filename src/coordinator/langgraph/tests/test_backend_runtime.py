"""Backend rounds with the real gate/SQLite, FakeFormalATP and scripted workers.

Fault receipts below are explicitly scripted contract fixtures, not remote
service executions or independent Lean replay.
"""
from contextlib import closing
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock

from commit_gate.ops import AddEdge, UpsertNode
from commit_gate.state import EdgeRecord, NodeRecord
from coordinator.langgraph import (
    BackendRoundRuntime, CoordinationState, CoordinatorRuntime,
    ProofReading, RoundRuntime, StaleAssessment,
)
from coordinator.langgraph.policy import RULES, classify
from coordinator.langgraph.progress import read_proof
from coordinator.langgraph.rounds import ACTIVE, STALLED, WAITING, assess, new_track, work
from coordinator.langgraph.snapshot import SnapshotView

from .fixtures import JournalCase, aligned_claim, critic, declaration, explorer, verified_target
from .test_policy import proof, stuck_run
from .test_rounds import candidate_certificate


class BackendOnly:
    """Expose exactly the backend contract, without a workspace/round API."""
    def __init__(self, local):
        self.proof_ids = local.proof_ids
        self.read_coordination = Mock(wraps=local.read_coordination)
        self.run_slot = Mock(wraps=local.run_slot)
        self.queue_escalations = Mock(wraps=local.queue_escalations)


class TestBackendRounds(JournalCase):
    def backend(self, **options):
        return BackendOnly(CoordinatorRuntime([self.workspace("p1", **options)]))

    def test_selected_slots_refresh_committed_revision_between_calls(self):
        self.seed("p1", declaration("p1"), *aligned_claim("p1"))
        backend = self.backend(handlers={"critic": critic()}, plan={"p1/fr-1": ["budget-exhausted"]})
        runtime = BackendRoundRuntime(backend)
        self.assertIsInstance(runtime, RoundRuntime)
        report = runtime.work("p1", 1)
        self.assertEqual([c.args for c in backend.run_slot.call_args_list],
                         [("p1", "formal-run"), ("p1", "critic-task")])
        self.assertEqual([c.kwargs["expected_revision"] for c in backend.run_slot.call_args_list], [1, 2])
        self.assertEqual([s["accepted"] for s in report["slots"]], [True, True])
        track, raised, _ = runtime.assess("p1", new_track(), report)
        self.assertEqual(track["progress"], [1, 0, 0])
        self.assertEqual(track["status"], ACTIVE)
        self.assertTrue(raised)
        json.dumps((report, track, raised))

    def test_backend_rounds_match_local_rounds_through_stagnation_and_restart(self):
        self.seed("p1", declaration("p1"))
        self.seed("p2", declaration("p2"))
        backend = self.backend(handlers={"llm-research": explorer()}, plan={"p1/fr-1": ["stagnated"]})
        runtime = BackendRoundRuntime(backend)
        local = self.workspace("p2", handlers={"llm-research": explorer()}, plan={"p2/fr-1": ["stagnated"]})
        left, right = new_track(), new_track()
        for n in range(1, 12):
            report = runtime.work("p1", n)
            left, raised, _ = runtime.assess("p1", left, report)
            right, direct, _ = assess(local, right, work(local, n))
            self.assertEqual(left, right)
            self.assertEqual([e["rule"] for e in raised], [e["rule"] for e in direct])
            if left["status"] != ACTIVE:
                break
        self.assertEqual(left["status"], STALLED)
        count = len(self.coordinator_events("p1"))
        # Restart with an empty projection and track; on-record identity wins.
        restarted = BackendRoundRuntime(self.backend())
        _, raised, _ = restarted.assess("p1", new_track(),
            {"proof_id": "p1", "round": 1, "before": [0, 0, 0], "slots": []})
        self.assertEqual(raised, [])
        self.assertEqual(len(self.coordinator_events("p1")), count)
        with closing(self.open_store()) as store:
            self.assertEqual(store.verify_chain("p1"), store.head("p1")[0])
        self.assertTrue(all(op.get("label") in {None, "ResearchState", "ResearchMove"}
                            for event in self.coordinator_events("p1") for op in event["ops"]))

    def test_repair_preserves_claim_target_and_policy_metadata(self):
        self.seed("p1", *aligned_claim("p1"),
            UpsertNode("Attempt", "p1/at-1", {"worker_class": "critic", "status": "refuted", "earliest_gap": "base case"}),
            AddEdge("REVIEWS_CLAIM", "p1/at-1", "p1/c-1", "p1/reviews"))
        runtime = BackendRoundRuntime(self.backend())
        _, raised, _ = runtime.assess("p1", new_track(),
            {"proof_id": "p1", "round": 7, "before": [0, 0, 0], "slots": []})
        self.assertEqual(raised[0]["rule"], "repair")
        (move,) = self.moves("p1", "repair")
        self.assertEqual((move.fields["round"], move.fields["basis"], move.fields["target_id"]),
                         (7, "0/0/0", "p1/at-1"))
        self.assertEqual(self.view_of("p1").edges_from(move.node_id, "MOVE_TARGETS")[0].dst_id, "p1/c-1")

    def test_stale_slot_defers_policy_without_retry_or_new_escalations(self):
        self.seed("p1", *aligned_claim("p1"))
        backend = self.backend()
        backend.run_slot.side_effect = StaleAssessment("scripted pre-dispatch race")
        runtime = BackendRoundRuntime(backend)
        report = runtime.work("p1", 1)
        track, raised, _ = runtime.assess("p1", {**new_track(), "stalled_rounds": 10}, report)
        self.assertEqual(track["status"], ACTIVE)
        self.assertEqual(track["stalled_rounds"], 10)
        self.assertEqual(raised, [])
        backend.run_slot.assert_called_once()
        backend.queue_escalations.assert_not_called()

    def test_unknown_worker_outcome_retains_lease_and_blocks_policy_writes(self):
        self.seed("p1", *aligned_claim("p1"))
        def timeout(lease, packet):
            raise TimeoutError("scripted lost acknowledgement")
        backend = self.backend(handlers={"critic": timeout})
        runtime = BackendRoundRuntime(backend)
        report = runtime.work("p1", 1)
        track, raised, _ = runtime.assess("p1", {**new_track(), "stalled_rounds": 10}, report)
        self.assertEqual(track["status"], WAITING)
        self.assertEqual(raised, [])
        backend.run_slot.assert_called_once()
        backend.queue_escalations.assert_not_called()
        with closing(self.open_store()) as store:
            self.assertEqual(len(store.active_leases("p1")), 1)
            self.assertEqual(store.head("p1")[0], 1)

    def test_stale_escalation_does_not_rebase_or_retry(self):
        self.seed("p1", declaration("p1"))
        backend = self.backend()
        original = backend.queue_escalations
        def race(*args, **options):
            self.seed("p1", UpsertNode("ResearchState", "p1/rs-race", {"status": "open"}))
            return original(*args, **options)
        backend.queue_escalations = Mock(side_effect=race)
        track, raised, _ = BackendRoundRuntime(backend).assess("p1", {**new_track(), "stalled_rounds": 4},
            {"proof_id": "p1", "round": 1, "before": [0, 0, 0], "slots": []})
        self.assertEqual(track["status"], ACTIVE)
        self.assertEqual(raised, [])
        backend.queue_escalations.assert_called_once()
        self.assertEqual(self.moves("p1"), [])

    def test_scripted_unknown_queue_receipt_waits_and_rejection_remains_active(self):
        self.seed("p1", declaration("p1"))
        for outcome, expected in (("execution-unknown", WAITING), ("rejected", ACTIVE)):
            with self.subTest(outcome=outcome):
                backend = self.backend()
                backend.queue_escalations.return_value = {"proof_id": "p1", "outcome": outcome,
                    "revision": 1, "commit": {"rejections": [{"reason": "scripted-rejection"}]}}
                track, raised, messages = BackendRoundRuntime(backend).assess("p1", {**new_track(), "stalled_rounds": 4},
                    {"proof_id": "p1", "round": 1, "before": [0, 0, 0], "slots": []})
                self.assertEqual(track["status"], expected)
                self.assertEqual(raised, [])
                self.assertTrue(any("unknown" in m or "scripted-rejection" in m for m in messages))
                backend.queue_escalations.assert_called_once()
        self.assertEqual(self.moves("p1"), [])

    def test_misleading_progress_and_foreign_proof_snapshots_fail_before_execution(self):
        self.seed("p1", declaration("p1"))
        backend = self.backend()
        state = backend.read_coordination("p1")
        for invalid in (replace(state, proof_id="p2"),
                        replace(state, progress=replace(state.progress, target_claims=1, verified_targets=1))):
            with self.subTest(invalid=invalid.proof_id):
                backend.read_coordination.return_value = invalid
                with self.assertRaises(ValueError):
                    BackendRoundRuntime(backend).work("p1", 1)
                backend.run_slot.assert_not_called()
                backend.queue_escalations.assert_not_called()

    def test_custom_worker_authority_is_rejected_before_backend_calls(self):
        from coordinator.langgraph.portfolio import Slot
        backend = self.backend()
        runtime = BackendRoundRuntime(backend)
        with self.assertRaisesRegex(ValueError, "registered portfolio"):
            runtime.work("p1", 1, slots=(Slot("exploit", "formal-run", "other-worker"),))
        backend.read_coordination.assert_not_called()

    def test_partial_acknowledgement_cannot_satisfy_an_atomic_escalation_batch(self):
        self.seed("p1", declaration("p1"))
        backend = self.backend()
        backend.queue_escalations.return_value = {"proof_id": "p1", "outcome": "accepted",
                                                   "revision": 1, "escalations": []}
        with self.assertRaisesRegex(ValueError, "complete escalation batch"):
            BackendRoundRuntime(backend).assess("p1", {**new_track(), "stalled_rounds": 4},
                {"proof_id": "p1", "round": 1, "before": [0, 0, 0], "slots": []})
        backend.queue_escalations.assert_called_once()

    def test_operation_success_alone_does_not_solve_a_registered_target(self):
        self.seed("p1", *aligned_claim("p1"), UpsertNode("Proof", "p1/proof", {}),
            AddEdge("HAS_TARGET", "p1/proof", "p1/c-1", "p1/target"))
        backend = self.backend()
        # Scripted acknowledgement without a journal change: no proof evidence.
        backend.run_slot.return_value = {"proof_id": "p1", "outcome": "accepted", "accepted": True}
        runtime = BackendRoundRuntime(backend)
        report = runtime.work("p1", 1)
        track, raised, _ = runtime.assess("p1", new_track(), report)
        self.assertEqual(track["status"], ACTIVE)
        self.assertEqual(track["progress"], [0, 0, 0])
        self.assertEqual(raised, [])
        self.assertEqual(self.view_of("p1").node("p1/c-1").fields["status"], "provisional")

    def test_bound_certificate_is_parked_pending_replay(self):
        # Synthetic certificate binding only; no Lean execution or replay.
        self.seed("p1", declaration("p1", status="certificate-produced"), *candidate_certificate("p1"))
        backend = self.backend()
        runtime = BackendRoundRuntime(backend)
        track, raised, _ = runtime.assess("p1", {**new_track(), "stalled_rounds": 10}, runtime.work("p1", 1))
        self.assertEqual(track["status"], "parked")
        self.assertEqual(raised, [])
        backend.run_slot.assert_not_called()
        backend.queue_escalations.assert_not_called()

    def test_same_revision_cannot_change_journal_identity_during_assessment(self):
        self.seed("p1", declaration("p1"))
        backend = self.backend()
        state = backend.read_coordination("p1")
        backend.read_coordination.side_effect = [state, replace(state, journal_hash="other-journal")]
        with self.assertRaisesRegex(ValueError, "journal identity changed"):
            BackendRoundRuntime(backend).assess("p1", new_track(),
                {"proof_id": "p1", "round": 1, "before": [0, 0, 0], "slots": []})
        backend.queue_escalations.assert_not_called()

    def test_concurrent_target_registration_prevents_a_stale_solved_ending(self):
        # Synthetic replay records only, as in the existing local ending tests.
        self.seed("p1", *verified_target("p1"))
        backend = self.backend()
        original = backend.read_coordination
        solved = original("p1")
        self.assertTrue(solved.progress.solved)
        def register_target():
            self.seed("p1", UpsertNode("Claim", "p1/new-target", {"status": "provisional"}),
                AddEdge("HAS_TARGET", "p1/proof", "p1/new-target", "p1/new-target-edge"))
            return original("p1")
        calls = iter((lambda: solved, register_target))
        backend.read_coordination = Mock(side_effect=lambda _: next(calls)())
        track, raised, messages = BackendRoundRuntime(backend).assess("p1", new_track(),
            {"proof_id": "p1", "round": 1, "before": [0, 0, 0], "slots": []})
        self.assertEqual(track["status"], ACTIVE)
        self.assertEqual(raised, [])
        self.assertFalse(any("solved;" in m for m in messages))
        self.assertTrue(any("became stale" in m for m in messages))

    def test_langgraph_runs_using_only_the_three_backend_calls(self):
        try:
            from coordinator.langgraph.graph import Coordinator, HALTED
        except ModuleNotFoundError as exc:
            if exc.name.split(".")[0] == "langgraph":
                self.skipTest("langgraph is not installed")
            raise
        self.seed("p1", declaration("p1"))
        backend = self.backend(handlers={"llm-research": explorer()}, plan={"p1/fr-1": ["stagnated"]})
        final = Coordinator(runtime=BackendRoundRuntime(backend), max_rounds=16).run()
        self.assertEqual(final["outcome"], HALTED)
        self.assertEqual(final["tracks"]["p1"]["status"], STALLED)
        self.assertTrue(backend.read_coordination.called)
        self.assertTrue(backend.run_slot.called)
        self.assertTrue(backend.queue_escalations.called)
        self.assertFalse(hasattr(backend, "workspaces"))
        json.dumps(final)


class TestPolicySnapshot(unittest.TestCase):
    def test_readonly_snapshot_matches_all_six_rules_and_alignment_hold(self):
        source = proof()
        source.add_node("p1/c-1", "Claim", {"status": "provisional"})
        source.add_node("p1/at-1", "Attempt", {"worker_class": "critic", "status": "refuted"})
        source.add_edge("REVIEWS_CLAIM", "p1/at-1", "p1/c-1", "p1/review")
        stuck_run(source, "p1/fr-1", ("p1/obs-1", "typeclass", "p1/fs-1", "s1"))
        stuck_run(source, "p1/fr-2", ("p1/obs-2", "typeclass", "p1/fs-2", "s2"))
        stuck_run(source, "p1/fr-3", ("p1/obs-3", "missing-lemma", "p1/fs-3", "s3"))
        source.add_node("p1/fd-2", "FormalDeclaration", {"status": "searching"})
        stuck_run(source, "p1/fr-4", ("p1/obs-4", "missing-lemma", "p1/fs-4", "s4"), declaration="p1/fd-2")
        source.add_node("p1/al-2", "Alignment", {"lifecycle": "reviewed", "verdict": "mismatch"})
        source.add_edge("ALIGNS_DECLARATION", "p1/al-2", "p1/fd-2", "p1/al-edge")
        reading = read_proof(source, "p1", 1)
        state = CoordinationState("p1", 1, "test-hash", reading,
                                  tuple(source.nodes.values()), tuple(source.edges.values()), (), ())
        view = SnapshotView(state)
        expected = classify(source, "p1", reading, stalled_rounds=4, frontier_empty=True)
        actual = classify(view, "p1", reading, stalled_rounds=4, frontier_empty=True)
        self.assertEqual(actual, expected)
        self.assertEqual([e.rule for e in actual], list(RULES))
        self.assertNotIn("p1/fs-4", [e.target_id for e in actual])
        self.assertEqual(view.snapshot("p1"), (1, view))
        with self.assertRaises(TypeError):
            view.nodes["new"] = None
        with self.assertRaises(TypeError):
            view.node("p1/fd-1").fields["status"] = "changed"
        with self.assertRaises(KeyError):
            view.snapshot("p2")
        self.assertIsNone(view.node("p2/fd-1"))
        self.assertFalse(hasattr(view, "apply"))

    def test_duplicate_obligations_and_alternatives_are_retained(self):
        nodes = (NodeRecord("p1/root", "FormalState", {"exact_state_hash": "root"}),
                 *(NodeRecord("p1/child-" + i, "FormalState", {"exact_state_hash": "same"})
                   for i in ("1", "2")),
                 *(NodeRecord("p1/tactic-" + i, "TacticApplication", {}) for i in ("1", "2")))
        edges = (*(EdgeRecord("p1/child-edge-" + i, "CHILD_OF", "p1/child-" + i,
                             "p1/tactic-1", {"index": int(i)}) for i in ("1", "2")),
                 *(EdgeRecord("p1/alternative-" + i, "APPLIES_TO", "p1/tactic-" + i,
                              "p1/root", {}) for i in ("1", "2")))
        state = CoordinationState("p1", 1, "test-hash", ProofReading(1), nodes, edges, (), ())
        view = SnapshotView(state)
        self.assertEqual(len(view.nodes), 5)
        self.assertEqual(len(view.edges_to("p1/tactic-1", "CHILD_OF")), 2)
        self.assertEqual(len(view.edges_to("p1/root", "APPLIES_TO")), 2)
        for invalid in (replace(state, nodes=nodes + (nodes[0],)),
                        replace(state, nodes=nodes + (NodeRecord("p2/other", "Claim", {}),))):
            with self.assertRaises(ValueError):
                SnapshotView(invalid)
