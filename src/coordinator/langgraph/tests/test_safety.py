"""Regression checks for assessment races and worker answer validation."""
from commit_gate.ops import UpsertNode
from coordinator.langgraph.escalate import commit_escalations
from coordinator.langgraph.mechanisms import check_mechanisms
from coordinator.langgraph.policy import Escalation, GLOBAL_AUDIT, FIVE_MECHANISMS
from coordinator.langgraph.progress import read_proof
from coordinator.langgraph.rounds import work
from coordinator.langgraph.workspace import project
from .fixtures import JournalCase, declaration, explorer


class TestSafety(JournalCase):
    def test_stale_assessment_cannot_be_rebased_or_revoke_a_lease(self):
        self.seed("p1", declaration("p1"))
        store = self.open_store()
        view = self.view_of("p1", store)
        reading = read_proof(view, "p1", 1)
        self.seed("p1", UpsertNode("ResearchState", "p1/rs-9", {"status": "open"}))
        audit = len(store.read_lease_events("p1"))
        result = commit_escalations(store, view, "p1", [Escalation(GLOBAL_AUDIT, "p1", "audit")],
            reading=reading, round_number=1)
        self.assertFalse(result.accepted)
        self.assertEqual(result.rejections[0].reason.value, "stale-base-revision")
        self.assertEqual(len(store.read_lease_events("p1")), audit)
        self.assertEqual(store.head("p1")[0], 2)

    def test_race_after_revision_check_still_rejects_at_gate(self):
        self.seed("p1", declaration("p1"))
        store = self.open_store()
        view = self.view_of("p1", store)
        acquire = store.acquire_lease
        def racing_acquire(*args, **kwargs):
            self.seed("p1", UpsertNode("ResearchState", "p1/rs-9", {"status": "open"}))
            return acquire(*args, **kwargs)
        store.acquire_lease = racing_acquire
        result = commit_escalations(store, view, "p1", [Escalation(GLOBAL_AUDIT, "p1", "audit")],
            reading=read_proof(view, "p1", 1), round_number=1)
        self.assertFalse(result.accepted)
        self.assertEqual(store.head("p1")[0], 2)

    def test_repeated_escalation_submission_is_a_journal_noop(self):
        self.seed("p1", declaration("p1"))
        store, view = self.open_store(), self.view_of("p1")
        items = [Escalation(GLOBAL_AUDIT, "p1", "audit")]
        self.assertTrue(commit_escalations(store, view, "p1", items,
            reading=read_proof(view, "p1", 1), round_number=1).accepted)
        revision = project(view, store, "p1")
        self.assertIsNone(commit_escalations(store, view, "p1", items,
            reading=read_proof(view, "p1", revision), round_number=2))
        self.assertEqual(store.head("p1")[0], revision)

    def test_incomplete_five_mechanism_worker_result_never_commits(self):
        self.seed("p1", declaration("p1", status="replay-accepted"))
        store, view = self.open_store(), self.view_of("p1")
        result = commit_escalations(store, view, "p1", [Escalation(FIVE_MECHANISMS, "p1", "five")],
            reading=read_proof(view, "p1", 1), round_number=1)
        self.assertTrue(result.accepted)
        before = store.head("p1")
        workspace = self.workspace("p1", handlers={"llm-research": explorer({FIVE_MECHANISMS: [{"detail": "incomplete"}]})})
        report = work(workspace, 2)
        self.assertEqual(report["slots"][0]["outcome"], "invalid-worker-result")
        self.assertEqual(store.head("p1"), before)
        self.assertEqual(self.moves("p1")[0].fields["status"], "queued")
        self.assertFalse(store.active_leases("p1"))

    def test_malformed_mechanism_answer_returns_diagnostics(self):
        self.assertTrue(check_mechanisms(None))
        self.assertTrue(check_mechanisms([None] * 5))
