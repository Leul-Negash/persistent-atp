"""Runtime injection with explicitly scripted rounds, not an ATP or replay test."""

import json
import threading
import unittest
from contextlib import closing

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover - exercised without the optional extra
    InMemorySaver = None

from coordinator.langgraph import CoordinatorRuntime, PORTFOLIO, RoundRuntime, StagnationPolicy
from coordinator.langgraph.rounds import ACTIVE, PARKED, STALLED

from .fixtures import JournalCase, declaration, explorer

if InMemorySaver is not None:
    from coordinator.langgraph.graph import Coordinator, HALTED


class ScriptedRoundRuntime:
    """Graph plumbing fixture: no stores, execution, proofs or verification."""

    def __init__(self):
        self.proof_ids = ("p2", "p1")
        self.calls = []
        self.barrier = threading.Barrier(2, timeout=10)

    def work(self, proof_id, round_number, *, slots):
        if round_number == 1:
            self.barrier.wait()
        self.calls.append(("work", proof_id, round_number, slots))
        return {"proof_id": proof_id, "round": round_number, "mode": "scripted-round"}

    def assess(self, proof_id, track, report, *, policy):
        self.calls.append(("assess", proof_id, report["round"], policy))
        status = PARKED if proof_id == "p2" else ACTIVE if report["round"] == 1 else STALLED
        return ({**track, "status": status}, [], [f"{proof_id}: scripted assessment"])


@unittest.skipIf(InMemorySaver is None, "langgraph is not installed")
class TestInjectedRuntime(unittest.TestCase):
    def test_runtime_without_workspaces_runs_parallel_rounds_and_plain_checkpoints(self):
        runtime = ScriptedRoundRuntime()
        self.assertIsInstance(runtime, RoundRuntime)
        self.assertFalse(hasattr(runtime, "workspaces"))
        policy = StagnationPolicy(non_progress_rounds=1)
        slots = (PORTFOLIO[2],)
        coordinator = Coordinator(runtime=runtime, policy=policy, slots=slots)
        self.assertEqual(tuple(coordinator.initial_state()["tracks"]), ("p2", "p1"))
        # Registration is stable even if the implementation later replaces its registry.
        runtime.proof_ids = ("other",)
        self.assertEqual(coordinator.proof_ids, ("p2", "p1"))
        with self.assertRaisesRegex(AttributeError, "local CoordinatorRuntime"):
            coordinator.workspaces

        app = coordinator.build(InMemorySaver())
        config = {"configurable": {"thread_id": "injected"}, "recursion_limit": 20}
        final = app.invoke(coordinator.initial_state(), config)
        self.assertEqual(final["outcome"], HALTED)
        self.assertEqual(final["round"], 2)
        self.assertEqual(final["tracks"]["p2"]["status"], PARKED)
        self.assertEqual(final["tracks"]["p1"]["status"], STALLED)
        for method in ("work", "assess"):
            calls = [c for c in runtime.calls if c[0] == method]
            self.assertEqual(sorted((c[1], c[2]) for c in calls),
                             [("p1", 1), ("p1", 2), ("p2", 1)])
            self.assertTrue(all(c[3] is (slots if method == "work" else policy) for c in calls))
        values = app.get_state(config).values
        self.assertEqual(set(values),
                         {"round", "tracks", "reports", "escalations", "milestones", "outcome"})
        json.dumps(values)

    def test_constructor_rejects_missing_ambiguous_or_incomplete_runtime(self):
        with self.assertRaisesRegex(ValueError, "either workspaces or runtime"):
            Coordinator()
        with self.assertRaisesRegex(ValueError, "either workspaces or runtime"):
            Coordinator([], runtime=ScriptedRoundRuntime())
        with self.assertRaisesRegex(TypeError, "proof_ids, work and assess"):
            Coordinator(runtime=object())

    def test_constructor_rejects_duplicate_or_invalid_proof_identity(self):
        for proof_ids in (("p1", "p1"), ("",), (1,)):
            with self.subTest(proof_ids=proof_ids):
                runtime = ScriptedRoundRuntime()
                runtime.proof_ids = proof_ids
                with self.assertRaisesRegex(ValueError, "runtime proof IDs"):
                    Coordinator(runtime=runtime)

    def test_misscoped_reports_are_rejected_before_assessment(self):
        for proof_id, round_number in (("other", 1), ("p1", 2)):
            with self.subTest(proof_id=proof_id, round_number=round_number):
                runtime = ScriptedRoundRuntime()
                runtime.work = lambda *args, **kwargs: {
                    "proof_id": proof_id, "round": round_number}
                coordinator = Coordinator(runtime=runtime)
                with self.assertRaisesRegex(ValueError, "another proof or round"):
                    coordinator._work({"proof_id": "p1", "round": 1})
                self.assertEqual(runtime.calls, [])


@unittest.skipIf(InMemorySaver is None, "langgraph is not installed")
class TestLocalRuntimeInjection(JournalCase):
    def test_local_workspace_constructor_and_injected_runtime_are_equivalent(self):
        workspace = self.workspace("p1")
        local = Coordinator([workspace])
        injected = Coordinator(runtime=CoordinatorRuntime([workspace]))
        self.assertIs(local.workspaces["p1"], workspace)
        self.assertIs(injected.workspaces["p1"], workspace)
        self.assertEqual(local.initial_state(), injected.initial_state())

    def test_injected_local_runtime_uses_real_gate_and_journal(self):
        self.seed("p1", declaration("p1"))
        workspace = self.workspace("p1", handlers={"llm-research": explorer()},
                                   plan={"p1/fr-1": ["stagnated"]})
        final = Coordinator(runtime=CoordinatorRuntime([workspace]), max_rounds=2).run()
        self.assertTrue(final["escalations"])
        with closing(self.open_store()) as store:
            self.assertEqual(store.verify_chain("p1"), store.head("p1")[0])
        self.assertTrue(self.moves("p1"))
