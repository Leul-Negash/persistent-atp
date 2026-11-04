"""Shared backend snapshots; no SDK, external service or proof executor."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock

from commit_gate.state import EdgeRecord, NodeRecord
from mathproof.scheduler import Candidate
from coordinator.langgraph import CoordinationState, CoordinatorRuntime, ProofBackend, StaleAssessment
from coordinator.langgraph.progress import ProofReading
from coordinator.langgraph.workspace import ProofWorkspace


class TestCoordinationContract(unittest.TestCase):
    def test_snapshot_owns_immutable_copies_and_export_is_detached(self):
        node_fields = {"nested": {"values": [1, 2]}}
        edge_fields = {"diagnostics": ["original"]}
        features = {"novelty": .5}
        conflicts = ["p1/c-1"]
        node = NodeRecord("p1/rm-1", "ResearchMove", node_fields)
        state = CoordinationState("p1", 1, "sha256:" + "a" * 64, ProofReading(1),
            (node,), (EdgeRecord("p1/e-1", "PROPOSES", "p1/rs-1", "p1/rm-1", edge_fields),),
            (Candidate("p1/rm-1", "research-move", "investigate", features, conflicts),),
            (node,))

        node_fields["nested"]["values"].append(3)
        edge_fields["diagnostics"].append("changed")
        features["novelty"] = 1.0
        conflicts.append("p1/c-2")
        self.assertEqual(state.nodes[0].fields["nested"]["values"], (1, 2))
        self.assertEqual(state.edges[0].fields["diagnostics"], ("original",))
        self.assertEqual(state.frontier[0].features["novelty"], .5)
        self.assertEqual(state.frontier[0].conflict_ids, ("p1/c-1",))
        self.assertEqual(state.escalations[0].fields["nested"]["values"], (1, 2))
        with self.assertRaises(TypeError):
            state.escalations[0].fields["nested"]["values"][0] = 9

        exported = state.to_dict()
        json.dumps(exported)
        exported["nodes"][0]["fields"]["nested"]["values"].append(99)
        exported["frontier"][0]["features"]["novelty"] = 0
        self.assertEqual(state.nodes[0].fields["nested"]["values"], (1, 2))
        self.assertEqual(state.frontier[0].features["novelty"], .5)

    def test_progress_cannot_silently_come_from_another_revision(self):
        with self.assertRaisesRegex(ValueError, "another coordination revision"):
            CoordinationState("p1", 2, "sha256:" + "a" * 64, ProofReading(1), (), (), (), ())
        for revision in (-1, True):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                CoordinationState("p1", revision, "sha256:" + "a" * 64,
                                  ProofReading(revision), (), (), (), ())

    def test_original_runtime_imports_remain_compatible(self):
        from coordinator.langgraph.runtime import CoordinationState as OldState
        from coordinator.langgraph.runtime import StaleAssessment as OldStale
        self.assertIs(OldState, CoordinationState)
        self.assertIs(OldStale, StaleAssessment)

    def test_registered_discovery_does_not_open_stores_or_run_workers(self):
        open_store, dispatcher = Mock(), Mock()
        runtime = CoordinatorRuntime([
            ProofWorkspace("p2", open_store, dispatcher),
            ProofWorkspace("p1", open_store, dispatcher),
        ])
        self.assertIsInstance(runtime, ProofBackend)
        self.assertEqual(runtime.proof_ids, ("p2", "p1"))
        open_store.assert_not_called()
        dispatcher.assert_not_called()

    def test_contract_import_does_not_require_langgraph_or_client_sdk(self):
        source_root = str(Path(__file__).resolve().parents[3])
        script = """
import importlib.abc
import sys
class RefuseOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'langgraph', 'persistent_atp_client'}:
            raise AssertionError('optional dependency imported: ' + fullname)
sys.meta_path.insert(0, RefuseOptionalDependencies())
sys.path.insert(0, sys.argv[1])
from coordinator.langgraph.contracts import CoordinationState, ProofBackend
assert 'coordinator.langgraph.graph' not in sys.modules
"""
        completed = subprocess.run([sys.executable, "-I", "-c", script, source_root],
                                   capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
