"""Opt-in read integration against a separately running compatible receiver.

The receiver and idle proof must be provisioned explicitly. This test submits no
operations, stores no credentials and makes no claims about real ATP/replay.
"""
import asyncio
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import persistent_atp_client
    from persistent_atp_client import AsyncClient, ProofAudit
    from persistent_atp_client.adapters.http import HTTPReceiverAdapter
    SDK_HTTP_AVAILABLE = True
except ImportError:
    SDK_HTTP_AVAILABLE = False

from coordinator.langgraph.sdk_backend import SDKBackend
from coordinator.langgraph.sdk_operations import DurableSDKOperations


@unittest.skipUnless(SDK_HTTP_AVAILABLE, "optional persistent-atp-client[http] is unavailable")
class TestSDKHTTPReads(unittest.TestCase):
    def configured_receiver(self):
        names = ("PERSISTENT_ATP_COORDINATION_TEST_URL", "PERSISTENT_ATP_COORDINATION_TEST_TOKEN",
                 "PERSISTENT_ATP_COORDINATION_TEST_WORKSPACE", "PERSISTENT_ATP_COORDINATION_TEST_PROOF")
        if not all(os.environ.get(name) for name in names):
            self.skipTest("configure PERSISTENT_ATP_COORDINATION_TEST_{URL,TOKEN,WORKSPACE,PROOF}")
        return tuple(os.environ[name] for name in names)

    def test_paginated_committed_graph_from_configured_receiver(self):
        url, token, workspace, proof = self.configured_receiver()
        async def scenario():
            async with AsyncClient(HTTPReceiverAdapter(url, token)) as client:
                backend = SDKBackend(client, workspace_id=workspace, proof_ids=(proof,), page_size=1)
                before = await backend.aread_coordination(proof)
                after = await backend.aread_coordination(proof)
                self.assertGreater(len(before.nodes), 0)
                self.assertGreater(before.revision, 0)
                self.assertTrue(before.component_mode)
                self.assertEqual(before.to_dict(), after.to_dict())
                self.assertEqual(before.progress.revision, before.revision)
        asyncio.run(scenario())

    def test_durable_audit_reattaches_from_a_fresh_client_process(self):
        url, token, workspace, proof = self.configured_receiver()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "client-intents.sqlite3"
            async def first_process():
                async with AsyncClient(HTTPReceiverAdapter(url, token)) as client:
                    backend = SDKBackend(client, workspace_id=workspace, proof_ids=(proof,))
                    before = await backend.aread_coordination(proof)
                    operations = DurableSDKOperations(client, path, receiver_id="reference-test",
                                                       workspace_id=workspace, proof_ids=(proof,))
                    prepared = operations.prepare("audit-restart", ProofAudit(proof))
                    result = await operations.advance(prepared.intent_id, timeout=5)
                    self.assertEqual(result.lifecycle, "succeeded")
                    after = await backend.aread_coordination(proof)
                    self.assertEqual(before.to_dict(), after.to_dict())
                    return operations.get(prepared.intent_id).checkpoint_ref()
            checkpoint = asyncio.run(first_process())
            code = """
import asyncio
import json
import os
import sys
sys.path[:0] = sys.argv[1:3]
from persistent_atp_client import AsyncClient
from persistent_atp_client.adapters.http import HTTPReceiverAdapter
from coordinator.langgraph.sdk_operations import DurableSDKOperations
async def main():
    prefix = 'PERSISTENT_ATP_COORDINATION_TEST_'
    async with AsyncClient(HTTPReceiverAdapter(os.environ[prefix+'URL'], os.environ[prefix+'TOKEN'])) as client:
        operations = DurableSDKOperations(client, sys.argv[3], receiver_id='reference-test',
            workspace_id=os.environ[prefix+'WORKSPACE'], proof_ids=(os.environ[prefix+'PROOF'],))
        result = await operations.advance('audit-restart', timeout=5)
        assert result.lifecycle == 'succeeded'
        print(json.dumps(operations.get('audit-restart').checkpoint_ref()))
asyncio.run(main())
"""
            source = str(Path(__file__).resolve().parents[3])
            sdk_source = str(Path(persistent_atp_client.__file__).resolve().parents[1])
            child = subprocess.run([sys.executable, "-I", "-c", code, source, sdk_source, str(path)],
                                   capture_output=True, text=True, timeout=15)
            self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
            self.assertEqual(json.loads(child.stdout), checkpoint)
