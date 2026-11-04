"""Opt-in read integration against a separately running compatible receiver.

The receiver and idle proof must be provisioned explicitly. This test submits no
operations, stores no credentials and makes no claims about real ATP/replay.
"""
import asyncio
import os
import unittest

try:
    from persistent_atp_client import AsyncClient
    from persistent_atp_client.adapters.http import HTTPReceiverAdapter
    SDK_HTTP_AVAILABLE = True
except ImportError:
    SDK_HTTP_AVAILABLE = False

from coordinator.langgraph.sdk_backend import SDKBackend


@unittest.skipUnless(SDK_HTTP_AVAILABLE, "optional persistent-atp-client[http] is unavailable")
class TestSDKHTTPReads(unittest.TestCase):
    def test_paginated_committed_graph_from_configured_receiver(self):
        names = ("PERSISTENT_ATP_COORDINATION_TEST_URL", "PERSISTENT_ATP_COORDINATION_TEST_TOKEN",
                 "PERSISTENT_ATP_COORDINATION_TEST_WORKSPACE", "PERSISTENT_ATP_COORDINATION_TEST_PROOF")
        if not all(os.environ.get(name) for name in names):
            self.skipTest("configure PERSISTENT_ATP_COORDINATION_TEST_{URL,TOKEN,WORKSPACE,PROOF}")
        url, token, workspace, proof = (os.environ[name] for name in names)
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
