from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from core.conversation_guard import TurnRecord
from core.jev_client import DEFAULT_JEV_ENDPOINT, DEFAULT_JEV_MODEL, JEVClient


class JEVClientTests(unittest.TestCase):
    def test_decide_sends_official_body_and_maps_choice_answer(self) -> None:
        response = Mock()
        response.json.return_value = {
            "model": "jev-1.13",
            "model_version": "jev-1.13",
            "answers": {
                "decision": {
                    "type": "choice",
                    "choice": "ALLOW",
                    "probabilities": {"ALLOW": 0.98, "SANITIZE": 0.01, "BLOCK": 0.01},
                    "confidence": 0.97,
                }
            },
            "usage": {"input_tokens": 42},
            "latency_ms": 120,
        }
        response.raise_for_status.return_value = None
        http = Mock()
        http.post.return_value = response
        event = TurnRecord(1, "2026-10-02T00:00:00Z", "previous", "safe", 0.9, "safe", False, "")

        with patch.dict(os.environ, {"JEV_API_KEY": "test-key"}):
            result = JEVClient(endpoint="https://jev.test", session=http).decide(
                "Explain Python lists.", [event]
            )

        request = http.post.call_args.kwargs
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(request["json"]["model"], DEFAULT_JEV_MODEL)
        self.assertEqual(request["json"]["state"]["current_instruction"], "Explain Python lists.")
        self.assertEqual(request["json"]["state"]["session_events"][0]["turn_index"], 1)
        self.assertEqual(request["json"]["questions"]["decision"]["type"], "choice")
        self.assertEqual(result["decision"], "ALLOW")
        self.assertEqual(result["probabilities"]["BLOCK"], 0.01)
        self.assertEqual(result["confidence"], 0.97)
        self.assertEqual(result["model"], "jev-1.13")
        self.assertEqual(result["model_version"], "jev-1.13")
        self.assertEqual(result["usage"], {"input_tokens": 42})
        self.assertEqual(result["latency_ms"], 120)
        self.assertGreaterEqual(result["elapsed_seconds"], 0.0)

    def test_defaults_use_current_official_endpoint(self) -> None:
        with patch.dict(os.environ, {"JEV_API_KEY": "test-key"}):
            client = JEVClient()
        self.assertEqual(client.endpoint, DEFAULT_JEV_ENDPOINT)
        self.assertEqual(client.model, DEFAULT_JEV_MODEL)

    @unittest.skipUnless(os.getenv("JEV_API_KEY"), "JEV_API_KEY is not configured")
    def test_live_jev_api(self) -> None:
        result = JEVClient().decide("Explain Python lists.", [])
        self.assertIn(result["decision"], {"ALLOW", "SANITIZE", "BLOCK"})
        self.assertIsInstance(result["probabilities"], dict)
        self.assertIsInstance(result["confidence"], (int, float))


if __name__ == "__main__":
    unittest.main(verbosity=2)