from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from core.jev_style_client import JevStyleClient


class JevStyleClientTests(unittest.TestCase):
    def test_serializes_compact_session_state(self) -> None:
        event = SimpleNamespace(
            turn_index=2,
            prompt_snippet="previous instruction",
            intent="coding",
            risk_level="safe",
            is_blocked=False,
            block_reason="",
        )
        state = JevStyleClient.serialize_state("current instruction", [event])
        self.assertEqual(state["current_instruction"], "current instruction")
        self.assertEqual(state["previous_events"][0]["instruction"], "previous instruction")
        self.assertEqual(state["previous_events"][0]["turn"], 2)
        self.assertNotIn("H_t", json.dumps(state))

    def test_parses_choice_result(self) -> None:
        response = Mock()
        response.json.return_value = {
            "model": "jev-style-2b-decision-v3",
            "answers": {"choice_1": {
                "type": "choice",
                "choice": "Safe/Educational",
                "probabilities": {"Safe/Educational": 0.9, "Jailbreak": 0.1},
                "confidence": 0.88,
            }},
            "usage": {"input_tokens": 12},
            "latency_ms": 42.5,
        }
        response.raise_for_status.return_value = None
        http = Mock()
        http.post.return_value = response

        with patch.dict("os.environ", {}, clear=False):
            result = JevStyleClient(http_client=http).decide("current", [])
        request = http.post.call_args.kwargs
        self.assertEqual(request["url"] if "url" in request else http.post.call_args.args[0],
                         "http://127.0.0.1:8765/v1/systemone")
        self.assertEqual(request["json"]["state"]["current_instruction"], "current")
        self.assertEqual(request["json"]["questions"]["choice_1"]["type"], "choice")
        self.assertEqual(result["decision"], "Safe/Educational")
        self.assertEqual(result["choice"], "Safe/Educational")
        self.assertEqual(result["probabilities"]["Jailbreak"], 0.1)
        self.assertEqual(result["confidence"], 0.88)
        self.assertEqual(result["model"], "jev-style-2b-decision-v3")
        self.assertEqual(result["usage"]["input_tokens"], 12)
        self.assertEqual(result["latency_ms"], 42.5)

    def test_server_unavailable_is_propagated_for_pipeline_fallback(self) -> None:
        http = Mock()
        http.post.side_effect = requests.ConnectionError("server unavailable")
        with self.assertRaises(requests.ConnectionError):
            JevStyleClient(http_client=http).decide("current", [])


if __name__ == "__main__":
    unittest.main(verbosity=2)