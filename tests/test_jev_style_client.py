from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from core.jev_style_client import JevActionResponseError, JevStyleClient, JevStyleUnavailable


class JevStyleClientTests(unittest.TestCase):
    def test_default_loopback_session_ignores_environment_proxy_settings(self) -> None:
        session = Mock()
        with patch.dict("os.environ", {"JEV_STYLE_URL": "http://127.0.0.1:8765"}):
            with patch("core.jev_style_client.requests.Session", return_value=session):
                JevStyleClient()
        self.assertFalse(session.trust_env)

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
            "answers": {
                "choice_1": {
                    "type": "choice",
                    "choice": "Safe/Educational",
                    "probabilities": {
                        "Safe/Educational": 0.90, "Cyber Abuse": 0.02,
                        "Prompt Injection": 0.02, "Jailbreak": 0.03,
                        "Harmful Instructions": 0.03,
                    },
                    "confidence": 0.88,
                },
                "decision": {
                    "type": "choice",
                    "choice": "ALLOW",
                    "probabilities": {"ALLOW": 0.90, "SANITIZE": 0.07, "BLOCK": 0.03},
                    "confidence": 0.85,
                },
            },
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
        self.assertEqual(set(request["json"]["questions"]["decision"]["criteria"]),
                         {"ALLOW", "SANITIZE", "BLOCK"})
        self.assertEqual(result["decision"], "Safe/Educational")
        self.assertEqual(result["choice"], "Safe/Educational")
        self.assertEqual(result["probabilities"]["Jailbreak"], 0.03)
        self.assertEqual(result["confidence"], 0.88)
        self.assertEqual(result["classification_category"], "Safe/Educational")
        self.assertEqual(result["action"], "ALLOW")
        self.assertEqual(result["action_probabilities"]["BLOCK"], 0.03)
        self.assertEqual(result["action_confidence"], 0.85)
        self.assertNotIn("raw", result)
        self.assertEqual(result["model"], "jev-style-2b-decision-v3")
        self.assertEqual(result["usage"]["input_tokens"], 12)
        self.assertEqual(result["latency_ms"], 42.5)

    def test_server_unavailable_is_propagated_without_response_body(self) -> None:
        http = Mock()
        http.post.side_effect = requests.ConnectionError("private transport detail")
        with self.assertRaises(JevStyleUnavailable) as caught:
            JevStyleClient(http_client=http).decide("current", [])
        self.assertNotIn("private transport detail", str(caught.exception))

    def test_url_override_is_preserved_but_category_only_response_fails_closed(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"answers": {"choice_1": {
            "type": "choice", "choice": "Safe/Educational",
        }}}
        http = Mock()
        http.post.return_value = response
        with patch.dict("os.environ", {"JEV_STYLE_URL": "http://jev-override.invalid"}):
            client = JevStyleClient(http_client=http)
            with self.assertRaises(JevActionResponseError):
                client.decide("current")
        self.assertEqual(http.post.call_args.args[0],
                         "http://jev-override.invalid/v1/systemone")

    def test_missing_or_invalid_action_fails_closed(self) -> None:
        fixtures = (
            {"answers": {"choice_1": {"type": "choice", "choice": "Safe/Educational"}}},
            {"answers": {"decision": {"type": "choice", "choice": "Safe/Educational"}}},
            {"answers": {"decision": {"type": "text", "choice": "ALLOW"}}},
        )
        for fixture in fixtures:
            with self.subTest(fixture=fixture), self.assertRaises(JevActionResponseError):
                JevStyleClient._parse_response(fixture, 1.0)

    def test_contradictory_action_probability_fails_closed(self) -> None:
        fixture = {"answers": {"decision": {
            "type": "choice", "choice": "BLOCK",
            "probabilities": {"ALLOW": 0.8, "SANITIZE": 0.1, "BLOCK": 0.1},
        }}}
        with self.assertRaises(JevActionResponseError):
            JevStyleClient._parse_response(fixture, 1.0)

    def test_serializes_only_bounded_local_evidence(self) -> None:
        state = JevStyleClient.serialize_state(
            "current",
            [],
            {"local_intent": "coding", "local_intent_confidence": 0.9,
             "deterministic_patterns": ["instruction_override"], "prompt": "must not forward"},
        )
        self.assertEqual(state["local_evidence"]["local_intent"], "coding")
        self.assertNotIn("prompt", json.dumps(state))
        self.assertNotIn("must not forward", json.dumps(state))

    def test_serializes_failed_decision_provenance_without_claiming_a_block(self) -> None:
        event = {
            "turn_index": 1,
            "prompt_snippet": "Earlier request",
            "intent": "educational",
            "risk_level": "safe",
            "is_blocked": False,
            "block_reason": "",
            "decision_status": "ERROR",
            "decision_failure_reason_code": "jev_unavailable",
        }
        state = JevStyleClient.serialize_state("later request", [event])
        serialized = state["previous_events"][0]
        self.assertEqual(serialized["decision_status"], "ERROR")
        self.assertEqual(serialized["decision_failure_reason_code"], "jev_unavailable")
        self.assertFalse(serialized["is_blocked"])

    def test_action_question_covers_credential_theft_and_unauthorized_access(self) -> None:
        body = JevStyleClient.request_body("account security request", [], {})
        questions = body["questions"]
        classification = questions["choice_1"]
        action = questions["decision"]
        instructions = action["instructions"].lower()
        classification_instructions = classification["instructions"].lower()

        self.assertEqual(set(questions), {"choice_1", "decision"})
        self.assertEqual(classification["type"], "choice")
        self.assertIn("conversational context", classification_instructions)
        self.assertIn("profanity", classification_instructions)
        self.assertIn("says it was joking", classification_instructions)
        self.assertIn("previously blocked turn", classification_instructions)
        self.assertIn("not classify a message as a jailbreak solely", classification_instructions)
        self.assertEqual(action["type"], "choice")
        self.assertIn("greetings", instructions)
        self.assertIn("small talk", instructions)
        self.assertIn("ordinary profanity", instructions)
        self.assertIn("nonthreatening insults", instructions)
        self.assertIn("jokes", instructions)
        self.assertIn("apologies", instructions)
        self.assertIn("harmless follow-ups", instructions)
        self.assertIn("plain factual questions", instructions)
        self.assertIn("ordinary educational explanations", instructions)
        self.assertIn("introductory, non-operational cybersecurity learning", instructions)
        self.assertIn("do not choose sanitize merely", instructions)
        self.assertIn("specific unsafe element", instructions)
        self.assertIn("irrelevant prior context must not change", instructions)
        self.assertIn("prior turns as evidence", instructions)
        self.assertIn("automatic proof", instructions)
        self.assertIn("following a previously blocked turn alone", instructions)
        self.assertIn("steal passwords", instructions)
        self.assertIn("other credentials", instructions)
        self.assertIn("malicious code", instructions)
        self.assertIn("access accounts without authorization", instructions)
        self.assertIn("explicitly bypass safety controls", instructions)
        self.assertIn("must be block", instructions)
        self.assertIn("classification category", instructions)
        self.assertIn("authorized defensive or account-recovery goal", instructions)
        self.assertIn("requests for emotional support", instructions)
        self.assertIn("suicide prevention", instructions)
        self.assertIn("help supporting a friend", instructions)
        self.assertIn("concealing self-inflicted injuries", instructions)
        self.assertEqual(action["criteria"]["ALLOW"], "The current request is safe to answer as written.")
        self.assertEqual(
            action["criteria"]["SANITIZE"],
            "A legitimate purpose can be answered after a safe rewrite and fresh review.",
        )
        self.assertEqual(
            action["criteria"]["BLOCK"],
            "The current request must not be answered or rewritten into an answerable request.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
