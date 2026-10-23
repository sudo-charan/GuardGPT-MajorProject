import json
import unittest
from unittest.mock import Mock

from core.output_auditor import OutputAuditor


class OutputAuditorTests(unittest.TestCase):
    def test_short_greeting_is_judged_against_the_actual_request(self):
        backend = Mock()
        backend.generate.return_value = json.dumps({
            "safe": True,
            "relevant": True,
            "categories": [],
        })

        verdict = OutputAuditor(backend).review("Hai there", "Hello! How can I help?")

        payload = json.loads(backend.generate.call_args.args[0])
        self.assertEqual(payload, {
            "request": "Hai there",
            "candidate": "Hello! How can I help?",
        })
        self.assertIn("a concise", backend.generate.call_args.kwargs["system_prompt"])
        self.assertTrue(verdict.safe)
        self.assertTrue(verdict.relevant)

    def test_short_prompt_does_not_make_unrelated_answer_relevant(self):
        backend = Mock()
        backend.generate.return_value = json.dumps({
            "safe": True,
            "relevant": False,
            "categories": ["irrelevant"],
        })

        verdict = OutputAuditor(backend).review("hello", "Python lists are ordered collections.")

        self.assertTrue(verdict.safe)
        self.assertFalse(verdict.relevant)
        self.assertEqual(verdict.categories, ["irrelevant"])


if __name__ == "__main__":
    unittest.main()
