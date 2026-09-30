import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from core.complete_pipeline import CompletePipeline, AuditLog, SUPPORT_MESSAGE
from core.output_auditor import AuditVerdict, OutputAuditor


def analysis(action="ALLOW", intent="coding"):
    return {"decision": dict(action=action, intent=intent, intent_confidence=0.9,
        risk_level="safe" if action == "ALLOW" else "high", reason_codes=[],
        category_scores={}, dataset_match_confidence=0.9, matched_record_id="r1",
        user_message="Blocked by input safety checks."),
        "signal": {"matched_record_intent": "safe"}, "detected_attacks": [], "matched_category_scores": {}}


class CompleteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.safety = Mock()
        self.safety.analyze.side_effect = lambda _: analysis()
        self.backend = Mock(model="test")
        self.backend.generate.return_value = "Python lists are ordered collections."
        self.auditor = Mock()
        self.auditor.review.return_value = AuditVerdict(safe=True, relevant=True, categories=[])
        self.app = CompletePipeline(self.safety, self.backend, self.auditor, AuditLog(self.path))

    def test_answer_released_only_after_audit(self):
        report = self.app.run("Explain Python lists")
        self.assertEqual(report["output_audit"], "PASSED")
        self.assertTrue(report["allowed"])
        self.assertEqual(report["response"], self.backend.generate.return_value)
        self.auditor.review.assert_called_once()
        entries = self.path.read_text().splitlines()
        self.assertEqual(len(entries), 1)
        self.assertNotIn("Python lists", entries[0])

    def test_block_never_calls_generator(self):
        self.safety.analyze.side_effect = lambda _: analysis("BLOCK", "jailbreak")
        report = self.app.run("blocked request")
        self.backend.generate.assert_not_called()
        self.auditor.review.assert_not_called()
        self.assertFalse(report["allowed"])

    def test_empty_rejected_without_models(self):
        report = self.app.run("  ")
        self.assertEqual(report["final_status"], "INVALID_INPUT")
        self.safety.analyze.assert_not_called()
        self.backend.generate.assert_not_called()

    def test_long_prompt_rejected(self):
        self.assertEqual(self.app.run("x" * 6001)["final_status"], "INVALID_INPUT")

    def test_check_only_never_calls_ollama(self):
        report = self.app.run("explain", check_only=True)
        self.assertEqual(report["output_audit"], "NOT_RUN")
        self.backend.generate.assert_not_called()

    def test_missing_safety_component_fails_closed(self):
        self.safety.analyze.side_effect = FileNotFoundError("private path")
        report = self.app.run("hello")
        self.assertEqual(report["final_status"], "ERROR")
        self.assertNotIn("private path", json.dumps(report))
        self.backend.generate.assert_not_called()

    def test_generation_timeout_fails_closed(self):
        self.backend.generate.side_effect = TimeoutError()
        report = self.app.run("hello")
        self.assertFalse(report["allowed"])
        self.assertIsNone(report["response"])

    def test_audit_error_does_not_release_candidate(self):
        self.auditor.review.side_effect = ValueError("bad JSON")
        report = self.app.run("hello")
        self.assertEqual(report["output_audit"], "ERROR")
        self.assertIsNone(report["response"])
        self.assertEqual(self.backend.generate.call_count, 1)

    def test_failed_candidate_regenerated_and_reaudited(self):
        self.backend.generate.side_effect = ["failed candidate", "safe candidate"]
        self.auditor.review.side_effect = [AuditVerdict(safe=False, relevant=True, categories=["harm"]), AuditVerdict(safe=True, relevant=True, categories=[])]
        report = self.app.run("hello")
        self.assertEqual(report["response"], "safe candidate")
        self.assertEqual(report["generation_attempts"], 2)
        self.assertNotIn("failed candidate", self.path.read_text())

    def test_retries_bounded_and_failed_text_withheld(self):
        self.auditor.review.return_value = AuditVerdict(safe=False, relevant=True, categories=["harm"])
        report = self.app.run("hello")
        self.assertEqual(self.backend.generate.call_count, 2)
        self.assertEqual(report["action"], "BLOCK")
        self.assertIsNone(report["response"])

    def test_irrelevant_output_not_released(self):
        self.auditor.review.return_value = AuditVerdict(safe=True, relevant=False, categories=["irrelevant"])
        self.assertFalse(self.app.run("hello")["allowed"])

    def test_rewritten_prompt_rechecked_and_used(self):
        self.safety.analyze.side_effect = [analysis("SANITIZE"), analysis()]
        self.backend.generate.side_effect = ['{"possible": true, "prompt": "Explain defensive coding"}', "answer"]
        report = self.app.run("original")
        self.assertEqual(report["sanitized_prompt"], "Explain defensive coding")
        self.assertEqual(self.backend.generate.call_args.args[0], "Explain defensive coding")
        self.assertEqual(self.auditor.review.call_args.args[0], "Explain defensive coding")
        self.assertEqual(report["action"], "SANITIZE")

    def test_unchanged_rewrite_not_generated(self):
        self.safety.analyze.side_effect = lambda _: analysis("SANITIZE")
        self.backend.generate.return_value = '{"possible": true, "prompt": "original"}'
        report = self.app.run("original")
        self.assertFalse(report["allowed"])
        self.auditor.review.assert_not_called()

    def test_unsafe_rewrite_not_generated(self):
        self.safety.analyze.side_effect = [analysis("SANITIZE"), analysis("BLOCK")]
        self.backend.generate.return_value = '{"possible": true, "prompt": "rewrite"}'
        self.assertIn("rewrite_rejected", self.app.run("original")["reasons"])
        self.assertEqual(self.backend.generate.call_count, 1)

    def test_support_response_without_generation(self):
        self.safety.analyze.side_effect = lambda _: analysis("BLOCK", "self_harm")
        report = self.app.run("distress")
        self.assertEqual(report["response"], SUPPORT_MESSAGE)
        self.assertEqual(report["final_status"], "SUPPORT")
        self.backend.generate.assert_not_called()

    def test_log_failure_withholds_answer(self):
        self.app.audit_log = Mock()
        self.app.audit_log.write.side_effect = OSError()
        report = self.app.run("hello")
        self.assertFalse(report["audit_logged"])
        self.assertIsNone(report["response"])

    def test_session_history_called_once(self):
        self.safety.apply_history.return_value = 1
        self.app.run("hello", "session")
        self.safety.apply_history.assert_called_once()

    def test_auditor_strict_boolean_and_extra_fields(self):
        backend = Mock()
        for value in ('{"safe":"true","relevant":true,"categories":[]}', '{"safe":true,"relevant":true,"categories":[],"extra":"x"}', 'not JSON'):
            backend.generate.return_value = value
            with self.assertRaises(ValueError):
                OutputAuditor(backend).review("prompt", "candidate")

    def test_auditor_conflicting_verdict_rejected(self):
        backend = Mock()
        backend.generate.return_value = '{"safe":true,"relevant":true,"categories":["harm"]}'
        with self.assertRaises(ValueError):
            OutputAuditor(backend).review("prompt", "candidate")


if __name__ == "__main__":
    unittest.main()
