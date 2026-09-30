"""
Tests for `core.risk_estimator` — the canonical risk helper.

Covers the `estimate_risk` and `should_preliminarily_block` helpers plus
the export surface of the constants. The legacy `0.45 threshold` is
intentionally documented in the boundary tests below to lock in the
behavior change.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.risk_estimator import (  # noqa: E402
    EDUCATIONAL_SEMANTIC_BLOCK_THRESHOLD,
    EDUCATIONAL_SEMANTIC_RISK_THRESHOLD,
    HIGH_RISK_INTENT_CONFIDENCE,
    LOW_SIMILARITY_THRESHOLD,
    SELF_HARM_CONFIDENCE,
    SELF_HARM_CRITICAL_CONFIDENCE,
    SEMANTIC_BLOCK_THRESHOLD,
    SEMANTIC_RISK_THRESHOLD,
    estimate_risk,
    should_preliminarily_block,
)


class EstimateRiskTests(unittest.TestCase):
    """Table-driven tests covering every branch of `estimate_risk`."""

    def test_self_harm_critical_at_or_above_threshold(self) -> None:
        self.assertEqual(
            estimate_risk("self_harm", max(SELF_HARM_CONFIDENCE, SELF_HARM_CRITICAL_CONFIDENCE), 0.0),
            "critical",
        )
        self.assertEqual(
            estimate_risk("self_harm_risk", 0.95, 0.0),
            "critical",
        )

    def test_self_harm_below_gate_uses_remaining_evidence(self) -> None:
        self.assertEqual(
            estimate_risk("self_harm", SELF_HARM_CONFIDENCE - 0.01, 0.0),
            "safe",
        )

    def test_harm_intent_high_at_or_above_threshold(self) -> None:
        for intent in ("harmful", "harmful_instructions", "cyber_abuse", "illegal",
                       "prompt_injection", "jailbreak"):
            with self.subTest(intent=intent):
                self.assertEqual(
                    estimate_risk(intent, HIGH_RISK_INTENT_CONFIDENCE, 0.0),
                    "high",
                )

    def test_harm_intent_medium_below_threshold(self) -> None:
        self.assertEqual(
            estimate_risk("harmful", HIGH_RISK_INTENT_CONFIDENCE - 0.01, 0.0),
            "medium",
        )

    def test_harmful_legacy_0_45_threshold_no_longer_blocks(self) -> None:
        """Pin the behavior change: confidence 0.50 used to score as `medium`,
        but after the threshold raise to 0.65 it still scores as `medium`.
        The legacy engine's `0.45` block threshold is what changed; this test
        documents that the canonical reference still treats <0.65 as not
        high-risk."""
        self.assertEqual(
            estimate_risk("harmful", 0.50, 0.0),
            "medium",
        )

    def test_semantic_high_general(self) -> None:
        self.assertEqual(
            estimate_risk("general", 0.0, SEMANTIC_BLOCK_THRESHOLD),
            "high",
        )

    def test_semantic_high_educational(self) -> None:
        self.assertEqual(
            estimate_risk("educational", 0.0, EDUCATIONAL_SEMANTIC_BLOCK_THRESHOLD),
            "high",
        )

    def test_semantic_medium_general(self) -> None:
        self.assertEqual(
            estimate_risk("general", 0.0, SEMANTIC_RISK_THRESHOLD),
            "medium",
        )

    def test_semantic_medium_educational(self) -> None:
        self.assertEqual(
            estimate_risk("educational", 0.0, EDUCATIONAL_SEMANTIC_RISK_THRESHOLD),
            "medium",
        )

    def test_semantic_low(self) -> None:
        self.assertEqual(
            estimate_risk("general", 0.0, LOW_SIMILARITY_THRESHOLD),
            "low",
        )

    def test_semantic_safe(self) -> None:
        self.assertEqual(
            estimate_risk("general", 0.0, LOW_SIMILARITY_THRESHOLD - 0.01),
            "safe",
        )


class ShouldPreliminarilyBlockTests(unittest.TestCase):
    """Boundary tests for the canonical preliminary-block predicate."""

    def test_self_harm_block_threshold(self) -> None:
        self.assertTrue(
            should_preliminarily_block(
                {"intent": "self_harm", "intent_confidence": SELF_HARM_CONFIDENCE}
            )
        )
        self.assertFalse(
            should_preliminarily_block(
                {"intent": "self_harm", "intent_confidence": SELF_HARM_CONFIDENCE - 0.01}
            )
        )

    def test_harmful_block_threshold_uses_0_65(self) -> None:
        # The legacy 0.45 threshold is gone; 0.65 is the canonical value.
        self.assertFalse(
            should_preliminarily_block(
                {"intent": "harmful", "intent_confidence": 0.45}
            )
        )
        self.assertTrue(
            should_preliminarily_block(
                {"intent": "harmful", "intent_confidence": HIGH_RISK_INTENT_CONFIDENCE}
            )
        )

    def test_harmful_instructions_block_threshold_uses_0_65(self) -> None:
        self.assertFalse(
            should_preliminarily_block(
                {"intent": "harmful_instructions", "intent_confidence": 0.45}
            )
        )
        self.assertTrue(
            should_preliminarily_block(
                {"intent": "harmful_instructions",
                 "intent_confidence": HIGH_RISK_INTENT_CONFIDENCE}
            )
        )

    def test_jailbreak_block_threshold_uses_0_65(self) -> None:
        self.assertFalse(
            should_preliminarily_block(
                {"intent": "jailbreak", "intent_confidence": 0.45}
            )
        )
        self.assertTrue(
            should_preliminarily_block(
                {"intent": "jailbreak", "intent_confidence": HIGH_RISK_INTENT_CONFIDENCE}
            )
        )

    def test_educational_block_threshold(self) -> None:
        self.assertFalse(
            should_preliminarily_block(
                {"intent": "educational", "dataset_match_confidence": SEMANTIC_BLOCK_THRESHOLD}
            )
        )
        self.assertTrue(
            should_preliminarily_block(
                {"intent": "educational",
                 "dataset_match_confidence": EDUCATIONAL_SEMANTIC_BLOCK_THRESHOLD}
            )
        )

    def test_matched_record_safe_blocks_even_at_high_similarity(self) -> None:
        self.assertFalse(
            should_preliminarily_block(
                {
                    "intent": "general",
                    "dataset_match_confidence": 0.99,
                    "matched_record_intent": "safe",
                }
            )
        )

    def test_accepts_dataclass_like_object(self) -> None:
        """`.get_value` should fall back to getattr for non-dict inputs."""

        class Fake:
            intent = "harmful"
            intent_confidence = HIGH_RISK_INTENT_CONFIDENCE
            dataset_match_confidence = 0.0
            matched_record_intent = None
            category_scores = {}

        self.assertTrue(should_preliminarily_block(Fake()))


class ReExportTests(unittest.TestCase):
    """The constants must still be importable from `core.decision_engine`."""

    def test_decision_engine_re_exports_constants(self) -> None:
        from core import decision_engine
        for name in (
            "HIGH_RISK_INTENT_CONFIDENCE",
            "SELF_HARM_CONFIDENCE",
            "SEMANTIC_BLOCK_THRESHOLD",
            "SEMANTIC_RISK_THRESHOLD",
            "EDUCATIONAL_SEMANTIC_BLOCK_THRESHOLD",
            "EDUCATIONAL_SEMANTIC_RISK_THRESHOLD",
        ):
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(decision_engine, name),
                    msg=f"core.decision_engine should re-export {name}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
