"""
Jailbreak Detection MCP tool - thin adapter around existing GuardGPT core.

It does NOT create a new detection engine. It:

  - classifies the prompt via the existing IntentClassifier
    (which already covers prompt_injection / jailbreak intents), and
  - adds small deterministic pattern hints for transparency / reasons.

The classifier remains the source of truth; patterns only contribute
evidence / reasons.
"""

from __future__ import annotations

import logging
from functools import lru_cache

# Importing the mcp_server package runs its __init__.py, which sets up
# sys.path so the following imports resolve correctly.
import mcp_server  # noqa: F401

from mcp_server.models.schemas import (  # noqa: E402
    JailbreakDetectionInput,
    JailbreakDetectionOutput,
)
from core.intent_classifier import IntentClassifier  # noqa: E402

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _classifier() -> IntentClassifier:
    return IntentClassifier()


from core.jailbreak_patterns import _pattern_hits


def detect_jailbreak(
    data: JailbreakDetectionInput,
) -> JailbreakDetectionOutput:
    prompt = data.prompt.strip() if data.prompt else ""
    prompt_lower = prompt.lower()

    if not prompt:
        return JailbreakDetectionOutput(
            detected=False,
            is_jailbreak=False,
            attack_type="none",
            detected_patterns=[],
            categories=[],
            confidence=0.0,
            risk_level="safe",
            reasons=["empty_input"],
            explanation="Empty prompt - no jailbreak signals.",
        )

    classifier_output = _classifier().classify(prompt)
    intent = str(classifier_output.get("intent", "unknown"))
    confidence = float(classifier_output.get("confidence", 0.0) or 0.0)

    pattern_hits = _pattern_hits(prompt_lower)

    classifier_flag = intent in {"prompt_injection", "jailbreak"}
    pattern_flag = bool(pattern_hits)

    detected = classifier_flag or pattern_flag

    categories: list[str] = []
    reasons: list[str] = []

    if classifier_flag:
        categories.append(intent)
        reasons.append(
            f"IntentClassifier labeled prompt as '{intent}' "
            f"(confidence={confidence:.3f})."
        )

    for hit in pattern_hits:
        categories.append(hit)
        reasons.append(f"Matched deterministic pattern: {hit}.")

    if not detected:
        attack_type = "none"
        risk_level = "safe"
        confidence_out = max(0.05, 1.0 - confidence)
    else:
        if intent == "prompt_injection":
            attack_type = "prompt_injection"
        elif intent == "jailbreak":
            attack_type = "jailbreak"
        elif "system_prompt_extraction" in pattern_hits:
            attack_type = "system_prompt_extraction"
        elif "instruction_override" in pattern_hits:
            attack_type = "instruction_override"
        elif "role_manipulation" in pattern_hits:
            attack_type = "role_manipulation"
        else:
            attack_type = "safety_bypass"

        if intent == "jailbreak" or len(pattern_hits) >= 2:
            risk_level = "high"
            confidence_out = max(confidence, 0.90)
        else:
            risk_level = "high" if classifier_flag else "medium"
            confidence_out = max(confidence, 0.75)

    return JailbreakDetectionOutput(
        detected=detected,
        is_jailbreak=detected,
        attack_type=attack_type,
        detected_patterns=pattern_hits,
        categories=categories,
        confidence=round(confidence_out, 4),
        risk_level=risk_level,
        reasons=reasons,
        explanation=(
            "The prompt triggered jailbreak / prompt-injection signals "
            "via the GuardGPT IntentClassifier and/or deterministic patterns."
            if detected
            else "No jailbreak or prompt-injection signals were detected."
        ),
    )
