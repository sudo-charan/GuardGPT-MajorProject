"""
Content Moderation MCP tool - thin adapter around existing GuardGPT core.

It does NOT invent a new taxonomy. It reuses:

  - core.intent_classifier.IntentClassifier
  - category scores from core.dataset_loader.DatasetLoader

and maps the core's existing risk signals to the moderation output schema.
"""

from __future__ import annotations

import logging
from functools import lru_cache

# Importing the mcp_server package runs its __init__.py, which sets up
# sys.path so the following imports resolve correctly.
import mcp_server  # noqa: F401

from mcp_server.models.schemas import (  # noqa: E402
    ContentModerationInput,
    ContentModerationOutput,
)
from core.intent_classifier import IntentClassifier  # noqa: E402
from core.dataset_loader import DatasetLoader  # noqa: E402

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _classifier() -> IntentClassifier:
    return IntentClassifier()


@lru_cache(maxsize=1)
def _loader() -> DatasetLoader:
    loader = DatasetLoader()
    try:
        loader.load()
    except Exception as error:
        logger.warning("DatasetLoader could not load: %s", error)
        raise RuntimeError("Dataset unavailable") from error
    return loader


_INTENT_TO_CATEGORY = {
    "self_harm": "self_harm",
    "self_harm_risk": "self_harm",
    "harmful": "harmful_instructions",
    "harmful_instructions": "harmful_instructions",
    "illegal": "illegal_activity",
    "cyber_abuse": "malicious_cyber_activity",
    "prompt_injection": "prompt_injection",
    "jailbreak": "jailbreak_attempt",
}


def _severity_from_risk(risk_level: str, intent: str) -> str:
    if intent in {"self_harm", "self_harm_risk"}:
        return "critical"
    if risk_level == "critical":
        return "critical"
    if risk_level == "high":
        return "high"
    if risk_level == "medium":
        return "medium"
    if risk_level == "low":
        return "low"
    return "none"


def moderate_content(
    data: ContentModerationInput,
) -> ContentModerationOutput:
    prompt = data.prompt.strip() if data.prompt else ""

    if not prompt:
        return ContentModerationOutput(
            is_unsafe=False,
            detected=False,
            categories=[],
            severity="none",
            risk_level="safe",
            reasons=["empty_input"],
            explanation="Empty prompt - no content moderation signals.",
        )

    classifier = _classifier().classify(prompt)
    intent = str(classifier.get("intent", "unknown"))
    confidence = float(classifier.get("confidence", 0.0) or 0.0)

    category_scores: dict = {}
    try:
        record = _loader().query(prompt)
        if record:
            scores = record.get("category_scores")
            if isinstance(scores, dict):
                category_scores = {str(k): float(v) for k, v in scores.items()}
    except Exception as error:
        logger.warning("Dataset query failed in content_moderation: %s", error)
        raise RuntimeError("Dataset query failed") from error

    categories: list[str] = []
    reasons: list[str] = []

    mapped = _INTENT_TO_CATEGORY.get(intent)
    if mapped:
        categories.append(mapped)
        reasons.append(
            f"IntentClassifier labeled prompt as '{intent}' "
            f"(confidence={confidence:.3f})."
        )

    for key, value in category_scores.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric >= 0.50 and key not in categories:
            categories.append(key)
            reasons.append(
                f"Dataset category score '{key}' = {numeric:.3f}."
            )

    detected = bool(categories)
    is_unsafe = detected

    if intent in {"self_harm", "self_harm_risk"} and confidence >= 0.40:
        risk_level = "critical"
    elif intent in {
        "harmful",
        "harmful_instructions",
        "cyber_abuse",
        "illegal",
        "prompt_injection",
        "jailbreak",
    } and confidence >= 0.45:
        risk_level = "high"
    elif category_scores:
        best = max(category_scores.values()) if category_scores else 0.0
        if best >= 0.70:
            risk_level = "high"
        elif best >= 0.40:
            risk_level = "medium"
        elif best >= 0.20:
            risk_level = "low"
        else:
            risk_level = "safe"
    else:
        risk_level = "safe"

    severity = _severity_from_risk(risk_level, intent)
    if not detected:
        severity = "none"

    return ContentModerationOutput(
        is_unsafe=is_unsafe,
        detected=detected,
        categories=categories,
        severity=severity,
        risk_level=risk_level,
        reasons=reasons,
        explanation=(
            "Prompt contains content matching one or more unsafe categories "
            "based on GuardGPT IntentClassifier and dataset category scores."
            if detected
            else "No unsafe content categories were detected."
        ),
    )
