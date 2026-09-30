"""
Canonical risk estimation and preliminary-block predicates.

This is the single source of truth for the GuardGPT risk vocabulary
(`safe / low / medium / high / critical`) and the block / risk thresholds
that were previously duplicated across `core.guard_engine`,
`mcp_server/tools/prompt_analysis`, and `core.decision_engine`.

Pure module — it MUST NOT import from `core.intent_classifier`,
`core.decision_engine`, `core.guard_engine`, or
`core.conversation_guard`, so it can be safely imported from any path
(including the MCP server which does not pull in the legacy engine).
"""

from __future__ import annotations

from typing import Any


# ============================================================
# Block / risk thresholds (single source of truth)
# ============================================================

# Minimum intent-classifier confidence required for an intent-based block.
HIGH_RISK_INTENT_CONFIDENCE = 0.65

# Confidence gate for a self-harm classifier label to count as risk evidence.
# This is a model score, not a calibrated probability.
SELF_HARM_CONFIDENCE = 0.65

# Semantic-match thresholds for general prompts.
SEMANTIC_BLOCK_THRESHOLD = 0.70
SEMANTIC_RISK_THRESHOLD = 0.55

# Higher thresholds required for blocking educational / conceptual queries.
EDUCATIONAL_SEMANTIC_BLOCK_THRESHOLD = 0.85
EDUCATIONAL_SEMANTIC_RISK_THRESHOLD = 0.80

# Minimum dataset similarity that counts as "low risk" for non-harm intents.
LOW_SIMILARITY_THRESHOLD = 0.40

# Self-harm critical-vs-high boundary (calibrated against the legacy engine).
SELF_HARM_CRITICAL_CONFIDENCE = 0.50


# ============================================================
# Intent sets
# ============================================================

SELF_HARM_INTENTS = {"self_harm", "self_harm_risk"}

EDUCATIONAL_INTENTS = {"educational", "coding", "benign", "account_recovery"}

HARM_INTENTS = {
    "harmful",
    "harmful_instructions",
    "cyber_abuse",
    "illegal",
    "prompt_injection",
    "jailbreak",
}


# ============================================================
# Public helpers
# ============================================================

def estimate_risk(
    intent: str,
    intent_confidence: float,
    similarity: float,
    matched_record_intent: str | None = None,
) -> str:
    """
    Map intent + confidence + dataset similarity to a risk level.

    Note: intent_confidence is a transformed cosine similarity score,
    not a calibrated probability of harmfulness.

    Returns one of: "safe", "low", "medium", "high", "critical".
    """
    if intent in SELF_HARM_INTENTS and intent_confidence >= SELF_HARM_CONFIDENCE:
        if intent_confidence >= SELF_HARM_CRITICAL_CONFIDENCE:
            return "critical"
        return "high"

    if intent in HARM_INTENTS:
        if intent_confidence >= HIGH_RISK_INTENT_CONFIDENCE:
            return "high"
        return "medium"

    if matched_record_intent == "safe":
        similarity = 0.0

    is_educational = intent in EDUCATIONAL_INTENTS
    high_threshold = (
        EDUCATIONAL_SEMANTIC_BLOCK_THRESHOLD
        if is_educational
        else SEMANTIC_BLOCK_THRESHOLD
    )
    med_threshold = (
        EDUCATIONAL_SEMANTIC_RISK_THRESHOLD
        if is_educational
        else SEMANTIC_RISK_THRESHOLD
    )

    if similarity >= high_threshold:
        return "high"
    if similarity >= med_threshold:
        return "medium"
    if not is_educational and similarity >= LOW_SIMILARITY_THRESHOLD:
        return "low"
    return "safe"


def should_preliminarily_block(result_or_mapping: Any) -> bool:
    """
    Cheap first-pass block check used by the legacy GuardEngine.
    """
    intent = _get_value(result_or_mapping, "intent", "unknown") or "unknown"
    intent = str(intent).lower()
    confidence = float(
        _get_value(result_or_mapping, "intent_confidence", 0.0) or 0.0
    )
    similarity = float(
        _get_value(result_or_mapping, "dataset_match_confidence", 0.0) or 0.0
    )
    matched_record_intent = _get_value(
        result_or_mapping, "matched_record_intent", None
    )
    category_scores = _get_value(
        result_or_mapping, "category_scores", {}
    ) or {}

    if intent in SELF_HARM_INTENTS and confidence >= SELF_HARM_CONFIDENCE:
        return True

    if intent in HARM_INTENTS:
        return confidence >= HIGH_RISK_INTENT_CONFIDENCE

    matched_record_is_safe = (
        matched_record_intent == "safe"
        or (float(category_scores.get("safe", 0.0) or 0.0) > 0.5)
    )
    is_educational = intent in EDUCATIONAL_INTENTS
    threshold = (
        EDUCATIONAL_SEMANTIC_BLOCK_THRESHOLD
        if is_educational
        else SEMANTIC_BLOCK_THRESHOLD
    )

    return similarity >= threshold and not matched_record_is_safe


# ============================================================
# Internal helpers
# ============================================================

def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    """Dict-aware attribute access (mirrors decision_engine._get_value)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


__all__ = [
    "HIGH_RISK_INTENT_CONFIDENCE",
    "SELF_HARM_CONFIDENCE",
    "SEMANTIC_BLOCK_THRESHOLD",
    "SEMANTIC_RISK_THRESHOLD",
    "EDUCATIONAL_SEMANTIC_BLOCK_THRESHOLD",
    "EDUCATIONAL_SEMANTIC_RISK_THRESHOLD",
    "LOW_SIMILARITY_THRESHOLD",
    "SELF_HARM_CRITICAL_CONFIDENCE",
    "SELF_HARM_INTENTS",
    "EDUCATIONAL_INTENTS",
    "HARM_INTENTS",
    "estimate_risk",
    "should_preliminarily_block",
]
