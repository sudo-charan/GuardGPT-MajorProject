"""
Prompt Analysis MCP tool - thin adapter around the existing GuardGPT core.

This tool does NOT introduce a new classifier. It reuses:

    core.intent_classifier.IntentClassifier
    core.dataset_loader.DatasetLoader

and returns the same intent / risk / category signals the core already
produces. It only formats them for the MCP wire format.
"""

from __future__ import annotations

import logging
from functools import lru_cache

# Importing the mcp_server package runs its __init__.py, which adds the
# project root and the mcp_server directory to sys.path so the following
# `models.*` and `core.*` imports resolve correctly.
import mcp_server  # noqa: F401

from mcp_server.models.schemas import (  # noqa: E402
    PromptAnalysisInput,
    PromptAnalysisOutput,
)
from core.intent_classifier import IntentClassifier  # noqa: E402
from core.dataset_loader import DatasetLoader  # noqa: E402
from core.risk_estimator import estimate_risk  # noqa: E402

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


def _record_id(record: dict) -> str | None:
    for key in ("request_id", "id", "record_id", "prompt_id", "uuid"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def _category_scores(record: dict) -> dict:
    for key in ("category_scores", "scores", "safety_scores"):
        value = record.get(key)
        if isinstance(value, dict):
            return {str(k): float(v) for k, v in value.items()}
    return {}


def analyze_prompt(data: PromptAnalysisInput) -> PromptAnalysisOutput:
    """
    Analyze a user prompt using existing GuardGPT core components.
    """
    prompt = data.prompt.strip() if data.prompt else ""

    if not prompt:
        return PromptAnalysisOutput(
            intent="unknown",
            intent_confidence=0.0,
            risk_level="safe",
            reason_codes=["empty_input"],
        )

    classifier = _classifier()
    classifier_output = classifier.classify(prompt)

    intent = str(classifier_output.get("intent", "unknown"))
    intent_confidence = float(classifier_output.get("confidence", 0.0) or 0.0)

    dataset_match_confidence = 0.0
    matched_record_id = None
    matched_record_intent = None
    category_scores: dict = {}

    try:
        loader = _loader()
        record = loader.query(prompt)
        if record:
            dataset_match_confidence = float(record.get("_similarity", 0.0) or 0.0)
            matched_record_id = _record_id(record)
            matched_record_intent = record.get("intent")
            category_scores = _category_scores(record)
    except Exception as error:
        logger.warning("Dataset query failed in prompt_analysis: %s", error)
        raise RuntimeError("Dataset query failed") from error

    risk_level = estimate_risk(
        intent=intent,
        intent_confidence=intent_confidence,
        similarity=dataset_match_confidence,
        matched_record_intent=matched_record_intent,
    )

    reason_codes: list[str] = []
    if intent in {
        "harmful",
        "harmful_instructions",
        "cyber_abuse",
        "illegal",
        "prompt_injection",
        "jailbreak",
    }:
        reason_codes.append("high_risk_intent")
    if intent in {"self_harm", "self_harm_risk"}:
        reason_codes.append("critical_intent")

    requires_jailbreak_check = intent in {
        "prompt_injection",
        "jailbreak",
    } or any(score >= 0.50 for score in category_scores.values())

    evidence = []
    if matched_record_id is not None:
        evidence.append(
            f"Matched dataset record {matched_record_id} "
            f"with similarity {dataset_match_confidence:.3f}"
        )
    if intent_confidence >= 0.50:
        evidence.append(
            f"IntentClassifier confidence for '{intent}' = {intent_confidence:.3f}"
        )

    return PromptAnalysisOutput(
        intent=intent,
        intent_confidence=round(intent_confidence, 4),
        risk_level=risk_level,
        category_scores=category_scores,
        evidence=evidence,
        reason_codes=reason_codes,
        matched_record_id=matched_record_id,
        matched_record_intent=matched_record_intent,
        dataset_match_confidence=round(dataset_match_confidence, 4),
        requires_jailbreak_check=requires_jailbreak_check,
    )
