"""Compatibility MCP decision tool using JEV-Style as action authority."""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache

# Importing the package sets up the existing project import path.
import mcp_server  # noqa: F401

from mcp_server.models.schemas import DecisionInput, DecisionOutput  # noqa: E402
from core.jev_style_client import (  # noqa: E402
    JevActionResponseError,
    JevStyleClient,
    JevStyleUnavailable,
)

logger = logging.getLogger(__name__)
JEV_ACTIONS = {"ALLOW", "SANITIZE", "BLOCK"}


@lru_cache(maxsize=1)
def _jev_client() -> JevStyleClient:
    return JevStyleClient()


def _failure_code(error: Exception) -> str:
    if isinstance(error, JevActionResponseError):
        return "jev_action_invalid"
    if isinstance(error, JevStyleUnavailable):
        return "jev_unavailable"
    return "jev_decision_failed"


def decide(data: DecisionInput) -> DecisionOutput:
    """Return a fail-closed decision from JEV, preserving the MCP wire model."""
    intent = data.intent or "unknown"
    risk_level = data.risk_level or "unknown"
    evidence = {
        "local_intent": intent,
        "local_intent_confidence": float(data.intent_confidence or 0.0),
        "local_risk_level": risk_level,
        "dataset_match_confidence": float(data.dataset_match_confidence or 0.0),
        "matched_record_id": data.matched_record_id,
        "matched_record_intent": data.matched_record_intent,
        "category_scores": dict(data.category_scores or {}),
        "deterministic_patterns": list(data.detected_attacks or []),
        "reason_codes": list(data.reasons or data.reason_codes or []),
        "local_preliminary_risk": bool(data.requires_jailbreak_check),
        "history_triggered": bool(data.history_triggered),
    }
    try:
        result = _jev_client().decide(data.prompt or "", (), evidence=evidence)
        action = result.get("action") if isinstance(result, dict) else None
        if not isinstance(action, str) or action not in JEV_ACTIONS:
            raise JevActionResponseError("Missing or invalid JEV action")
    except Exception as error:
        code = _failure_code(error)
        logger.warning("JEV decision unavailable or invalid (%s)", code)
        return DecisionOutput(
            action="BLOCK",
            final_status="ERROR",
            allowed=False,
            intent=intent,
            risk_level=risk_level,
            reason_codes=[code],
            technical_reason="JEV did not provide a validated action.",
            user_message="The required safety decision could not be validated.",
            category_scores=dict(data.category_scores or {}),
            detected_attacks=list(data.detected_attacks or []),
            audit_id=uuid.uuid4().hex,
            jev_style={"status": "invalid" if code == "jev_action_invalid" else "unavailable",
                       "action": None, "failure_reason_code": code},
        )

    final_status = {"ALLOW": "SAFE", "SANITIZE": "CAUTION", "BLOCK": "UNSAFE"}[action]
    messages = {
        "ALLOW": "JEV approved the request for processing.",
        "SANITIZE": "JEV requires a safe rewrite before processing.",
        "BLOCK": "JEV blocked the request.",
    }
    reason_codes = list(data.reasons or data.reason_codes or [])
    if action == "BLOCK":
        reason_codes.append("jev_action_block")
    jev_style = {
        "status": "available",
        "decision": result.get("classification_category"),
        "choice": result.get("classification_category"),
        "confidence": result.get("classification_confidence"),
        "probabilities": result.get("classification_probabilities"),
        "classification_category": result.get("classification_category"),
        "classification_confidence": result.get("classification_confidence"),
        "classification_probabilities": result.get("classification_probabilities"),
        "action": action,
        "action_confidence": result.get("action_confidence"),
        "action_probabilities": result.get("action_probabilities"),
        "decision_provenance": result.get("decision_provenance"),
        "model": result.get("model"),
        "latency_ms": result.get("latency_ms"),
    }
    return DecisionOutput(
        action=action,
        final_status=final_status,
        allowed=action != "BLOCK",
        intent=intent,
        risk_level=risk_level,
        reason_codes=list(dict.fromkeys(reason_codes)),
        technical_reason="Action selected by JEV-Style using the supplied local evidence.",
        user_message=messages[action],
        category_scores=dict(data.category_scores or {}),
        detected_attacks=list(data.detected_attacks or []),
        sanitized_prompt=None,
        audit_id=uuid.uuid4().hex,
        jev_style=jev_style,
    )
