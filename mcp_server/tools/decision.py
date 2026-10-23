"""MCP decision tool: use JEV-Style as the sole prompt-action authority."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import lru_cache

import mcp_server  # noqa: F401
from mcp_server.models.schemas import DecisionInput, DecisionOutput  # noqa: E402
from core.complete_pipeline import AuditLog  # noqa: E402
from core.jev_style_client import JevStyleClient  # noqa: E402

logger = logging.getLogger(__name__)
JEV_ACTIONS = {"ALLOW", "SANITIZE", "BLOCK"}
JEV_FAILURE_MESSAGE = "A valid JEV-Style decision could not be obtained. No answer was generated."


@lru_cache(maxsize=1)
def _jev_style() -> JevStyleClient:
    return JevStyleClient()


@lru_cache(maxsize=1)
def _audit_log() -> AuditLog:
    return AuditLog()


def _attack_labels(data: DecisionInput) -> list[str]:
    labels = {
        "jailbreak", "prompt_injection", "instruction_override", "system_prompt_extraction",
        "role_manipulation", "safety_bypass", "jailbreak_attempt", "harm", "toxicity",
        "harmful_instructions", "cyber_abuse", "illegal", "violence", "hate_speech",
    }
    return list(dict.fromkeys(value for value in data.detected_attacks if value in labels))


def decide(data: DecisionInput) -> DecisionOutput:
    """Ask JEV-Style for one explicit action, then audit that result."""
    prompt = data.prompt or ""
    request_id = data.request_id or "req_" + uuid.uuid4().hex[:12]
    audit_id = uuid.uuid4().hex
    reasons = list(dict.fromkeys([*data.reason_codes, *data.reasons]))
    evidence = {
        "intent": data.intent,
        "intent_confidence": data.intent_confidence,
        "risk_level": data.risk_level,
        "dataset_match_confidence": data.dataset_match_confidence,
        "matched_record_id": data.matched_record_id,
        "matched_record_intent": data.matched_record_intent,
        "category_scores": data.category_scores,
        "detected_attacks": data.detected_attacks,
        "evidence": data.evidence,
        "reason_codes": reasons,
        "history_triggered": data.history_triggered,
        "history_block_reason": data.history_block_reason,
    }

    jev_result: dict = {}
    action = "BLOCK"
    final_status = "DECISION_UNAVAILABLE"
    allowed = False
    user_message = JEV_FAILURE_MESSAGE
    technical_reason = "JEV-Style did not provide a valid enforcement action."
    if not prompt.strip():
        jev_result = {"status": "invalid_input"}
        reasons.append("empty_input")
        final_status = "INVALID_INPUT"
        user_message = "Please enter a prompt."
    else:
        try:
            result = _jev_style().decide(prompt, data.session_events, analysis=evidence)
        except Exception as error:
            logger.warning("JEV-Style decision failed: %s", type(error).__name__)
            result = None
            jev_result = {"status": "unavailable", "error": type(error).__name__}

        if result is not None:
            if isinstance(result, Mapping):
                raw_action = result.get("decision")
                if isinstance(raw_action, str) and raw_action.strip().upper() in JEV_ACTIONS:
                    action = raw_action.strip().upper()
                    final_status = {"ALLOW": "SAFE", "SANITIZE": "CAUTION", "BLOCK": "UNSAFE"}[action]
                    allowed = action == "ALLOW"
                    user_message = {
                        "ALLOW": "JEV-Style allowed this request.",
                        "SANITIZE": "A rewrite is required. Use complete_request to rewrite, recheck and generate.",
                        "BLOCK": "JEV-Style blocked this request.",
                    }[action]
                    technical_reason = f"JEV-Style selected {action}."
                    jev_result = dict(result, status="ok", decision=action, action=action)
                else:
                    jev_result = {"status": "invalid_decision", "model": result.get("model")}
            else:
                jev_result = {"status": "invalid_response"}
        if action == "BLOCK" and final_status == "DECISION_UNAVAILABLE":
            reasons.append("jev_style_decision_unavailable")
        elif action == "SANITIZE":
            reasons.append("jev_style_sanitize")
        elif action == "BLOCK":
            reasons.append("jev_style_block")

    reasons = list(dict.fromkeys(reasons))
    valid_jev_decision = jev_result.get("status") == "ok" and action in JEV_ACTIONS
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "audit_id": audit_id,
        "session_id": data.session_id,
        "turn_index": data.turn_index,
        "intent": data.intent,
        "intent_confidence": data.intent_confidence,
        "risk_level": data.risk_level,
        "action": action,
        "final_status": final_status,
        "allowed": allowed,
        "reason_codes": reasons,
        "decision_source": "jev-style" if valid_jev_decision else None,
    }
    if valid_jev_decision:
        event["input_action"] = action
    audit_logged = True
    try:
        _audit_log().write(event)
    except Exception:
        logger.exception("Could not write JEV-Style decision audit event")
        action, final_status, allowed = "BLOCK", "ERROR", False
        user_message = "The audit log could not be saved; no answer was released."
        technical_reason = "The decision audit could not be saved."
        reasons.append("audit_log_failed")
        audit_logged = False

    return DecisionOutput(
        request_id=request_id,
        action=action,
        final_status=final_status,
        allowed=allowed,
        intent=data.intent,
        intent_confidence=data.intent_confidence,
        risk_level=data.risk_level,
        dataset_match_confidence=data.dataset_match_confidence,
        matched_record_id=data.matched_record_id,
        reason_codes=list(dict.fromkeys(reasons)),
        technical_reason=technical_reason,
        user_message=user_message,
        category_scores=dict(data.category_scores),
        detected_attacks=_attack_labels(data),
        sanitized_prompt=None,
        audit_id=audit_id,
        audit_logged=audit_logged,
        jev_style=jev_result,
    )
