"""
GuardGPT Agent nodes.

Each node is a thin function over the MCP client. The Agent does NOT
implement its own safety classifier, jailbreak detector, content
moderator, or decision engine. Those live in the GuardGPT core and are
exposed through MCP tools.

Workflow:

    ReceivePrompt
        |
        v
    PromptAnalysis   -> MCP tool: prompt_analysis
        |
        v
    JailbreakDetect  -> MCP tool: jailbreak_detection
        |
        v
    ModerateContent  -> MCP tool: content_moderation
        |
        v
    CombineResults
        |
        v
    Decision         -> MCP tool: decision
        |
        v
    AuditLog         -> MCP tool: audit_logger
        |
        v
    BuildReport      -> final Guard Report
        |
        v
    END
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from agent.mcp_client import (
    MCPConnectionError,
    MCPClientError,
    MCPToolError,
    call_tool,
)

from agent.state import GuardState


logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def _safe_call(
    state: GuardState,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Invoke an MCP tool safely.

    - Logs and records the error in `state["errors"]` on failure
    - Returns an empty dict on failure so downstream nodes can keep going
    """
    url = state.get("mcp_url")
    try:
        result = call_tool(tool_name, arguments, url=url) if url else call_tool(tool_name, arguments)
        return result.data
    except MCPConnectionError as error:
        msg = f"{tool_name}: MCP server unreachable - {error}"
        logger.warning(msg)
        state.setdefault("errors", []).append(msg)
        return {"_error": "mcp_connection_error", "_message": str(error)}
    except MCPToolError as error:
        msg = f"{tool_name}: tool failure - {error}"
        logger.warning(msg)
        state.setdefault("errors", []).append(msg)
        return {"_error": "mcp_tool_error", "_message": str(error)}
    except MCPClientError as error:
        msg = f"{tool_name}: MCP client error - {error}"
        logger.warning(msg)
        state.setdefault("errors", []).append(msg)
        return {"_error": "mcp_client_error", "_message": str(error)}


def _prompt_argument(prompt: str) -> dict[str, Any]:
    return {"data": {"prompt": prompt}}


# ============================================================
# Node 1: ReceivePrompt
# ============================================================

def receive_prompt(state: GuardState) -> GuardState:
    """Initialize the workflow state for a new prompt."""
    if not state.get("request_id"):
        state["request_id"] = _new_request_id()
    if "errors" not in state:
        state["errors"] = []
    state["completed"] = False
    return state


# ============================================================
# Node 2: PromptAnalysis
# ============================================================

def run_prompt_analysis(state: GuardState) -> GuardState:
    """Call MCP `prompt_analysis` and store its result in state."""
    prompt = state.get("prompt", "") or ""
    if not prompt:
        state["prompt_analysis"] = {
            "intent": "unknown",
            "intent_confidence": 0.0,
            "risk_level": "safe",
            "category_scores": {},
            "evidence": [],
            "reason_codes": ["empty_input"],
            "dataset_match_confidence": 0.0,
            "requires_jailbreak_check": False,
        }
        return state

    state["prompt_analysis"] = _safe_call(
        state,
        "prompt_analysis",
        _prompt_argument(prompt),
    )
    return state


# ============================================================
# Node 3: JailbreakDetection
# ============================================================

def run_jailbreak_detection(state: GuardState) -> GuardState:
    """Call MCP `jailbreak_detection`."""
    prompt = state.get("prompt", "") or ""
    state["jailbreak_analysis"] = _safe_call(
        state,
        "jailbreak_detection",
        _prompt_argument(prompt),
    )
    return state


# ============================================================
# Node 4: ContentModeration
# ============================================================

def run_content_moderation(state: GuardState) -> GuardState:
    """Call MCP `content_moderation`."""
    prompt = state.get("prompt", "") or ""
    state["moderation_analysis"] = _safe_call(
        state,
        "content_moderation",
        _prompt_argument(prompt),
    )
    return state


# ============================================================
# Node 5: CombineResults
# ============================================================

def combine_results(state: GuardState) -> GuardState:
    """
    Merge the three analysis outputs into a single payload to feed the
    DecisionEngine MCP tool.

    The combination preserves the original signals - it does NOT
    classify anything new. It just unifies categories / detected attacks /
    reasons into a single dict the decision tool can consume.
    """
    prompt_analysis = state.get("prompt_analysis") or {}
    jailbreak_analysis = state.get("jailbreak_analysis") or {}
    moderation_analysis = state.get("moderation_analysis") or {}

    intent = str(prompt_analysis.get("intent", "unknown") or "unknown")
    intent_confidence = float(prompt_analysis.get("intent_confidence", 0.0) or 0.0)
    risk_level = str(prompt_analysis.get("risk_level", "safe") or "safe")
    dataset_match_confidence = float(
        prompt_analysis.get("dataset_match_confidence", 0.0) or 0.0
    )

    category_scores: dict[str, float] = {}
    for source in (prompt_analysis, moderation_analysis):
        scores = source.get("category_scores")
        if isinstance(scores, dict):
            for key, value in scores.items():
                try:
                    category_scores[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

    detected_attacks: list[str] = []
    for source in (jailbreak_analysis, moderation_analysis):
        for key in ("categories", "detected_patterns"):
            values = source.get(key)
            if isinstance(values, list):
                for value in values:
                    value_str = str(value)
                    if value_str and value_str not in detected_attacks:
                        detected_attacks.append(value_str)

    reasons: list[str] = []
    for source in (prompt_analysis, jailbreak_analysis, moderation_analysis):
        for key in ("reason_codes", "reasons"):
            values = source.get(key)
            if isinstance(values, list):
                for value in values:
                    value_str = str(value)
                    if value_str and value_str not in reasons:
                        reasons.append(value_str)

    matched_record_id = prompt_analysis.get("matched_record_id")
    matched_record_intent = prompt_analysis.get("matched_record_intent")

    state["combined_analysis"] = {
        "intent": intent,
        "intent_confidence": intent_confidence,
        "risk_level": risk_level,
        "dataset_match_confidence": dataset_match_confidence,
        "matched_record_id": matched_record_id,
        "matched_record_intent": matched_record_intent,
        "category_scores": category_scores,
        "detected_attacks": detected_attacks,
        "reasons": reasons,
        "history_triggered": bool(state.get("history_triggered", False)),
        "history_block_reason": state.get("history_block_reason", "") or "",
        "turn_index": int(state.get("turn_index", 0) or 0),
    }
    return state


# ============================================================
# Node 6: Decision
# ============================================================

def run_decision(state: GuardState) -> GuardState:
    """Call MCP `decision` with the combined analysis."""
    combined = state.get("combined_analysis") or {}
    prompt = state.get("prompt", "") or ""

    decision_input = {
        "data": {
            "prompt": prompt,
            "intent": combined.get("intent", "unknown"),
            "intent_confidence": combined.get("intent_confidence", 0.0),
            "risk_level": combined.get("risk_level", "safe"),
            "dataset_match_confidence": combined.get("dataset_match_confidence", 0.0),
            "matched_record_id": combined.get("matched_record_id"),
            "matched_record_intent": combined.get("matched_record_intent"),
            "category_scores": combined.get("category_scores", {}),
            "detected_attacks": combined.get("detected_attacks", []),
            "reasons": combined.get("reasons", []),
            "history_triggered": combined.get("history_triggered", False),
            "history_block_reason": combined.get("history_block_reason", ""),
            "turn_index": combined.get("turn_index", 0),
        }
    }

    state["decision_output"] = _safe_call(state, "decision", decision_input)
    return state


# ============================================================
# Node 7: AuditLog
# ============================================================

def run_audit_log(state: GuardState) -> GuardState:
    """
    No-op audit step.

    `audit_id` is generated by `core.decision_engine.DecisionEngine.decide()`
    and surfaced via `state["decision_output"]["audit_id"]`. The MCP
    `audit_logger` tool is now a stub; this node exists to preserve the
    graph shape and set the final `completed` flag.
    """
    state["audit_output"] = {}
    state["completed"] = True
    return state


# ============================================================
# Node 8: BuildReport
# ============================================================

def build_report_node(state: GuardState) -> GuardState:
    """Final node - flatten the workflow state into a Guard Report."""
    report = build_report(state)
    state.update({
        "intent": report.get("intent", "unknown"),
        "intent_confidence": report.get("intent_confidence", 0.0),
        "risk_level": report.get("risk_level", "safe"),
        "category_scores": report.get("category_scores", {}),
        "detected_attacks": report.get("detected_attacks", []),
        "reasons": report.get("reasons", []),
        "action": report.get("action", "ALLOW"),
        "final_status": report.get("final_status", "SAFE"),
        "sanitized_prompt": report.get("sanitized_prompt"),
        "audit_id": report.get("audit_id"),
    })
    return state


# ============================================================
# Guard Report builder (shared by Audit and Report nodes)
# ============================================================

def build_report(state: GuardState) -> dict[str, Any]:
    """
    Assemble the final Guard Report from the workflow state.

    Fields:
      - request_id
      - prompt
      - intent
      - intent_confidence
      - risk_level
      - category_scores
      - detected_attacks
      - reasons
      - action        (ALLOW | SANITIZE | BLOCK)
      - final_status  (SAFE | CAUTION | UNSAFE)
      - sanitized_prompt
      - audit_id
      - timestamp
    """
    decision = state.get("decision_output") or {}
    audit = state.get("audit_output") or {}
    combined = state.get("combined_analysis") or {}
    prompt_analysis = state.get("prompt_analysis") or {}

    if state.get("errors") or decision.get("_error") or not decision:
        return {"request_id": state.get("request_id"), "prompt": state.get("prompt", ""),
                "action": "BLOCK", "final_status": "ERROR", "allowed": False,
                "intent": "unknown", "risk_level": "unknown", "detected_attacks": [],
                "reasons": ["analysis_failed"], "response": None,
                "user_message": "A required input check failed."}

    action = str(decision.get("action", "BLOCK") or "BLOCK")
    final_status = str(decision.get("final_status", "SAFE") or "SAFE")
    allowed = bool(decision.get("allowed", action != "BLOCK"))

    category_scores = decision.get("category_scores")
    if not isinstance(category_scores, dict):
        category_scores = combined.get("category_scores", {})
    if not isinstance(category_scores, dict):
        category_scores = {}

    detected_attacks = decision.get("detected_attacks")
    if not isinstance(detected_attacks, list):
        detected_attacks = combined.get("detected_attacks", [])
    if not isinstance(detected_attacks, list):
        detected_attacks = []

    reasons = decision.get("reason_codes")
    if not isinstance(reasons, list):
        reasons = combined.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = []

    technical_reason = decision.get("technical_reason", "")
    user_message = decision.get("user_message", "")

    sanitized_prompt = decision.get("sanitized_prompt")

    audit_id = decision.get("audit_id") or audit.get("audit_id")

    return {
        "request_id": state.get("request_id"),
        "prompt": state.get("prompt", ""),
        "timestamp": _now_iso(),
        "intent": decision.get("intent")
        or combined.get("intent")
        or prompt_analysis.get("intent", "unknown"),
        "intent_confidence": float(
            decision.get("intent_confidence")
            if decision.get("intent_confidence") is not None
            else combined.get("intent_confidence", 0.0)
        ),
        "risk_level": decision.get("risk_level")
        or combined.get("risk_level")
        or "safe",
        "category_scores": category_scores,
        "detected_attacks": [str(x) for x in detected_attacks],
        "reasons": [str(x) for x in reasons],
        "technical_reason": technical_reason,
        "user_message": user_message,
        "action": action,
        "final_status": final_status,
        "allowed": allowed,
        "sanitized_prompt": sanitized_prompt,
        "audit_id": audit_id,
    }
