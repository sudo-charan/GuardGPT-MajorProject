"""MCP audit tool writing compact, prompt-free security events."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import mcp_server  # noqa: F401
from mcp_server.models.schemas import AuditLoggerInput, AuditLoggerOutput  # noqa: E402
from core.complete_pipeline import AuditLog  # noqa: E402

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _audit_log_path() -> Path:
    configured_path = os.getenv("GUARDGPT_AUDIT_LOG_PATH")
    if configured_path:
        return Path(configured_path)
    root = Path(os.getenv("GUARDGPT_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
    return root / "logs" / "guardgpt_audit.jsonl"


def log_audit_event(data: AuditLoggerInput) -> AuditLoggerOutput:
    """Persist the supplied report's decision metadata without prompt text."""
    report = data.report or {}
    decision_source = report.get("decision_source", data.tool_name)
    input_action = report.get("input_action")
    jev_style = report.get("jev_style")
    valid_input_decision = (
        isinstance(jev_style, Mapping)
        and jev_style.get("status") == "ok"
        and jev_style.get("decision") in {"ALLOW", "SANITIZE", "BLOCK"}
        and input_action == jev_style.get("decision")
    )
    if decision_source == "jev-style" and not valid_input_decision:
        decision_source = None
    record_jev_provenance = decision_source == "jev-style" and valid_input_decision
    audit_id = str(report.get("audit_id") or uuid.uuid4())
    request_id = data.request_id or report.get("request_id") or "req_" + uuid.uuid4().hex[:12]
    log_path = _audit_log_path()
    event = {
        "timestamp": report.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "audit_id": audit_id,
        "session_id": report.get("session_id"),
        "turn_index": report.get("turn_index"),
        "intent": report.get("intent", "unknown"),
        "intent_confidence": report.get("intent_confidence", 0.0),
        "risk_level": report.get("risk_level", "unknown"),
        "action": report.get("action", "BLOCK"),
        "final_status": report.get("final_status", "ERROR"),
        "allowed": bool(report.get("allowed", False)),
        "reason_codes": report.get("reason_codes", report.get("reasons", [])),
        "decision_source": decision_source,
    }
    if record_jev_provenance:
        event["input_action"] = input_action
        recheck = report.get("jev_style_recheck")
        if (
            input_action == "SANITIZE"
            and isinstance(recheck, Mapping)
            and recheck.get("status") == "ok"
            and recheck.get("decision") in {"ALLOW", "SANITIZE", "BLOCK"}
        ):
            event["jev_style_recheck_action"] = recheck["decision"]
    try:
        AuditLog(log_path).write(event)
    except Exception:
        logger.exception("Could not write MCP audit event")
        raise
    return AuditLoggerOutput(
        success=True,
        audit_id=audit_id,
        log_path=str(log_path),
        message="Decision metadata was written to the GuardGPT audit log.",
    )
