"""
Audit Logger MCP tool - stub.

The canonical writer for `logs/guardgpt_audit.jsonl` is
`core.decision_engine.DecisionEngine._write_audit_log`, which is invoked
inside the MCP `decision` tool on every request.

This module keeps the `audit_logger` MCP tool registered for backwards
compatibility with callers that still expect to invoke it and receive an
`audit_id`. It mints and returns a synthesized `audit_id` but does not
touch the audit file.
"""

from __future__ import annotations

import logging
import os
import uuid
from functools import lru_cache
from pathlib import Path

# Importing the mcp_server package runs its __init__.py, which sets up
# sys.path so the following `models.*` imports resolve correctly.
import mcp_server  # noqa: F401

from mcp_server.models.schemas import (  # noqa: E402
    AuditLoggerInput,
    AuditLoggerOutput,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _audit_log_path() -> Path:
    root = Path(os.getenv("GUARDGPT_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
    return root / "logs" / "guardgpt_audit.jsonl"


def log_audit_event(data: AuditLoggerInput) -> AuditLoggerOutput:
    """
    Stub audit-logger for the MCP pipeline.

    `DecisionEngine._write_audit_log` (invoked via the MCP `decision` tool)
    is the sole writer of `logs/guardgpt_audit.jsonl`. This tool exists for
    backwards compatibility with callers that still expect to invoke an
    `audit_logger` MCP tool and receive an `audit_id`. It mints and returns
    a synthesized `audit_id` but does not touch the file.
    """
    log_path = _audit_log_path()
    audit_id = str(uuid.uuid4())
    return AuditLoggerOutput(
        success=True,
        audit_id=audit_id,
        log_path=str(log_path),
        message="Audit handled by core.decision_engine; this tool is a stub.",
    )