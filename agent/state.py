"""
GuardGPT Agent state.

The state is the structured payload passed between LangGraph nodes. Every
node reads from and writes to this state. Final output of the workflow is
the `Guard Report` carried in this dict.

The Agent MUST NOT do its own classification - it only coordinates MCP
tool results. The fields below describe what each tool contributes.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class GuardState(TypedDict, total=False):
    # ----------------------------------------------------------
    # Input
    # ----------------------------------------------------------
    request_id: str
    prompt: str
    mcp_url: Optional[str]

    # ----------------------------------------------------------
    # Tool outputs (populated by each MCP node)
    # ----------------------------------------------------------
    prompt_analysis: dict[str, Any]
    jailbreak_analysis: dict[str, Any]
    moderation_analysis: dict[str, Any]
    combined_analysis: dict[str, Any]
    decision_output: dict[str, Any]
    audit_output: dict[str, Any]

    # ----------------------------------------------------------
    # Final Guard Report fields (denormalized for easy consumption)
    # ----------------------------------------------------------
    intent: str
    intent_confidence: float
    risk_level: str
    category_scores: dict[str, float]
    detected_attacks: list[str]
    reasons: list[str]
    action: str            # ALLOW | SANITIZE | BLOCK
    final_status: str      # SAFE | CAUTION | UNSAFE
    sanitized_prompt: Optional[str]
    audit_id: Optional[str]

    # ----------------------------------------------------------
    # Operational
    # ----------------------------------------------------------
    errors: list[str]
    history_triggered: bool
    turn_index: int
    completed: bool
