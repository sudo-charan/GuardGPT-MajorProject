"""
Decision MCP tool - thin MCP wrapper around core.decision_engine.DecisionEngine.

The DecisionEngine remains the single source of truth for ALLOW / SANITIZE /
BLOCK. This tool only:

  - translates the DecisionInput into the engine's expected inputs,
  - calls DecisionEngine.decide(),
  - and shapes the DecisionOutput back to MCP wire format.

It MUST NOT introduce competing logic.
"""

from __future__ import annotations

import logging
from functools import lru_cache

# Importing the mcp_server package runs its __init__.py, which sets up
# sys.path so the following imports resolve correctly.
import mcp_server  # noqa: F401

from mcp_server.models.schemas import (  # noqa: E402
    DecisionInput,
    DecisionOutput,
)
from core.intent_classifier import GuardResult  # noqa: E402
from core.decision_engine import DecisionEngine  # noqa: E402

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _engine() -> DecisionEngine:
    return DecisionEngine()


def decide(data: DecisionInput) -> DecisionOutput:
    prompt = data.prompt or ""

    result = GuardResult(
        prompt=prompt,
        intent=data.intent or "unknown",
        intent_confidence=float(data.intent_confidence or 0.0),
        risk_level=data.risk_level or "safe",
        category_scores=dict(data.category_scores or {}),
        dataset_match_confidence=float(data.dataset_match_confidence or 0.0),
        matched_record_id=data.matched_record_id,
        matched_record_intent=data.matched_record_intent,
        final_blocked=False,
        block_reason="",
        reason_codes=list(data.reasons or []),
        history_triggered=bool(data.history_triggered),
        history_block_reason=data.history_block_reason or "",
    )

    decision = _engine().decide(result, turn_index=int(data.turn_index or 0))

    labels = {"jailbreak", "prompt_injection", "instruction_override", "system_prompt_extraction",
              "role_manipulation", "safety_bypass", "jailbreak_attempt", "harm", "toxicity",
              "harmful_instructions", "cyber_abuse", "illegal", "violence", "hate_speech"}
    detected_attacks = list(dict.fromkeys(x for x in (data.detected_attacks or []) if x in labels))
    if decision.action == "SANITIZE":
        decision.sanitized_prompt = None
        decision.user_message = "A rewrite is required. Use complete_request to rewrite, recheck and generate."

    return DecisionOutput(
        action=str(decision.action),
        final_status=str(decision.final_status),
        allowed=bool(decision.allowed),
        intent=str(decision.intent),
        risk_level=str(decision.risk_level),
        reason_codes=list(decision.reason_codes),
        technical_reason=str(decision.technical_reason),
        user_message=str(decision.user_message),
        category_scores=dict(decision.category_scores or {}),
        detected_attacks=detected_attacks,
        sanitized_prompt=decision.sanitized_prompt,
        audit_id=decision.audit_id,
    )
