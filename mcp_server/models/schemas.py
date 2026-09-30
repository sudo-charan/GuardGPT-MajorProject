"""Strict, JSON-safe input and output contracts for the MCP tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PromptAnalysisInput(WireModel):
    prompt: str = Field(default="", max_length=6000)


class PromptAnalysisOutput(WireModel):
    intent: str
    intent_confidence: float = 0.0
    risk_level: str = "safe"
    category_scores: dict[str, float] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    matched_record_id: str | None = None
    matched_record_intent: str | None = None
    dataset_match_confidence: float = 0.0
    requires_jailbreak_check: StrictBool = False


class JailbreakDetectionInput(WireModel):
    prompt: str = Field(default="", max_length=6000)


class JailbreakDetectionOutput(WireModel):
    detected: StrictBool
    is_jailbreak: StrictBool
    attack_type: str
    detected_patterns: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    risk_level: str = "safe"
    reasons: list[str] = Field(default_factory=list)
    explanation: str = ""


class ContentModerationInput(WireModel):
    prompt: str = Field(default="", max_length=6000)


class ContentModerationOutput(WireModel):
    is_unsafe: StrictBool
    detected: StrictBool
    categories: list[str] = Field(default_factory=list)
    severity: str
    risk_level: str
    reasons: list[str] = Field(default_factory=list)
    explanation: str = ""


class DecisionInput(WireModel):
    prompt: str = Field(default="", max_length=6000)
    intent: str = "unknown"
    intent_confidence: float = 0.0
    risk_level: str = "safe"
    evidence: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    requires_jailbreak_check: StrictBool = False
    dataset_match_confidence: float = 0.0
    matched_record_id: str | None = None
    matched_record_intent: str | None = None
    category_scores: dict[str, float] = Field(default_factory=dict)
    detected_attacks: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    history_triggered: StrictBool = False
    history_block_reason: str = ""
    turn_index: int = 0


from typing import Any, Optional

class SessionReport(WireModel):
    session_id: str
    turn_count: int
    intent_history: list[str]
    current_intent: str
    previous_intent: Optional[str]
    transitions: list[dict[str, str]]
    session_risk_score: float
    session_summary: str


class DecisionOutput(WireModel):
    action: str
    final_status: str
    allowed: StrictBool
    intent: str
    risk_level: str
    reason_codes: list[str] = Field(default_factory=list)
    technical_reason: str = ""
    user_message: str = ""
    category_scores: dict[str, float] = Field(default_factory=dict)
    detected_attacks: list[str] = Field(default_factory=list)
    sanitized_prompt: Optional[str] = None
    audit_id: Optional[str] = None
    prompt_report: Optional['DecisionOutput'] = None
    session_report: Optional[SessionReport] = None


class AuditLoggerInput(WireModel):
    request_id: str | None = None
    prompt: str = Field(default="", max_length=6000)
    tool_name: str = "guardgpt"
    report: dict[str, Any] = Field(default_factory=dict)


class AuditLoggerOutput(WireModel):
    success: StrictBool
    audit_id: str
    log_path: str
    message: str
