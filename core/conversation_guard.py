import logging
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from core.risk_estimator import (
    HIGH_RISK_INTENT_CONFIDENCE,
    SELF_HARM_CONFIDENCE,
    SELF_HARM_INTENTS,
)
from core.temporal_intent import TemporalIntentModel

logger = logging.getLogger(__name__)

HISTORY_WINDOW = 10
CONVERSATION_CONTEXT_MAX_BYTES = 4000
CONVERSATION_MESSAGE_MAX_BYTES = 1800
UNSAFE_RATIO_WARNING = 0.40
UNSAFE_RATIO_FLAG = 0.70
MIN_HISTORY_FOR_RATIO = 3

UNSAFE_INTENTS = {
    "harmful",
    "self_harm",
    "illegal",
    "cyber_abuse",
    "prompt_injection",
    "jailbreak",
}


@dataclass
class TurnRecord:
    turn_index: int
    timestamp: str
    prompt_snippet: str
    intent: str
    intent_confidence: float
    risk_level: str
    is_blocked: bool
    block_reason: str
    semantic_similarity: float = 0.0
    category_scores: dict = field(default_factory=dict)
    decision_status: str = "UNRECORDED"
    decision_failure_reason_code: str = ""


@dataclass
class ConversationResult:
    history_triggered: bool
    history_block_reason: str
    unsafe_ratio: float
    previous_block: bool
    session_flagged: bool
    session_flag_reason: str
    turn_index: int


class ConversationGuard:
    """Tracks multi-turn conversations and flags repeated recent unsafe inputs."""

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self._history = deque(maxlen=HISTORY_WINDOW)
        self._dialogue = deque(maxlen=HISTORY_WINDOW)
        self._turn_counter = 0
        self._flagged = False
        self._flag_reason = ""
        self.temporal_model = TemporalIntentModel()
        self.temporal_state = None

    def evaluate(
        self,
        prompt: str,
        intent: str,
        intent_confidence: float = 0.0,
        risk_level: str = "safe",
        is_blocked: bool = False,
        semantic_similarity: float = 0.0,
        category_scores: dict | None = None,
        block_reason: str = "",
    ) -> ConversationResult:
        current_turn = self._turn_counter + 1

        if self._flagged:
            history_reason = "Conversation session was flagged due to repeated unsafe activity."
            self._save_turn(
                prompt=prompt,
                intent=intent,
                intent_confidence=intent_confidence,
                risk_level=risk_level,
                is_blocked=True,
                block_reason=history_reason,
                semantic_similarity=semantic_similarity,
                category_scores=category_scores,
            )
            return ConversationResult(
                history_triggered=True,
                history_block_reason=history_reason,
                unsafe_ratio=self._unsafe_ratio(),
                previous_block=True,
                session_flagged=True,
                session_flag_reason=self._flag_reason,
                turn_index=current_turn,
            )

        previous_block = any(turn.is_blocked for turn in self._history)
        history_triggered = False
        history_reason = ""

        supported_intent = intent in UNSAFE_INTENTS and (
            intent not in SELF_HARM_INTENTS or intent_confidence >= SELF_HARM_CONFIDENCE
        )
        if previous_block and supported_intent:
            history_triggered = True
            history_reason = "Unsafe intent continued from a previous blocked message."
        elif previous_block and not is_blocked:
            ratio = self._unsafe_ratio()
            clearly_safe = risk_level == "safe" and (
                not category_scores or all(float(s) < 0.20 for s in category_scores.values())
            )
            if not clearly_safe and ratio >= UNSAFE_RATIO_WARNING:
                history_triggered = True
                history_reason = f"{ratio:.0%} of recent messages were unsafe."

        final_blocked = is_blocked or history_triggered
        final_reason = history_reason or block_reason

        self._save_turn(
            prompt=prompt,
            intent=intent,
            intent_confidence=intent_confidence,
            risk_level=risk_level,
            is_blocked=final_blocked,
            block_reason=final_reason,
            semantic_similarity=semantic_similarity,
            category_scores=category_scores,
        )

        ratio = self._unsafe_ratio()
        if self._decided_turn_count() >= MIN_HISTORY_FOR_RATIO and ratio >= UNSAFE_RATIO_FLAG:
            self._flagged = True
            self._flag_reason = "Repeated unsafe messages detected."

        return ConversationResult(
            history_triggered=history_triggered,
            history_block_reason=history_reason,
            unsafe_ratio=ratio,
            previous_block=previous_block,
            session_flagged=self._flagged,
            session_flag_reason=self._flag_reason,
            turn_index=self._turn_counter,
        )

    def evaluate_result(self, result: Any) -> ConversationResult:
        return self.evaluate(
            prompt=self._get_value(result, "prompt", ""),
            intent=self._get_value(result, "intent", "unknown"),
            intent_confidence=float(self._get_value(result, "intent_confidence", 0.0) or 0.0),
            risk_level=self._get_value(result, "risk_level", "safe"),
            is_blocked=bool(
                self._get_value(
                    result, "final_blocked", self._get_value(result, "is_blocked", False)
                )
            ),
            semantic_similarity=float(
                self._get_value(result, "dataset_match_confidence", 0.0) or 0.0
            ),
            category_scores=self._get_value(result, "category_scores", {}),
            block_reason=self._get_value(result, "block_reason", ""),
        )

    def _save_turn(
        self,
        prompt: str,
        intent: str,
        intent_confidence: float,
        risk_level: str,
        is_blocked: bool,
        block_reason: str,
        semantic_similarity: float,
        category_scores: dict | None,
        decision_status: str = "UNRECORDED",
        decision_failure_reason_code: str = "",
    ) -> None:
        self._turn_counter += 1
        record = TurnRecord(
            turn_index=self._turn_counter,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_snippet=str(prompt)[:80],
            intent=str(intent) if intent else "unknown",
            intent_confidence=round(float(intent_confidence), 4),
            risk_level=str(risk_level) if risk_level else "safe",
            is_blocked=bool(is_blocked),
            block_reason=str(block_reason) if block_reason else "",
            semantic_similarity=round(float(semantic_similarity), 4),
            category_scores=dict(category_scores) if isinstance(category_scores, dict) else {},
            decision_status=str(decision_status),
            decision_failure_reason_code=str(decision_failure_reason_code),
        )
        self._history.append(record)

    def _unsafe_ratio(self) -> float:
        decided = [turn for turn in self._history if turn.decision_status != "ERROR"]
        if len(decided) < MIN_HISTORY_FOR_RATIO:
            return 0.0
        return sum(turn.is_blocked for turn in decided) / len(decided)

    def _decided_turn_count(self) -> int:
        return sum(turn.decision_status != "ERROR" for turn in self._history)

    def _refresh_session_flag(self) -> None:
        """Keep the session flag aligned with the bounded recent-history window."""
        self._flagged = (
            self._decided_turn_count() >= MIN_HISTORY_FOR_RATIO
            and self._unsafe_ratio() >= UNSAFE_RATIO_FLAG
        )
        self._flag_reason = (
            "Repeated JEV-blocked messages detected." if self._flagged else ""
        )

    def get_recent_history(self, limit: int = HISTORY_WINDOW) -> list[TurnRecord]:
        limit = max(1, min(int(limit), len(self._history)))
        return list(self._history)[-limit:]

    def preview_history_evidence(self, signal: dict) -> dict[str, Any]:
        """Summarize prior-session risk for JEV without recording or deciding."""
        self._refresh_session_flag()
        previous_block = any(turn.is_blocked for turn in self._history)
        triggered = False
        reason = ""
        if self._flagged:
            triggered = True
            reason = "session_flagged"
        else:
            intent = str(signal.get("intent", "unknown"))
            confidence = float(signal.get("intent_confidence", 0.0) or 0.0)
            risk_level = str(signal.get("risk_level", "safe"))
            category_scores = signal.get("category_scores", {})
            confidence_gate = (
                SELF_HARM_CONFIDENCE
                if intent in SELF_HARM_INTENTS
                else HIGH_RISK_INTENT_CONFIDENCE
            )
            supported_intent = (
                intent in UNSAFE_INTENTS and confidence >= confidence_gate
            )
            if previous_block and supported_intent:
                triggered = True
                reason = "unsafe_intent_after_prior_block"
            elif previous_block:
                ratio = self._unsafe_ratio()
                clearly_safe = risk_level == "safe" and (
                    not category_scores or all(float(s) < 0.20 for s in category_scores.values())
                )
                if not clearly_safe and ratio >= UNSAFE_RATIO_WARNING:
                    triggered = True
                    reason = "unsafe_recent_history"
        return {
            "history_triggered": triggered,
            "history_reason_code": reason,
            "previous_block": previous_block,
            "recent_unsafe_ratio": self._unsafe_ratio(),
            "session_flagged": self._flagged,
            "recent_turn_count": len(self._history),
        }

    def record_jev_outcome(
        self,
        signal: dict,
        action: str,
        decision_failure_reason_code: str = "",
    ) -> ConversationResult:
        """Record one turn exactly as JEV decided, without history overrides."""
        if action not in {"ALLOW", "SANITIZE", "BLOCK"}:
            raise ValueError("Invalid JEV action for session history")
        evidence = self.preview_history_evidence(signal)
        is_blocked = action == "BLOCK"
        self._save_turn(
            prompt=self._get_value(signal, "prompt", ""),
            intent=self._get_value(signal, "intent", "unknown"),
            intent_confidence=float(self._get_value(signal, "intent_confidence", 0.0) or 0.0),
            risk_level=self._get_value(signal, "risk_level", "safe"),
            is_blocked=is_blocked,
            block_reason="jev_block" if is_blocked else "",
            semantic_similarity=float(self._get_value(signal, "dataset_match_confidence", 0.0) or 0.0),
            category_scores=self._get_value(signal, "category_scores", {}),
            decision_status=action,
            decision_failure_reason_code=decision_failure_reason_code,
        )
        self._refresh_session_flag()
        ratio = self._unsafe_ratio()
        return ConversationResult(
            history_triggered=evidence["history_triggered"],
            history_block_reason=evidence["history_reason_code"],
            unsafe_ratio=ratio,
            previous_block=evidence["previous_block"],
            session_flagged=self._flagged,
            session_flag_reason=self._flag_reason,
            turn_index=self._turn_counter,
        )

    def record_jev_failure(self, signal: dict, reason_code: str) -> ConversationResult:
        """Record an unavailable/invalid decision without inventing a JEV action."""
        if reason_code not in {
            "jev_action_invalid", "jev_unavailable", "jev_decision_failed",
        }:
            reason_code = "jev_decision_failed"
        evidence = self.preview_history_evidence(signal)
        self._save_turn(
            prompt=self._get_value(signal, "prompt", ""),
            intent=self._get_value(signal, "intent", "unknown"),
            intent_confidence=float(self._get_value(signal, "intent_confidence", 0.0) or 0.0),
            risk_level=self._get_value(signal, "risk_level", "safe"),
            is_blocked=False,
            block_reason="",
            semantic_similarity=float(self._get_value(signal, "dataset_match_confidence", 0.0) or 0.0),
            category_scores=self._get_value(signal, "category_scores", {}),
            decision_status="ERROR",
            decision_failure_reason_code=reason_code,
        )
        self._refresh_session_flag()
        return ConversationResult(
            history_triggered=evidence["history_triggered"],
            history_block_reason=evidence["history_reason_code"],
            unsafe_ratio=self._unsafe_ratio(),
            previous_block=evidence["previous_block"],
            session_flagged=self._flagged,
            session_flag_reason=self._flag_reason,
            turn_index=self._turn_counter,
        )

    @staticmethod
    def _bounded_text(value: str, max_bytes: int) -> str:
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    def add_conversation_turn(self, user_message: str, assistant_message: str) -> None:
        """Keep a bounded, in-memory copy of one successfully answered turn."""
        self._dialogue.append({
            "user": self._bounded_text(str(user_message), CONVERSATION_MESSAGE_MAX_BYTES),
            "assistant": self._bounded_text(str(assistant_message), CONVERSATION_MESSAGE_MAX_BYTES),
        })

    def get_conversation_context(
        self, max_bytes: int = CONVERSATION_CONTEXT_MAX_BYTES
    ) -> list[dict[str, str]]:
        """Return the newest complete turns that fit the serialized byte cap."""
        max_bytes = max(0, min(int(max_bytes), CONVERSATION_CONTEXT_MAX_BYTES))
        selected: list[dict[str, str]] = []
        for turn in reversed(self._dialogue):
            candidate = [turn, *selected]
            size = len(json.dumps(candidate, ensure_ascii=False).encode("utf-8"))
            if size > max_bytes:
                break
            selected = candidate
        return [dict(turn) for turn in selected]

    def reset(self) -> None:
        self._history.clear()
        self._dialogue.clear()
        self._turn_counter = 0
        self._flagged = False
        self._flag_reason = ""
        self.temporal_model.reset()
        self.temporal_state = None

    def generate_session_report(self, request_id: str) -> dict:
        """Generate a session-level summary from the TurnRecord history."""
        history = self.history
        turn_count = len(history)
        intent_history = [turn.intent for turn in history]

        current_intent = intent_history[-1] if intent_history else "unknown"
        previous_intent = intent_history[-2] if len(intent_history) >= 2 else None

        transitions = []
        for i in range(1, len(intent_history)):
            prev = intent_history[i-1]
            curr = intent_history[i]
            if prev != curr:
                transitions.append({"from": prev, "to": curr})

        risk_score = self._unsafe_ratio()

        # Deterministic summary string
        summary = (
            f"Session consists of {turn_count} turns. "
            f"Risk score: {risk_score:.2%}. "
            f"Intent sequence: {', '.join(intent_history)}."
        )

        return {
            "session_id": self.session_id,
            "request_id": request_id,
            "turn_count": turn_count,
            "intent_history": intent_history,
            "current_intent": current_intent,
            "previous_intent": previous_intent,
            "transitions": transitions,
            "session_risk_score": risk_score,
            "session_summary": summary,
        }


    @staticmethod
    def _get_value(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    @property
    def history(self) -> list[TurnRecord]:
        return list(self._history)

    @property
    def unsafe_ratio(self) -> float:
        return self._unsafe_ratio()
