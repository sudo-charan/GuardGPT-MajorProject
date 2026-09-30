# ============================================================
# GuardGPT - decision_engine.py
# ============================================================
# PURPOSE:
#   Make the final ALLOW / BLOCK decision using the signals
#   produced by the IntentClassifier, DatasetLoader and
#   ConversationGuard.
#
# PIPELINE:
#   IntentClassifier + DatasetLoader/FAISS + ConversationGuard
#          ↓
#   DecisionEngine
#          ↓
#   ALLOW / BLOCK
# ============================================================

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

# Block / risk thresholds live in `core.risk_estimator` (single source of
# truth). They are re-exported here for backward compatibility with any
# caller that does `from core.decision_engine import HIGH_RISK_INTENT_CONFIDENCE`.
from core.risk_estimator import (  # noqa: F401
    HIGH_RISK_INTENT_CONFIDENCE,
    SELF_HARM_CONFIDENCE,
    SELF_HARM_CRITICAL_CONFIDENCE,
    SEMANTIC_BLOCK_THRESHOLD,
    SEMANTIC_RISK_THRESHOLD,
    EDUCATIONAL_SEMANTIC_BLOCK_THRESHOLD,
    EDUCATIONAL_SEMANTIC_RISK_THRESHOLD,
)

# Conversation history signal (decision-engine-specific).
HISTORY_BLOCK_ENABLED = True


# ============================================================
# DATASET LABELS & INTENT GROUPS
# ============================================================

UNSAFE_DATASET_LABELS = {
    "prompt_injection",
    "jailbreak",
    "harmful_instructions",
    "manipulation",
    "self_harm_risk",
    "harm",
    "toxicity",
}

UNSAFE_INTENTS = {
    "unsafe",
    "harmful",
    "harmful_instructions",
    "cyber_abuse",
    "cyberattack_intent",
    "illegal",
    "prompt_injection",
    "jailbreak",
    "jailbreak_attempt",
    "self_harm",
    "self_harm_risk",
    "hate_speech_harassment",
}

HIGH_RISK_INTENTS = {
    "harmful",
    "harmful_instructions",
    "cyber_abuse",
    "illegal",
    "prompt_injection",
    "jailbreak",
    "manipulation",
}

CRITICAL_INTENTS = {
    "self_harm",
    "self_harm_risk",
}


# ============================================================
# DECISION OUTPUT
# ============================================================

@dataclass
class DecisionOutput:
    """Final security decision returned by DecisionEngine."""

    allowed: bool
    intent: str
    risk_level: str
    user_message: str
    technical_reason: str
    category_scores: dict = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    dataset_match_confidence: float = 0.0
    matched_record_id: Optional[str] = None
    history_triggered: bool = False
    turn_index: int = 0
    intent_confidence: float = 0.0
    semantic_match: bool = False
    action: str = "ALLOW"
    final_status: str = "SAFE"
    sanitized_prompt: Optional[str] = None
    audit_id: Optional[str] = None


# ============================================================
# DECISION ENGINE
# ============================================================

class DecisionEngine:
    """
    Produces the final ALLOW / BLOCK decision.

    Combines signals produced by IntentClassifier, DatasetLoader,
    and ConversationGuard.
    """

    def __init__(
        self,
        audit_log_path: str = "logs/guardgpt_audit.jsonl",
        write_audit: bool = True,
    ) -> None:
        import os

        resolved = Path(audit_log_path)
        if not resolved.is_absolute():
            root = os.getenv("GUARDGPT_PROJECT_ROOT")
            if root:
                resolved = Path(root).resolve() / resolved
        self.audit_log_path = resolved
        self.write_audit = write_audit
        logger.info("DecisionEngine initialized.")

    # ========================================================
    # FINAL DECISION
    # ========================================================

    def decide(
        self,
        result: Any,
        turn_index: int = 0,
    ) -> DecisionOutput:
        """Produce the final security decision."""

        # ----------------------------------------------------
        # Extract values safely
        # ----------------------------------------------------

        intent = str(
            self._get_value(result, "intent", "unknown") or "unknown"
        ).lower()

        intent_confidence = self._safe_float(
            self._get_value(result, "intent_confidence", 0.0)
        )

        risk_level = str(
            self._get_value(result, "risk_level", "safe") or "safe"
        ).lower()

        similarity = self._safe_float(
            self._get_value(result, "dataset_match_confidence", 0.0)
        )

        matched_record_id = self._get_value(result, "matched_record_id", None)
        matched_record_intent = self._get_value(result, "matched_record_intent", None)

        category_scores = self._get_value(result, "category_scores", {})
        if not isinstance(category_scores, dict):
            category_scores = {}

        history_triggered = bool(self._get_value(result, "history_triggered", False))
        history_reason = str(
            self._get_value(result, "history_block_reason", "") or ""
        )

        preliminary_block = bool(self._get_value(result, "final_blocked", False))
        preliminary_reason = str(
            self._get_value(result, "block_reason", "") or ""
        )

        incoming_reason_codes = self._get_value(result, "reason_codes", [])
        if not isinstance(incoming_reason_codes, list):
            incoming_reason_codes = []

        reason_codes = list(dict.fromkeys(map(str, incoming_reason_codes)))

        # ====================================================
        # SECURITY SIGNALS
        # ====================================================

        high_risk_intent = (
            intent in HIGH_RISK_INTENTS
            and intent_confidence >= HIGH_RISK_INTENT_CONFIDENCE
        )

        critical_intent = (
            intent in CRITICAL_INTENTS
            and intent_confidence >= SELF_HARM_CONFIDENCE
        )

        history_block = HISTORY_BLOCK_ENABLED and history_triggered

        dataset_label = self._get_strongest_dataset_label(category_scores)
        dataset_label_unsafe = (
            dataset_label in UNSAFE_DATASET_LABELS
            and self._safe_float(category_scores.get(dataset_label, 0.0)) >= 0.5
        )

        # Educational / Conceptual intent check
        is_educational = intent in {"educational", "coding", "benign", "account_recovery"}

        # Dynamic Threshold Selection based on Intent Context
        required_block_threshold = (
            EDUCATIONAL_SEMANTIC_BLOCK_THRESHOLD
            if is_educational
            else SEMANTIC_BLOCK_THRESHOLD
        )

        required_risk_threshold = (
            EDUCATIONAL_SEMANTIC_RISK_THRESHOLD
            if is_educational
            else SEMANTIC_RISK_THRESHOLD
        )

        semantic_match = similarity >= required_block_threshold
        strong_semantic_risk = similarity >= required_risk_threshold

        # ====================================================
        # FINAL BLOCK DECISION
        # ====================================================

        should_block = False
        technical_reasons = []

        # Rule 1: Existing preliminary block
        if preliminary_block:
            should_block = True
            if preliminary_reason:
                technical_reasons.append(preliminary_reason)

        # Rule 2: Critical intent
        if critical_intent:
            should_block = True
            reason_codes.append("critical_intent")
            technical_reasons.append("High-confidence critical safety intent detected.")

        # Rule 3: High-risk intent
        elif high_risk_intent:
            should_block = True
            reason_codes.append("high_risk_intent")
            technical_reasons.append("High-confidence unsafe intent detected.")

        matched_record_is_safe = (
            matched_record_intent == "safe"
            or (category_scores.get("safe", 0.0) > 0.5)
        )

        matched_record_unsafe = (
            (matched_record_intent in UNSAFE_INTENTS)
            or dataset_label_unsafe
        )

        # Keep raw similarity in the report; safe matches are not risk evidence.
        risk_similarity = 0.0 if matched_record_is_safe else similarity

        # Rule 4: Strong semantic dataset match
        if semantic_match and not matched_record_is_safe:
            should_block = True
            reason_codes.append("strong_semantic_match")
            technical_reasons.append(
                "Prompt strongly matches a harmful-prompt dataset example."
            )

        # Rule 5: Unsafe dataset category
        if matched_record_unsafe and strong_semantic_risk and not matched_record_is_safe:
            should_block = True
            reason_codes.append("unsafe_dataset_category")
            technical_reasons.append(
                f"Semantic match belongs to unsafe dataset category: {matched_record_intent or dataset_label}."
            )

        # Rule 6: Conversation history
        if history_block:
            should_block = True
            reason_codes.append("history_unsafe")
            if history_reason:
                technical_reasons.append(history_reason)
            else:
                technical_reasons.append("Unsafe behaviour continued across conversation turns.")

        # ====================================================
        # REMOVE DUPLICATES & CALC RISK
        # ====================================================

        reason_codes = list(dict.fromkeys(reason_codes))
        technical_reasons = list(dict.fromkeys(technical_reasons))

        final_risk = self._calculate_final_risk(
            intent=intent,
            intent_confidence=intent_confidence,
            similarity=risk_similarity,
            history_triggered=history_triggered,
            should_block=should_block,
        )

        # ====================================================
        # ALLOW / BLOCK
        # ====================================================

        if not should_block:
            allowed = True
            user_message = "Your prompt is safe to process."
            technical_reason = "No high-confidence security violation was detected."
        else:
            allowed = False
            user_message = self._build_block_message(
                intent="unknown" if intent in CRITICAL_INTENTS and not critical_intent else intent,
                history_triggered=history_triggered,
            )
            technical_reason = (
                "; ".join(technical_reasons)
                if technical_reasons
                else "Security policy violation detected."
            )

        # ====================================================
        # CREATE OUTPUT
        # ====================================================

        action, final_status = self._derive_action_and_status(
            allowed=allowed,
            final_risk=final_risk,
            intent=intent,
            similarity=risk_similarity,
        )

        sanitized_prompt = (
            self._sanitize_prompt(
                prompt=self._get_value(result, "prompt", "") or "",
                intent=intent,
                final_risk=final_risk,
            )
            if action == "SANITIZE"
            else None
        )

        allowed = action != "BLOCK"

        if action == "BLOCK" and not reason_codes:
            if high_risk_intent:
                reason_codes.append("high_risk_intent")
            elif critical_intent:
                reason_codes.append("critical_intent")
            elif preliminary_block:
                reason_codes.append("preliminary_block_triggered")
            else:
                reason_codes.append("security_policy_violation")

        if action == "SANITIZE" and not reason_codes:
            if final_risk == "medium":
                reason_codes.append("medium_risk_similarity")
            elif intent in UNSAFE_INTENTS:
                reason_codes.append("unsafe_intent_caution")
            else:
                reason_codes.append("sanitization_required")

        reason_codes = list(dict.fromkeys(reason_codes))

        output = DecisionOutput(
            allowed=allowed,
            intent=intent,
            risk_level=final_risk,
            user_message=user_message,
            technical_reason=technical_reason,
            category_scores=dict(category_scores),
            reason_codes=reason_codes,
            dataset_match_confidence=similarity,
            matched_record_id=str(matched_record_id) if matched_record_id is not None else None,
            history_triggered=history_triggered,
            turn_index=turn_index,
            intent_confidence=intent_confidence,
            semantic_match=semantic_match,
            action=action,
            final_status=final_status,
            sanitized_prompt=sanitized_prompt,
            audit_id=str(uuid.uuid4()),
        )

        # ====================================================
        # AUDIT LOG
        # ====================================================

        if self.write_audit:
            self._write_audit_log(result=result, decision=output)

        logger.info(
            "Decision=%s | intent=%s | risk=%s | similarity=%.3f",
            "ALLOW" if allowed else "BLOCK",
            intent,
            final_risk,
            similarity,
        )

        return output

    # ========================================================
    # FINAL RISK & ACTION
    # ========================================================

    @staticmethod
    def _derive_action_and_status(
        allowed: bool,
        final_risk: str,
        intent: str,
        similarity: float,
    ) -> tuple[str, str]:
        """
        Map the existing block decision to the spec's 3-way action:

            ALLOW    -> safe to send downstream as-is
            SANITIZE -> risky but not blocked; needs downstream rewrite
            BLOCK    -> disallowed

        Preserves existing safety rules (allowed = not should_block).
        """
        if not allowed:
            return "BLOCK", "UNSAFE"

        if final_risk in {"critical", "high"}:
            return "BLOCK", "UNSAFE"

        if final_risk == "medium":
            return "SANITIZE", "CAUTION"

        if final_risk == "low":
            unsafe_intents = {
                "harmful",
                "harmful_instructions",
                "cyber_abuse",
                "illegal",
                "prompt_injection",
                "jailbreak",
                "self_harm",
                "self_harm_risk",
            }
            if intent in unsafe_intents:
                return "SANITIZE", "CAUTION"

        return "ALLOW", "SAFE"

    @staticmethod
    def _sanitize_prompt(
        prompt: str,
        intent: str,
        final_risk: str,
    ) -> Optional[str]:
        """
        Produce a minimal, conservative sanitization of the prompt for the
        SANITIZE action.
        """
        text = str(prompt or "").strip()
        if not text:
            return None

        if final_risk == "critical" or intent in {
            "self_harm",
            "self_harm_risk",
            "harmful",
            "illegal",
            "cyber_abuse",
        }:
            return (
                "I cannot help with that request. "
                "If you are in crisis, please contact local emergency "
                "services or a crisis helpline."
            )

        if intent in {"prompt_injection", "jailbreak"}:
            return (
                "Please rephrase your request without instructions that "
                "attempt to override my safety guidelines."
            )

        return text

    @staticmethod
    def _calculate_final_risk(
        intent: str,
        intent_confidence: float,
        similarity: float,
        history_triggered: bool,
        should_block: bool,
    ) -> str:
        if intent in CRITICAL_INTENTS and intent_confidence >= SELF_HARM_CONFIDENCE:
            return "critical" if intent_confidence >= SELF_HARM_CRITICAL_CONFIDENCE else "high"

        if should_block:
            if intent in HIGH_RISK_INTENTS or similarity >= 0.85 or history_triggered:
                return "high"
            return "medium"

        is_educational = intent in {"educational", "coding", "benign", "account_recovery"}
        med_thresh = EDUCATIONAL_SEMANTIC_RISK_THRESHOLD if is_educational else SEMANTIC_RISK_THRESHOLD

        if similarity >= med_thresh:
            return "medium"
        if not is_educational and similarity >= 0.55:
            return "low"

        return "safe"

    # ========================================================
    # BLOCK MESSAGE
    # ========================================================

    @staticmethod
    def _build_block_message(
        intent: str,
        history_triggered: bool,
    ) -> str:
        if history_triggered:
            return (
                "I can't help with this request because "
                "the conversation contains repeated unsafe "
                "or restricted content."
            )

        if intent in {"self_harm", "self_harm_risk"}:
            return "I can't assist with instructions or content that could facilitate self-harm."

        if intent in {"prompt_injection", "jailbreak"}:
            return "I can't follow instructions intended to bypass or override safety controls."

        if intent == "cyber_abuse":
            return "I can't assist with harmful or unauthorized cyber activity."

        if intent == "illegal":
            return "I can't provide instructions that facilitate illegal activity."

        if intent in {"harmful", "harmful_instructions"}:
            return "I can't provide instructions that could facilitate harmful activity."

        if intent == "manipulation":
            return "I can't assist with harmful or deceptive manipulation."

        return "I can't process this request because it was identified as unsafe."

    # ========================================================
    # DATASET LABEL
    # ========================================================

    @staticmethod
    def _get_strongest_dataset_label(
        category_scores: dict,
    ) -> Optional[str]:
        if not category_scores:
            return None

        best_label = None
        best_score = float("-inf")

        for label, score in category_scores.items():
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue

            if numeric_score > best_score:
                best_score = numeric_score
                best_label = str(label).lower()

        return best_label

    # ========================================================
    # AUDIT LOG
    # ========================================================

    def _write_audit_log(
        self,
        result: Any,
        decision: DecisionOutput,
    ) -> None:
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            prompt = self._get_value(result, "prompt", "")

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "turn_index": decision.turn_index,
                "allowed": decision.allowed,
                "intent": decision.intent,
                "intent_confidence": decision.intent_confidence,
                "risk_level": decision.risk_level,
                "reason_codes": decision.reason_codes,
                "technical_reason": decision.technical_reason,
                "dataset_match_confidence": decision.dataset_match_confidence,
                "matched_record_id": decision.matched_record_id,
                "history_triggered": decision.history_triggered,
                "category_scores": decision.category_scores,
                "prompt_snippet": str(prompt)[:120],
            }

            for optional_key, optional_value in (
                ("action", getattr(decision, "action", None)),
                ("final_status", getattr(decision, "final_status", None)),
                ("audit_id", getattr(decision, "audit_id", None)),
            ):
                if optional_value is not None:
                    entry[optional_key] = optional_value

            with self.audit_log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception as error:
            logger.warning("Unable to write audit log: %s", error)

    # ========================================================
    # SAFE VALUE ACCESS
    # ========================================================

    @staticmethod
    def _get_value(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    # ========================================================
    # SAFE FLOAT
    # ========================================================

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
