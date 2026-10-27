"""Input checks shared by the complete request pipeline."""
from functools import lru_cache
from core.risk_estimator import estimate_risk, should_preliminarily_block
import numpy as np


def _self_harm_context(prompt: str) -> str | None:
    """Label support/facilitation context as evidence, never as an action."""
    text = " ".join(prompt.lower().split())
    facilitation_cues = (
        "instructions for seriously injuring", "instructions to injure",
        "instructions for self-injury", "how to injure myself",
        "seriously injure myself", "hide self-inflicted injuries",
        "conceal self-inflicted injuries", "hide my injuries",
        "conceal my injuries", "cover up self-inflicted injuries",
    )
    if any(cue in text for cue in facilitation_cues):
        return "facilitation"

    friend_support_cues = (
        "help a friend", "how can i help a friend", "help someone who is suicidal",
        "talking about suicide",
    )
    if any(cue in text for cue in friend_support_cues):
        return "supporting_another_person"

    if any(cue in text for cue in ("healthy ways to cope", "cope with emotional pain")):
        return "coping_support"

    support_cues = (
        "i feel hopeless", "help me figure out what to do", "help me stay safe",
        "thoughts of hurting myself", "suicide prevention",
    )
    if any(cue in text for cue in support_cues):
        return "support_seeking"
    return None


@lru_cache(maxsize=1)
def resources():
    # Lazy imports allow diagnostics, help and offline tests without model loading.
    from core.dataset_loader import DatasetLoader
    from core.intent_classifier import IntentClassifier
    loader = DatasetLoader()
    loader.load()  # Missing or invalid artifacts must stop analysis.
    classifier = IntentClassifier()
    classifier._model = loader._model
    return loader, classifier


class SafetyService:
    def __init__(self, resource_provider=resources):
        self.resource_provider = resource_provider

    def analyze(self, prompt):
        loader, classifier = self.resource_provider()
        classified = classifier.classify(prompt)
        intent = str(classified.get("intent", "unknown"))
        confidence = float(classified.get("confidence", 0.0))
        record = loader.query(prompt)
        if not record:
            raise ValueError("Semantic search returned no record")
        similarity = float(record["_similarity"])
        matched_intent = record["intent"]
        raw_scores = dict(record.get("category_scores", {}))
        threshold = 0.8 if intent in {"coding", "educational", "benign", "account_recovery"} else 0.55
        scores = raw_scores if similarity >= threshold else {}

        from core.jailbreak_patterns import _pattern_hits
        patterns = _pattern_hits(prompt.lower())
        risk = estimate_risk(intent, confidence, similarity, matched_intent)
        signal = dict(prompt=prompt, intent=intent, intent_confidence=confidence,
            risk_level=risk, dataset_match_confidence=similarity,
            matched_record_id=record.get("request_id"), matched_record_intent=matched_intent,
            category_scores=scores, reason_codes=[],
            self_harm_context=_self_harm_context(prompt))
        preliminary = should_preliminarily_block(signal)
        if patterns:
            preliminary = True
            signal["reason_codes"].extend(patterns)
        # These are evidence for JEV-Style. They do not decide or enforce an
        # action in the active request pipeline.
        signal["local_preliminary_risk"] = preliminary
        embedding = getattr(classifier, "last_embedding", None)
        signal["_temporal_embedding"] = (
            np.asarray(embedding, dtype=np.float64).copy()
            if isinstance(embedding, np.ndarray) and embedding.shape == (384,)
            else np.zeros(384, dtype=np.float64)
        )
        attacks = list(patterns)
        if confidence >= 0.65 and intent in {"jailbreak", "prompt_injection", "harmful", "cyber_abuse", "illegal"}:
            attacks.append(intent)
        # Self-harm is a support need, not an attack on the system.
        evidence = {
            "local_intent": intent,
            "local_intent_confidence": confidence,
            "local_risk_level": risk,
            "dataset_match_confidence": similarity,
            "matched_record_id": record.get("request_id"),
            "category_scores": scores,
            "deterministic_patterns": list(patterns),
            "reason_codes": list(dict.fromkeys(signal["reason_codes"])),
            "local_preliminary_risk": preliminary,
        }
        if signal["self_harm_context"] is not None:
            evidence["self_harm_context"] = signal["self_harm_context"]
        # Keep a retrieval label as JEV evidence only when the same existing
        # similarity gate used for category scores accepts the match. The full
        # matched intent remains on signal for reporting and local analysis.
        if similarity >= threshold:
            evidence["matched_record_intent"] = matched_intent
        return dict(signal=signal, evidence=evidence,
                    detected_attacks=list(dict.fromkeys(attacks)),
                    matched_category_scores=raw_scores)

    def apply_history(self, analysis, guard, action, decision_failure_reason_code=""):
        """Record one JEV outcome while keeping history as evidence only."""
        signal = dict(analysis["signal"])
        if action is None:
            history = guard.record_jev_failure(signal, decision_failure_reason_code)
        else:
            history = guard.record_jev_outcome(
                signal, action, decision_failure_reason_code
            )  # Exactly one recorded input turn.
        analysis["history_evidence"] = {
            "history_triggered": history.history_triggered,
            "history_block_reason": history.history_block_reason,
            "unsafe_ratio": history.unsafe_ratio,
            "session_flagged": history.session_flagged,
        }
        if history.history_triggered:
            signal.update(history_triggered=True, history_block_reason=history.history_block_reason)
        update = guard.temporal_model.update(
            signal["_temporal_embedding"], signal["dataset_match_confidence"]
        )
        guard.temporal_state = update
        analysis["temporal_state"] = update
        return history.turn_index
