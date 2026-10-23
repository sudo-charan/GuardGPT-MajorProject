"""Prompt analysis and conversation evidence shared by request paths.

This service supplies evidence only. The caller sends that evidence to
JEV-Style, which owns the prompt action.
"""
from functools import lru_cache
from core.risk_estimator import estimate_risk, should_preliminarily_block
import numpy as np


def _dataset_match_threshold(intent):
    return 0.8 if intent in {"coding", "educational", "benign", "account_recovery"} else 0.55


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
        threshold = _dataset_match_threshold(intent)
        scores = raw_scores if similarity >= threshold else {}

        from core.jailbreak_patterns import _pattern_hits
        patterns = _pattern_hits(prompt.lower())
        risk = estimate_risk(intent, confidence, similarity, matched_intent)
        signal = dict(prompt=prompt, intent=intent, intent_confidence=confidence,
            risk_level=risk, dataset_match_confidence=similarity,
            matched_record_id=record.get("request_id"), matched_record_intent=matched_intent,
            category_scores=scores, reason_codes=[])
        preliminary = should_preliminarily_block(signal)
        if patterns:
            preliminary = True
            signal["reason_codes"].extend(patterns)
        signal["preliminary_block"] = preliminary
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
        return dict(signal=signal, detected_attacks=list(dict.fromkeys(attacks)),
                    matched_category_scores=raw_scores)

    def finalize_turn(
        self,
        analysis,
        guard,
        final_action,
        final_status,
        *,
        qualifies_for_recovery=False,
    ):
        """
        Finalize the session state after a request completes.
        Updates history, behavioral flags, and temporal intent.
        """
        if not analysis or not guard:
            return None

        # A turn is blocked if JEV said BLOCK or if the request did not complete safely.
        is_blocked = (final_action == "BLOCK" or final_status != "SAFE")

        try:
            history = guard.record_decision(
                prompt=analysis["signal"]["prompt"],
                intent=analysis["signal"]["intent"],
                intent_confidence=analysis["signal"]["intent_confidence"],
                risk_level=analysis["signal"]["risk_level"],
                is_blocked=is_blocked,
                semantic_similarity=analysis["signal"]["dataset_match_confidence"],
                category_scores=analysis["signal"]["category_scores"],
                block_reason=(
                    "JEV-Style returned BLOCK."
                    if final_action == "BLOCK"
                    else "Request did not complete successfully."
                    if is_blocked
                    else ""
                ),
            )

            embedding = analysis["signal"].get("_temporal_embedding")
            if embedding is not None:
                update = guard.temporal_model.update(embedding, analysis["signal"]["dataset_match_confidence"])
                guard.temporal_state = update
        except Exception:
            guard.update_behavioral_state(qualifies_for_recovery=False)
            raise

        guard.update_behavioral_state(
            qualifies_for_recovery=bool(qualifies_for_recovery and not is_blocked)
        )

        return history.turn_index

    def apply_history(self, analysis, guard, action):
        signal = dict(analysis["signal"])
        is_blocked = action == "BLOCK"
        history = guard.record_decision(
            prompt=signal["prompt"],
            intent=signal["intent"],
            intent_confidence=signal["intent_confidence"],
            risk_level=signal["risk_level"],
            is_blocked=is_blocked,
            semantic_similarity=signal["dataset_match_confidence"],
            category_scores=signal["category_scores"],
            block_reason="JEV-Style returned BLOCK." if is_blocked else "",
        )
        update = guard.temporal_model.update(
            signal["_temporal_embedding"], signal["dataset_match_confidence"]
        )
        guard.temporal_state = update
        analysis["temporal_state"] = update
        analysis["history"] = {
            "turn_index": history.turn_index,
            "previous_block": history.previous_block,
            "session_flagged": history.session_flagged,
            "unsafe_ratio": history.unsafe_ratio,
        }
        return history.turn_index
