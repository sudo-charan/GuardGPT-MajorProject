"""Input checks shared by the complete request pipeline."""
from dataclasses import asdict
from functools import lru_cache
from core.decision_engine import DecisionEngine
from core.risk_estimator import estimate_risk, should_preliminarily_block


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
        self.engine = DecisionEngine(write_audit=False)

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
            category_scores=scores, reason_codes=[])
        preliminary = should_preliminarily_block(signal)
        if patterns:
            preliminary = True
            signal["reason_codes"].extend(patterns)
        signal["final_blocked"] = preliminary
        signal["block_reason"] = "Input safety check triggered." if preliminary else ""
        decision = asdict(self.engine.decide(signal))
        attacks = list(patterns)
        if confidence >= 0.65 and intent in {"jailbreak", "prompt_injection", "harmful", "cyber_abuse", "illegal"}:
            attacks.append(intent)
        # Self-harm is a support need, not an attack on the system.
        return dict(signal=signal, decision=decision,
                    detected_attacks=list(dict.fromkeys(attacks)),
                    matched_category_scores=raw_scores)

    def apply_history(self, analysis, guard):
        signal = dict(analysis["signal"])
        signal["final_blocked"] = analysis["decision"]["action"] == "BLOCK"
        history = guard.evaluate_result(signal)  # Exactly one recorded input turn.
        if history.history_triggered:
            signal.update(history_triggered=True, history_block_reason=history.history_block_reason)
            analysis["decision"] = asdict(self.engine.decide(signal, turn_index=history.turn_index))
        return history.turn_index
