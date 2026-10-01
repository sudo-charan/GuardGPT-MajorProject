import unittest

import numpy as np
from unittest.mock import Mock

from core.conversation_guard import ConversationGuard
from core.safety_service import SafetyService
from core.temporal_intent import EMBEDDING_DIM, INTENT_DIM, LABELS, TemporalIntentModel


class TemporalIntentTests(unittest.TestCase):
    def test_labels_and_dimensions(self):
        model = TemporalIntentModel()
        self.assertEqual(model.labels, LABELS)
        self.assertEqual(LABELS, ("safe", "prompt_injection", "jailbreak", "harmful_instructions", "manipulation", "self_harm_risk"))
        self.assertEqual(model.H_t.shape, (384,))
        self.assertEqual(model.I_t.shape, (6,))
        self.assertEqual(model.W_c.shape, (6, 384))
        self.assertEqual(model.W_h.shape, (6, 384))
        self.assertEqual(model.W_r.shape, (6, 1))
        self.assertEqual(model.W_I.shape, (6, 6))
        self.assertEqual(model.b.shape, (6,))
        self.assertEqual(model.W_g.shape, (1, 775))
        self.assertEqual(model.b_g.shape, (1,))

    def test_update_equations_and_probabilities(self):
        model = TemporalIntentModel()
        previous_H, previous_I = model.H_t.copy(), model.I_t.copy()
        embedding = np.linspace(-1.0, 1.0, 384)
        update = model.update(embedding, 0.7)
        expected_H = model.lambda_ * previous_H + (1 - model.lambda_) * embedding
        self.assertTrue(np.allclose(update.H_t, expected_H))
        next_embedding = np.linspace(1.0, -1.0, 384)
        previous_H, previous_I = update.H_t.copy(), update.I_t.copy()
        second = model.update(next_embedding, 0.3)
        expected_second_H = model.lambda_ * previous_H + (1 - model.lambda_) * next_embedding
        self.assertTrue(np.allclose(second.H_t, expected_second_H))
        self.assertAlmostEqual(float(update.I_hat_t.sum()), 1.0)
        self.assertAlmostEqual(float(update.I_t.sum()), 1.0)
        self.assertGreaterEqual(update.g_t, 0.0)
        self.assertLessEqual(update.g_t, 1.0)

        model.W_c.fill(0)
        model.W_h.fill(0)
        model.W_r.fill(0)
        model.b.fill(0)
        model.W_g.fill(0)
        model.W_I.fill(0)
        model.W_I[0, 0] = 10
        model.reset()
        candidate = model.update(np.zeros(384), 0.0).I_hat_t
        self.assertGreater(candidate[0], candidate[1])

    def test_candidate_gate_and_intent_update_equations(self):
        model = TemporalIntentModel()
        model.update(np.ones(384), 0.4)
        previous_H, previous_I = model.H_t.copy(), model.I_t.copy()
        embedding, confidence = np.full(384, 0.25), 0.7
        update = model.update(embedding, confidence)
        logits = (model.W_c @ embedding + model.W_h @ previous_H
                  + model.W_r[:, 0] * confidence + model.W_I @ previous_I + model.b)
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        gate_input = np.concatenate((embedding, previous_H, [confidence], previous_I))
        gate = 1 / (1 + np.exp(-((model.W_g @ gate_input)[0] + model.b_g[0])))
        expected_intent = (1 - gate) * previous_I + gate * probabilities
        self.assertTrue(np.allclose(update.I_hat_t, probabilities))
        self.assertAlmostEqual(update.g_t, gate)
        self.assertTrue(np.allclose(update.I_t, expected_intent))

    def test_state_persists_reset_is_deterministic_and_sessions_are_isolated(self):
        model = TemporalIntentModel()
        first = model.update(np.ones(384), 0.2)
        second = model.update(np.full(384, 2.0), 0.8)
        self.assertFalse(np.allclose(first.H_t, second.H_t))
        model.reset()
        self.assertTrue(np.array_equal(model.H_t, np.zeros(384)))
        self.assertTrue(np.array_equal(model.I_t, np.full(6, 1 / 6)))
        session_a, session_b = ConversationGuard("a"), ConversationGuard("b")
        session_a.temporal_model.update(np.ones(384), 0.5)
        self.assertTrue(np.array_equal(session_b.temporal_model.H_t, np.zeros(384)))
        session_a.reset()
        self.assertTrue(np.array_equal(session_a.temporal_model.H_t, session_b.temporal_model.H_t))
        self.assertTrue(np.array_equal(session_a.temporal_model.I_t, session_b.temporal_model.I_t))

    def test_sequential_updates_are_deterministic(self):
        inputs = [(np.linspace(0, 1, 384), 0.2), (np.linspace(1, 0, 384), 0.8)]
        first = TemporalIntentModel()
        second = TemporalIntentModel()
        first_updates = [first.update(vector, confidence) for vector, confidence in inputs]
        second_updates = [second.update(vector, confidence) for vector, confidence in inputs]
        for left, right in zip(first_updates, second_updates):
            self.assertTrue(np.array_equal(left.H_t, right.H_t))
            self.assertTrue(np.array_equal(left.I_t, right.I_t))
            self.assertEqual(left.g_t, right.g_t)

    def test_temporal_update_is_observational(self):
        loader, classifier = Mock(), Mock()
        loader.query.return_value = {"request_id": "r", "intent": "safe", "_similarity": 0.99,
            "category_scores": {"harm": 0.0}}
        classifier.classify.return_value = {"intent": "coding", "confidence": 0.9}
        service = SafetyService(lambda: (loader, classifier))
        analysis = service.analyze("Explain lists")
        original_action = analysis["decision"]["action"]
        guard = ConversationGuard("observational")
        service.apply_history(analysis, guard)
        self.assertEqual(analysis["decision"]["action"], original_action)
        self.assertIsNotNone(analysis["temporal_state"])

    def test_input_validation(self):
        model = TemporalIntentModel()
        with self.assertRaises(ValueError):
            model.update(np.zeros(383), 0.5)
        with self.assertRaises(ValueError):
            model.update(np.zeros(384), 1.1)


if __name__ == "__main__":
    unittest.main()