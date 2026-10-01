"""Deterministic observational temporal intent layer."""

from dataclasses import dataclass

import numpy as np


LABELS = (
    "safe",
    "prompt_injection",
    "jailbreak",
    "harmful_instructions",
    "manipulation",
    "self_harm_risk",
)
EMBEDDING_DIM = 384
INTENT_DIM = 6


@dataclass(frozen=True)
class TemporalUpdate:
    H_t: np.ndarray
    I_t: np.ndarray
    g_t: float
    I_hat_t: np.ndarray


class TemporalIntentModel:
    """Mathematical temporal layer; its parameters are not trained."""

    def __init__(self, lambda_: float = 0.8, seed: int = 0) -> None:
        if not 0.0 <= float(lambda_) <= 1.0:
            raise ValueError("lambda_ must be between 0 and 1")
        self.lambda_ = float(lambda_)
        generator = np.random.default_rng(seed)
        self.W_c = generator.normal(0.0, 0.01, (INTENT_DIM, EMBEDDING_DIM))
        self.W_h = generator.normal(0.0, 0.01, (INTENT_DIM, EMBEDDING_DIM))
        self.W_r = generator.normal(0.0, 0.01, (INTENT_DIM, 1))
        self.W_I = generator.normal(0.0, 0.01, (INTENT_DIM, INTENT_DIM))
        self.b = np.zeros(INTENT_DIM, dtype=np.float64)
        self.W_g = generator.normal(0.0, 0.01, (1, 775))
        self.b_g = np.zeros(1, dtype=np.float64)
        self.reset()

    @property
    def labels(self) -> tuple[str, ...]:
        return LABELS

    def reset(self) -> None:
        """Restore H_0 = 0 and the deterministic uniform I_0 distribution."""
        self.H_t = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        self.I_t = np.full(INTENT_DIM, 1.0 / INTENT_DIM, dtype=np.float64)

    @staticmethod
    def _vector(value, shape: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    def update(self, c_t, R_t: float) -> TemporalUpdate:
        c_t = self._vector(c_t, (EMBEDDING_DIM,), "c_t")
        confidence = float(R_t)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("R_t must be a finite float in [0, 1]")

        previous_H = self.H_t.copy()
        previous_I = self.I_t.copy()
        H_t = self.lambda_ * previous_H + (1.0 - self.lambda_) * c_t
        logits = (
            self.W_c @ c_t
            + self.W_h @ previous_H
            + self.W_r[:, 0] * confidence
            + self.W_I @ previous_I
            + self.b
        )
        shifted = logits - np.max(logits)
        I_hat_t = np.exp(shifted)
        I_hat_t /= np.sum(I_hat_t)
        gate_input = np.concatenate((c_t, previous_H, np.array([confidence]), previous_I))
        gate_logit = float((self.W_g @ gate_input)[0] + self.b_g[0])
        g_t = 1.0 / (1.0 + np.exp(-gate_logit))
        I_t = (1.0 - g_t) * previous_I + g_t * I_hat_t

        self.H_t = H_t
        self.I_t = I_t
        return TemporalUpdate(H_t=H_t.copy(), I_t=I_t.copy(), g_t=g_t, I_hat_t=I_hat_t.copy())
