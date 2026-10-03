"""Local Jev-Style 2B decision adapter.

This adapter is observational. It returns a Jev-Style result but never owns
or changes GuardGPT's safety decision.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import requests


DECISION_QUESTION = (
    "Classify the current intent::"
    "Safe/Educational,Cyber Abuse,Prompt Injection,Jailbreak,Harmful Instructions"
)
DEFAULT_RELEASE = "2b-v3"
DEFAULT_BACKEND = "torch"
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = "float32"
DEFAULT_SERVER_URL = "http://127.0.0.1:8765"
QUESTION_ID = "choice_1"
CHOICE_OPTIONS = (
    "Safe/Educational",
    "Cyber Abuse",
    "Prompt Injection",
    "Jailbreak",
    "Harmful Instructions",
)


class JevStyleClient:
    """Call the persistent local Jev-Style server over HTTP."""

    def __init__(
        self,
        executable: str | None = None,
        release: str | None = None,
        backend: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        timeout: float = 300.0,
        runner: Callable[..., Any] | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.executable = executable or os.getenv("JEV_STYLE_EXECUTABLE")
        self.release = release or os.getenv("JEV_STYLE_RELEASE", DEFAULT_RELEASE)
        self.backend = backend or os.getenv("JEV_STYLE_BACKEND", DEFAULT_BACKEND)
        self.device = device or os.getenv("JEV_STYLE_DEVICE", DEFAULT_DEVICE)
        self.dtype = dtype or os.getenv("JEV_STYLE_DTYPE", DEFAULT_DTYPE)
        self.timeout = timeout
        self.server_url = (os.getenv("JEV_STYLE_URL", DEFAULT_SERVER_URL)).rstrip("/")
        self.endpoint = f"{self.server_url}/v1/systemone"
        self.http_client = http_client or requests.Session()
        self.runner = runner

    @staticmethod
    def serialize_state(current_instruction: str, session_events: Sequence[Any] | None) -> dict[str, Any]:
        events = []
        for event in session_events or ():
            if isinstance(event, Mapping):
                values = event
                events.append({
                    "turn": values.get("turn_index", values.get("turn")),
                    "instruction": values.get("prompt_snippet", values.get("instruction", "")),
                    "intent": values.get("intent"),
                    "risk_level": values.get("risk_level"),
                    "is_blocked": values.get("is_blocked", values.get("blocked", False)),
                    "block_reason": values.get("block_reason", ""),
                })
            elif hasattr(event, "turn_index"):
                events.append({
                    "turn": event.turn_index,
                    "instruction": event.prompt_snippet,
                    "intent": event.intent,
                    "risk_level": event.risk_level,
                    "is_blocked": event.is_blocked,
                    "block_reason": event.block_reason,
                })
            else:
                events.append({"instruction": str(event)})
        return {"previous_events": events, "current_instruction": current_instruction}

    @classmethod
    def request_body(cls, current_instruction: str, session_events: Sequence[Any] | None) -> dict[str, Any]:
        return {
            "state": cls.serialize_state(current_instruction, session_events),
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": "Classify the current intent",
                    "criteria": {option: None for option in CHOICE_OPTIONS},
                }
            },
        }

    @staticmethod
    def _parse_response(raw: Mapping[str, Any], elapsed_ms: float) -> dict[str, Any]:
        answers = raw.get("answers", {})
        answer = answers.get(QUESTION_ID, {}) if isinstance(answers, Mapping) else {}
        if not isinstance(answer, Mapping) or answer.get("type") != "choice":
            raise ValueError("Jev-Style response did not contain a choice answer")
        return {
            "raw": raw,
            "question_id": QUESTION_ID,
            "decision": answer.get("choice"),
            "choice": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence"),
            "model": raw.get("model"),
            "latency_ms": raw.get("latency_ms", elapsed_ms),
            "usage": raw.get("usage"),
            "elapsed_ms": elapsed_ms,
        }

    def decide(self, current_instruction: str, session_events: Sequence[Any] | None = None) -> dict[str, Any]:
        if not isinstance(current_instruction, str) or not current_instruction.strip():
            raise ValueError("current_instruction must be a non-empty string")

        started = time.perf_counter()
        response = self.http_client.post(
            self.endpoint,
            json=self.request_body(current_instruction, session_events),
            timeout=self.timeout,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        return self._parse_response(response.json(), elapsed_ms)