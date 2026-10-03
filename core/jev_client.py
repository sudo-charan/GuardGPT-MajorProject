"""Standalone client for the official TypeSafe JEV API."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import requests


DEFAULT_JEV_ENDPOINT = "https://jev-ai.org/api/v1/systemone/"
DEFAULT_JEV_MODEL = "jev-1.13"
DECISION_OPTIONS = ("ALLOW", "SANITIZE", "BLOCK")


class JEVClient:
    """Call JEV without coupling it to GuardGPT's canonical pipeline."""

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = os.getenv("JEV_API_KEY")
        self.endpoint = endpoint or os.getenv("JEV_API_ENDPOINT", DEFAULT_JEV_ENDPOINT)
        self.model = model or os.getenv("JEV_MODEL", DEFAULT_JEV_MODEL)
        self.timeout = timeout
        self.session = session or requests.Session()

    @staticmethod
    def serialize_state(current_instruction: str, session_events: Sequence[Any] | None) -> dict[str, Any]:
        """Represent the instruction and existing events as ordinary JSON state."""
        events = []
        for event in session_events or ():
            if isinstance(event, Mapping):
                events.append(dict(event))
            elif hasattr(event, "__dataclass_fields__"):
                from dataclasses import asdict

                events.append(asdict(event))
            else:
                events.append(str(event))
        return {"current_instruction": current_instruction, "session_events": events}

    @classmethod
    def request_body(cls, current_instruction: str, session_events: Sequence[Any] | None, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "state": cls.serialize_state(current_instruction, session_events),
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Choose the single safety decision for the current instruction.",
                    "criteria": {
                        "ALLOW": "The instruction is safe to answer as requested.",
                        "SANITIZE": "The legitimate topic can be answered after removing unsafe or disallowed details.",
                        "BLOCK": "The instruction must not be answered.",
                    },
                }
            },
        }

    def decide(self, current_instruction: str, session_events: Sequence[Any] | None = None) -> dict[str, Any]:
        if not isinstance(current_instruction, str) or not current_instruction.strip():
            raise ValueError("current_instruction must be a non-empty string")
        if not self.api_key:
            raise RuntimeError("JEV_API_KEY is not configured")

        body = self.request_body(current_instruction, session_events, self.model)
        started = time.perf_counter()
        response = self.session.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout,
        )
        elapsed_seconds = time.perf_counter() - started
        response.raise_for_status()
        raw = response.json()
        answer = raw.get("answers", {}).get("decision", {})
        return {
            "raw": raw,
            "decision": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence"),
            "model": raw.get("model"),
            "model_version": raw.get("model_version"),
            "usage": raw.get("usage"),
            "latency_ms": raw.get("latency_ms"),
            "elapsed_seconds": elapsed_seconds,
        }