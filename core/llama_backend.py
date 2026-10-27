import logging
import os
from typing import Optional
import requests

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
DEFAULT_TIMEOUT = 120
DEFAULT_TEMPERATURE = 0.2


def _tag_generation_failure(error, category, status_code=None):
    """Attach bounded diagnostic metadata without retaining backend text."""
    error.generation_failure_category = category
    if (isinstance(status_code, int) and not isinstance(status_code, bool)
            and 100 <= status_code <= 599):
        error.generation_http_status = status_code
        # Preserve the attribute used by the existing rewrite diagnostics.
        error.status_code = status_code
    return error


class LlamaBackend:
    """Ollama backend REST client."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

        try:
            self.timeout = int(timeout or os.getenv("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            self.timeout = DEFAULT_TIMEOUT

        try:
            self.temperature = float(
                temperature
                if temperature is not None
                else os.getenv("OLLAMA_TEMPERATURE", DEFAULT_TEMPERATURE)
            )
        except (TypeError, ValueError):
            self.temperature = DEFAULT_TEMPERATURE

        self._session = requests.Session()
        from urllib.parse import urlparse
        if urlparse(self.base_url).hostname in {"127.0.0.1", "localhost", "::1"}:
            self._session.trust_env = False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, *, schema=None, temperature=None) -> str:
        prompt = str(prompt).strip() if prompt else ""
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature if temperature is None else temperature,
                        "num_predict": 768, "num_ctx": 8192,
                        # Discourages the degenerate token-repetition loops seen with
                        # smaller/general-purpose models under grammar-constrained JSON
                        # decoding (e.g. an enum array cycling the same 2-3 values).
                        "repeat_penalty": 1.3, "repeat_last_n": 64},
        }

        if system_prompt:
            payload["system"] = str(system_prompt)
        if schema is not None:
            payload["format"] = schema

        url = f"{self.base_url}/api/generate"

        try:
            response = self._session.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout as error:
            failure = RuntimeError("Ollama generation failed or timed out.")
            raise _tag_generation_failure(failure, "timeout") from error
        except requests.exceptions.ConnectionError as error:
            failure = ConnectionError(
                "Cannot reach Ollama. Start Ollama and run python main.py --status."
            )
            raise _tag_generation_failure(failure, "connection_error") from error
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            category = (
                "http_error"
                if isinstance(status_code, int) and not isinstance(status_code, bool)
                and 100 <= status_code <= 599
                else "transport_error"
            )
            failure = RuntimeError("Ollama generation failed or timed out.")
            raise _tag_generation_failure(failure, category, status_code) from error

        if response.status_code != 200:
            error = RuntimeError(f"Ollama returned HTTP {response.status_code}. Check installed models.")
            status_code = response.status_code
            _tag_generation_failure(error, "http_error", status_code)
            raise error

        try:
            data = response.json()
        except Exception as error:
            failure = RuntimeError("Invalid Ollama response JSON.")
            raise _tag_generation_failure(failure, "invalid_json") from error
        if not isinstance(data, dict) or not isinstance(data.get("response"), str):
            failure = RuntimeError("Invalid Ollama response format.")
            raise _tag_generation_failure(failure, "invalid_response")
        if data.get("done") is not True or data.get("done_reason") == "length":
            failure = RuntimeError("Ollama response was incomplete or exceeded the token limit.")
            raise _tag_generation_failure(failure, "incomplete_response")
        generated_text = data["response"].strip()

        if not generated_text:
            failure = RuntimeError("Empty response received from backend.")
            raise _tag_generation_failure(failure, "empty_response")

        return generated_text

    def is_available(self) -> bool:
        try:
            return self._session.get(f"{self.base_url}/api/tags", timeout=5).status_code == 200
        except Exception:
            return False
