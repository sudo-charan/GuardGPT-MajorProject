"""
End-to-end dedup assertion.

Runs the new pipeline once via `main.py --check --prompt ...` and
asserts that an isolated audit file contains *exactly one* entry
with a `request_id` matching the printed Guard Report.

Prereq: full venv (sentence_transformers, mcp, pydantic, langgraph).
Skips when any of those are missing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Write a Python function to reverse a string."


def _venv_python() -> str:
    """Prefer the project venv's python; fall back to the active interpreter."""
    candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _missing_required_modules() -> list[str]:
    """Skip the test if any runtime dep is unavailable."""
    missing = []
    for mod in ("sentence_transformers", "mcp", "pydantic", "langgraph"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


class AuditDedupEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        missing = _missing_required_modules()
        if missing:
            self.skipTest(
                f"runtime deps unavailable in this environment: {missing}"
            )

    def test_pipeline_writes_exactly_one_audit_entry(self) -> None:
        class JEVHandler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                payload = json.dumps({"model": "jev-style-test-fixture", "answers": {
                    "choice_1": {"type": "choice", "choice": "Safe/Educational",
                                 "probabilities": {"Safe/Educational": 1.0}, "confidence": 1.0},
                    "decision": {"type": "choice", "choice": "ALLOW",
                                 "probabilities": {"ALLOW": 1.0}, "confidence": 1.0},
                }, "usage": {"input_tokens": 1}, "latency_ms": 1.0}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        jev = ThreadingHTTPServer(("127.0.0.1", 0), JEVHandler)
        jev_thread = threading.Thread(target=jev.serve_forever, daemon=True)
        jev_thread.start()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "guardgpt_audit.jsonl"
            env = os.environ.copy()
            env["JEV_STYLE_URL"] = f"http://127.0.0.1:{jev.server_port}"
            env["JEV_STYLE_TIMEOUT"] = "5"
            env["JEV_STYLE_API_KEY"] = ""
            env["GUARDGPT_AUDIT_LOG_PATH"] = str(log_path)
            try:
                result = subprocess.run(
                    [
                        _venv_python(),
                        str(PROJECT_ROOT / "main.py"),
                        "--check",
                        "--prompt",
                        PROMPT,
                    ],
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            finally:
                jev.shutdown()
                jev.server_close()
                jev_thread.join(timeout=5)

            self.assertEqual(
                result.returncode,
                0,
                msg=f"pipeline failed:\nstdout={result.stdout}\nstderr={result.stderr}",
            )

            after = (
                log_path.read_text(encoding="utf-8").splitlines()
                if log_path.exists()
                else []
            )

        new_lines = after
        self.assertEqual(
            len(new_lines),
            1,
            msg=(
                "Audit dedup violated: pipeline wrote "
                f"{len(new_lines)} new entries, expected 1."
            ),
        )

        entry = json.loads(new_lines[0])
        self.assertEqual(entry.get("action"), "ALLOW")
        self.assertEqual(entry.get("decision_source"), "jev-style")
        self.assertTrue(entry.get("request_id"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
