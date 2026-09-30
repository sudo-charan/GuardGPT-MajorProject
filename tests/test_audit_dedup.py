"""
End-to-end dedup assertion.

Runs the new pipeline once via `main.py --pipeline --prompt ...` and
asserts that `logs/guardgpt_audit.jsonl` grew by *exactly one* entry
with a `request_id` matching the printed Guard Report.

Prereq: full venv (sentence_transformers, mcp, pydantic, langgraph).
Skips when any of those are missing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
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
        log_path = PROJECT_ROOT / "logs" / "guardgpt_audit.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        before = (
            log_path.read_text(encoding="utf-8").splitlines()
            if log_path.exists()
            else []
        )

        result = subprocess.run(
            [
                _venv_python(),
                str(PROJECT_ROOT / "main.py"),
                "--pipeline",
                "--prompt",
                PROMPT,
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
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
        new_lines = after[len(before):]
        self.assertEqual(
            len(new_lines),
            1,
            msg=(
                "Audit dedup violated: pipeline wrote "
                f"{len(new_lines)} new entries, expected 1."
            ),
        )

        entry = json.loads(new_lines[0])
        report = json.loads(result.stdout)
        self.assertEqual(
            entry.get("audit_id"),
            report.get("audit_id"),
            msg=(
                "Audit entry audit_id does not match the printed "
                "Guard Report audit_id (they must be the same UUID)."
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)