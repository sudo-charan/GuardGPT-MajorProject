"""
Smoke test for the per-CLI-run `ConversationGuard` wiring.

The actual end-to-end pipeline depends on the MCP server and the
Sentence-BERT classifier, which is heavy to start in a unit test. This
test instead exercises the `run_agent_with_history` helper in `main.py`
with a fake agent, verifying:

  - First pass: state has `history_triggered=False`, `turn_index=1`.
  - ConversationGuard advances the turn counter.
  - When the guard raises `history_triggered=True`, the helper re-invokes
    the agent with `history_triggered=True` and `history_block_reason`
    populated, and the second report's state carries the signal.

The helper is intentionally pure: it depends only on the injected
`agent` and `guard`, so a stub is enough.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_main_helper():
    """Import `run_agent_with_history` from `main.py` without importing the
    heavy `core.guard_engine` chain (which pulls in `sentence_transformers`)
    or the `agent.nodes` chain (which pulls in `mcp`)."""
    import importlib.util
    import types

    # Stub the surfaces main.py reaches at import time or first call.
    fake_guard_engine = types.ModuleType("core.guard_engine")
    fake_guard_engine.GuardEngine = object
    fake_guard_engine.print_response = lambda *_a, **_k: None
    sys.modules["core.guard_engine"] = fake_guard_engine

    # Stub `agent` packages so the lazy `from agent.nodes import build_report`
    # inside `run_agent_with_history` resolves without pulling in `mcp`.
    # The trick: pre-register stubs in `sys.modules` *and* give the parent
    # stub a `__path__` so the package machinery treats it as a package,
    # but the submodule lookup finds our stub via `sys.modules` first.
    fake_agent = types.ModuleType("agent")
    fake_agent.__path__ = [str(PROJECT_ROOT / "agent")]  # type: ignore[attr-defined]
    sys.modules["agent"] = fake_agent

    fake_agent_nodes = types.ModuleType("agent.nodes")
    fake_agent_nodes.build_report = lambda report: report
    sys.modules["agent.nodes"] = fake_agent_nodes

    fake_agent_state = types.ModuleType("agent.state")
    # GuardState is referenced only as a type annotation in the helper,
    # but `from agent.state import GuardState` is still evaluated at
    # runtime; expose a stub object so the import succeeds.
    fake_agent_state.GuardState = dict  # type: ignore[attr-defined]
    sys.modules["agent.state"] = fake_agent_state

    spec = importlib.util.spec_from_file_location("main", PROJECT_ROOT / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Stubs MUST remain registered: `run_agent_with_history` does a lazy
    # `from agent.nodes import build_report` on every call. Pop them only
    # in cleanup_helper, exposed via sys.modules cleanup at process exit.
    return module.run_agent_with_history


run_agent_with_history = _load_main_helper()


class _RecordingAgent:
    """Records every invocation it receives and returns a caller-supplied
    report keyed by the call number."""

    def __init__(self, reports: list[dict]) -> None:
        self.reports = list(reports)
        self.calls: list[dict] = []

    def invoke(self, state: dict) -> dict:
        self.calls.append(dict(state))
        idx = min(len(self.calls) - 1, len(self.reports) - 1)
        return self.reports[idx]


class _FakeGuardResult:
    """Just enough shape for `ConversationGuard.evaluate_result` to work."""

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _fake_history_triggered_guard():
    """Return a `ConversationGuard` subclass whose `evaluate_result` always
    flags `history_triggered=True` *after* the first call."""
    from core.conversation_guard import ConversationGuard

    class _TriggerGuard(ConversationGuard):
        def __init__(self) -> None:
            super().__init__(session_id="unit-test")
            self.calls = 0

        def evaluate_result(self, result):  # type: ignore[override]
            self.calls += 1
            if self.calls == 1:
                # Pre-second-call evaluation: pretend history tripped.
                from dataclasses import dataclass

                @dataclass
                class _R:
                    history_triggered: bool
                    history_block_reason: str
                    unsafe_ratio: float
                    previous_block: bool
                    session_flagged: bool
                    session_flag_reason: str
                    turn_index: int

                return _R(
                    history_triggered=True,
                    history_block_reason="Previous turn was unsafe.",
                    unsafe_ratio=0.5,
                    previous_block=True,
                    session_flagged=False,
                    session_flag_reason="",
                    turn_index=self._turn_counter + 1,
                )
            self._turn_counter += 1
            from dataclasses import dataclass

            @dataclass
            class _R:
                history_triggered: bool
                history_block_reason: str
                unsafe_ratio: float
                previous_block: bool
                session_flagged: bool
                session_flag_reason: str
                turn_index: int

            return _R(
                history_triggered=False,
                history_block_reason="",
                unsafe_ratio=0.0,
                previous_block=False,
                session_flagged=False,
                session_flag_reason="",
                turn_index=self._turn_counter,
            )

    return _TriggerGuard()


class HistoryAwareInvocationTests(unittest.TestCase):
    def test_first_pass_uses_default_state(self) -> None:
        from core.conversation_guard import ConversationGuard

        guard = ConversationGuard(session_id="unit-test")
        agent = _RecordingAgent(
            [
                {
                    "prompt": "Explain how Python lists work.",
                    "intent": "coding",
                    "allowed": True,
                    "final_blocked": False,
                    "risk_level": "safe",
                    "category_scores": {},
                    "dataset_match_confidence": 0.1,
                }
            ]
        )

        report = run_agent_with_history(agent, guard, "Explain how Python lists work.", "mcp://test")

        self.assertEqual(len(agent.calls), 1)
        state = agent.calls[0]
        self.assertEqual(state["prompt"], "Explain how Python lists work.")
        self.assertFalse(state["history_triggered"])
        self.assertEqual(state["turn_index"], 1)
        self.assertEqual(report["prompt"], "Explain how Python lists work.")

    def test_turn_counter_advances_between_prompts(self) -> None:
        from core.conversation_guard import ConversationGuard

        guard = ConversationGuard(session_id="unit-test")
        agent = _RecordingAgent(
            [
                {
                    "prompt": "first",
                    "allowed": True,
                    "final_blocked": False,
                    "risk_level": "safe",
                    "category_scores": {},
                    "dataset_match_confidence": 0.1,
                },
                {
                    "prompt": "second",
                    "allowed": True,
                    "final_blocked": False,
                    "risk_level": "safe",
                    "category_scores": {},
                    "dataset_match_confidence": 0.1,
                },
            ]
        )

        run_agent_with_history(agent, guard, "first", "mcp://test")
        first_state = agent.calls[0]
        run_agent_with_history(agent, guard, "second", "mcp://test")
        second_state = agent.calls[1]

        self.assertEqual(first_state["turn_index"], 1)
        self.assertEqual(second_state["turn_index"], 2)

    def test_second_pass_runs_when_history_triggered(self) -> None:
        guard = _fake_history_triggered_guard()
        agent = _RecordingAgent(
            [
                {"prompt": "x", "allowed": True, "final_blocked": False,
                 "risk_level": "safe", "category_scores": {},
                 "dataset_match_confidence": 0.0},
                {"prompt": "x", "allowed": False, "final_blocked": True,
                 "risk_level": "high", "category_scores": {},
                 "dataset_match_confidence": 0.0},
            ]
        )

        report = run_agent_with_history(agent, guard, "x", "mcp://test")

        self.assertEqual(len(agent.calls), 2)
        second_state = agent.calls[1]
        self.assertTrue(second_state["history_triggered"])
        self.assertEqual(second_state["history_block_reason"], "Previous turn was unsafe.")
        self.assertEqual(report["prompt"], "x")


if __name__ == "__main__":
    unittest.main(verbosity=2)