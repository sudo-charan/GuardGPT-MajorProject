"""
GuardGPT Agent - LangGraph workflow.

START
  |
  v
ReceivePrompt
  |
  v
PromptAnalysis      (MCP tool)
  |
  v
JailbreakDetection  (MCP tool)
  |
  v
ContentModeration   (MCP tool)
  |
  v
CombineResults
  |
  v
Decision            (MCP tool - thin wrapper around core.decision_engine)
  |
  v
AuditLog            (MCP tool)
  |
  v
BuildReport
  |
  v
END

The Agent MUST NOT do its own classification; it only coordinates the
MCP tools that wrap the existing GuardGPT core.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, StateGraph

from agent.state import GuardState

from agent.nodes import (
    receive_prompt,
    run_prompt_analysis,
    run_jailbreak_detection,
    run_content_moderation,
    combine_results,
    run_decision,
    run_audit_log,
    build_report_node,
)


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def build_agent_graph():
    """Compile and return the GuardGPT LangGraph workflow."""
    workflow = StateGraph(GuardState)

    workflow.add_node("ReceivePrompt", receive_prompt)
    workflow.add_node("PromptAnalysis", run_prompt_analysis)
    workflow.add_node("JailbreakDetection", run_jailbreak_detection)
    workflow.add_node("ContentModeration", run_content_moderation)
    workflow.add_node("CombineResults", combine_results)
    workflow.add_node("Decision", run_decision)
    workflow.add_node("AuditLog", run_audit_log)
    workflow.add_node("BuildReport", build_report_node)

    workflow.set_entry_point("ReceivePrompt")
    workflow.add_edge("ReceivePrompt", "PromptAnalysis")
    workflow.add_edge("PromptAnalysis", "JailbreakDetection")
    workflow.add_edge("JailbreakDetection", "ContentModeration")
    workflow.add_edge("ContentModeration", "CombineResults")
    workflow.add_edge("CombineResults", "Decision")
    workflow.add_edge("Decision", "AuditLog")
    workflow.add_edge("AuditLog", "BuildReport")
    workflow.add_edge("BuildReport", END)

    return workflow.compile()


agent = build_agent_graph()


__all__ = ["agent", "build_agent_graph"]
