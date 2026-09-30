"""
GuardGPT Agent <-> MCP Server client.

Responsibilities:

  - connect to the MCP Server over Streamable HTTP
  - (optionally) discover tools registered on the server
  - invoke named MCP tools with structured arguments
  - parse the structured `CallToolResult` into a clean dict
  - validate responses
  - handle connection failures and tool failures gracefully
  - expose clean sync methods for the Agent nodes (Phase 3)

The Agent nodes MUST use this client to talk to the MCP server.
They MUST NOT import GuardGPT safety components directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import timedelta
import httpx
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


logger = logging.getLogger(__name__)


MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"


def configured_server_url() -> str:
    """Read the endpoint at call time so runtime configuration changes apply."""
    return os.getenv("GUARDGPT_MCP_URL", MCP_SERVER_URL)


# ============================================================
# Exceptions
# ============================================================

class MCPClientError(Exception):
    """Base error raised by the GuardGPT MCP client."""


class MCPConnectionError(MCPClientError):
    """The MCP server is unreachable or did not complete initialization."""


class MCPToolError(MCPClientError):
    """The MCP tool returned an error or an invalid response."""


# ============================================================
# Result
# ============================================================

@dataclass
class ToolResult:
    """Cleaned-up tool result returned to Agent nodes."""

    tool_name: str
    data: dict[str, Any]
    raw_text: str
    is_error: bool


class SmartTimeout(float):
    """Compatible timeout wrapper for both MCP 1.x (expects timedelta.total_seconds())
    and MCP 2.x / anyio (expects float for deadline math)."""
    def total_seconds(self) -> float:
        return float(self)


async def _call_mcp_tool_async(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    url: Optional[str] = None,
    read_timeout_seconds: Optional[float] = None,
) -> ToolResult:
    url = url or configured_server_url()
    try:
        local = urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(read_timeout_seconds or 900), trust_env=not local) as http_client, streamable_http_client(url, http_client=http_client) as streams:
            read_stream, write_stream = streams[:2]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                sec = float(read_timeout_seconds.total_seconds() if isinstance(read_timeout_seconds, timedelta) else (read_timeout_seconds or 900))
                result = await session.call_tool(
                    tool_name,
                    arguments=arguments,
                    read_timeout_seconds=SmartTimeout(sec),
                )



    except asyncio.TimeoutError as error:
        raise MCPConnectionError(
            f"Timed out calling tool '{tool_name}' at {url}"
        ) from error
    except Exception as error:
        message = str(error).lower()
        if (
            "connect" in message
            or "connection" in message
            or "refused" in message
            or "taskgroup" in message
            or "all connection attempts failed" in message
        ):
            raise MCPConnectionError(
                f"Cannot reach MCP server at {url}: {error}"
            ) from error
        raise MCPClientError(
            f"MCP call to '{tool_name}' failed: {error}"
        ) from error

    return _coerce_tool_result(tool_name, result)


async def _list_mcp_tools_async(
    *,
    url: Optional[str] = None,
) -> list[str]:
    url = url or configured_server_url()
    try:
        local = urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}
        async with httpx.AsyncClient(trust_env=not local) as http_client, streamable_http_client(url, http_client=http_client) as streams:
            read_stream, write_stream = streams[:2]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await session.list_tools()
    except Exception as error:
        raise MCPConnectionError(
            f"Cannot list tools on MCP server at {url}: {error}"
        ) from error

    return [tool.name for tool in listed.tools]


# ============================================================
# Result coercion
# ============================================================

def _coerce_tool_result(tool_name: str, result: Any) -> ToolResult:
    """
    Convert the MCP `CallToolResult` into a `ToolResult` with a parsed dict.

    The GuardGPT server tools return their Pydantic output models, which
    FastMCP serializes as a single JSON `TextContent` block.
    """
    if result is None:
        raise MCPToolError(f"Tool '{tool_name}' returned no result.")

    is_error = bool(getattr(result, "isError", getattr(result, "is_error", False)))
    content = getattr(result, "content", None)
    structured = getattr(result, "structuredContent", getattr(result, "structured_content", None))
    if is_error:
        raise MCPToolError(f"Tool '{tool_name}' failed. No result was accepted.")

    if structured:
        if isinstance(structured, dict):
            return ToolResult(
                tool_name=tool_name,
                data=structured,
                raw_text=json.dumps(structured, ensure_ascii=False),
                is_error=is_error,
            )

    if not content:
        if is_error:
            raise MCPToolError(f"Tool '{tool_name}' returned an error with no content.")
        raise MCPToolError(f"Tool '{tool_name}' returned empty content.")

    text_chunks: list[str] = []
    for chunk in content:
        text = getattr(chunk, "text", None)
        if text:
            text_chunks.append(text)

    raw_text = "\n".join(text_chunks).strip()

    if not raw_text:
        if is_error:
            raise MCPToolError(
                f"Tool '{tool_name}' returned an error with no text content."
            )
        raise MCPToolError(
            f"Tool '{tool_name}' returned non-text content: {content!r}"
        )

    parsed: Any
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = {"_raw_text": raw_text}

    if is_error:
        raise MCPToolError(
            f"Tool '{tool_name}' reported error: {raw_text}"
        )

    if not isinstance(parsed, dict):
        raise MCPToolError(
            f"Tool '{tool_name}' returned non-object JSON: {type(parsed).__name__}"
        )

    return ToolResult(
        tool_name=tool_name,
        data=parsed,
        raw_text=raw_text,
        is_error=False,
    )


# ============================================================
# Synchronous facade for Agent nodes (Phase 3+)
# ============================================================

def call_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    url: Optional[str] = None,
    read_timeout_seconds: Optional[float] = None,
) -> ToolResult:
    """Synchronously call an MCP tool and return a `ToolResult`."""
    try:
        return asyncio.run(
            _call_mcp_tool_async(
                tool_name,
                arguments,
                url=url,
                read_timeout_seconds=read_timeout_seconds,
            )
        )
    except MCPClientError:
        raise
    except Exception as error:
        raise MCPClientError(
            f"MCP client failed for tool '{tool_name}': {error}"
        ) from error


def list_tools(*, url: str = MCP_SERVER_URL) -> list[str]:
    """Synchronously list tool names registered on the MCP server."""
    try:
        return asyncio.run(_list_mcp_tools_async(url=url))
    except MCPClientError:
        raise
    except Exception as error:
        raise MCPClientError(
            f"MCP client failed while listing tools: {error}"
        ) from error


# ============================================================
# Backwards-compatible helpers used by the existing test_mcp.py
# ============================================================

def run_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Backwards-compatible sync helper.

    The legacy Agent code wrapped tool responses in `{"data": {...}}` and
    expected a dict back. This helper keeps that surface so the existing
    `test_mcp.py` script and Phase 3 Agent nodes can rely on it.
    """
    return {"data": call_tool(tool_name, arguments, url=url).data}


def known_tool_names() -> Iterable[str]:
    """Return the canonical GuardGPT tool names handled by the MCP server."""
    return (
        "health",
        "complete_request",
        "prompt_analysis",
        "jailbreak_detection",
        "content_moderation",
        "decision",
        "audit_logger",
    )


__all__ = [
    "MCP_SERVER_URL",
    "configured_server_url",
    "MCPClientError",
    "MCPConnectionError",
    "MCPToolError",
    "ToolResult",
    "call_tool",
    "list_tools",
    "run_mcp_tool",
    "known_tool_names",
]
