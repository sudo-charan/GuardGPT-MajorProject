"""
GuardGPT MCP Server package.

Importing this package makes the GuardGPT project root and the
`mcp_server/` directory importable on `sys.path`, so the MCP tools can
import `core.*` and `models.schemas` without per-tool sys.path hacks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_PROJECT_ROOT_HINT = os.getenv("GUARDGPT_PROJECT_ROOT")


def project_root() -> Path:
    """Return the GuardGPT project root directory."""
    if _PROJECT_ROOT_HINT:
        return Path(_PROJECT_ROOT_HINT).resolve()
    return Path(__file__).resolve().parent.parent


def _ensure_paths() -> None:
    """Insert the project root and `mcp_server/` directory into sys.path."""
    root = project_root()
    mcp_server_dir = root / "mcp_server"
    for path_str in (str(root), str(mcp_server_dir)):
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_paths()
