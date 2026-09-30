"""
Subprocess manager for the GuardGPT MCP server.

The end-to-end entry point needs to ensure an MCP server is running.
This helper will:

  - check whether the configured GUARDGPT_MCP_URL already has a listener
  - if not, spawn `python -m mcp_server.server` on a free port
  - configure GUARDGPT_MCP_URL and GUARDGPT_PROJECT_ROOT so the
    Agent and the subprocess can find each other
  - shut the subprocess down on exit

It is intentionally lightweight so tests and the CLI can share it.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.settimeout(1.0)
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


class MCPServerManager:
    """Context manager that ensures an MCP server is reachable."""

    def __init__(
        self,
        *,
        python_executable: Optional[str] = None,
        auto_start: bool = True,
        url: Optional[str] = None,
        startup_timeout: float = 90.0,
    ) -> None:
        self.python_executable = python_executable or sys.executable
        self.auto_start = auto_start
        self.startup_timeout = startup_timeout
        self.process: Optional[subprocess.Popen] = None
        self.url: str = url or os.getenv("GUARDGPT_MCP_URL", "http://127.0.0.1:8000/mcp")

    def __enter__(self) -> str:
        if _port_reachable(self.url):
            os.environ["GUARDGPT_MCP_URL"] = self.url
            return self.url

        if not self.auto_start:
            return self.url

        port = _free_port()
        self.url = f"http://127.0.0.1:{port}/mcp"
        os.environ["GUARDGPT_MCP_URL"] = self.url

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["GUARDGPT_PROJECT_ROOT"] = str(PROJECT_ROOT)
        env["GUARDGPT_MCP_URL"] = self.url

        log_path = Path(os.getenv("TEMP", "/tmp")) / f"guardgpt_mcp_server_{port}.log"
        log_file = open(log_path, "wb")

        try:
            self.process = subprocess.Popen(
                [
                    self.python_executable,
                    "-u",
                    "-m",
                    "mcp_server.server",
                ],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        finally:
            log_file.close()

        if not _wait_for_port(port, timeout=self.startup_timeout):
            self._terminate()
            raise RuntimeError(
                f"MCP server failed to start on port {port}. Log: {log_path}"
            )

        return self.url

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._terminate()

    def _terminate(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


def _port_reachable(url: str) -> bool:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
    except Exception:
        return False
    if not parsed.hostname or not parsed.port:
        return False
    family = socket.AF_INET6 if ":" in parsed.hostname else socket.AF_INET
    address = (parsed.hostname, parsed.port, 0, 0) if family == socket.AF_INET6 else (parsed.hostname, parsed.port)
    try:
        with closing(socket.socket(family, socket.SOCK_STREAM)) as sock:
            sock.settimeout(1.0)
            return sock.connect_ex(address) == 0
    except OSError:
        return False
