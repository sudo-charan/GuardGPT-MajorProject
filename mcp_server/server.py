"""Local MCP server. complete_request is the supported end-to-end entry point."""
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server.mcpserver import MCPServer
    class FastMCP(MCPServer):
        def __init__(self, name: str, host: str = "127.0.0.1", port: int = 8000, **kwargs):
            super().__init__(name=name)
            self.host = host
            self.port = port

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("GUARDGPT_PROJECT_ROOT", str(ROOT))
parsed = urlparse(os.getenv("GUARDGPT_MCP_URL", "http://127.0.0.1:8000/mcp"))
if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
    raise ValueError("GuardGPT's unauthenticated MCP server must bind to loopback.")
mcp = FastMCP("GuardGPT", host=parsed.hostname, port=parsed.port or 8000)


@lru_cache(maxsize=1)
def pipeline():
    from core.complete_pipeline import CompletePipeline
    return CompletePipeline()

@mcp.tool()
def health() -> dict:
    return {"service": "GuardGPT", "version": "2.0", "complete_pipeline": True}

@mcp.tool()
def complete_request(prompt: str, session_id: str | None = None, check_only: bool = False) -> dict:
    """Check input, generate via Ollama, audit output, and log one final report."""
    if session_id is not None and (len(session_id) > 100 or not session_id):
        raise ValueError("Invalid session id")
    return pipeline().run(prompt, session_id=session_id, check_only=check_only)

# Retain the original five tools for existing report-only clients.
@mcp.tool()
def prompt_analysis(data: dict) -> dict:
    from mcp_server.models.schemas import PromptAnalysisInput
    from mcp_server.tools.prompt_analysis import analyze_prompt
    return analyze_prompt(PromptAnalysisInput(**data)).model_dump()

@mcp.tool()
def jailbreak_detection(data: dict) -> dict:
    from mcp_server.models.schemas import JailbreakDetectionInput
    from mcp_server.tools.jailbreak_detection import detect_jailbreak
    return detect_jailbreak(JailbreakDetectionInput(**data)).model_dump()

@mcp.tool()
def content_moderation(data: dict) -> dict:
    from mcp_server.models.schemas import ContentModerationInput
    from mcp_server.tools.content_moderation import moderate_content
    return moderate_content(ContentModerationInput(**data)).model_dump()

@mcp.tool()
def decision(data: dict) -> dict:
    from mcp_server.models.schemas import DecisionInput
    from mcp_server.tools.decision import decide
    return decide(DecisionInput(**data)).model_dump()

@mcp.tool()
def audit_logger(data: dict) -> dict:
    from mcp_server.models.schemas import AuditLoggerInput
    from mcp_server.tools.audit_logger import log_audit_event
    return log_audit_event(AuditLoggerInput(**data)).model_dump()

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
