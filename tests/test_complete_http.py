"""Real MCP HTTP and Ollama-shaped HTTP; deterministic safety/model fixtures.

No model weights, API keys or production dataset required.
"""
import json
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import uvicorn
from agent.mcp_client import call_tool, MCPToolError
from core.llama_backend import LlamaBackend
from core.complete_pipeline import CompletePipeline, AuditLog
from core.output_auditor import OutputAuditor
from tests.test_complete_pipeline import analysis


class HTTPIntegrationTests(unittest.TestCase):
    def test_real_mcp_transport_and_backend_contract(self):
        calls = []
        class Ollama(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(payload)
                text = json.dumps({"safe": True, "relevant": True, "categories": []}) if "format" in payload else "Lists store ordered values."
                response = json.dumps({"response": text, "done": True, "done_reason": "stop"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
        ollama = ThreadingHTTPServer(("127.0.0.1", 0), Ollama)
        thread = threading.Thread(target=ollama.serve_forever, daemon=True)
        thread.start()
        from mcp_server import server
        with TemporaryDirectory() as tmp:
            safety = Mock()
            safety.analyze.side_effect = lambda _: analysis()
            backend = LlamaBackend(base_url=f"http://127.0.0.1:{ollama.server_port}", model="fixture", timeout=5)
            app = CompletePipeline(safety, backend, OutputAuditor(backend), AuditLog(Path(tmp)/"audit.jsonl"))
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            runner = uvicorn.Server(uvicorn.Config(server.mcp.streamable_http_app(), log_level="error"))
            with patch.object(server, "pipeline", return_value=app):
                runner_thread = threading.Thread(target=runner.run, kwargs={"sockets": [sock]}, daemon=True)
                runner_thread.start()
                try:
                    deadline = time.monotonic() + 10
                    while not runner.started and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(runner.started)
                    url = f"http://127.0.0.1:{port}/mcp"
                    report = call_tool("complete_request", {"prompt": "Explain lists"}, url=url).data
                    self.assertEqual(report["response"], "Lists store ordered values.")
                    self.assertEqual(report["output_audit"], "PASSED")
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(calls[0]["model"], "fixture")
                    self.assertIn("format", calls[1])
                    safety.analyze.side_effect = lambda _: analysis("BLOCK", "jailbreak")
                    blocked = call_tool("complete_request", {"prompt": "blocked"}, url=url).data
                    self.assertFalse(blocked["allowed"])
                    self.assertEqual(len(calls), 2)
                    with self.assertRaises(MCPToolError):
                        call_tool("complete_request", {"wrong": "field"}, url=url)
                    self.assertEqual(len((Path(tmp)/"audit.jsonl").read_text().splitlines()), 2)
                finally:
                    runner.should_exit = True
                    runner_thread.join(10)
                    sock.close()
                    ollama.shutdown()
                    ollama.server_close()
                    thread.join(5)

    def test_incomplete_backend_output_rejected(self):
        backend = LlamaBackend()
        response = Mock(status_code=200)
        response.json.return_value = {"response": "partial", "done": True, "done_reason": "length"}
        backend._session.post = Mock(return_value=response)
        with self.assertRaises(RuntimeError):
            backend.generate("prompt")


if __name__ == "__main__":
    unittest.main()
