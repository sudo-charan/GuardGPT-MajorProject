import importlib
import unittest
from unittest.mock import Mock, patch


def _main_module():
    with patch("dotenv.load_dotenv"):
        return importlib.import_module("main")


class MainStatusTests(unittest.TestCase):
    def setUp(self):
        self.main = _main_module()

    def _backend(self, *, model_names=None, fail=False):
        backend = Mock()
        backend.model = "generation-model"
        backend.base_url = "http://127.0.0.1:11434"
        if fail:
            backend._session.get.side_effect = RuntimeError("private connection detail")
        else:
            response = Mock()
            response.json.return_value = {"models": [{"name": name} for name in model_names or []]}
            backend._session.get.return_value = response
        return backend

    def test_endpoint_reachability_uses_tcp_without_sending_a_decision(self):
        with patch("main.socket.create_connection") as connect:
            self.assertTrue(self.main._endpoint_reachable("http://jev.test:8765/v1/systemone"))

        connect.assert_called_once_with(("jev.test", 8765), timeout=5)

    def test_ollama_status_failure_keeps_models_list_and_returns_unready(self):
        backend = self._backend(fail=True)
        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("core.llama_backend.LlamaBackend", return_value=backend),
            patch("core.jev_style_client.JevStyleClient", return_value=Mock(endpoint="http://jev.test:8765/v1/systemone")),
            patch.object(self.main, "_endpoint_reachable", return_value=True),
        ):
            result = self.main.status()

        self.assertEqual(result["installed_models"], [])
        self.assertFalse(result["ollama_reachable"])
        self.assertFalse(result["models_ready"])
        self.assertFalse(result["ready_for_local_validation"])

    def test_jev_unreachable_prevents_ready_status(self):
        backend = self._backend(model_names=["generation-model"])
        with (
            patch.dict("os.environ", {"OLLAMA_AUDIT_MODEL": "generation-model"}),
            patch("pathlib.Path.is_file", return_value=True),
            patch("core.llama_backend.LlamaBackend", return_value=backend),
            patch("core.jev_style_client.JevStyleClient", return_value=Mock(endpoint="http://jev.test:8765/v1/systemone")),
            patch.object(self.main, "_endpoint_reachable", return_value=False),
        ):
            result = self.main.status()

        self.assertTrue(result["ollama_reachable"])
        self.assertTrue(result["models_ready"])
        self.assertFalse(result["jev_endpoint_tcp_reachable"])
        self.assertFalse(result["ready_for_local_validation"])

    def test_cli_shows_jev_model_and_action_probabilities(self):
        report = {
            "status": "available",
            "classification_category": "Prompt Injection",
            "classification_confidence": 0.435,
            "action": "BLOCK",
            "action_confidence": 0.114,
            "model": "fixture-jev-model",
            "action_probabilities": {"ALLOW": 0.1, "SANITIZE": 0.2, "BLOCK": 0.7},
            "decision_history": [],
        }
        with patch("builtins.print") as output:
            self.main.print_jev_style({"jev_style": report})
        rendered = " ".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("JEV model: fixture-jev-model", rendered)
        self.assertIn("JEV action probabilities: ALLOW 10.0%, SANITIZE 20.0%, BLOCK 70.0%", rendered)


if __name__ == "__main__":
    unittest.main()
