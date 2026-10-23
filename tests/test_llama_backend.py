import unittest
from unittest.mock import Mock

from core.llama_backend import LlamaBackend, OllamaHTTPError


class LlamaBackendTests(unittest.TestCase):
    def test_non_success_status_is_retained_without_reading_response_body(self):
        backend = LlamaBackend(base_url="http://127.0.0.1:11434", model="fixture")
        response = Mock(status_code=400)
        backend._session.post = Mock(return_value=response)

        with self.assertRaises(OllamaHTTPError) as raised:
            backend.generate("hello")

        self.assertEqual(raised.exception.diagnostic_category, "ollama_http_error")
        self.assertEqual(raised.exception.status_code, 400)
        self.assertNotIn("400", str(raised.exception))
        response.json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
