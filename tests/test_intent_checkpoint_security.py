import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.intent_classifier import IntentClassifier


class IntentCheckpointSecurityTests(unittest.TestCase):
    def test_checkpoint_load_uses_restricted_weights_only_mode(self):
        class FakeTensor:
            shape = (2, 3)

        encoder = Mock()
        head = Mock()
        fake_torch = types.ModuleType("torch")
        fake_torch.load = Mock(return_value={
            "labels": ["safe", "unsafe"],
            "model_state_dict": {
                "encoder.layer.weight": FakeTensor(),
                "classifier.weight": FakeTensor(),
                "classifier.bias": FakeTensor(),
            },
        })
        fake_torch.nn = SimpleNamespace(Linear=Mock(return_value=head))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "fixture.pt"
            checkpoint_path.write_bytes(b"mock checkpoint path")
            classifier = IntentClassifier()
            classifier._model = [SimpleNamespace(auto_model=encoder)]
            with patch.dict(os.environ, {"GUARDGPT_INTENT_MODEL": str(checkpoint_path)}), \
                 patch.dict(sys.modules, {"torch": fake_torch}):
                classifier._load_trained_safety_model()

        self.assertTrue(fake_torch.load.call_args.kwargs["weights_only"])
        self.assertEqual(classifier._trained_labels, ["safe", "unsafe"])
        self.assertIs(classifier._trained_encoder, encoder)
        self.assertIs(classifier._trained_head, head)


if __name__ == "__main__":
    unittest.main(verbosity=2)
