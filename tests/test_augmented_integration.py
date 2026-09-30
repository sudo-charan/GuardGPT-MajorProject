"""Offline regression tests: real FAISS, deterministic test embeddings."""
import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import faiss
import numpy as np

from core.decision_engine import DecisionEngine
from core.risk_estimator import estimate_risk, should_preliminarily_block


class FakeEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), 384), dtype="float32")
        for i, text in enumerate(texts):
            vectors[i, 0 if text == "safe example" else 1] = 1
        return vectors


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = DecisionEngine(str(self.root / "audit.jsonl"))

    def result(self, **changes):
        value = dict(intent="coding", intent_confidence=0.9,
                     matched_record_intent="safe", dataset_match_confidence=0.99,
                     category_scores={"harm": 0.0}, prompt="safe example")
        value.update(changes)
        return value

    def test_safe_match_allows_without_sanitizing(self):
        out = self.engine.decide(self.result())
        self.assertEqual((out.action, out.risk_level), ("ALLOW", "safe"))
        self.assertEqual(out.dataset_match_confidence, 0.99)

    def test_safe_match_does_not_block_preliminary(self):
        self.assertFalse(should_preliminarily_block(self.result()))

    def test_safe_match_does_not_raise_risk(self):
        self.assertEqual(estimate_risk("coding", 0.9, 0.99, "safe"), "safe")

    def test_classifier_attack_still_blocks(self):
        value = self.result(intent="jailbreak")
        self.assertTrue(should_preliminarily_block(value))
        self.assertEqual(self.engine.decide(value).action, "BLOCK")
        self.assertEqual(estimate_risk("jailbreak", 0.9, 0.99, "safe"), "high")

    def test_history_still_blocks(self):
        self.assertEqual(self.engine.decide(self.result(history_triggered=True)).action, "BLOCK")

    def test_preliminary_block_is_preserved(self):
        self.assertEqual(self.engine.decide(self.result(final_blocked=True)).action, "BLOCK")

    def test_unsafe_match_blocks(self):
        self.assertEqual(self.engine.decide(self.result(matched_record_intent="unsafe", category_scores={"harm": 1.0})).action, "BLOCK")

    def artifacts(self):
        records = []
        for i, intent in enumerate(("safe", "unsafe")):
            records.append(dict(request_id=str(i), input_text=f"{intent} example",
                intent=intent, target_class="test", target_verdict="ALLOW" if i == 0 else "BLOCK",
                category_scores=dict(harm=float(i), toxicity=0.0, jailbreak=0.0, prompt_injection=0.0)))
        path = self.root / "guardgpt_augmented_clean.json"
        # Dataset order intentionally differs from vector order.
        path.write_text(json.dumps(records[::-1]), encoding="utf-8")
        (self.root / "guardgpt_id_map.json").write_text(json.dumps(dict(enumerate(records))), encoding="utf-8")
        index = faiss.IndexFlatIP(384)
        index.add(FakeEmbedder().encode([r["input_text"] for r in records]))
        faiss.write_index(index, str(self.root / "guardgpt_faiss.index"))
        return path

    def loader(self, path):
        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = FakeEmbedder
        with patch.dict(sys.modules, {"sentence_transformers": fake}):
            module = importlib.import_module("core.dataset_loader")
        with patch.object(module, "SentenceTransformer", FakeEmbedder):
            loader = module.DatasetLoader(path)
            loader.load()
        return loader

    def test_map_order_and_queries(self):
        loader = self.loader(self.artifacts())
        self.assertEqual(loader.query("safe example")["request_id"], "0")
        self.assertEqual(loader.query("unsafe example")["request_id"], "1")
        self.assertEqual(len(loader.query_top_k("safe example", 5)), 2)

    def test_clear_keeps_artifacts(self):
        loader = self.loader(self.artifacts())
        loader.clear_cache()
        self.assertTrue((self.root / "guardgpt_faiss.index").exists())
        self.assertTrue((self.root / "guardgpt_id_map.json").exists())

    def test_missing_index_rejected(self):
        path = self.artifacts()
        (self.root / "guardgpt_faiss.index").unlink()
        with self.assertRaises(FileNotFoundError):
            self.loader(path)

    def test_model_download_requires_explicit_opt_in(self):
        module = importlib.import_module("core.dataset_loader")
        with patch.object(module, "SentenceTransformer", side_effect=OSError("offline")):
            with patch.dict("os.environ", {"GUARDGPT_ALLOW_MODEL_DOWNLOAD": ""}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "not available locally"):
                    module._load_embedding_model("missing-model")

    def test_changed_mapping_rejected(self):
        path = self.artifacts()
        mapping_path = self.root / "guardgpt_id_map.json"
        mapping = json.loads(mapping_path.read_text())
        mapping["0"]["input_text"] = "wrong text"
        mapping_path.write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError, "changed dataset"):
            self.loader(path)

    def test_conflicting_labels_rejected(self):
        path = self.artifacts()
        records = json.loads(path.read_text())
        records[0]["target_verdict"] = "ALLOW"
        path.write_text(json.dumps(records))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.loader(path)

    def test_mcp_adapter_preserves_safe_label(self):
        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = FakeEmbedder
        with patch.dict(sys.modules, {"sentence_transformers": fake}):
            analysis = importlib.import_module("mcp_server.tools.prompt_analysis")
            decision = importlib.import_module("mcp_server.tools.decision")
        schemas = importlib.import_module("models.schemas")
        loader = self.loader(self.artifacts())
        classifier = types.SimpleNamespace(classify=lambda prompt: {"intent": "coding", "confidence": 0.9})
        with patch.object(analysis, "_loader", return_value=loader), patch.object(analysis, "_classifier", return_value=classifier):
            report = analysis.analyze_prompt(schemas.PromptAnalysisInput(prompt="safe example"))
        payload = report.model_dump()
        self.assertEqual(payload["matched_record_intent"], "safe")
        self.assertEqual(payload["risk_level"], "safe")
        with patch.object(decision, "_engine", return_value=self.engine):
            output = decision.decide(schemas.DecisionInput(prompt="safe example", **payload))
        self.assertEqual(output.action, "ALLOW")


if __name__ == "__main__":
    unittest.main()
