import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import faiss
import numpy as np

from core import dataset_loader


class FakeEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def get_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), 384), dtype="float32")
        for position, text in enumerate(texts):
            vectors[position, position % 384] = 1.0
        return vectors


class DatasetLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset_path = self.root / "guardgpt_dataset.jsonl"
        self.records = [
            {
                "request_id": "one",
                "input_text": "safe example",
                "intent": "safe",
                "target_verdict": "ALLOW",
                "category_scores": {
                    "harm": 0.0,
                    "toxicity": 0.0,
                    "jailbreak": 0.0,
                    "prompt_injection": 0.0,
                },
            },
            {
                "request_id": "two",
                "input_text": "unsafe example",
                "intent": "unsafe",
                "target_verdict": "BLOCK",
                "category_scores": {
                    "harm": 1.0,
                    "toxicity": 0.0,
                    "jailbreak": 0.0,
                    "prompt_injection": 0.0,
                },
            },
        ]
        self.dataset_path.write_text(
            "\n".join(json.dumps(record) for record in self.records) + "\n",
            encoding="utf-8",
        )

    def loader(self):
        return dataset_loader.DatasetLoader(self.dataset_path)

    def write_artifacts(self, records=None, dimension=384, mapping=None):
        records = records or self.records
        index = faiss.IndexFlatIP(dimension)
        vectors = np.zeros((len(records), dimension), dtype="float32")
        for position in range(len(records)):
            vectors[position, position % dimension] = 1.0
        index.add(vectors)
        faiss.write_index(index, str(self.root / "guardgpt_faiss.index"))
        if mapping is None:
            mapping = {str(position): record for position, record in enumerate(records)}
        (self.root / "guardgpt_id_map.json").write_text(
            json.dumps(mapping), encoding="utf-8"
        )

    def load(self):
        with patch.object(dataset_loader, "SentenceTransformer", FakeEmbedder):
            loader = self.loader()
            loader.load()
            return loader

    def test_jsonl_loading_without_prebuilt_artifacts(self):
        loader = self.load()
        self.assertEqual(loader.record_count, 2)
        self.assertEqual(loader.embedding_dimension, 384)
        self.assertEqual(loader.query("safe example")["request_id"], "one")

    def test_prebuilt_faiss_uses_jsonl_records_and_id_map_order(self):
        normalized = self.records
        self.write_artifacts(records=[normalized[1], normalized[0]])
        # The vectors are deliberately in a different order from the JSONL file.
        mapping = {"0": normalized[1], "1": normalized[0]}
        (self.root / "guardgpt_id_map.json").write_text(json.dumps(mapping), encoding="utf-8")
        loader = self.load()
        self.assertEqual(loader.record_count, 2)
        self.assertEqual(loader.query("anything")["request_id"], "two")

    def test_partial_prebuilt_artifacts_are_rejected(self):
        (self.root / "guardgpt_faiss.index").write_bytes(b"partial")
        with self.assertRaisesRegex(FileNotFoundError, "supplied together"):
            self.load()

    def test_mismatched_index_dimension_is_rejected(self):
        self.write_artifacts(dimension=3)
        with self.assertRaisesRegex(ValueError, "384-dimensional"):
            self.load()

    def test_mismatched_id_map_length_is_rejected(self):
        self.write_artifacts(mapping={"0": self.records[0]})
        with self.assertRaisesRegex(ValueError, "counts/keys"):
            self.load()

    def test_invalid_jsonl_record_is_rejected(self):
        self.dataset_path.write_text(
            json.dumps({"intent": "safe"}) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "no prompt text"):
            self.load()


if __name__ == "__main__":
    unittest.main()
