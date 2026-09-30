"""Validate local artifacts and check model/vector alignment without an LLM."""
import numpy as np
from core.dataset_loader import DatasetLoader


def main():
    loader = DatasetLoader()
    loader.load()
    print(f"Records loaded: {loader.record_count:,}")
    print(f"Dimensions: {loader.embedding_dimension}")
    positions = np.linspace(0, loader.record_count - 1, min(16, loader.record_count), dtype=int)
    texts = [loader._records[int(i)]["input_text"] for i in positions]
    vectors = loader._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    alignment = [float(np.dot(vector, loader._index.reconstruct(int(i))))
                 for vector, i in zip(vectors, positions)]
    print(f"Minimum sampled model/index alignment: {min(alignment):.6f}")
    if min(alignment) < 0.99:
        raise ValueError("Model/preprocessing or vector-to-record alignment differs. Do not use this index yet.")
    print("PASS: sampled embeddings agree with the supplied index.")
    print("This validates integration, not dataset label quality or classifier accuracy.")


if __name__ == "__main__":
    main()
