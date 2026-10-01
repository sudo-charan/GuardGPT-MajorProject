# Data Directory

This directory contains the supplied GuardGPT dataset and its validated vector
search artifacts:

- `guardgpt_dataset.jsonl` - 19,200 JSONL records with `input_text`, `intent`,
	`confidence`, and `reason` fields.
- `guardgpt_faiss.index` - FAISS `IndexFlatIP` containing 19,200 normalized
	384-dimensional vectors.
- `guardgpt_id_map.json` - position-to-record mapping validated against the
	JSONL dataset and FAISS index.
- `guardgpt_id_map.json.bak` - backup copy of the ID map; it is not used unless
	explicitly restored outside the application.

`core.dataset_loader.DatasetLoader` loads the JSONL records, validates the
matching index and ID map, and returns nearest-neighbor evidence as
`dataset_match_confidence`. The evidence is contextual input to the existing
safety pipeline; it is not a replacement for `IntentClassifier`.

The index uses the normalized 384-dimensional embeddings from
`all-MiniLM-L6-v2`. Do not edit, regenerate, or reorder these artifacts without
also rebuilding and validating the complete dataset/index/ID-map set.
