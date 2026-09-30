# Data directory

This directory is reserved for the GuardGPT dataset artifacts and generated vector/index files.

Expected local artifacts:
- `guardgpt_dataset.jsonl` or a project-root equivalent
- `guardgpt_augmented_clean.json`
- `guardgpt_faiss.index`
- `guardgpt_id_map.json`

Keep large or generated files out of version control; use `.gitignore` to ignore them while preserving the directory structure.
