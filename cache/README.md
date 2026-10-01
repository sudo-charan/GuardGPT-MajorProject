# Cache Directory

This directory is reserved for optional local runtime caches created by tools
or development workflows. The current repository contains only `.gitkeep`.

The production dataset and FAISS artifacts are stored in `data/`, not here:

- `data/guardgpt_dataset.jsonl`
- `data/guardgpt_faiss.index`
- `data/guardgpt_id_map.json`

`DatasetLoader` validates and uses those supplied artifacts directly. Do not
move or replace them with files in this directory. Any future cache files are
local runtime artifacts and must not be treated as authoritative dataset or
index sources.
