# MedAgent Testing Suite

This guide describes how to install dependencies, download models/datasets, and run the full test suite under different modes (offline stubs, light mode, and full heavy mode with FAISS + Transformers).


## Test Modes

- Offline stubs (default): tests run without installing heavy deps. `tests/conftest.py` provides in-memory stubs for `faiss`, `openai`, `torch`, `transformers`, and `tenacity`. Good for CI and quick feedback.
- Light mode: Uses deterministic embeddings (dimension 128) and avoids network. Enable with `LIGHT_TESTS=1` (already defaulted in tests). Good for logic coverage and reproducible runs.
- Heavy mode: Uses real FAISS, Transformers, and Torch. Requires GPU/CPU capable environment and internet access to download models.


## Local Setup

- Recommended: Python 3.11 and UV
- Install project with dev extras:
  - `uv pip install -e ".[dev,cli,data]"`
- Optional: pre-commit hooks
  - `pre-commit install`


## Running Tests

- Quick run (offline stubs, light mode):
  - `pytest -q --disable-warnings --maxfail=1`
- With coverage report:
  - `pytest -q --disable-warnings --maxfail=1 --cov=medagent --cov-report=term-missing`
- Full matrix (tox):
  - `tox`

Environment flags:
- `LIGHT_TESTS=1` — deterministic embeddings; no model downloads
- `MEDAGENT_API_KEY=secret` — protects API endpoints for auth tests
- `MEDAGENT_DATA_DIR=$PWD` — safe root for `/ingest` paths


## API Security Tests

- Health:
  - `curl http://127.0.0.1:8000/health` → 200
- Info (with auth):
  - `curl -H "X-API-Key: secret" http://127.0.0.1:8000/info` → 200, with secrets redacted
- Ingest safety:
  - `curl -X POST -H "X-API-Key: secret" "http://127.0.0.1:8000/ingest?jsonl_path=scripts/sample_questions.jsonl"`
  - Absolute paths or traversal should return 400; large files should return 413.


## Prefetching Embedding Models

Use the helper script to pre-download embedding models to a local cache (no runtime network).

- Choose cache location: `export HF_HOME=$PWD/.hf-cache`
- Example:
  - `python scripts/prefetch_models.py allenai/scibert_scivocab_uncased`
- Multiple models:
  - `python scripts/prefetch_models.py allenai/scibert_scivocab_uncased nomic-ai/nomic-embed-text-v1`

Notes:
- The script uses `huggingface_hub.snapshot_download`, which avoids importing `torch` or loading the model weights in memory.
- For Ollama/LM Studio (OpenAI-compatible), pull models with their respective CLIs.


## Prefetching Datasets

If you use HuggingFace datasets, run:
- `python scripts/prefetch_datasets.py medmcqa` (or any HF dataset name)
- For local JSON → JSONL conversion:
  - `python scripts/convert_json_to_jsonl.py input.json output.jsonl`


## Heavy Mode (Real FAISS + Transformers)

Prerequisites:
- `faiss-cpu`, `torch`, `transformers`, and `datasets` installed (included in dev/data extras)
- Internet access for initial downloads (models/datasets)

Steps:
1) Install: `uv pip install -e ".[dev,cli,data]"`
2) Set `LIGHT_TESTS=0` to enable real embeddings.
3) Prefetch models: `python scripts/prefetch_models.py allenai/scibert_scivocab_uncased`
4) Run tests: `pytest -q`


## CI Details

- Workflow: `.github/workflows/ci.yml`
- Runs on Python 3.10 and 3.11
- Steps: ruff lint + format check, mypy strict, pytest with coverage


## Troubleshooting

- Missing heavy deps in local runs: keep `LIGHT_TESTS=1` or run `uv pip install -e ".[dev,cli,data]"`
- Slow model downloads: set `HF_HOME=$PWD/.hf-cache`
- Path issues in tests: Ensure `src` is on `PYTHONPATH` or rely on `tests/conftest.py` which injects it automatically.


## What the Suite Verifies

- Embeddings: shape and determinism in light mode
- FAISS store: build/search roundtrip and index I/O
- CLI: help renders and core commands import
- API: health endpoint, auth enforcement, safe ingest
- Agent: robust JSON parsing under messy model outputs (stubbed)
- Index dedup: xfail placeholder test to guide the upcoming refactor

