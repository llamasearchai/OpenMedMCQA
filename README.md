# MedAgent (MedMCQA RAG + API)

Author: Nik Jois

MedAgent is a production-grade RAG system for MedMCQA-style multiple-choice questions. It combines deterministic embeddings, FAISS retrieval, strict JSON outputs, and a clean CLI/API. It supports Hugging Face models locally and OpenAI-compatible embedding/generation servers (OpenAI, Ollama, LM Studio).

Features
- CLI: info, ingest, build-index, ask, evaluate, serve-datasette, llm-cmd, menu, embeddings-info, pull-ollama-model
- API: FastAPI service for programmatic access (info, ingest, index, ask)
- Retrieval: FAISS cosine-similarity index with normalized vectors
- Embeddings: HF models (SciBERT/medBERT) and OpenAI-compatible servers
- Storage: SQLite with schema-managed tables (sqlite-utils)
- Determinism: LIGHT_TESTS stub embeddings; generation at temperature 0.0
- Tooling: Datasette browsing and llm CLI integration
 - Observability & Security: Auth via API key, CORS, structured JSON logs, Prometheus metrics, rate limiting + daily quotas
 - New endpoints: `/ask/batch`, `/models`, `/models/select`, enriched `/info`
 - New CLI: `models`, `export`, `report`, `report-compare`, `dashboard`

Quickstart
- Create venv and install: `uv pip install -e ".[dev,cli,data]"` or Python venv below
- Ingest sample: `python scripts/convert_json_to_jsonl.py scripts/sample_questions.json scripts/sample_questions.jsonl && medagent ingest scripts/sample_questions.jsonl`
- Build index: `medagent build-index`
- Ask: `medagent ask --question "Which vitamin deficiency causes scurvy?" --A "Vitamin A" --B "Vitamin C" --C "Vitamin D" --D "Vitamin K"`
- API: `uvicorn medagent.api:app --host 0.0.0.0 --port 8000`

Environment
- Required: `OPENAI_API_KEY`
- Optional: `MEDAGENT_MODEL=gpt-4o-mini`, `MEDAGENT_EMBEDDING_MODEL=allenai/scibert_scivocab_uncased`, `MEDAGENT_DB=medagent.db`, `MEDAGENT_INDEX_DIR=.index`, `MEDAGENT_TEMPERATURE=0.0`, `LIGHT_TESTS=1`, `MEDAGENT_USE_ASSISTANTS=0`, `MEDAGENT_EMBEDDING_BACKEND=hf`, `MEDAGENT_EMBEDDING_API_BASE=`, `MEDAGENT_API_BASE=`

Embedding backends
- HF (default): set `MEDAGENT_EMBEDDING_BACKEND=hf` and `MEDAGENT_EMBEDDING_MODEL` to a HF model (e.g., `allenai/scibert_scivocab_uncased`).
- OpenAI-compatible (OpenAI, Ollama, LM Studio): set `MEDAGENT_EMBEDDING_BACKEND=openai`, `MEDAGENT_EMBEDDING_API_BASE` (e.g., `http://localhost:11434/v1`), and `MEDAGENT_EMBEDDING_MODEL` (e.g., `nomic-embed-text`).
- Inspect: `medagent embeddings-info`
- Pull Ollama: `medagent pull-ollama-model nomic-embed-text`

Generation via local OpenAI-compatible servers
- Point chat completions to a local server with `MEDAGENT_API_BASE` (e.g., `http://localhost:11434/v1`). Set `OPENAI_API_KEY` if required.

Data schema (JSONL)
- Each line: `id`, `question`, `options.A..D`, `correct_answer` (optional: `explanation`, `subject`, `topic`, `difficulty`, `source`).
- Convert array JSON: `python scripts/convert_json_to_jsonl.py scripts/sample_questions.json scripts/sample_questions.jsonl`

CLI reference
- info — print configuration
- ingest — load questions and seed contexts
- build-index — embed contexts and build/update FAISS
- ask — answer a single question
- evaluate — batch evaluation and DB logging
- serve-datasette — explore SQLite in browser
- llm-cmd — run a quick prompt through the `llm` CLI
- embeddings-info — show embedding backend and vector size
- pull-ollama-model — pre-pull an Ollama model

API reference (FastAPI)
- GET /health — health check
- GET /info — configuration
- POST /ingest?jsonl_path=... — ingest dataset
- POST /index — build/update index
- POST /ask — body: `{ "question": str, "A": str, "B": str, "C": str, "D": str }`
 - POST /ask/batch — body: `[AskPayload, ...]` returns `{count, items}`
 - GET /models — model/embedding presets
 - POST /models/select — persist overrides for model/backend/api base

Quality gates
- Tests: `pytest -q`
- Lint/format: `ruff check src tests` / `ruff format src tests`
- Types: `mypy src`
- Tox matrix: `tox`
 - CI: GitHub Actions (`.github/workflows/ci.yml`, `.github/workflows/docker.yml`)

Docker
- Build: `docker build -t medagent:latest .`
- Run: `docker run --rm -p 8000:8000 --env-file .env -v $PWD:/app medagent:latest`
- Compose: `docker compose up --build`

Troubleshooting
- First index run crash: rebuild after applying first-run store check
- Ingest returns 0: use JSONL (not array JSON)
- Slow HF downloads: set `HF_HOME=$PWD/.hf-cache`
- Port busy: use `--port` option for datasette or API
- Local servers: verify `MEDAGENT_EMBEDDING_API_BASE` and `MEDAGENT_API_BASE`
 - 429 errors: check rate limit headers and daily quota; adjust `MEDAGENT_RATE_LIMIT_*` and `MEDAGENT_QUOTA_DAILY`

Assistants note
- Default is Chat Completions. An Assistants-compatible class is present but uses the same path; full threads/runs are not implemented.

License
- MIT
