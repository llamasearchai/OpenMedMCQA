# Master Review, Gap Analysis, and Roadmap — MedAgent (OpenMedMCQA)

Status: Initial pass based on repository inspection only. Dynamic tests, benchmarks, and external API calls not executed in this environment.


## Executive Summary

MedAgent is a focused, production-leaning MedMCQA RAG system with a clear module layout, a practical CLI and API, and a sensible dependency stack. Core strengths include: deterministic testing mode, FastAPI + Typer ergonomics, SQLite schema management, FAISS retrieval, and a simple Docker/Compose setup. The codebase is compact and readable with type hints and modern tooling (ruff, mypy, tox).

Key gaps to address for production-grade rigor:
- Security: API is unauthenticated; ingest takes arbitrary file paths; no rate limiting/CORS/authZ.
- Reliability/Performance: FAISS index may duplicate entries across runs; no embedding caching; index build re-embeds everything; limited error handling and no file locking.
- Observability: Minimal logging and no metrics/tracing; limited latency/quality instrumentation.
- CI/CD & Quality Gates: No CI workflows; coverage is limited; types not enforced in CI.
- Test Depth: Unit/integration tests cover smoke paths only; no retrieval/agent behavior tests under stubs; no API contract/property tests.
- Documentation & DX: Good README quickstart; missing architecture and operations docs; no pre-commit; no devcontainer; no contribution guide.

Priority recommendations (first 2–3 weeks):
1) Lock down the API: add auth, path restrictions, CORS, rate limiting, request size limits; harden input validation. 2) Fix index duplication and add embedding caching with sane rebuild/update semantics. 3) Add CI (lint, type, test matrix) and pre-commit. 4) Add structured logging + Prometheus metrics and basic traces. 5) Extend tests with stubs to validate retrieval, agent JSON output, and evaluation loop without external network.

Expected impact: Substantial improvements to security posture, reliability, and operability with low-to-moderate implementation complexity. Provides confidence to pass external audits and independent QA.


## Architecture Overview (Evidence: src/medagent/*, pyproject.toml)

- Core modules
  - `config.py`: Pydantic-based settings from env; ensures dirs.
  - `db.py`: SQLite schema management (sqlite-utils) with runs/predictions.
  - `datasets.py`: JSONL ingestion; seeds retrieval contexts from Q/A content.
  - `embeddings.py`: HF or OpenAI-compatible embeddings; light (deterministic) mode.
  - `retrieval.py`: FAISS IndexFlatIP store with cosine via normalized vectors.
  - `rag.py`: Indexing workflow; first-run guard; similarity voting utility.
  - `agent.py`: Chat Completions agent (tenacity retry, strict JSON extraction).
  - `assistants_agent.py`: Parity path for Assistants API (placeholder, no retries).
  - `agents.py`: Switch between agent variants by flag.
  - `evaluation.py`: Batch evaluation loop, progress UI, DB logging.
  - `api.py`: FastAPI service (`/health`, `/info`, `/ingest`, `/index`, `/ask`).
  - `cli.py`: Typer commands for info/ingest/index/ask/evaluate/datasette/llm/menu.
  - `datasette_integration.py`: Launch Datasette against SQLite DB.
  - `llm_integration.py`: Shell to `llm` CLI command.
- Data stores
  - SQLite file for questions, contexts, runs, predictions (`medagent.db`).
  - On-disk FAISS index and context ID array under `.index/`.
- Interfaces
  - CLI entrypoints: `medagent`, `medagent-api` (pyproject scripts).
  - REST API (FastAPI): JSON payloads for ask; path param for ingest; server info/health.
  - Docker + Compose for containerized API; Makefile and tox for local workflows.
- Dependencies (pyproject)
  - Runtime: FastAPI/Uvicorn, transformers/torch, faiss-cpu, sqlite-utils, openai, httpx/requests, pydantic, numpy/pandas/sklearn, jinja2, orjson, tenacity, datasette, typer/rich.
  - Dev: pytest (+cov/xdist), ruff, mypy, typeshed packages. Optional: `llm`, `datasets`.


## Data Flows (Evidence: datasets.py, db.py, rag.py, retrieval.py, agent.py, api.py)

- Ingest: JSONL → `datasets.load_medmcqa_jsonl` → `db.insert_questions` → `seed_contexts_from_questions` → `db.insert_contexts`.
- Index: DB `contexts` → embed in batches (512) → `FAISSStore.build` on first chunk, then `add` for the rest → files under `.index/`.
- Retrieval: Query embeds for question and choices → FAISS search → top-k per query → score aggregation vote → fetch texts from DB → prompt LLM.
- Answer: Agent builds system+user messages, calls Chat Completions, extracts strict JSON, returns answer, confidence, explanation, context IDs, and latency.
- Evaluation: Iterates questions, logs predictions + latencies, computes accuracy.


## Findings and Evidence

- Security
  - Unauthenticated API: No auth on any endpoint (`api.py`). Risk for public deployments.
  - Ingest path injection: `/ingest?jsonl_path=...` accepts arbitrary server path; no sandbox or whitelisting; potential data exfiltration or denial.
  - No rate limiting, CORS, or request size limits. Potential abuse vectors; missing CSRF/CORS policy if served cross-origin.
  - Secrets: Uses env vars; fine, but recommend explicit secrets management and not exposing `OPENAI_API_KEY` via `/info` (currently `/info` dumps settings; verify it does not leak secrets—`settings.model_dump()` will include empty/open string for `openai_api_key` but check risk if non-empty).
- Reliability & Performance
  - Index duplication: `build_index` embeds all contexts and repeatedly calls `index_contexts`. If index already exists, subsequent runs `add` duplicates, growing index and degrading retrieval.
  - No embedding caching/persistence: Re-embeds same contexts; slow startup and expensive CPU/GPU consumption.
  - No file locking on index writes: Concurrent runs may corrupt index files.
  - Assistants path lacks retries/backoff; risk of transient failures.
  - Error handling: Limited validation for malformed JSONL ingest or empty datasets.
- Privacy & Compliance
  - Prompt includes a safety note; OK for academic use. No PII flows. For regulated environments, clarify PHI is unsupported and document data handling and retention.
- Test Strategy & Coverage
  - Tests: smoke for API health and CLI help, FAISS roundtrip, light embeddings shape. No behavioral tests for agent JSON parsing, retrieval quality under stubs, or evaluation logging assertions.
  - No golden tests for strict JSON conformance; brittle JSON extraction fallback substring could raise.
- CI/CD & Build
  - No CI workflows. Dockerfile is decent but single-stage; size optimizations possible. No SBOM or image scanning.
- Observability
  - No structured logging, metrics, or traces. Latency is measured per request but not exported.
- Documentation & DX
  - README is solid. Missing: architecture diagram, ops runbook, API auth story, perf guidelines, contribution guide, changelog, pre-commit config, devcontainer.


## Gap Analysis → Prioritized, Actionable Improvements

P0 — Security & Safety (High impact, Low–Med complexity)
- Add API authentication: per-request bearer token; support `X-API-Key` header. Optionally OAuth2 for multi-user.
- Restrict ingest: require upload body or restrict to whitelisted data dir; reject absolute/out-of-tree paths.
- Add CORS policy, rate limiting, request body size limit, and sensible timeouts.
- Ensure `/info` redacts secrets; add safe mode.

P0 — Reliability & Index Hygiene (High impact, Low–Med complexity)
- Rebuild vs. update: detect existing index and either rebuild from scratch or only add new contexts based on DB `ctx_id` diff.
- Embedding cache: persist embeddings to disk keyed by `(model_name, ctx_id, text_hash)`; reuse on rebuild.
- File locking: adopt advisory locks around `.index/*` writes.

P1 — Observability (High impact, Low complexity)
- Structured logging (JSON) and enriched context (run_id, request_id).
- Prometheus metrics: request latency, errors, retrieval depth, FAISS timings, token usage.
- Basic OpenTelemetry traces for `ask` path.

P1 — CI/CD & Quality Gates (Med impact, Low complexity)
- GitHub Actions: lint (ruff), type (mypy), tests (pytest) on 3.10/3.11; coverage threshold.
- Build & publish wheel; optional Docker image build with SBOM/scan.

P2 — Test Suite Expansion (Med impact, Med complexity)
- Add stubs for OpenAI client in LIGHT_TESTS to validate JSON parsing and agent behavior.
- Property tests for JSONL ingest; DB schema invariants; retrieval contract tests.
- API contract tests with pydantic models and negative cases.

P2 — DevEx & Docs (Med impact, Low complexity)
- Pre-commit with ruff, black/ruff format, mypy, end-of-file-fixer.
- Architecture and ops docs, security hardening guide, contribution guide, changelog.

P3 — Features & Enhancements (User/Dev value)
- Batch ask endpoint and CLI (evaluate-like for ad-hoc sets).
- Retrieval debug mode: return scored contexts and overlaps; interactive UI via Datasette queries.
- Model/embedding registry with presets; environment introspection endpoint.
- Export results to CSV/Parquet; simple dashboard with accuracy over time.


## Implementation Sketches (pseudocode/diffs)

1) API authentication and safe ingest
```python
# src/medagent/api.py
from fastapi import Depends, Header

API_KEY = os.environ.get("MEDAGENT_API_KEY", "")

def require_api_key(x_api_key: str = Header("") ):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(401, "invalid or missing api key")

@app.get("/info")
def info(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    s = settings.model_dump()
    s["openai_api_key"] = "***redacted***" if settings.openai_api_key else ""
    return s

@app.post("/ingest")
def ingest(jsonl_path: str, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    if os.path.isabs(jsonl_path) or ".." in jsonl_path:
        raise HTTPException(400, "path not allowed")
    allowed_root = os.environ.get("MEDAGENT_DATA_DIR", ".")
    full = os.path.abspath(os.path.join(allowed_root, jsonl_path))
    if not full.startswith(os.path.abspath(allowed_root)):
        raise HTTPException(400, "path outside data dir")
    # proceed...
```

2) Prevent FAISS index duplication and add rebuild flag
```python
# src/medagent/cli.py
@app.command()
def build_index(rebuild: bool = typer.Option(False, "--rebuild")):
    # if rebuild, start fresh index
    if rebuild and os.path.exists(store.index_path):
        os.remove(store.index_path)
        if os.path.exists(store.ids_path):
            os.remove(store.ids_path)
    # embed only new ctxs if not rebuild
    existing_ids = set(store.ids.tolist()) if len(store.ids) else set()
    rows = [r for r in db.query("select ctx_id, text, source, meta from contexts") if r["ctx_id"] not in existing_ids]
    # then batch + index_contexts as today
```

3) Embedding cache
```python
# src/medagent/embeddings.py
class MedEmbedder:
    def __init__(...):
        self._cache_dir = Path(os.environ.get("MEDAGENT_EMB_CACHE", ".emb_cache"))
        self._cache_dir.mkdir(exist_ok=True)

    def _cache_key(self, text: str) -> Path:
        import hashlib
        h = hashlib.sha256((self.model_name + "\0" + text).encode()).hexdigest()
        return self._cache_dir / f"{h}.npy"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # check per-text cache; batch compute for misses only
```

4) Prometheus metrics (FastAPI)
```python
# app wiring (e.g., api.py)
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(...)
Instrumentator().instrument(app).expose(app)
```

5) GitHub Actions (CI)
```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix: {python: ["3.10", "3.11"]}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: ${{ matrix.python }}}
      - run: pip install uv
      - run: uv pip install -e ".[dev,cli,data]"
      - run: ruff check src tests
      - run: ruff format --check src tests
      - run: mypy src
      - run: pytest -q --disable-warnings --maxfail=1 --cov=medagent --cov-report=xml
      - uses: codecov/codecov-action@v4
        if: ${{ always() }}
        with: {files: coverage.xml}
```

6) Pre-commit config
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks: [{id: ruff}, {id: ruff-format}]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.0
    hooks: [{id: mypy}]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
```


## Test Plan and Coverage

- Unit tests
  - `embeddings`: light mode shapes; caching roundtrip; backend selection logic.
  - `retrieval`: build/add/rebuild semantics; dedup; search edge cases on empty index.
  - `datasets`: malformed lines; missing required fields; property tests for schema.
  - `db`: schema creation idempotence; run/prediction inserts; VACUUM/PRAGMAs if added.
  - `agent`: JSON strictness parsing (with stubbed OpenAI client); retry behavior and backoff.
  - `api`: auth middleware; input validation; path restriction; request size limits.
- Integration tests
  - `ingest → index → ask` on LIGHT_TESTS with a stub LLM returning deterministic JSON.
  - Batch evaluation correctness: accuracy computation and run logging.
- Performance tests
  - Index build throughput with LIGHT_TESTS; FAISS search latency distribution.
- Coverage target
  - Initial: 80% lines in `src/medagent/`, excluding CLI UI code.
  - Increase to 90% after caching/index refactors.
- Tooling
  - `pytest -q --cov=medagent --cov-report=term-missing`
  - Type coverage: `mypy --strict src` must pass.

Note: In this environment, tests were not executed. See Validation Results for commands to run locally/CI.


## Validation Results (current run)

- Static review completed across modules and configs. No dynamic execution.
- Suggested commands to validate locally:
  - `make setup && make lint && make type && make test`
  - `uvicorn medagent.api:app --host 0.0.0.0 --port 8000` then `GET /health`.
  - `medagent ingest scripts/sample_questions.jsonl && medagent build-index --rebuild && medagent ask --question ... --A ... --B ... --C ... --D ...`
- If environment lacks `faiss-cpu`/`torch/transformers`, use `LIGHT_TESTS=1` and skip heavy paths.


## Risks, Mitigations, and Rollback

- Risk: Unauthenticated API exposure → Mitigation: API key/OAuth; CORS + rate limiting; Rollback: toggle feature flag to disable auth for dev only.
- Risk: Index corruption on concurrent writes → Mitigation: file locks; atomic writes; Rollback: restore from last known-good snapshot.
- Risk: Duplicate contexts causing degraded accuracy → Mitigation: rebuild-only or dedup logic; Rollback: clear `.index/` and rebuild.
- Risk: Dependency CVEs (torch/faiss/transformers) → Mitigation: Dependabot + pinned upper bounds; image scanning; Rollback: pin to last known-good versions.
- Risk: Prompt/JSON parsing brittleness → Mitigation: function calling or JSON schema validation; Rollback: strict fallback parser with logging.


## Roadmap (Phases, Tasks, Owners, Timelines)

Legend: Impact (H/M/L), Complexity (H/M/L). Owners: Backend, Infra, Security, QA, Docs.

Phase 0 (Week 0–1): Baseline Hardening
- API auth and safe ingest (Impact H, Complexity M) — Owner: Security/Backend
  - Add API key middleware and redact `/info`.
  - Restrict ingest path to data dir; size/time limits.
  - Checklist: auth enabled in prod; 401 on missing/invalid; `/info` hides secrets.
- CI setup (Impact H, Complexity L) — Owner: Infra
  - GH Actions with lint/type/test/coverage; codecov optional.
  - Gate on coverage ≥ 80%.
- Pre-commit + formatting (Impact M, Complexity L) — Owner: DevEx

Phase 1 (Week 1–2): Reliability & Observability
- Index hygiene and dedup (Impact H, Complexity M) — Owner: Backend
  - `--rebuild` flag; only add new `ctx_id`s; locking.
  - Embedding cache.
- Observability (Impact H, Complexity L) — Owner: Infra
  - Structured logging; Prometheus metrics; basic traces.

Phase 2 (Week 2–3): Tests & Docs
- Expanded tests (Impact M, Complexity M) — Owner: QA/Backend
  - Stubs for OpenAI; retrieval/agent/evaluation tests; negative API tests.
- Docs (Impact M, Complexity L) — Owner: Docs
  - Architecture diagram; ops runbook; security guide; contribution guide.

Phase 3 (Week 3–5): Features
- Batch ask endpoint/CLI; results export; retrieval debug; model registry. Owners: Backend/DevEx.

Dependencies & Prereqs
- CI secrets: `MEDAGENT_API_KEY`, limited dummy `OPENAI_API_KEY` for smoke (optional; prefer stubs).
- Test data: sample JSONL present; expand with fixtures.


## Issue-by-Issue Iterative Loop (examples)

1) Unauthenticated API and unsafe ingest path
- Root cause: Missing auth and path validation in `api.py`.
- Options: API key header; OAuth2 (fastapi-security); reverse-proxy auth.
- Decision: Start with API key + CORS + limits; optional OAuth later.
- Implementation steps: Add dependency `require_api_key`; redact `/info`; path whitelist for `/ingest`; add `--max-request-size` via server config.
- Tests: 401 on missing/invalid key; path traversal blocked; success cases.
- Validate: `pytest -q` green; manual curl with/without key.

2) Index duplication and rebuild semantics
- Root cause: `build_index` unconditionally adds all contexts when index exists.
- Options: Always rebuild; track last-indexed `ctx_id`; maintain index metadata.
- Decision: Implement `--rebuild` and dedup by `ctx_id` diff; add locks.
- Implementation steps: Compute `existing_ids` from store; filter rows; add file locks; optional `VACUUM` DB.
- Tests: Re-run build twice; index size constant; search correctness.
- Validate: coverage for rebuild path; manual inspection of `.index/*`.

3) Embedding caching
- Root cause: Repeated embedding for same text; slow builds.
- Options: On-disk `.npy` cache; SQLite table; external vector DB.
- Decision: Simple `.npy` cache keyed by `(model, hash(text))`.
- Implementation steps: Cache lookup; batch misses; write after compute; invalidate on model change.
- Tests: cache hit/miss counts; time savings in LIGHT_TESTS stub.
- Validate: benchmark harness compares build times with/without cache.

4) Observability
- Root cause: No metrics/tracing/logging standards.
- Options: Prometheus FastAPI instrumentator; structlog/loguru; OpenTelemetry.
- Decision: Add Prometheus + JSON logging; basic spans around retrieval and LLM call.
- Implementation steps: Wire instrumentator; add middleware for request_id; log timings.
- Tests: Metrics endpoint exports; unit test logger context fields.
- Validate: scrape locally; confirm latency histograms.

5) Tests and JSON strictness
- Root cause: Fragile substring JSON recovery.
- Options: Function calling or JSON schema-based extraction; robust JSON fixer.
- Decision: Keep function calling option for OpenAI; add schema validation and fallback fixer with logs.
- Implementation steps: Pydantic model for LLM output; parse/validate; retry once with stricter system message if invalid.
- Tests: malformed content handled; output validated; no crashes.
- Validate: deterministic stub returns; error paths covered.


## Compliance, Privacy, Accessibility

- Compliance: Not designed for PHI; document non-HIPAA usage. Add user-visible disclaimers and ToS in README/API root.
- Privacy: Avoid logging question payloads at INFO; add configurable sampling/redaction.
- Accessibility: CLI and API; no UI; document API schemas; ensure readable CLI output.


## Infrastructure & IaC

- Docker: Consider multi-stage build to shrink image; pin slim tag; enable build cache mounts for `uv`. Add non-root (already present) and healthcheck (present). Optionally publish image via CI.
- Compose: Add `read_only: true` for container fs except data mounts; resource limits; `tmpfs` for ephemeral.
- Devcontainer: `.devcontainer/devcontainer.json` with Python 3.11, UV, FAISS deps; consistent DX.


## Documentation Updates

- Add `SECURITY.md` (reporting, supported versions) and `CONTRIBUTING.md` (dev setup, tests, style).
- Architecture doc with module diagram and data flow; operations runbook (common failures, index rebuilds, backups, recovery).
- API auth and usage examples (curl + Python clients).
- Changelog with semantic versioning policy.


## New Feature Concepts (value, acceptance criteria, metrics)

- Batch Ask API + CLI
  - Value: Faster ad-hoc evaluation without full runs.
  - Acceptance: `POST /ask/batch` accepts array; returns array with timing; 10k QPS sustained on LIGHT_TESTS.
  - Metrics: Throughput, P95 latency, error rate.

- Retrieval Debug Mode
  - Value: Improves trust/inspection; shows scored contexts.
  - Acceptance: `?debug=1` returns scores and normalized similarities; CLI `--debug` prints table.
  - Metrics: Developer usage; reduced triage time.

- Model/Embedding Presets
  - Value: Easy switching across HF/OpenAI/Ollama backends.
  - Acceptance: `GET /models` lists presets; CLI `medagent embeddings-info` resolves preset → model.
  - Metrics: Fewer config support requests.

- Results Dashboard
  - Value: Track accuracy over time per run/model.
  - Acceptance: Simple HTML report or Datasette queries; export CSV.
  - Metrics: Adoption by researchers.


## Assumptions and Required Artifacts

- Assumptions: Single-node SQLite/FAISS; no PHI; moderate dataset sizes; environments with Python 3.10+; GPU optional.
- Missing info to finalize: Production deployment topology, SSO/IAM requirements, data retention/SLA/SLOs, expected dataset sizes and throughput targets, approved models/backends list.
- Artifacts needed: CI secrets (`MEDAGENT_API_KEY`), staging environment URL, logging/metrics stack (Prometheus/Grafana) if used.


## Next Steps

- Approve Phase 0/1 plan and owners.
- Implement API auth + ingest restrictions; add CI and pre-commit.
- Ship index dedup + `--rebuild` and embedding cache.
- Add logging/metrics; expand tests with stubs.
- Re-run validation with coverage and share results.

