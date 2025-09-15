# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-09-14

Added
- API: Rate limiting (token bucket) and daily per-key quotas with response headers.
- API: Batch ask endpoint `POST /ask/batch` with per-item error handling.
- API: `/models` presets and `POST /models/select` to persist overrides.
- API: `/info` now includes index stats and embedding cache stats; config overrides echoed.
- Observability: Structured JSON logging with request IDs; Prometheus metrics for ask/ingest/index latency and counts; embedding cache hit/miss; FAISS search latency.
- CLI: `models`, `export`, `report`, `report-compare`, `dashboard` commands.
- Reliability: FAISS index deduplication; `--rebuild` flag in `build-index`.
- Security: API key auth on sensitive endpoints; safe ingest path + file-size limit; CORS control.
- Dev: Multi-stage Docker build; Docker CI workflow with Trivy scan; Devcontainer.
- Tests: Broad coverage for API, CLI, rate limiting, exports, reports, models, parsing robustness.

Changed
- Stricter JSON parsing for LLM responses with schema validation and normalization.
- Embedding system now supports on-disk per-text caching (disabled in light mode).

## [0.1.0] - Initial
- Initial MedAgent release with CLI, API, FAISS retrieval, embeddings, evaluation, and tests.

