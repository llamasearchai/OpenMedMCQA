from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import os
from fastapi import FastAPI, HTTPException, Depends, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uuid
import time
import logging
from pydantic import BaseModel

from .config import settings
from .db import connect, insert_questions, insert_contexts, config_set, config_get, quota_consume
from .datasets import load_medmcqa_jsonl, seed_contexts_from_questions
from .embeddings import MedEmbedder
from .retrieval import FAISSStore
from .rag import index_contexts
from .agents import get_agent
from .evaluation import evaluate_run
from .models import EMBEDDING_PRESETS, CHAT_PRESETS
from .logging_utils import get_logger
from .metrics import ASK_REQUESTS, ASK_LATENCY
from .rate_limit import RateLimiter, Quota


app = FastAPI(title="MedAgent API", version="0.1.0")

# CORS configuration (allow list via env, comma-separated)
_cors_origins = os.environ.get("MEDAGENT_CORS_ORIGINS", "*")
if _cors_origins:
    origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"]
    )


def require_api_key(x_api_key: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)) -> None:
    """Require API key when MEDAGENT_API_KEY is set.
    Accepts either X-API-Key: <key> or Authorization: Bearer <key>.
    """
    expected = os.environ.get("MEDAGENT_API_KEY", "")
    if not expected:
        return  # No key required when not configured
    supplied = None
    if x_api_key:
        supplied = x_api_key
    elif authorization and authorization.lower().startswith("bearer "):
        supplied = authorization.split(" ", 1)[1].strip()
    if supplied != expected:
        raise HTTPException(status_code=401, detail="invalid or missing api key")


# Optional in-process rate limiter (disabled by default)
_rl_enabled = os.environ.get("MEDAGENT_RATE_LIMIT", "0") == "1"
_rl_rps = float(os.environ.get("MEDAGENT_RATE_LIMIT_RPS", "5"))
_rl_burst = int(os.environ.get("MEDAGENT_RATE_LIMIT_BURST", "10"))
_rate_limiter = RateLimiter(_rl_rps, _rl_burst) if _rl_enabled else None

# Optional daily quota per API key
_quota_daily = int(os.environ.get("MEDAGENT_QUOTA_DAILY", "0"))
_quota = Quota(_quota_daily) if _quota_daily > 0 else None


def require_rate_limit(x_api_key: Optional[str] = Header(default=None), response: Response = ... ) -> None:  # type: ignore[assignment]
    key = x_api_key or "global"
    # Rate limit token bucket (if enabled)
    if _rate_limiter is not None:
        allowed = _rate_limiter.check(key)
        remaining, reset_s = _rate_limiter.state(key)
        limit_str = f"{_rl_rps:.3g}rps;burst={_rl_burst}"
        response.headers["X-RateLimit-Limit"] = limit_str
        response.headers["X-RateLimit-Remaining"] = str(int(remaining))
        response.headers["X-RateLimit-Reset"] = str(int(round(reset_s)))
        response.headers["X-RateLimit-Policy"] = "token-bucket"
        if not allowed:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
    # Quota check (after rate limit) if enabled
    if _quota is not None:
        if not _quota.consume(key):
            response.headers["X-Quota-Limit"] = str(_quota_daily)
            response.headers["X-Quota-Remaining"] = str(_quota.remaining(key))
            response.headers["X-Quota-Reset"] = str(_quota.reset_seconds())
            raise HTTPException(status_code=429, detail="quota exceeded")
        else:
            response.headers["X-Quota-Limit"] = str(_quota_daily)
            response.headers["X-Quota-Remaining"] = str(_quota.remaining(key))
            response.headers["X-Quota-Reset"] = str(_quota.reset_seconds())


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    logger = get_logger("medagent.api")
    start = time.perf_counter()
    try:
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        rec = logging.LogRecord(
            name="medagent.api",
            level=logging.INFO,
            pathname=__file__,
            lineno=0,
            msg=f"{request.method} {request.url.path}",
            args=(),
            exc_info=None,
        )
        rec.request_id = rid  # type: ignore[attr-defined]
        rec.method = request.method  # type: ignore[attr-defined]
        rec.path = request.url.path  # type: ignore[attr-defined]
        rec.status_code = getattr(response, "status_code", 200)  # type: ignore[attr-defined]
        rec.latency_ms = round(latency_ms, 2)  # type: ignore[attr-defined]
        logger.handle(rec)
        response.headers["X-Request-ID"] = rid
        return response
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000.0
        rec = logging.LogRecord(
            name="medagent.api",
            level=logging.INFO,
            pathname=__file__,
            lineno=0,
            msg=f"{request.method} {request.url.path}",
            args=(),
            exc_info=None,
        )
        rec.request_id = rid  # type: ignore[attr-defined]
        rec.method = request.method  # type: ignore[attr-defined]
        rec.path = request.url.path  # type: ignore[attr-defined]
        rec.status_code = 500  # type: ignore[attr-defined]
        rec.latency_ms = round(latency_ms, 2)  # type: ignore[attr-defined]
        logger.handle(rec)
        raise

# Prometheus metrics (optional)
try:  # pragma: no cover - optional dependency
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)
except Exception:
    pass


class AskPayload(BaseModel):
    question: str
    A: str
    B: str
    C: str
    D: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    settings.ensure_dirs()
    s = settings.model_dump()
    # Redact secrets
    if s.get("openai_api_key"):
        s["openai_api_key"] = "***redacted***"
    # Include index and cache stats
    try:
        embed_cache_dir = os.environ.get("MEDAGENT_EMB_CACHE_DIR", ".emb_cache")
        total_bytes = 0
        total_files = 0
        if os.path.isdir(embed_cache_dir):
            for root, _, files in os.walk(embed_cache_dir):
                for fn in files:
                    p = os.path.join(root, fn)
                    try:
                        total_bytes += os.path.getsize(p)
                        total_files += 1
                    except OSError:
                        pass
        s["embedding_cache_stats"] = {"dir": embed_cache_dir, "files": total_files, "bytes": total_bytes}
    except Exception:
        s["embedding_cache_stats"] = {"dir": "", "files": 0, "bytes": 0}
    try:
        idx_dir = os.environ.get("MEDAGENT_INDEX_DIR", settings.index_dir)
        store = FAISSStore(dim=128, index_dir=idx_dir)  # dim unused when reading existing index
        index_bytes = os.path.getsize(store.index_path) if os.path.exists(store.index_path) else 0
        ids_bytes = os.path.getsize(store.ids_path) if os.path.exists(store.ids_path) else 0
        vectors = len(store.ids) if os.path.exists(store.ids_path) else 0
        s["index_stats"] = {"index_path": store.index_path, "ids_path": store.ids_path, "vectors": int(vectors), "bytes": index_bytes + ids_bytes}
    except Exception:
        s["index_stats"] = {"vectors": 0, "bytes": 0}
    # Include persisted config
    try:
        db = connect()
        s["config_overrides"] = {
            "openai_model": config_get(db, "openai_model", s.get("openai_model", "")),
            "embedding_backend": config_get(db, "embedding_backend", s.get("embedding_backend", "")),
            "embedding_model": config_get(db, "embedding_model", s.get("embedding_model", "")),
            "openai_api_base": config_get(db, "openai_api_base", s.get("openai_api_base", "")),
            "embedding_api_base": config_get(db, "embedding_api_base", s.get("embedding_api_base", "")),
        }
    except Exception:
        s["config_overrides"] = {}
    return s


@app.post("/ingest")
def ingest(jsonl_path: str, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    db = connect()
    _t0 = time.perf_counter()
    # Ingest safety: allow only files within MEDAGENT_DATA_DIR (default .)
    data_root = os.path.abspath(os.environ.get("MEDAGENT_DATA_DIR", "."))
    path = jsonl_path
    if os.path.isabs(path) or ".." in path.replace("\\", "/"):
        raise HTTPException(status_code=400, detail="path not allowed; use relative path within data dir")
    full = os.path.abspath(os.path.join(data_root, path))
    if not full.startswith(data_root + os.sep) and full != data_root:
        raise HTTPException(status_code=400, detail="path outside MEDAGENT_DATA_DIR")
    if not os.path.isfile(full):
        raise HTTPException(status_code=400, detail="file not found")
    max_bytes = int(os.environ.get("MEDAGENT_MAX_INGEST_BYTES", str(50 * 1024 * 1024)))
    try:
        size = os.path.getsize(full)
    except OSError:
        size = 0
    if size > max_bytes:
        raise HTTPException(status_code=413, detail="file too large")
    rows = load_medmcqa_jsonl(full)
    insert_questions(db, rows)
    ctx = seed_contexts_from_questions(rows)
    insert_contexts(db, ctx)
    latency_ms = (time.perf_counter() - _t0) * 1000.0
    try:
        ASK_REQUESTS.labels(route="ingest", status="ok").inc()
        ASK_LATENCY.labels(route="ingest").observe(latency_ms)
    except Exception:
        pass
    return {"questions": len(rows), "contexts": len(ctx), "latency_ms": latency_ms}


@app.post("/index")
def build_index(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    settings.ensure_dirs()
    _t0 = time.perf_counter()
    db = connect()
    rows = list(db.query("select ctx_id, text, source, meta from contexts"))
    embedder = MedEmbedder(settings.embedding_model, light_mode=settings.light_tests)
    store = FAISSStore(dim=embedder.dim)
    batch = 512
    buffer = []
    for r in rows:
        buffer.append(r)
        if len(buffer) >= batch:
            index_contexts(embedder, store, buffer)
            buffer.clear()
    if buffer:
        index_contexts(embedder, store, buffer)
    latency_ms = (time.perf_counter() - _t0) * 1000.0
    try:
        ASK_REQUESTS.labels(route="index", status="ok").inc()
        ASK_LATENCY.labels(route="index").observe(latency_ms)
    except Exception:
        pass
    return {"indexed": len(rows), "latency_ms": latency_ms}


@app.post("/ask")
def ask(p: AskPayload, response: Response, x_api_key: Optional[str] = Header(default=None), _: None = Depends(require_api_key)) -> Dict[str, Any]:
    # Apply rate limit and quota
    require_rate_limit(x_api_key, response)  # type: ignore[arg-type]
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")
    embedder = MedEmbedder(settings.embedding_model, light_mode=settings.light_tests)
    store = FAISSStore(dim=embedder.dim)
    agent = get_agent(embedder, store)
    q = {"id": "api", "question": p.question, "A": p.A, "B": p.B, "C": p.C, "D": p.D, "correct": "?", "explanation": "", "subject": "", "topic": "", "difficulty": "", "source": ""}
    try:
        ans, conf, expl, ctx_ids, raw, latency_ms = agent.answer_medmcqa(q)
        ASK_REQUESTS.labels(route="ask", status="ok").inc()
        ASK_LATENCY.labels(route="ask").observe(latency_ms)
        return {"answer": ans, "confidence": conf, "explanation": expl, "contexts": ctx_ids, "latency_ms": latency_ms}
    except HTTPException:
        ASK_REQUESTS.labels(route="ask", status="error").inc()
        raise


@app.post("/ask/batch")
def ask_batch(payload: List[AskPayload], response: Response, x_api_key: Optional[str] = Header(default=None), _: None = Depends(require_api_key)) -> Dict[str, Any]:
    # Apply rate limit and quota
    require_rate_limit(x_api_key, response)  # type: ignore[arg-type]
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")
    embedder = MedEmbedder(settings.embedding_model, light_mode=settings.light_tests)
    store = FAISSStore(dim=embedder.dim)
    agent = get_agent(embedder, store)
    results: List[Dict[str, Any]] = []
    for p in payload:
        q = {"id": "api", "question": p.question, "A": p.A, "B": p.B, "C": p.C, "D": p.D, "correct": "?", "explanation": "", "subject": "", "topic": "", "difficulty": "", "source": ""}
        try:
            ans, conf, expl, ctx_ids, raw, latency_ms = agent.answer_medmcqa(q)
            ASK_REQUESTS.labels(route="ask_batch", status="ok").inc()
            ASK_LATENCY.labels(route="ask_batch").observe(latency_ms)
            results.append({"answer": ans, "confidence": conf, "explanation": expl, "contexts": ctx_ids, "latency_ms": latency_ms})
        except Exception as e:  # per-item error handling for batch
            ASK_REQUESTS.labels(route="ask_batch", status="error").inc()
            results.append({"error": str(e)})
    return {"count": len(results), "items": results}


@app.get("/models")
def models() -> Dict[str, Any]:
    return {"embedding_presets": EMBEDDING_PRESETS, "chat_presets": CHAT_PRESETS}


def run() -> None:
    import uvicorn
    uvicorn.run("medagent.api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
class ModelSelectPayload(BaseModel):
    embedding_backend: Optional[str] = None
    embedding_model: Optional[str] = None
    openai_model: Optional[str] = None
    openai_api_base: Optional[str] = None
    embedding_api_base: Optional[str] = None


@app.post("/models/select")
def models_select(p: ModelSelectPayload, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    db = connect()
    # Persist overrides
    if p.embedding_backend:
        config_set(db, "embedding_backend", p.embedding_backend)
        settings.embedding_backend = p.embedding_backend
    if p.embedding_model:
        config_set(db, "embedding_model", p.embedding_model)
        settings.embedding_model = p.embedding_model
    if p.openai_model:
        config_set(db, "openai_model", p.openai_model)
        settings.openai_model = p.openai_model
    if p.openai_api_base is not None:
        config_set(db, "openai_api_base", p.openai_api_base)
        settings.openai_api_base = p.openai_api_base
    if p.embedding_api_base is not None:
        config_set(db, "embedding_api_base", p.embedding_api_base)
        settings.embedding_api_base = p.embedding_api_base
    return {"ok": True, "settings": settings.model_dump()}
