from __future__ import annotations

from typing import Any


class _Noop:
    def labels(self, *args: Any, **kwargs: Any):  # noqa: D401
        return self

    def inc(self, *args: Any, **kwargs: Any) -> None:
        return None

    def observe(self, *args: Any, **kwargs: Any) -> None:
        return None


try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter, Histogram

    ASK_REQUESTS = Counter(
        "medagent_ask_requests_total", "Total ask requests", ["route", "status"]
    )
    ASK_LATENCY = Histogram(
        "medagent_ask_latency_ms", "Ask latency in ms", ["route"], buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500)
    )
    EMB_CACHE_HITS = Counter(
        "medagent_emb_cache_hits_total", "Embedding cache hits", ["backend"]
    )
    EMB_CACHE_MISSES = Counter(
        "medagent_emb_cache_misses_total", "Embedding cache misses", ["backend"]
    )
    FAISS_SEARCH_LATENCY = Histogram(
        "medagent_faiss_search_latency_ms", "FAISS search latency in ms", buckets=(0.1, 0.5, 1, 2, 5, 10, 25, 50, 100)
    )
except Exception:  # pragma: no cover
    ASK_REQUESTS = _Noop()
    ASK_LATENCY = _Noop()
    EMB_CACHE_HITS = _Noop()
    EMB_CACHE_MISSES = _Noop()
    FAISS_SEARCH_LATENCY = _Noop()
