import os
import pytest

from medagent.embeddings import MedEmbedder
from medagent.retrieval import FAISSStore
from medagent.rag import index_contexts


def test_index_deduplication(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHT_TESTS", "1")
    e = MedEmbedder("dummy", light_mode=True)
    store = FAISSStore(dim=e.dim, index_dir=str(tmp_path))
    rows = [
        {"ctx_id": "c1", "text": "x1", "source": "s", "meta": ""},
        {"ctx_id": "c2", "text": "x2", "source": "s", "meta": ""},
        {"ctx_id": "c3", "text": "x3", "source": "s", "meta": ""},
    ]
    index_contexts(e, store, rows)
    # Re-index same rows should be a no-op once dedup is added
    index_contexts(e, store, rows)
    assert len(store.ids) == len({r["ctx_id"] for r in rows})
