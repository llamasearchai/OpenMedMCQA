from __future__ import annotations

from fastapi.testclient import TestClient
from medagent.api import app
from medagent.embeddings import MedEmbedder
from medagent.retrieval import FAISSStore


def test_info_includes_stats(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDAGENT_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("MEDAGENT_INDEX_DIR", str(tmp_path / ".index"))
    # Build a small index to populate stats
    monkeypatch.setenv("LIGHT_TESTS", "1")
    e = MedEmbedder("dummy", light_mode=True)
    store = FAISSStore(dim=e.dim, index_dir=str(tmp_path / ".index"))
    embs = e.embed_texts(["a", "b", "c"])  # noqa: F841
    store.build(e.embed_texts(["a", "b", "c"]), ["i1", "i2", "i3"])

    client = TestClient(app)
    r = client.get("/info", headers={"X-API-Key": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert "index_stats" in body and "vectors" in body["index_stats"]
    assert "embedding_cache_stats" in body

    # Quota/persistence behavior covered in separate tests; here we only validate stats presence
