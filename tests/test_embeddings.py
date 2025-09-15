import os
from medagent.embeddings import MedEmbedder
from medagent.config import settings


def test_light_backend_dim_and_shape(monkeypatch):
    monkeypatch.setenv("LIGHT_TESTS", "1")
    e = MedEmbedder(settings.embedding_model, light_mode=True)
    vecs = e.embed_texts(["a", "b", "c"])
    assert vecs.shape == (3, e.dim)


