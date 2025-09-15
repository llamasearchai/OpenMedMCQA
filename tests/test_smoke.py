import os
from typer.testing import CliRunner

from medagent.cli import app
from medagent.config import settings
from medagent.embeddings import MedEmbedder
from medagent.retrieval import FAISSStore


runner = CliRunner()


def test_embedder_light():
    os.environ["LIGHT_TESTS"] = "1"
    e = MedEmbedder(settings.embedding_model, light_mode=True)
    vecs = e.embed_texts(["hello", "world"])
    assert vecs.shape == (2, e.dim)


def test_faiss_roundtrip(tmp_path):
    os.environ["LIGHT_TESTS"] = "1"
    e = MedEmbedder(settings.embedding_model, light_mode=True)
    store = FAISSStore(dim=e.dim, index_dir=str(tmp_path))
    embs = e.embed_texts(["a", "b", "c"])
    store.build(embs, ["x1", "x2", "x3"])
    q = e.embed_texts(["a"])[0].reshape(1, -1)
    s, idx, ids = store.search(q, 2)
    assert idx.shape == (1, 2)
    assert ids.shape[0] == 3


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0


