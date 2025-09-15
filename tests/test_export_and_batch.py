import json
from pathlib import Path

from typer.testing import CliRunner
from fastapi.testclient import TestClient

from medagent.cli import app as cli
from medagent.api import app
from medagent.db import connect, new_run, record_prediction


def test_export_cli_generates_csv(tmp_path, monkeypatch):
    db_path = tmp_path / "exp.db"
    monkeypatch.setenv("MEDAGENT_DB", str(db_path))
    db = connect()
    new_run(db, "runX", notes="test")
    record_prediction(
        db,
        run_id="runX",
        q_id="q1",
        predicted="A",
        confidence=0.9,
        explanation="",
        chosen_ctx_ids=["c1"],
        raw={"answer": "A", "confidence": 0.9},
        is_correct=True,
        latency_ms=12.3,
    )
    out = tmp_path / "pred.csv"
    runner = CliRunner()
    res = runner.invoke(cli, ["export", "--table", "predictions", "--format", "csv", "--out", str(out)])
    assert res.exit_code == 0
    assert out.exists() and out.read_text().startswith("run_id,")


def test_ask_batch_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDAGENT_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("MEDAGENT_INDEX_DIR", str(tmp_path / "idx"))
    client = TestClient(app)
    payload = [{"question": "q?", "A": "a", "B": "b", "C": "c", "D": "d"}]
    r = client.post("/ask/batch", headers={"X-API-Key": "secret"}, json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert isinstance(body["items"], list) and len(body["items"]) == 1
