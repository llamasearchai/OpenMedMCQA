import os
from pathlib import Path

from fastapi.testclient import TestClient

from medagent.api import app
import medagent.config as config_mod


def test_health_open():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_info_requires_api_key(monkeypatch):
    monkeypatch.setenv("MEDAGENT_API_KEY", "secret")
    client = TestClient(app)

    r = client.get("/info")
    assert r.status_code == 401

    r = client.get("/info", headers={"X-API-Key": "secret"})
    assert r.status_code == 200
    data = r.json()
    # Secret should be redacted (not leaked)
    assert data.get("openai_api_key", "") in ("", "***redacted***")


def test_ingest_safe_path_and_limits(tmp_path, monkeypatch):
    # Route DB to a temp file and restrict data dir
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(config_mod.settings, "db_path", str(db_path))
    monkeypatch.setenv("MEDAGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MEDAGENT_API_KEY", "secret")

    # Create minimal valid JSONL
    p = tmp_path / "valid.jsonl"
    p.write_text(
        "\n".join([
            '{"id":"q1","question":"q?","options":{"A":"a","B":"b","C":"c","D":"d"},"correct_answer":"A"}'
        ]),
        encoding="utf-8",
    )

    client = TestClient(app)
    r = client.post(f"/ingest?jsonl_path={p.name}", headers={"X-API-Key": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("questions") == 1
    assert body.get("contexts") >= 4

    # Absolute path should be rejected
    r2 = client.post(f"/ingest?jsonl_path={p.resolve()}", headers={"X-API-Key": "secret"})
    assert r2.status_code == 400

