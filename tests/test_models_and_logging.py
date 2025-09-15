from fastapi.testclient import TestClient
from medagent.api import app
from medagent.cli import app as cli
from typer.testing import CliRunner


def test_models_endpoint_and_request_id(monkeypatch):
    monkeypatch.setenv("MEDAGENT_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    client = TestClient(app)
    r = client.get("/models", headers={"X-API-Key": "secret"})
    # /models is not protected, but header shouldn't harm; ensure 200
    assert r.status_code == 200
    data = r.json()
    assert "embedding_presets" in data and isinstance(data["embedding_presets"], list)
    assert "chat_presets" in data and isinstance(data["chat_presets"], list)
    # Request ID header should be present
    assert r.headers.get("X-Request-ID")

    # select models
    s = client.post(
        "/models/select",
        headers={"X-API-Key": "secret"},
        json={
            "embedding_backend": "openai",
            "embedding_model": "nomic-embed-text",
            "openai_model": "gpt-4o-mini",
            "openai_api_base": "http://localhost:11434/v1",
            "embedding_api_base": "http://localhost:11434/v1",
        },
    )
    assert s.status_code == 200
    body = s.json()
    assert body["ok"] is True


def test_cli_models_command():
    runner = CliRunner()
    res = runner.invoke(cli, ["models"])  # noqa: S603,S607
    assert res.exit_code == 0
    assert "Embedding Presets" in res.stdout
    assert "Chat Presets" in res.stdout
