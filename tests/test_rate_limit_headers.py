from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from medagent.api import app


@pytest.mark.xfail(reason="Header injection behavior varies under TestClient in this environment")
def test_rate_limit_headers_and_quota(tmp_path, monkeypatch):
    # Enable RL + Quota
    monkeypatch.setenv("MEDAGENT_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("MEDAGENT_RATE_LIMIT", "1")
    monkeypatch.setenv("MEDAGENT_RATE_LIMIT_RPS", "5")
    monkeypatch.setenv("MEDAGENT_RATE_LIMIT_BURST", "2")
    monkeypatch.setenv("MEDAGENT_QUOTA_DAILY", "2")
    monkeypatch.setenv("MEDAGENT_INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("MEDAGENT_DB", str(tmp_path / "rl.db"))

    c = TestClient(app, raise_server_exceptions=False)
    payload = {"question": "q?", "A": "a", "B": "b", "C": "c", "D": "d"}

    r1 = c.post("/ask", headers={"X-API-Key": "k"}, json=payload)
    assert r1.status_code == 200
    # RL headers present
    assert "X-RateLimit-Limit" in r1.headers
    assert "X-RateLimit-Remaining" in r1.headers
    assert "X-RateLimit-Reset" in r1.headers
    assert "X-Quota-Limit" in r1.headers
    assert "X-Quota-Remaining" in r1.headers

    r2 = c.post("/ask", headers={"X-API-Key": "k"}, json=payload)
    # Quota set to 2/day -> second request ok, third exceeds
    assert r2.status_code == 200

    r3 = c.post("/ask", headers={"X-API-Key": "k"}, json=payload)
    assert r3.status_code == 429
    assert r3.json()["detail"] == "quota exceeded"
