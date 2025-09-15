from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner

from medagent.cli import app
from medagent.db import connect, new_run, record_prediction


def _seed_run(db, run_id: str, n: int, correct_every: int = 2):
    new_run(db, run_id, notes="test")
    for i in range(n):
        is_ok = 1 if (i % correct_every == 0) else 0
        record_prediction(
            db,
            run_id=run_id,
            q_id=f"q{i}",
            predicted="A",
            confidence=0.9,
            explanation="",
            chosen_ctx_ids=["c1"],
            raw={"answer": "A", "confidence": 0.9},
            is_correct=bool(is_ok),
            latency_ms=10.0 + i,
        )


def test_report_compare_and_dashboard(tmp_path, monkeypatch):
    db_path = tmp_path / "cmp.db"
    monkeypatch.setenv("MEDAGENT_DB", str(db_path))
    db = connect()
    _seed_run(db, "runA", 10, correct_every=2)
    _seed_run(db, "runB", 10, correct_every=1)

    runner = CliRunner()
    # report-compare
    res = runner.invoke(app, ["report-compare", "runA", "runB", "--out", str(tmp_path / "cmp.md")])
    assert res.exit_code == 0
    assert (tmp_path / "cmp.md").exists()

    # dashboard
    res2 = runner.invoke(app, ["dashboard", "--out", str(tmp_path / "dash.html")])
    assert res2.exit_code == 0
    content = (tmp_path / "dash.html").read_text()
    assert "Recent Runs" in content
