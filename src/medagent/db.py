from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, Iterable, List

from sqlite_utils import Database

from .config import settings


def connect() -> Database:
    db = Database(settings.db_path)
    ensure_schema(db)
    return db


def ensure_schema(db: Database) -> None:
    # Questions (source of truth), Contexts (retrieval corpus), Predictions, Runs
    db["questions"].create(
        {
            "id": str,
            "question": str,
            "A": str,
            "B": str,
            "C": str,
            "D": str,
            "correct": str,
            "subject": str,
            "topic": str,
            "difficulty": str,
            "source": str,
            "explanation": str,
        },
        pk="id",
        if_not_exists=True,
    )
    db["contexts"].create(
        {
            "ctx_id": str,
            "text": str,
            "source": str,
            "meta": str,
        },
        pk="ctx_id",
        if_not_exists=True,
    )
    db["runs"].create(
        {
            "run_id": str,
            "created_at": str,
            "model": str,
            "embedding_model": str,
            "notes": str,
        },
        pk="run_id",
        if_not_exists=True,
    )
    db["predictions"].create(
        {
            "run_id": str,
            "q_id": str,
            "predicted": str,
            "confidence": float,
            "explanation": str,
            "chosen_ctx_ids": str,
            "raw": str,
            "is_correct": int,
            "latency_ms": float,
        },
        pk=("run_id", "q_id"),
        if_not_exists=True,
    )
    # Configuration overrides (persist runtime selections)
    db["config"].create(
        {
            "key": str,
            "value": str,
        },
        pk="key",
        if_not_exists=True,
    )
    # API usage tracking (rate limits/quotas)
    db["api_usage"].create(
        {
            "date": str,
            "api_key": str,
            "count": int,
        },
        pk=("date", "api_key"),
        if_not_exists=True,
    )
    # Config key-value store
    db["config"].create(
        {
            "key": str,
            "value": str,
        },
        pk="key",
        if_not_exists=True,
    )
    # Quota usage per key/day
    db["quota_usage"].create(
        {
            "api_key": str,
            "day": int,
            "count": int,
        },
        pk=("api_key", "day"),
        if_not_exists=True,
    )


def insert_questions(db: Database, rows: Iterable[Dict[str, Any]]) -> None:
    db["questions"].upsert_all(rows, pk="id")


def insert_contexts(db: Database, rows: Iterable[Dict[str, Any]]) -> None:
    db["contexts"].upsert_all(rows, pk="ctx_id")


def new_run(db: Database, run_id: str, notes: str = "") -> None:
    db["runs"].insert(
        {
            "run_id": run_id,
            "created_at": dt.datetime.utcnow().isoformat(),
            "model": settings.openai_model,
            "embedding_model": settings.embedding_model,
            "notes": notes,
        },
        pk="run_id",
        replace=True,
    )


def set_config(db: Database, key: str, value: str) -> None:
    db["config"].upsert({"key": key, "value": value}, pk="key")


def get_config(db: Database) -> Dict[str, str]:
    return {r["key"]: r["value"] for r in db["config"].rows}


def bump_api_usage(db: Database, api_key: str) -> int:
    import datetime as dt
    today = dt.date.today().isoformat()
    row = db["api_usage"].get((today, api_key), default=None)
    count = 1 if row is None else int(row["count"]) + 1
    db["api_usage"].upsert({"date": today, "api_key": api_key, "count": count}, pk=("date", "api_key"))
    return count


def record_prediction(
    db: Database,
    run_id: str,
    q_id: str,
    predicted: str,
    confidence: float,
    explanation: str,
    chosen_ctx_ids: List[str],
    raw: Dict[str, Any],
    is_correct: bool,
    latency_ms: float,
    ) -> None:
    db["predictions"].insert(
        {
            "run_id": run_id,
            "q_id": q_id,
            "predicted": predicted,
            "confidence": confidence,
            "explanation": explanation,
            "chosen_ctx_ids": json.dumps(chosen_ctx_ids),
            "raw": json.dumps(raw),
            "is_correct": 1 if is_correct else 0,
            "latency_ms": latency_ms,
        },
        pk=("run_id", "q_id"),
        replace=True,
    )


def config_set(db: Database, key: str, value: str) -> None:
    db["config"].upsert({"key": key, "value": value}, pk="key")


def config_get(db: Database, key: str, default: str = "") -> str:
    row = db["config"].get(key, default=None)
    if row is None:
        return default
    return str(row["value"]) if isinstance(row, dict) else str(row[1])


def quota_consume(db: Database, api_key: str, day: int, daily_limit: int) -> tuple[bool, int]:
    """Consume 1 unit for api_key/day if below limit. Returns (allowed, remaining)."""
    cur = db.conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        row = cur.execute(
            "SELECT count FROM quota_usage WHERE api_key=? AND day=?", (api_key, day)
        ).fetchone()
        count = row[0] if row else 0
        if daily_limit > 0 and count >= daily_limit:
            remaining = max(0, daily_limit - count)
            cur.execute("COMMIT")
            return False, remaining
        new_count = count + 1
        cur.execute(
            "INSERT INTO quota_usage(api_key, day, count) VALUES(?,?,?) ON CONFLICT(api_key,day) DO UPDATE SET count=excluded.count",
            (api_key, day, new_count),
        )
        cur.execute("COMMIT")
        remaining = max(0, daily_limit - new_count) if daily_limit > 0 else 2**31 - 1
        return True, remaining
    except Exception:
        cur.execute("ROLLBACK")
        raise

