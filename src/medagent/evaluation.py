from __future__ import annotations

from typing import Any, Dict, List

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

from .db import connect, new_run, record_prediction
from .agent import OpenAIFunctionAgent
from .embeddings import MedEmbedder
from .retrieval import FAISSStore
from .config import settings


def evaluate_run(run_id: str, question_ids: List[str] | None = None, notes: str = "") -> Dict[str, Any]:
    db = connect()
    new_run(db, run_id, notes=notes)

    rows = list(db.query("select * from questions" + ("" if not question_ids else " where id in ({})".format(
        ",".join(["?"] * len(question_ids))
    )), question_ids or []))
    embedder = MedEmbedder(settings.embedding_model, light_mode=settings.light_tests)
    store = FAISSStore(dim=embedder.dim)
    agent = OpenAIFunctionAgent(embedder, store)

    total = len(rows)
    correct = 0
    latencies: List[float] = []

    with Progress(
        TextColumn("[bold blue]Evaluating[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        t = progress.add_task("eval", total=total)
        for r in rows:
            ans, conf, expl, ctx_ids, raw, latency_ms = agent.answer_medmcqa(r)
            is_correct = (ans.strip().upper() == r["correct"].strip().upper())
            record_prediction(db, run_id, r["id"], ans, conf, expl, ctx_ids, raw, is_correct, latency_ms)
            if is_correct:
                correct += 1
            latencies.append(latency_ms)
            progress.advance(t)

    acc = correct / total if total else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {"run_id": run_id, "accuracy": acc, "avg_latency_ms": avg_latency, "count": total}


