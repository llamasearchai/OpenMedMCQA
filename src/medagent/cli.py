from __future__ import annotations

import json
import random
import string
from typing import Optional

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import track

from .config import settings
from .db import connect, insert_questions, insert_contexts
from .embeddings import MedEmbedder
from .retrieval import FAISSStore
from .rag import index_contexts
from .datasets import load_medmcqa_jsonl, seed_contexts_from_questions
from .agents import get_agent
from .evaluation import evaluate_run
from .datasette_integration import serve_datasette
from .llm_integration import llm_available, run_llm_cmd
from .models import EMBEDDING_PRESETS, CHAT_PRESETS
from .db import connect
import pandas as pd
from collections import Counter
from pathlib import Path


app = typer.Typer(add_completion=False, help="MedMCQA Agentic RAG CLI")


def _rand_id(prefix: str) -> str:
    return prefix + "-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


@app.command()
def info():
    """Show current configuration."""
    settings.ensure_dirs()
    t = Table(title="MedAgent Configuration")
    for k, v in settings.model_dump().items():
        t.add_row(k, str(v))
    rprint(t)


@app.command()
def ingest(jsonl_path: str):
    """Ingest MedMCQA JSONL into SQLite and seed contexts."""
    db = connect()
    rows = load_medmcqa_jsonl(jsonl_path)
    insert_questions(db, rows)
    ctx = seed_contexts_from_questions(rows)
    insert_contexts(db, ctx)
    rprint(Panel.fit(f"Ingested {len(rows)} questions and {len(ctx)} seed contexts", title="Ingest"))


@app.command()
def build_index(rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild index from scratch")):
    """Build or update FAISS index from contexts."""
    settings.ensure_dirs()
    db = connect()
    rows = list(db.query("select ctx_id, text, source, meta from contexts"))
    embedder = MedEmbedder(settings.embedding_model, light_mode=settings.light_tests)
    store = FAISSStore(dim=embedder.dim)
    if rebuild:
        import os
        for p in [store.index_path, store.ids_path]:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        # Reset any in-memory state
        store._index = None  # type: ignore[attr-defined]
        store._ids = None  # type: ignore[attr-defined]

    batch = 512
    buffer = []
    for r in track(rows, description="Embedding contexts"):
        buffer.append(r)
        if len(buffer) >= batch:
            index_contexts(embedder, store, buffer)
            buffer.clear()
    if buffer:
        index_contexts(embedder, store, buffer)

    rprint(Panel.fit("FAISS index built/updated", title="Index"))


@app.command("embeddings-info")
def embeddings_info():
    """Show embedding backend configuration and a sample vector size."""
    e = MedEmbedder(settings.embedding_model, light_mode=settings.light_tests)
    vec = e.embed_texts(["probe"])[0]
    t = Table(title="Embedding Backend")
    t.add_row("backend", settings.embedding_backend)
    t.add_row("model", settings.embedding_model)
    t.add_row("dim", str(e.dim))
    t.add_row("api_base", settings.embedding_api_base or "")
    rprint(t)


@app.command("pull-ollama-model")
def pull_ollama_model(model: str = typer.Argument(...)):
    """Pull an Ollama embedding model (requires `ollama` on PATH)."""
    import shutil, subprocess
    if shutil.which("ollama") is None:
        rprint("[red]`ollama` not found. Install from https://ollama.com/ and try again.")
        raise typer.Exit(1)
    subprocess.run(["ollama", "pull", model], check=True)
    rprint(Panel.fit(f"Pulled Ollama model: {model}", title="Ollama"))


@app.command()
def ask(
    question: str = typer.Option(..., "--question", "-q"),
    A: str = typer.Option(..., "--A"),
    B: str = typer.Option(..., "--B"),
    C: str = typer.Option(..., "--C"),
    D: str = typer.Option(..., "--D"),
):
    """Ask a single MedMCQA-style question."""
    embedder = MedEmbedder(settings.embedding_model, light_mode=settings.light_tests)
    store = FAISSStore(dim=embedder.dim)
    agent = get_agent(embedder, store)
    agent = get_agent(embedder, store)
    payload = {"id": _rand_id("tmp"), "question": question, "A": A, "B": B, "C": C, "D": D, "correct": "?", "explanation": "", "subject": "", "topic": "", "difficulty": "", "source": ""}
    ans, conf, expl, ctx_ids, raw, latency_ms = agent.answer_medmcqa(payload)

    t = Table(title="Answer")
    t.add_row("Predicted", ans)
    t.add_row("Confidence", f"{conf:.2f}")
    t.add_row("Latency (ms)", f"{latency_ms:.1f}")
    rprint(t)
    rprint(Panel(expl, title="Explanation"))
    rprint(Panel("\n".join(ctx_ids), title="Contexts"))


@app.command()
def evaluate(
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    limit: int = typer.Option(100, "--limit", help="Number of questions to evaluate (use 0 for all)"),
    subject: Optional[str] = typer.Option(None, "--subject"),
):
    """Evaluate on a subset of questions and store predictions."""
    db = connect()
    where = []
    params = []
    if subject:
        where.append("subject = ?")
        params.append(subject)
    if limit > 0:
        where.append("id in (select id from questions limit ?)")
        params.append(limit)
    sql = "select id from questions"
    if where:
        sql += " where " + " and ".join(where)
    ids = [r["id"] for r in db.query(sql, params)]
    rid = run_id or _rand_id("run")
    res = evaluate_run(rid, ids, notes=f"limit={limit}, subject={subject or 'ALL'}")
    rprint(Panel.fit(json.dumps(res, indent=2), title="Evaluation"))


@app.command()
def serve_datasette_cmd(port: int = 8080):
    """Serve the SQLite DB with Datasette."""
    serve_datasette(open_browser=True, port=port)


@app.command("serve-datasette")
def serve_datasette_alias(port: int = 8080):
    """Alias for 'serve-datasette-cmd' to match docs/Makefile."""
    serve_datasette_cmd(port)


@app.command()
def llm_cmd(prompt: str, model: str = "openai:gpt-4o-mini"):
    """Run a quick LLM prompt through the llm CLI."""
    if not llm_available():
        rprint("[red]`llm` CLI not found. Install with: pipx install llm && llm plugins install llm-cmd[/]")
        raise typer.Exit(1)
    out = run_llm_cmd(prompt, model=model, system="You are a helpful MedMCQA coding assistant.")
    rprint(Panel(out, title="llm output"))


@app.command()
def menu():
    """Interactive menu with progress bars."""
    rprint(Panel.fit("MedAgent CLI", subtitle="Agentic RAG for MedMCQA"))
    while True:
        choice = Prompt.ask(
            "Choose",
            choices=["info", "ingest", "index", "ask", "evaluate", "datasette", "llm", "quit"],
            default="info",
        )
        if choice == "info":
            info()
        elif choice == "ingest":
            path = Prompt.ask("Path to MedMCQA JSONL")
            ingest(path)
        elif choice == "index":
            build_index()
        elif choice == "ask":
            q = Prompt.ask("Question")
            A = Prompt.ask("Choice A")
            B = Prompt.ask("Choice B")
            C = Prompt.ask("Choice C")
            D = Prompt.ask("Choice D")
            ask(q, A, B, C, D)
        elif choice == "evaluate":
            limit = int(Prompt.ask("Limit (0 for all)", default="100"))
            evaluate(limit=limit)
        elif choice == "datasette":
            port = int(Prompt.ask("Port", default="8080"))
            serve_datasette_cmd(port)
        elif choice == "llm":
            prompt = Prompt.ask("Prompt to llm CLI")
            llm_cmd(prompt)
        elif choice == "quit":
            break


@app.command()
def models():
    """List model and embedding presets."""
    t1 = Table(title="Embedding Presets")
    t1.add_row("name", "backend", "model")
    for p in EMBEDDING_PRESETS:
        t1.add_row(p["name"], p["backend"], p["model"])
    t2 = Table(title="Chat Presets")
    t2.add_row("name", "provider", "model")
    for p in CHAT_PRESETS:
        t2.add_row(p["name"], p["provider"], p["model"])
    rprint(t1)
    rprint(t2)


@app.command()
def export(
    table: str = typer.Option("predictions", "--table", help="Table to export: predictions|runs|questions|contexts"),
    fmt: str = typer.Option("csv", "--format", help="Format: csv|json|parquet"),
    out: str = typer.Option("export.out", "--out", help="Output file path"),
):
    """Export a table to CSV/JSON/Parquet for analysis."""
    db = connect()
    if table not in {"predictions", "runs", "questions", "contexts"}:
        rprint(f"[red]Unsupported table: {table}[/]")
        raise typer.Exit(1)
    rows = list(db.query(f"select * from {table}"))
    df = pd.DataFrame(rows)
    if fmt == "csv":
        df.to_csv(out, index=False)
    elif fmt == "json":
        df.to_json(out, orient="records", lines=True)
    elif fmt == "parquet":
        df.to_parquet(out, index=False)
    else:
        rprint(f"[red]Unsupported format: {fmt}[/]")
        raise typer.Exit(1)
    rprint(Panel.fit(f"Exported {len(df)} rows from {table} to {out}", title="Export"))


@app.command()
def report(run_id: str | None = typer.Option(None, "--run-id"), out: str | None = typer.Option(None, "--out")):
    """Generate a summary report for predictions.

    - If --run-id is given: summarize that run
    - Else: summarize last run
    - If --out is given: write Markdown report to file
    """
    db = connect()
    if run_id is None:
        rows = list(db.query("select run_id, created_at from runs order by created_at desc limit 1"))
        if not rows:
            rprint("[red]No runs found[/]")
            raise typer.Exit(1)
        run_id = rows[0]["run_id"]

    preds = list(db.query("select * from predictions where run_id = ?", [run_id]))
    if not preds:
        rprint(f"[red]No predictions for run {run_id}[/]")
        raise typer.Exit(1)
    acc = sum(p["is_correct"] for p in preds) / len(preds)
    avg_latency = sum(p["latency_ms"] for p in preds) / len(preds)

    # Subject breakdown (if available)
    subs: Counter[str] = Counter()
    correct_subs: Counter[str] = Counter()
    for p in preds:
        q = list(db.query("select subject from questions where id = ?", [p["q_id"]]))
        subj = (q[0]["subject"] if q else "") or "(unknown)"
        subs[subj] += 1
        if p["is_correct"]:
            correct_subs[subj] += 1

    # Render tables
    t = Table(title=f"Report for run {run_id}")
    t.add_row("accuracy", f"{acc:.4f}")
    t.add_row("count", str(len(preds)))
    t.add_row("avg_latency_ms", f"{avg_latency:.2f}")
    rprint(t)

    tb = Table(title="Accuracy by subject")
    tb.add_row("subject", "n", "accuracy")
    for s, n in subs.most_common():
        a = (correct_subs[s] / n) if n else 0.0
        tb.add_row(s, str(n), f"{a:.4f}")
    rprint(tb)

    if out:
        md = [
            f"# Report for run {run_id}",
            f"- accuracy: {acc:.4f}",
            f"- count: {len(preds)}",
            f"- avg_latency_ms: {avg_latency:.2f}",
            "\n## Accuracy by subject",
        ]
        for s, n in subs.most_common():
            a = (correct_subs[s] / n) if n else 0.0
            md.append(f"- {s}: n={n}, accuracy={a:.4f}")
        Path(out).write_text("\n".join(md), encoding="utf-8")
        rprint(Panel.fit(f"Report written to {out}", title="Report"))


@app.command("report-compare")
def report_compare(run1: str = typer.Argument(...), run2: str = typer.Argument(...), out: str | None = typer.Option(None, "--out")):
    """Compare two runs and summarize differences."""
    db = connect()
    p1 = list(db.query("select * from predictions where run_id = ?", [run1]))
    p2 = list(db.query("select * from predictions where run_id = ?", [run2]))
    if not p1 or not p2:
        rprint("[red]Missing predictions for one of the runs[/]")
        raise typer.Exit(1)
    def acc(preds):
        return sum(p["is_correct"] for p in preds) / len(preds)
    a1, a2 = acc(p1), acc(p2)
    l1 = sum(p["latency_ms"] for p in p1) / len(p1)
    l2 = sum(p["latency_ms"] for p in p2) / len(p2)
    t = Table(title=f"Compare {run1} vs {run2}")
    t.add_row("accuracy", f"{a1:.4f}", f"{a2:.4f}", f"Δ={(a2-a1):+.4f}")
    t.add_row("avg_latency_ms", f"{l1:.2f}", f"{l2:.2f}", f"Δ={(l2-l1):+.2f}")
    rprint(t)
    if out:
        Path(out).write_text("\n".join([
            f"# Compare {run1} vs {run2}",
            f"- accuracy: {a1:.4f} → {a2:.4f} (Δ={(a2-a1):+.4f})",
            f"- avg_latency_ms: {l1:.2f} → {l2:.2f} (Δ={(l2-l1):+.2f})",
        ]), encoding="utf-8")
        rprint(Panel.fit(f"Comparison written to {out}", title="ReportCompare"))


@app.command()
def dashboard(out: str = typer.Option("dashboard.html", "--out")):
    """Generate a minimal static HTML dashboard summarizing runs."""
    db = connect()
    runs = list(db.query("select run_id, created_at from runs order by created_at desc limit 20"))
    rows = []
    for r in runs:
        preds = list(db.query("select is_correct, latency_ms from predictions where run_id = ?", [r["run_id"]]))
        if preds:
            acc = sum(p["is_correct"] for p in preds) / len(preds)
            lat = sum(p["latency_ms"] for p in preds) / len(preds)
            rows.append((r["run_id"], acc, lat))
    labels = [rid for rid, _, _ in rows]
    accs = [acc for _, acc, _ in rows]
    lats = [lat for _, _, lat in rows]
    html = [
        "<!doctype html>",
        "<meta charset='utf-8'><title>MedAgent Dashboard</title>",
        "<h1>Recent Runs</h1>",
        "<table border='1' cellpadding='6'><tr><th>run_id</th><th>accuracy</th><th>avg_latency_ms</th></tr>",
    ]
    for rid, acc, lat in rows:
        html.append(f"<tr><td>{rid}</td><td>{acc:.4f}</td><td>{lat:.2f}</td></tr>")
    html.append("</table>")
    # Charts (Chart.js via CDN)
    html.append("<h2>Accuracy</h2><canvas id='accuracyChart' width='600' height='250'></canvas>")
    html.append("<h2>Latency (ms)</h2><canvas id='latencyChart' width='600' height='250'></canvas>")
    html.append("<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>")
    html.append("<script>")
    html.append(f"const labels = {labels!r}; const accs = {accs!r}; const lats = {lats!r};")
    html.append(
        "new Chart(document.getElementById('accuracyChart'), {type:'line', data:{labels, datasets:[{label:'accuracy', data: accs, borderColor:'green'}]}});"
    )
    html.append(
        "new Chart(document.getElementById('latencyChart'), {type:'line', data:{labels, datasets:[{label:'avg_latency_ms', data: lats, borderColor:'blue'}]}});"
    )
    html.append("</script>")
    Path(out).write_text("\n".join(html), encoding="utf-8")
    rprint(Panel.fit(f"Dashboard written to {out}", title="Dashboard"))
