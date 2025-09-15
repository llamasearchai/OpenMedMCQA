from __future__ import annotations

from typing import Dict, List, Any, Iterable

import numpy as np

from .embeddings import MedEmbedder
from .retrieval import FAISSStore


def index_contexts(embedder: MedEmbedder, store: FAISSStore, rows: Iterable[Dict[str, Any]]) -> None:
    texts, ids = [], []
    for r in rows:
        texts.append(r["text"])
        ids.append(r["ctx_id"])
    embs = embedder.embed_texts(texts)
    # Fix: safe first-run check per runbook
    if getattr(store, "_ids", None) is None or len(store._ids) == 0:
        store.build(embs, ids)
    else:
        store.add(embs, ids)


def vectorize_questions_and_choices(embedder: MedEmbedder, questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for q in questions:
        out[q["id"]] = embedder.embed_pairwise_question_choices(q["question"], {k: q[k] for k in ["A", "B", "C", "D"]})
    return out


def simple_similarity_vote(q_embeds: Dict[str, np.ndarray], ctx_embeds: np.ndarray) -> str:
    ctx_mean = ctx_embeds.mean(axis=0)

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    scores = {opt: cos(vec, ctx_mean) for opt, vec in q_embeds.items()}
    return max(scores.items(), key=lambda kv: kv[1])[0]


