from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict

import faiss
import numpy as np
from sqlite_utils import Database

from .config import settings
from .metrics import FAISS_SEARCH_LATENCY
import time


class FAISSStore:
    def __init__(self, dim: int, index_dir: str | None = None):
        self.dim = dim
        self.index_dir = index_dir or settings.index_dir
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        self.index_path = os.path.join(self.index_dir, "contexts.faiss")
        self.ids_path = os.path.join(self.index_dir, "context_ids.npy")
        self._index = None
        self._ids: np.ndarray | None = None

    @property
    def index(self) -> faiss.IndexFlatIP:
        if self._index is None:
            if os.path.exists(self.index_path):
                self._index = faiss.read_index(self.index_path)
            else:
                self._index = faiss.IndexFlatIP(self.dim)
        return self._index

    @property
    def ids(self) -> np.ndarray:
        if self._ids is None:
            if os.path.exists(self.ids_path):
                self._ids = np.load(self.ids_path, allow_pickle=False)
            else:
                self._ids = np.array([], dtype="object")
        return self._ids

    def save(self) -> None:
        faiss.write_index(self.index, self.index_path)
        np.save(self.ids_path, self.ids)

    def build(self, embeddings: np.ndarray, ids: List[str]) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embs = embeddings / norms
        self._index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)
        self._ids = np.array(ids, dtype="object")
        self.save()

    def add(self, embeddings: np.ndarray, ids: List[str]) -> None:
        # Deduplicate by ctx_id to avoid index growth on repeated builds
        existing = set(self.ids.tolist()) if self.ids is not None and len(self.ids) > 0 else set()
        if len(existing) > 0:
            keep_mask = [i for i, cid in enumerate(ids) if cid not in existing]
        else:
            keep_mask = list(range(len(ids)))
        if not keep_mask:
            return
        ids_kept = [ids[i] for i in keep_mask]
        embs_in = embeddings[keep_mask]
        norms = np.linalg.norm(embs_in, axis=1, keepdims=True) + 1e-8
        embs = embs_in / norms
        self.index.add(embs)
        if self._ids is None or len(self._ids) == 0:
            self._ids = np.array(ids_kept, dtype="object")
        else:
            self._ids = np.concatenate([self._ids, np.array(ids_kept, dtype="object")])
        self.save()

    def search(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8
        q = query_embeddings / norms
        scores, idxs = self.index.search(q, top_k)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        try:
            FAISS_SEARCH_LATENCY.observe(dt_ms)
        except Exception:
            pass
        return scores, idxs, self.ids

    def get_texts_from_ids(self, db: Database, ids: List[str]) -> List[Dict]:
        rows = []
        for ctx_id in ids:
            row = db["contexts"].get(ctx_id)
            rows.append({"ctx_id": ctx_id, "text": row["text"], "source": row["source"], "meta": row["meta"]})
        return rows
