from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
from pathlib import Path
import hashlib
import os
from openai import OpenAI
from .config import settings
from .metrics import EMB_CACHE_HITS, EMB_CACHE_MISSES


class MedEmbedder:
    def __init__(self, model_name: str, device: str | None = None, light_mode: bool = False):
        self.model_name = model_name
        self.light_mode = light_mode
        self.backend = settings.embedding_backend.lower()
        self._client: OpenAI | None = None
        # Optional on-disk cache for embeddings (disabled in light mode)
        self._cache_dir = Path(os.environ.get("MEDAGENT_EMB_CACHE_DIR", ".emb_cache"))
        if not self.light_mode:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        if self.light_mode:
            rng = np.random.default_rng(42)
            self.dim = 128
            self._rng = rng
            self.device = "cpu"
            self.tokenizer = None
            self.model = None
            return

        if self.backend == "openai":
            # OpenAI-compatible embeddings endpoint (OpenAI, Ollama/OpenAI, LM Studio)
            self._client = OpenAI(api_key=settings.openai_api_key, base_url=(settings.embedding_api_base or None))
            # Dimension is backend-dependent; default to 768 if unknown
            self.dim = 768
            self.tokenizer = None
            self.model = None
        else:
            # Hugging Face models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.dim = self.model.config.hidden_size
            if device:
                self.device = device
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.light_mode:
            return self._rng.normal(0, 1, size=(len(texts), self.dim)).astype("float32")

        # Resolve cache for inputs; compute only misses, preserve order
        cached: Dict[int, np.ndarray] = {}
        misses: List[tuple[int, str]] = []
        for i, t in enumerate(texts):
            arr = self._cache_get(t)
            if arr is not None:
                cached[i] = arr
                try:
                    EMB_CACHE_HITS.labels(backend=self.backend).inc()
                except Exception:
                    pass
            else:
                misses.append((i, t))
                try:
                    EMB_CACHE_MISSES.labels(backend=self.backend).inc()
                except Exception:
                    pass

        miss_vecs: Dict[int, np.ndarray] = {}
        if misses:
            miss_texts = [t for _, t in misses]
            if self.backend == "openai":
                assert self._client is not None
                resp = self._client.embeddings.create(model=self.model_name, input=miss_texts)
                embs = np.array([d.embedding for d in resp.data], dtype="float32")
                if not hasattr(self, "dim") or self.dim != embs.shape[1]:
                    self.dim = int(embs.shape[1])
            else:
                tokens = self.tokenizer(
                    miss_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**tokens)
                    last_hidden = outputs.last_hidden_state
                    input_mask = tokens["attention_mask"].unsqueeze(-1)
                    masked = last_hidden * input_mask
                    summed = masked.sum(dim=1)
                    counts = input_mask.sum(dim=1).clamp(min=1)
                    embs = (summed / counts).cpu().numpy().astype("float32")

            # Persist misses to cache and collect
            for (idx, _), vec in zip(misses, embs, strict=False):
                self._cache_put(texts[idx], vec)
                miss_vecs[idx] = vec

        # Assemble output in original order
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i in range(len(texts)):
            if i in cached:
                out[i] = cached[i]
            else:
                out[i] = miss_vecs[i]
        return out

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokens)
            last_hidden = outputs.last_hidden_state
            input_mask = tokens["attention_mask"].unsqueeze(-1)
            masked = last_hidden * input_mask
            summed = masked.sum(dim=1)
            counts = input_mask.sum(dim=1).clamp(min=1)
            emb = (summed / counts).cpu().numpy().astype("float32")
            return emb

    def embed_pairwise_question_choices(self, question: str, choices: Dict[str, str]) -> Dict[str, np.ndarray]:
        texts = [f"Question: {question} Choice {k}: {v}" for k, v in choices.items()]
        embs = self.embed_texts(texts)
        return {k: embs[i] for i, k in enumerate(choices.keys())}

    def _cache_key(self, text: str) -> Path:
        h = hashlib.sha256((self.model_name + "\0" + text).encode("utf-8")).hexdigest()
        return self._cache_dir / f"{h}.npy"

    def _cache_get(self, text: str) -> np.ndarray | None:
        if self.light_mode:
            return None
        path = self._cache_key(text)
        if path.exists():
            try:
                arr = np.load(path, allow_pickle=False)
                # Dim might be set lazily for OpenAI
                if not hasattr(self, "dim") or self.dim != arr.shape[1]:
                    self.dim = int(arr.shape[1])
                return arr
            except Exception:
                return None
        return None

    def _cache_put(self, text: str, vec: np.ndarray) -> None:
        if self.light_mode:
            return
        try:
            np.save(self._cache_key(text), vec.astype("float32"))
        except Exception:
            pass
