from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, ValidationError

from openai import OpenAI

from .config import settings
from .prompts import SYSTEM_MEDMCQA, USER_QA_TEMPLATE
from .retrieval import FAISSStore
from .embeddings import MedEmbedder
from .db import connect


class OpenAIAssistantsAgent:
    """Optional Assistants API variant. Requires org/feature access.
    Matches signature of OpenAIFunctionAgent.answer_medmcqa.
    """

    def __init__(self, embedder: MedEmbedder, store: FAISSStore):
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=(settings.openai_api_base or None))
        self.embedder = embedder
        self.store = store
        self.db = connect()

    def retrieve_context(self, question: str, choices: Dict[str, str], k: int | None = None) -> List[Dict]:
        k = k or settings.max_ctx
        texts = [question] + [f"{question} {v}" for v in choices.values()]
        embs = self.embedder.embed_texts(texts)
        scores, idxs, ids = self.store.search(embs, top_k=settings.top_k)
        ctx_votes: Dict[str, float] = {}
        for i in range(idxs.shape[0]):
            for j in range(idxs.shape[1]):
                idx = idxs[i, j]
                if idx < 0 or idx >= len(ids):
                    continue
                ctx_id = ids[idx]
                score = float(scores[i, j])
                if ctx_id not in ctx_votes or score > ctx_votes[ctx_id]:
                    ctx_votes[ctx_id] = score
        top_ctx_ids = [cid for cid, _ in sorted(ctx_votes.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        return self._fetch_contexts(top_ctx_ids)

    def _fetch_contexts(self, ctx_ids: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for cid in ctx_ids:
            try:
                row = self.db["contexts"].get(cid)
                out.append({"ctx_id": cid, "text": row["text"], "source": row["source"], "meta": row["meta"]})
            except Exception:
                continue
        return out

    def answer_medmcqa(self, q: Dict[str, Any]) -> Tuple[str, float, str, List[str], Dict[str, Any], float]:
        # Placeholder implementation using Chat Completions parity path for now.
        # Switch to true Assistants threading/runs if available in your org.
        start = time.perf_counter()
        question = q["question"]
        choices = {k: q[k] for k in ["A", "B", "C", "D"]}
        contexts = self.retrieve_context(question, choices, k=settings.max_ctx)
        ctx_text = "\n\n".join([f"[{c['ctx_id']}] {c['text']}" for c in contexts])
        content = self.client.chat.completions.create(
            model=settings.openai_model,
            temperature=settings.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_MEDMCQA},
                {"role": "user", "content": USER_QA_TEMPLATE.format(context=ctx_text, **q)},
            ],
        ).choices[0].message.content or "{}"
        def parse(content: str) -> Dict[str, Any]:
            try:
                data = json.loads(content)
            except Exception:
                s = content[content.find("{") : content.rfind("}") + 1]
                data = json.loads(s)
            class QAResponse(BaseModel):
                answer: str = "A"
                confidence: float = 0.5
                explanation: str = ""
            try:
                v = QAResponse.model_validate(data)
                data = v.model_dump()
            except ValidationError:
                pass
            ans = str(data.get("answer", "A")).strip().upper()[:1]
            if ans not in {"A", "B", "C", "D"}:
                ans = "A"
            try:
                conf = float(data.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            expl = str(data.get("explanation", ""))
            return {"answer": ans, "confidence": conf, "explanation": expl}
        data = parse(content)
        ans = data["answer"]
        conf = data["confidence"]
        expl = data["explanation"]
        latency_ms = (time.perf_counter() - start) * 1000.0
        return ans, conf, expl, [c["ctx_id"] for c in contexts], data, latency_ms
