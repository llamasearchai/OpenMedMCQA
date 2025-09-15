from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


def load_medmcqa_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            rows.append({
                "id": x["id"],
                "question": x["question"],
                "A": x["options"]["A"],
                "B": x["options"]["B"],
                "C": x["options"]["C"],
                "D": x["options"]["D"],
                "correct": x["correct_answer"],
                "explanation": x.get("explanation", ""),
                "subject": x.get("subject", ""),
                "topic": x.get("topic", ""),
                "difficulty": x.get("difficulty", ""),
                "source": x.get("source", ""),
            })
    return rows


def seed_contexts_from_questions(questions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for q in questions:
        base = f"Q: {q['question']}\nA) {q['A']}\nB) {q['B']}\nC) {q['C']}\nD) {q['D']}\n"
        if q.get("explanation"):
            txt = q["explanation"]
            contexts.append({"ctx_id": f"{q['id']}:exp", "text": txt, "source": "medmcqa_explanation", "meta": ""})
        for opt in ["A", "B", "C", "D"]:
            contexts.append({"ctx_id": f"{q['id']}:{opt}", "text": f"{base}\nCHOICE {opt}: {q[opt]}", "source": "medmcqa_choice", "meta": ""})
    return contexts


