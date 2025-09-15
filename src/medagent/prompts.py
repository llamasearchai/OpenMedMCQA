SYSTEM_MEDMCQA = """You are a careful medical multiple-choice assistant.
-  Use only the provided CONTEXT to reason.
-  Always return a JSON object: {"answer": "A|B|C|D", "confidence": 0-1, "explanation": "..."}.
-  If context is insufficient, pick the best answer with uncertainty and explain why.
-  Be concise and safe; do not give medical advice beyond academic exam scope."""

USER_QA_TEMPLATE = """QUESTION:
{question}

CHOICES:
A) {A}
B) {B}
C) {C}
D) {D}

CONTEXT:
{context}

Return strictly valid JSON only.
"""


