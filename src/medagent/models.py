from __future__ import annotations

from typing import Dict, List


EMBEDDING_PRESETS: List[Dict[str, str]] = [
    {"name": "SciBERT", "backend": "hf", "model": "allenai/scibert_scivocab_uncased"},
    {"name": "nomic-embed-text", "backend": "openai", "model": "nomic-embed-text"},
]

CHAT_PRESETS: List[Dict[str, str]] = [
    {"name": "gpt-4o-mini", "provider": "openai", "model": "gpt-4o-mini"},
    {"name": "llama3.1-8b-instruct", "provider": "ollama", "model": "llama3.1:8b-instruct"},
]

