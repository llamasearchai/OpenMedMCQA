from __future__ import annotations

from .config import settings
from .embeddings import MedEmbedder
from .retrieval import FAISSStore
from .agent import OpenAIFunctionAgent
from .assistants_agent import OpenAIAssistantsAgent


def get_agent(embedder: MedEmbedder, store: FAISSStore):
    if settings.use_assistants:
        return OpenAIAssistantsAgent(embedder, store)
    return OpenAIFunctionAgent(embedder, store)


