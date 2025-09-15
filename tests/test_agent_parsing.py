import types

from medagent.embeddings import MedEmbedder
from medagent.retrieval import FAISSStore
from medagent.agent import OpenAIFunctionAgent
import medagent.agent as agent_mod


def test_agent_parses_strict_or_messy_json(monkeypatch):
    # Ensure we don't hit real DB or network
    monkeypatch.setenv("LIGHT_TESTS", "1")
    # Avoid creating a real DB; not used due to retrieval patch
    monkeypatch.setattr(agent_mod, "connect", lambda: object())

    e = MedEmbedder("dummy", light_mode=True)
    store = FAISSStore(dim=e.dim)
    agent = OpenAIFunctionAgent(e, store)

    # Patch retrieval to skip FAISS usage and return no contexts
    agent.retrieve_context = types.MethodType(lambda self, q, c, k=None: [], agent)

    # Case 1: strict JSON
    agent._chat = types.MethodType(lambda self, msgs: '{"answer":"B","confidence":0.8,"explanation":"ok"}', agent)
    q = {"id": "t1", "question": "q?", "A": "a", "B": "b", "C": "c", "D": "d", "correct": "A", "explanation": "", "subject": "", "topic": "", "difficulty": "", "source": ""}
    ans, conf, expl, ctx_ids, raw, _ = agent.answer_medmcqa(q)
    assert ans == "B"
    assert conf == 0.8
    assert isinstance(expl, str)
    assert ctx_ids == []

    # Case 2: messy content requiring substring extraction
    agent._chat = types.MethodType(lambda self, msgs: 'prefix\n{"answer":"C","confidence":0.55,"explanation":"m"}\nsuffix', agent)
    ans2, conf2, _, ctx_ids2, _, _ = agent.answer_medmcqa(q)
    assert ans2 == "C"
    assert 0.0 <= conf2 <= 1.0
    assert ctx_ids2 == []

