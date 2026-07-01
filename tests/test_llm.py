from spebt_agent.brain.llm import NullLLMClient, build_llm_client


def test_llm_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = build_llm_client({})
    assert isinstance(client, NullLLMClient)
