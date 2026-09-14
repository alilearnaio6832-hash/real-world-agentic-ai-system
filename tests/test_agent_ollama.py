from app.agent.agent import Agent
from app.llm.ollama import OllamaLLMClient


def test_agent_with_ollama():
    llm_client = OllamaLLMClient()
    agent = Agent(llm_client)

    response = agent.run(
        "Reply with exactly: AGENT_OK"
    )

    assert isinstance(response, str)
    assert response.strip()