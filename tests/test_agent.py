import pytest

from app.agent.agent import Agent
from app.llm.client import LLMClient


class MockLLMClient(LLMClient):
    """Mock LLM client for Agent unit tests."""

    def generate(self, prompt: str) -> str:
        return f"Mock response: {prompt}"


def test_agent_runs_task():
    agent = Agent(MockLLMClient())

    result = agent.run("Calculate 2 + 2")

    assert result == "Mock response: Calculate 2 + 2"


def test_agent_rejects_empty_task():
    agent = Agent(MockLLMClient())

    with pytest.raises(ValueError, match="Task cannot be empty."):
        agent.run("")


def test_agent_rejects_whitespace_task():
    agent = Agent(MockLLMClient())

    with pytest.raises(ValueError, match="Task cannot be empty."):
        agent.run("   ")