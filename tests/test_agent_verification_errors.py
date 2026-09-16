import pytest

from app.agent.agent import Agent
from app.agent.verification import Verifier
from app.llm.client import LLMClient
from app.llm.models import LLMResponse


class MockLLM(LLMClient):
    """Mock LLM for verification error tests."""

    def generate(self, prompt: str) -> str:
        return "Mock response."

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        return LLMResponse(
            content="Mock response."
        )


class MockVerifier(Verifier):
    """Mock verifier used to ensure verifier configuration."""

    def verify(
        self,
        task: str,
        result: str,
    ):
        raise AssertionError(
            "Verifier should not be called."
        )


def test_agent_rejects_verification_without_verifier():
    agent = Agent(
        llm_client=MockLLM(),
    )

    with pytest.raises(
        RuntimeError,
        match="Verifier is not configured",
    ):
        agent.run_with_verification(
            "Calculate 25 * 4"
        )


def test_agent_does_not_call_verifier_when_not_configured():
    agent = Agent(
        llm_client=MockLLM(),
        verifier=None,
    )

    with pytest.raises(
        RuntimeError,
        match="Verifier is not configured",
    ):
        agent.run_with_verification(
            "Calculate 25 * 4"
        )


def test_agent_rejects_empty_task_before_verification():
    agent = Agent(
        llm_client=MockLLM(),
        verifier=MockVerifier(),
    )

    with pytest.raises(
        ValueError,
        match="Task cannot be empty",
    ):
        agent.run_with_verification("")