import pytest

from app.agent.agent import Agent
from app.agent.state import ExecutionState
from app.llm.client import LLMClient
from app.llm.models import LLMResponse


class MockEmptyThenValidLLM(LLMClient):
    """Mock LLM that returns an empty response once, then a valid one."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        return "Mock response."

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        self.call_count += 1

        if self.call_count == 1:
            return LLMResponse(content="")

        return LLMResponse(content="The answer is 100.")


class MockAlwaysEmptyLLM(LLMClient):
    """Mock LLM that always returns an empty response."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        return "Mock response."

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content="")


def test_run_with_tools_retries_once_on_empty_response():
    llm = MockEmptyThenValidLLM()
    agent = Agent(llm_client=llm)

    state = ExecutionState()

    response = agent.run_with_tools(
        "Calculate 25 * 4",
        tools=[],
        state=state,
    )

    assert response.content == "The answer is 100."
    assert llm.call_count == 2
    assert state.empty_response_retries == 1


def test_run_with_tools_raises_after_repeated_empty_response():
    llm = MockAlwaysEmptyLLM()
    agent = Agent(llm_client=llm)

    with pytest.raises(
        RuntimeError,
        match="empty response",
    ):
        agent.run_with_tools(
            "Calculate 25 * 4",
            tools=[],
        )

    assert llm.call_count == 2


def test_run_with_tools_does_not_retry_on_valid_response():
    class ValidLLM(LLMClient):
        def __init__(self) -> None:
            self.call_count = 0

        def generate(self, prompt: str) -> str:
            return "Mock response."

        def generate_with_tools(
            self,
            prompt: str,
            tools: list[dict],
        ) -> LLMResponse:
            self.call_count += 1
            return LLMResponse(content="100")

    llm = ValidLLM()
    agent = Agent(llm_client=llm)

    state = ExecutionState()

    agent.run_with_tools(
        "Calculate 25 * 4",
        tools=[],
        state=state,
    )

    assert llm.call_count == 1
    assert state.empty_response_retries == 0