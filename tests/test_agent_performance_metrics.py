import time

from app.agent.agent import Agent
from app.agent.state import ExecutionState
from app.llm.client import LLMClient
from app.llm.models import LLMResponse


class MockTimedLLM(LLMClient):
    """Mock LLM that reports token usage and takes a small, real delay."""

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
        time.sleep(0.02)

        return LLMResponse(content="100", token_count=50)


def test_run_with_tools_tracks_elapsed_time():
    llm = MockTimedLLM()
    agent = Agent(llm_client=llm)

    state = ExecutionState()

    agent.run_with_tools(
        "Calculate 25 * 4",
        tools=[],
        state=state,
    )

    assert state.elapsed_seconds > 0.0
    assert state.elapsed_seconds < 5.0


def test_run_with_tools_tracks_token_usage():
    llm = MockTimedLLM()
    agent = Agent(llm_client=llm)

    state = ExecutionState()

    agent.run_with_tools(
        "Calculate 25 * 4",
        tools=[],
        state=state,
    )

    assert state.total_tokens == 50


def test_llm_response_defaults_token_count_to_zero():
    response = LLMResponse(content="100")

    assert response.token_count == 0