from app.agent.agent import Agent
from app.agent.state import ExecutionState
from app.agent.verification import CalculatorVerifier
from app.llm.client import LLMClient
from app.llm.models import LLMResponse, ToolCall
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class MockOneToolCallLLM(LLMClient):
    """Mock LLM that makes exactly one tool call, then answers."""

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
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="calculator",
                        arguments={"expression": "25 * 4"},
                    )
                ],
            )

        return LLMResponse(content="The answer is 100.")


def test_run_with_tools_tracks_iterations_and_tool_calls():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    llm = MockOneToolCallLLM()
    agent = Agent(llm_client=llm, tool_registry=registry)

    state = ExecutionState()

    agent.run_with_tools(
        "Calculate 25 * 4",
        tools=[{"name": "calculator"}],
        state=state,
    )

    assert state.iterations == 2
    assert state.tool_calls_made == 1


def test_run_with_tools_without_state_does_not_fail():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    llm = MockOneToolCallLLM()
    agent = Agent(llm_client=llm, tool_registry=registry)

    response = agent.run_with_tools(
        "Calculate 25 * 4",
        tools=[{"name": "calculator"}],
    )

    assert response.content == "The answer is 100."


def test_run_with_verification_attaches_execution_state():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    llm = MockOneToolCallLLM()
    verifier = CalculatorVerifier()

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
        verifier=verifier,
    )

    result = agent.run_with_verification(
        "Calculate 25 * 4",
        tools=[{"name": "calculator"}],
    )

    assert result.state.iterations == 2
    assert result.state.tool_calls_made == 1


class MockWrongThenRightLLM(LLMClient):
    """Mock LLM that answers incorrectly first, then correctly."""

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
            return LLMResponse(content="The answer is 101.")

        return LLMResponse(content="The answer is 100.")


def test_run_with_recovery_tracks_retries_in_state():
    llm = MockWrongThenRightLLM()
    verifier = CalculatorVerifier()

    agent = Agent(
        llm_client=llm,
        verifier=verifier,
    )

    result = agent.run_with_recovery(
        "Calculate 25 * 4",
        max_retries=1,
    )

    assert result.state.retries == 1
    assert result.verification.passed is True