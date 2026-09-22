from app.agent.agent import Agent
from app.agent.state import ExecutionState
from app.llm.client import LLMClient
from app.llm.models import LLMResponse, ToolCall
from app.tools.base import Tool
from app.tools.registry import ToolRegistry


class FlakyTool(Tool):
    """A tool that fails a fixed number of times, then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.call_count = 0

    @property
    def name(self) -> str:
        return "flaky_tool"

    @property
    def description(self) -> str:
        return "A tool that fails a configurable number of times."

    def run(self, tool_input: str) -> str:
        self.call_count += 1

        if self.call_count <= self.fail_times:
            raise ValueError("Transient failure.")

        return f"success: {tool_input}"


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
                        name="flaky_tool",
                        arguments={"input": "abc"},
                    )
                ],
            )

        return LLMResponse(content="Done.")


def test_tool_call_recovers_after_one_transient_failure():
    tool = FlakyTool(fail_times=1)
    registry = ToolRegistry()
    registry.register(tool)

    llm = MockOneToolCallLLM()
    agent = Agent(llm_client=llm, tool_registry=registry)

    state = ExecutionState()

    response = agent.run_with_tools(
        "Use the flaky tool.",
        tools=[{"name": "flaky_tool"}],
        state=state,
    )

    assert response.content == "Done."
    assert tool.call_count == 2
    assert state.tool_retries == 1


def test_tool_call_reports_error_after_exhausting_tool_retries():
    tool = FlakyTool(fail_times=5)
    registry = ToolRegistry()
    registry.register(tool)

    llm = MockOneToolCallLLM()
    agent = Agent(llm_client=llm, tool_registry=registry)

    state = ExecutionState()

    response = agent.run_with_tools(
        "Use the flaky tool.",
        tools=[{"name": "flaky_tool"}],
        state=state,
    )

    assert response.content == "Done."
    assert tool.call_count == 2
    assert state.tool_retries == 1


def test_tool_call_succeeds_immediately_without_retry():
    tool = FlakyTool(fail_times=0)
    registry = ToolRegistry()
    registry.register(tool)

    llm = MockOneToolCallLLM()
    agent = Agent(llm_client=llm, tool_registry=registry)

    state = ExecutionState()

    agent.run_with_tools(
        "Use the flaky tool.",
        tools=[{"name": "flaky_tool"}],
        state=state,
    )

    assert tool.call_count == 1
    assert state.tool_retries == 0