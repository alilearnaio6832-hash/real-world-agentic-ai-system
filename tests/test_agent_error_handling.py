import pytest

from app.agent.agent import Agent
from app.llm.client import LLMClient
from app.llm.models import LLMResponse, ToolCall
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calculate basic arithmetic expressions.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to calculate",
                }
            },
            "required": ["expression"],
        },
    },
}


class MockFailingToolLLM(LLMClient):
    """Mock LLM that requests an unknown tool."""

    def generate(self, prompt: str) -> str:
        return f"Mock response: {prompt}"

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_unknown",
                    name="unknown_tool",
                    arguments={
                        "input": "test",
                    },
                )
            ],
        )


class MockInfiniteToolLLM(LLMClient):
    """Mock LLM that always requests the calculator."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        return f"Mock response: {prompt}"

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        self.call_count += 1

        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"call_{self.call_count}",
                    name="calculator",
                    arguments={
                        "expression": "25 * 4",
                    },
                )
            ],
        )


def test_agent_handles_unknown_tool_error():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(
        llm_client=MockFailingToolLLM(),
        tool_registry=registry,
    )

    with pytest.raises(RuntimeError, match="maximum iterations"):
        agent.run_with_tools(
            "Use the unknown tool.",
            [CALCULATOR_TOOL],
        )


def test_agent_respects_max_iterations():
    llm = MockInfiniteToolLLM()

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
    )

    with pytest.raises(
        RuntimeError,
        match="maximum iterations",
    ):
        agent.run_with_tools(
            "Calculate 25 * 4",
            [CALCULATOR_TOOL],
            max_iterations=3,
        )

    assert llm.call_count == 3


def test_agent_rejects_invalid_max_iterations():
    agent = Agent(
        llm_client=MockInfiniteToolLLM(),
    )

    with pytest.raises(
        ValueError,
        match="max_iterations must be greater than zero",
    ):
        agent.run_with_tools(
            "Calculate 25 * 4",
            [CALCULATOR_TOOL],
            max_iterations=0,
        )