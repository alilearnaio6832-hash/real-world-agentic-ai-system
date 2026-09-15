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


class MockExecutionLLM(LLMClient):
    """Mock LLM for execution loop tests."""

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

        if self.call_count == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="calculator",
                        arguments={
                            "expression": "25 * 4",
                        },
                    )
                ],
            )

        return LLMResponse(
            content="The answer is 100."
        )


def create_agent() -> Agent:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    return Agent(
        llm_client=MockExecutionLLM(),
        tool_registry=registry,
    )


def test_agent_execution_loop():
    agent = create_agent()

    result = agent.run_with_tools(
        "Calculate 25 * 4",
        [CALCULATOR_TOOL],
    )

    assert result.content == "The answer is 100."


def test_agent_execution_loop_calls_llm_again():
    llm = MockExecutionLLM()

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
    )

    result = agent.run_with_tools(
        "Calculate 25 * 4",
        [CALCULATOR_TOOL],
    )

    assert result.content == "The answer is 100."
    assert llm.call_count == 2