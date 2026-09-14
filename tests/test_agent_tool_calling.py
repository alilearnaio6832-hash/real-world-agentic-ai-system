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


class MockToolCallingLLM(LLMClient):
    """Mock LLM that requests the calculator tool."""

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
                    id="call_test",
                    name="calculator",
                    arguments={
                        "expression": "25 * 4",
                    },
                )
            ],
        )


def test_agent_requests_tool_call_from_llm():
    agent = Agent(
        llm_client=MockToolCallingLLM(),
    )

    response = agent.run_with_tools(
        "Calculate 25 * 4",
        [CALCULATOR_TOOL],
    )

    assert response.has_tool_calls
    assert len(response.tool_calls) == 1

    tool_call = response.tool_calls[0]

    assert tool_call.name == "calculator"
    assert tool_call.arguments["expression"] == "25 * 4"


def test_agent_executes_requested_tool():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(
        llm_client=MockToolCallingLLM(),
        tool_registry=registry,
    )

    response = agent.run_with_tools(
        "Calculate 25 * 4",
        [CALCULATOR_TOOL],
    )

    tool_call = response.tool_calls[0]

    result = agent.run_tool(
        tool_call.name,
        tool_call.arguments["expression"],
    )

    assert result == "100"