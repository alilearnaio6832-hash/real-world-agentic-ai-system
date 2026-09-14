from app.agent.agent import Agent
from app.llm.client import LLMClient
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class MockLLMClient(LLMClient):
    """Mock LLM client for Agent tool tests."""

    def generate(self, prompt: str) -> str:
        return f"Mock response: {prompt}"


def create_agent_with_calculator() -> Agent:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    return Agent(
        llm_client=MockLLMClient(),
        tool_registry=registry,
    )


def test_agent_runs_calculator():
    agent = create_agent_with_calculator()

    result = agent.run_tool(
        "calculator",
        "25 * 4",
    )

    assert result == "100"


def test_agent_runs_calculator_with_parentheses():
    agent = create_agent_with_calculator()

    result = agent.run_tool(
        "calculator",
        "(10 + 5) * 3",
    )

    assert result == "45"


def test_agent_rejects_unknown_tool():
    agent = create_agent_with_calculator()

    try:
        agent.run_tool(
            "unknown_tool",
            "10 + 5",
        )
    except KeyError as exc:
        assert "Tool not found: unknown_tool" in str(exc)
    else:
        raise AssertionError(
            "Expected KeyError for unknown tool."
        )

def test_agent_propagates_calculator_error():
    agent = create_agent_with_calculator()

    try:
        agent.run_tool(
            "calculator",
            "10 / 0",
        )
    except ValueError as exc:
        assert "Invalid calculator expression" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError from calculator."
        )