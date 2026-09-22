from app.agent.agent import Agent
from app.llm.client import LLMClient
from app.llm.models import LLMResponse
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry
from app.tools.text_analyzer import TextAnalyzerTool


class MockLLMClient(LLMClient):
    """Mock LLM client for Agent tool tests."""

    def generate(self, prompt: str) -> str:
        return f"Mock response: {prompt}"

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        return LLMResponse(
            content=f"Mock tool response: {prompt}"
        )


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


def test_agent_runs_text_analyzer():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TextAnalyzerTool())

    agent = Agent(
        llm_client=MockLLMClient(),
        tool_registry=registry,
    )

    result = agent.run_tool(
        "text_analyzer",
        "Hello world",
    )

    assert result == "words: 2, characters: 11"