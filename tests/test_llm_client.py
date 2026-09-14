import pytest

from app.llm.client import LLMClient
from app.llm.models import LLMResponse


class MockLLMClient(LLMClient):
    """Test implementation of the LLM interface."""

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


def test_llm_client_requires_generate_implementation():
    with pytest.raises(TypeError):
        LLMClient()


def test_llm_client_implementation():
    client = MockLLMClient()

    result = client.generate("Hello")

    assert result == "Mock response: Hello"


def test_llm_client_tool_implementation():
    client = MockLLMClient()

    result = client.generate_with_tools(
        "Calculate 2 + 2",
        [],
    )

    assert isinstance(result, LLMResponse)
    assert result.content == "Mock tool response: Calculate 2 + 2"
    assert not result.has_tool_calls