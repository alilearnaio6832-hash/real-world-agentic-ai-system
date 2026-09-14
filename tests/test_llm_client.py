import pytest

from app.llm.client import LLMClient


class MockLLMClient(LLMClient):
    """Test implementation of the LLM interface."""

    def generate(self, prompt: str) -> str:
        return f"Mock response: {prompt}"


def test_llm_client_requires_generate_implementation():
    with pytest.raises(TypeError):
        LLMClient()


def test_llm_client_implementation():
    client = MockLLMClient()

    result = client.generate("Hello")

    assert result == "Mock response: Hello"