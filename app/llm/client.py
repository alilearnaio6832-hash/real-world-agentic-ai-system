from abc import ABC, abstractmethod

from app.llm.models import LLMResponse


class LLMError(Exception):
    """Base exception for LLM-related errors."""


class LLMClient(ABC):
    """Provider-agnostic interface for Large Language Models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a text response from the language model.

        Args:
            prompt: Input prompt for the model.

        Returns:
            Generated text response.

        Raises:
            LLMError: If the LLM provider cannot generate a response.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        """
        Generate a response with optional tool calling.

        Args:
            prompt: Input prompt for the model.
            tools: Tool definitions available to the model.

        Returns:
            Normalized LLM response containing text and/or tool calls.

        Raises:
            LLMError: If the LLM provider cannot generate a response.
        """
        raise NotImplementedError