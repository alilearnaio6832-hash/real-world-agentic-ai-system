from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Provider-agnostic interface for Large Language Models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the language model.

        Args:
            prompt: Input prompt for the model.

        Returns:
            Generated text response.
        """
        raise NotImplementedError