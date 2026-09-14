from abc import ABC, abstractmethod


class Tool(ABC):
    """Base interface for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique tool name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable tool description."""
        raise NotImplementedError

    @abstractmethod
    def run(self, tool_input: str) -> str:
        """Execute the tool with the given input."""
        raise NotImplementedError