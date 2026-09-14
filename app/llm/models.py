from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Represents a tool call requested by an LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response returned by an LLM client."""

    content: str
    tool_calls: list[ToolCall] = field(
        default_factory=list
    )

    @property
    def has_tool_calls(self) -> bool:
        """Return True when the response contains tool calls."""

        return bool(self.tool_calls)