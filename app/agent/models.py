from dataclasses import dataclass


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""

    tool_call_id: str
    name: str
    result: str | None = None
    error: str | None = None

    @property
    def is_success(self) -> bool:
        """Return True when the tool execution succeeded."""

        return self.error is None