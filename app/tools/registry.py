from app.tools.base import Tool


class ToolRegistry:
    """Registry for managing available agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by its unique name."""

        if tool.name in self._tools:
            raise ValueError(
                f"Tool already registered: {tool.name}"
            )

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a registered tool by name."""

        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(
                f"Tool not found: {name}"
            ) from exc

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""

        return list(self._tools.keys())

    def run(self, name: str, tool_input: str) -> str:
        """Execute a registered tool."""

        tool = self.get(name)

        return tool.run(tool_input)