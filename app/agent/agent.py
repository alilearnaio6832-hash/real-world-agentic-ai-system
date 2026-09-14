from app.llm.client import LLMClient
from app.llm.models import LLMResponse
from app.tools.registry import ToolRegistry


class Agent:
    """Core agent responsible for LLM and tool orchestration."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry or ToolRegistry()

    def run(self, task: str) -> str:
        """Execute a task using the configured LLM."""

        if not task.strip():
            raise ValueError("Task cannot be empty.")

        return self.llm_client.generate(task)

    def run_tool(
        self,
        tool_name: str,
        tool_input: str,
    ) -> str:
        """Execute a registered tool."""

        return self.tool_registry.run(
            tool_name,
            tool_input,
        )

    def run_with_tools(
        self,
        task: str,
        tools: list[dict],
    ) -> LLMResponse:
        """Ask the LLM to execute a task using available tools."""

        if not task.strip():
            raise ValueError("Task cannot be empty.")

        return self.llm_client.generate_with_tools(
            task,
            tools,
        )