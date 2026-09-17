from dataclasses import dataclass

from app.agent.models import ToolResult
from app.agent.verification import (
    VerificationResult,
    Verifier,
)
from app.llm.client import LLMClient
from app.llm.models import LLMResponse
from app.tools.registry import ToolRegistry


@dataclass
class VerifiedAgentResult:
    """Represents an agent response together with verification."""

    response: LLMResponse
    verification: VerificationResult

    @property
    def content(self) -> str:
        """Return the final agent response content."""

        return self.response.content


class Agent:
    """Core agent responsible for LLM, tools, verification, and recovery."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry or ToolRegistry()
        self.verifier = verifier

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

    def run_with_verification(
        self,
        task: str,
        tools: list[dict] | None = None,
        max_iterations: int = 5,
    ) -> VerifiedAgentResult:
        """
        Execute a task through the full execution loop and verify
        the final result.

        The method requires a verifier to be configured.
        """

        if not task.strip():
            raise ValueError("Task cannot be empty.")

        if self.verifier is None:
            raise RuntimeError(
                "Verifier is not configured."
            )

        response = self.run_with_tools(
            task,
            tools or [],
            max_iterations=max_iterations,
        )

        verification = self.verifier.verify(
            task,
            response.content,
        )

        return VerifiedAgentResult(
            response=response,
            verification=verification,
        )

    def run_with_recovery(
        self,
        task: str,
        tools: list[dict] | None = None,
        max_retries: int = 1,
        max_iterations: int = 5,
    ) -> VerifiedAgentResult:
        """
        Execute a task through the full execution loop, verify the
        result, and retry the entire loop on verification failure.

        Recovery is currently implemented as a simple retry strategy.
        """

        if not task.strip():
            raise ValueError("Task cannot be empty.")

        if self.verifier is None:
            raise RuntimeError(
                "Verifier is not configured."
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries must be greater than or equal to zero."
            )

        tools = tools or []

        for _ in range(max_retries + 1):
            response = self.run_with_tools(
                task,
                tools,
                max_iterations=max_iterations,
            )

            verification = self.verifier.verify(
                task,
                response.content,
            )

            result = VerifiedAgentResult(
                response=response,
                verification=verification,
            )

            if verification.passed:
                return result

        return result

    def run_with_tools(
        self,
        task: str,
        tools: list[dict],
        max_iterations: int = 5,
    ) -> LLMResponse:
        """
        Execute a task using an LLM and registered tools.
        The agent repeatedly asks the LLM for a decision. If the
        LLM requests a tool, the agent executes it, creates a
        ToolResult, adds the observation to the context, and asks
        the LLM again.
        """

        if not task.strip():
            raise ValueError("Task cannot be empty.")

        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than zero."
            )

        context = task

        for _ in range(max_iterations):
            response = self.llm_client.generate_with_tools(
                context,
                tools,
            )

            if not response.has_tool_calls:
                return response

            tool_results: list[ToolResult] = []

            for tool_call in response.tool_calls:
                tool_result = self._execute_tool_call(tool_call)

                tool_results.append(tool_result)

            context = self._build_observation_context(
                context,
                response,
                tool_results,
            )

        raise RuntimeError(
            "Agent execution exceeded maximum iterations."
        )

    def _execute_tool_call(self, tool_call) -> ToolResult:
        """Execute one LLM-requested tool call."""

        try:
            tool_input = self._serialize_tool_arguments(
                tool_call.arguments
            )

            result = self.run_tool(
                tool_call.name,
                tool_input,
            )

            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=result,
            )

        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                error=str(exc),
            )

    def _serialize_tool_arguments(
        self,
        arguments: dict,
    ) -> str:
        """Convert tool arguments into the tool input format."""

        if "expression" in arguments:
            return str(arguments["expression"])

        if len(arguments) == 1:
            return str(next(iter(arguments.values())))

        return str(arguments)

    def _build_observation_context(
        self,
        context: str,
        response: LLMResponse,
        tool_results: list[ToolResult],
    ) -> str:
        """Build the next LLM context using tool observations."""

        observations = []

        for tool_result in tool_results:
            if tool_result.is_success:
                observations.append(
                    f"Tool '{tool_result.name}' returned: "
                    f"{tool_result.result}"
                )
            else:
                observations.append(
                    f"Tool '{tool_result.name}' failed: "
                    f"{tool_result.error}"
                )

        observation_text = "\n".join(observations)

        return (
            f"{context}\n\n"
            f"Previous assistant tool calls were executed.\n"
            f"Tool observations:\n"
            f"{observation_text}\n\n"
            f"Use these observations to continue the task."
        )