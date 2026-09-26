from dataclasses import dataclass

from app.agent.models import ToolResult
from app.agent.planning import Planner
from app.agent.state import ExecutionState
from app.agent.verification import (
    VerificationResult,
    Verifier,
)
from app.llm.client import LLMClient
from app.llm.models import LLMResponse
from app.tools.registry import ToolRegistry


@dataclass
class VerifiedAgentResult:
    """Represents an agent response together with verification and state."""

    response: LLMResponse
    verification: VerificationResult
    state: ExecutionState

    @property
    def content(self) -> str:
        """Return the final agent response content."""

        return self.response.content


@dataclass
class PlanningResult:
    """Represents the outcome of executing a task through a plan."""

    step_results: list[str]
    state: ExecutionState

    @property
    def content(self) -> str:
        """Return the final step's result as the overall answer."""

        return self.step_results[-1]


@dataclass
class _AttemptRecord:
    """Internal record of one failed recovery attempt."""

    answer: str
    reason: str


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

    def run_with_planning(
        self,
        task: str,
        tools: list[dict],
        planner: Planner,
        max_iterations: int = 5,
    ) -> PlanningResult:
        """
        Break a task into an ordered plan, then execute each step in
        sequence through the full execution loop.

        Each step's task includes the results of all previous steps
        as context, so later steps can build on earlier results. The
        final step's result is treated as the overall answer.
        """

        if not task.strip():
            raise ValueError("Task cannot be empty.")

        plan = planner.create_plan(task)

        state = ExecutionState()
        step_results: list[str] = []

        for step in plan.steps:
            step_task = self._build_step_task(step, step_results)

            response = self.run_with_tools(
                step_task,
                tools,
                max_iterations=max_iterations,
                state=state,
            )

            step_results.append(response.content)
            state.steps_executed += 1

        return PlanningResult(
            step_results=step_results,
            state=state,
        )

    def _build_step_task(
        self,
        step: str,
        previous_results: list[str],
    ) -> str:
        """Build one step's task, including prior steps' results as context."""

        if not previous_results:
            return step

        lines = [step, "", "Context from previous steps:"]

        for index, result in enumerate(previous_results, start=1):
            lines.append(f"Step {index} result: {result}")

        return "\n".join(lines)

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

        state = ExecutionState()

        response = self.run_with_tools(
            task,
            tools or [],
            max_iterations=max_iterations,
            state=state,
        )

        verification = self.verifier.verify(
            task,
            response.content,
        )

        return VerifiedAgentResult(
            response=response,
            verification=verification,
            state=state,
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
        result, and retry on failure.

        Every failed attempt is recorded. Each retry's task includes
        feedback built from the full attempt history, so a repeated
        wrong answer is explicitly flagged to the LLM instead of
        being retried blindly. Iteration, tool-call, tool-retry, and
        retry counts are accumulated into a single ExecutionState
        across all attempts.
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
        current_task = task
        history: list[_AttemptRecord] = []
        state = ExecutionState()

        for attempt_index in range(max_retries + 1):
            state.retries = attempt_index

            response = self.run_with_tools(
                current_task,
                tools,
                max_iterations=max_iterations,
                state=state,
            )

            verification = self.verifier.verify(
                task,
                response.content,
            )

            result = VerifiedAgentResult(
                response=response,
                verification=verification,
                state=state,
            )

            if verification.passed:
                return result

            history.append(
                _AttemptRecord(
                    answer=response.content,
                    reason=verification.reason,
                )
            )

            current_task = self._build_retry_task(
                task,
                history,
            )

        return result

    def _build_retry_task(
        self,
        task: str,
        history: list[_AttemptRecord],
    ) -> str:
        """
        Build the next attempt's task, including feedback built from
        every previous failed attempt so far.
        """

        lines = [task, ""]

        for attempt in history:
            lines.append(
                f"You already tried '{attempt.answer}' and it "
                f"was rejected."
            )
            lines.append(f"Reason: {attempt.reason}")

        lines.append(
            "Please try a different answer and correct the mistake."
        )

        return "\n".join(lines)

    def run_with_tools(
        self,
        task: str,
        tools: list[dict],
        max_iterations: int = 5,
        state: ExecutionState | None = None,
    ) -> LLMResponse:
        """
        Execute a task using an LLM and registered tools.

        The agent repeatedly asks the LLM for a decision. If the
        LLM requests a tool, the agent executes it, creates a
        ToolResult, adds the observation to the context, and asks
        the LLM again.

        If an ExecutionState is provided, iteration, tool-call, and
        tool-retry counts are accumulated into it; otherwise state
        tracking is skipped entirely and behavior is unchanged.
        """
        if not task.strip():
            raise ValueError("Task cannot be empty.")

        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than zero."
            )

        context = task

        for _ in range(max_iterations):
            if state is not None:
                state.iterations += 1

            response = self.llm_client.generate_with_tools(
                context,
                tools,
            )

            if not response.has_tool_calls:
                return response

            tool_results: list[ToolResult] = []

            for tool_call in response.tool_calls:
                tool_result = self._execute_tool_call(
                    tool_call,
                    state,
                )

                if state is not None:
                    state.tool_calls_made += 1

                tool_results.append(tool_result)

            context = self._build_observation_context(
                context,
                response,
                tool_results,
            )

        raise RuntimeError(
            "Agent execution exceeded maximum iterations."
        )

    def _execute_tool_call(
        self,
        tool_call,
        state: ExecutionState | None = None,
        max_tool_retries: int = 1,
    ) -> ToolResult:
        """
        Execute one LLM-requested tool call.

        If the tool raises an exception, it is retried up to
        max_tool_retries times with the same input before the
        failure is reported in the ToolResult. This handles
        transient tool failures without involving the LLM or the
        verification/recovery layer.
        """

        tool_input = self._serialize_tool_arguments(
            tool_call.arguments
        )

        last_error: Exception | None = None

        for attempt in range(max_tool_retries + 1):
            try:
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
                last_error = exc

                if attempt < max_tool_retries and state is not None:
                    state.tool_retries += 1

        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            error=str(last_error),
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

    