from dataclasses import dataclass

from app.llm.client import LLMClient

_PLAN_PROMPT_TEMPLATE = (
    "Break the following task into an ordered list of simple, "
    "concrete steps. If the task is already simple and needs only "
    "one step, output exactly one step.\n\n"
    "Task: {task}\n\n"
    "Respond with ONLY the steps, one per line, each line starting "
    "with 'STEP: '. Do not include any other text, explanation, or "
    "numbering."
)


@dataclass
class Plan:
    """An ordered sequence of steps to execute a task."""

    steps: list[str]

    @property
    def is_single_step(self) -> bool:
        return len(self.steps) <= 1


class Planner:
    """
    Breaks a task into an ordered list of steps using an LLM.

    The output format is strictly constrained (one 'STEP: ' line per
    step) so parsing is deterministic and testable, independent of
    the LLM's natural-language style.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def create_plan(self, task: str) -> Plan:
        """Ask the LLM to break the task into steps and parse the result."""

        if not task.strip():
            raise ValueError("Task cannot be empty.")

        prompt = _PLAN_PROMPT_TEMPLATE.format(task=task)

        raw_response = self.llm_client.generate(prompt)

        steps = self._parse_steps(raw_response)

        if not steps:
            return Plan(steps=[task])

        return Plan(steps=steps)

    def _parse_steps(self, raw_response: str) -> list[str]:
        """Extract 'STEP: ...' lines from the raw LLM response."""

        steps = []

        for line in raw_response.splitlines():
            stripped = line.strip()

            if stripped.startswith("STEP:"):
                step_text = stripped[len("STEP:"):].strip()

                if step_text:
                    steps.append(step_text)

        return steps