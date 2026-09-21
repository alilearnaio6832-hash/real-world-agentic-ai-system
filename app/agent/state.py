from dataclasses import dataclass


@dataclass
class ExecutionState:
    """
    Tracks measurable facts about one agent execution.

    This is populated during run_with_tools / run_with_verification /
    run_with_recovery and exposed to the caller for observability and
    evaluation purposes. It is not persisted across calls.
    """

    iterations: int = 0
    tool_calls_made: int = 0
    retries: int = 0