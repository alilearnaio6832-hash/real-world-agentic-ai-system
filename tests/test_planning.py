from app.agent.planning import Plan, Planner
from app.llm.client import LLMClient


class MockPlanningLLM(LLMClient):
    """Mock LLM that returns a fixed planning response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.received_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.received_prompt = prompt
        return self.response

    def generate_with_tools(self, prompt: str, tools: list[dict]):
        raise NotImplementedError


def test_planner_parses_multiple_steps():
    llm = MockPlanningLLM(
        "STEP: Calculate 25 * 4\n"
        "STEP: Count the words in the result"
    )
    planner = Planner(llm_client=llm)

    plan = planner.create_plan(
        "Calculate 25 * 4, then count the words in the result."
    )

    assert plan.steps == [
        "Calculate 25 * 4",
        "Count the words in the result",
    ]
    assert plan.is_single_step is False


def test_planner_handles_single_step_task():
    llm = MockPlanningLLM("STEP: Calculate 25 * 4")
    planner = Planner(llm_client=llm)

    plan = planner.create_plan("Calculate 25 * 4")

    assert plan.steps == ["Calculate 25 * 4"]
    assert plan.is_single_step is True


def test_planner_ignores_non_step_lines():
    llm = MockPlanningLLM(
        "Here is the plan:\n"
        "STEP: Calculate 25 * 4\n"
        "That's it."
    )
    planner = Planner(llm_client=llm)

    plan = planner.create_plan("Calculate 25 * 4")

    assert plan.steps == ["Calculate 25 * 4"]


def test_planner_falls_back_to_original_task_when_no_steps_found():
    llm = MockPlanningLLM("I don't understand the format.")
    planner = Planner(llm_client=llm)

    plan = planner.create_plan("Calculate 25 * 4")

    assert plan.steps == ["Calculate 25 * 4"]


def test_planner_rejects_empty_task():
    llm = MockPlanningLLM("STEP: something")
    planner = Planner(llm_client=llm)

    try:
        planner.create_plan("")
    except ValueError as exc:
        assert "Task cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty task.")