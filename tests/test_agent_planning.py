from app.agent.agent import Agent
from app.agent.planning import Planner
from app.llm.client import LLMClient
from app.llm.models import LLMResponse


class MockPlanningAndExecutionLLM(LLMClient):
    """
    Mock LLM that returns a fixed 2-step plan via generate(), and
    distinct final answers per step via generate_with_tools(),
    based on which step's text appears in the prompt.
    """

    def __init__(self) -> None:
        self.execution_prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        return (
            "STEP: Calculate 25 * 4\n"
            "STEP: Describe the result in one word"
        )

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        self.execution_prompts.append(prompt)

        if "Calculate 25 * 4" in prompt and "Describe" not in prompt:
            return LLMResponse(content="100")

        return LLMResponse(content="Large")


def test_run_with_planning_executes_all_steps_in_order():
    llm = MockPlanningAndExecutionLLM()
    planner = Planner(llm_client=llm)

    agent = Agent(llm_client=llm)

    result = agent.run_with_planning(
        "Calculate 25 * 4, then describe the result.",
        tools=[],
        planner=planner,
    )

    assert len(result.step_results) == 2
    assert result.step_results[0] == "100"
    assert result.step_results[1] == "Large"
    assert result.content == "Large"
    assert result.state.steps_executed == 2


def test_run_with_planning_passes_previous_results_as_context():
    llm = MockPlanningAndExecutionLLM()
    planner = Planner(llm_client=llm)

    agent = Agent(llm_client=llm)

    agent.run_with_planning(
        "Calculate 25 * 4, then describe the result.",
        tools=[],
        planner=planner,
    )

    # The second step's execution prompt must include the first
    # step's result, so the LLM has context for the next step.
    assert "100" in llm.execution_prompts[1]


def test_run_with_planning_single_step_task_runs_once():
    class SingleStepLLM(LLMClient):
        def __init__(self) -> None:
            self.execution_prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            return "STEP: Calculate 25 * 4"

        def generate_with_tools(
            self,
            prompt: str,
            tools: list[dict],
        ) -> LLMResponse:
            self.execution_prompts.append(prompt)
            return LLMResponse(content="100")

    llm = SingleStepLLM()
    planner = Planner(llm_client=llm)
    agent = Agent(llm_client=llm)

    result = agent.run_with_planning(
        "Calculate 25 * 4",
        tools=[],
        planner=planner,
    )

    assert result.step_results == ["100"]
    assert result.content == "100"
    assert len(llm.execution_prompts) == 1