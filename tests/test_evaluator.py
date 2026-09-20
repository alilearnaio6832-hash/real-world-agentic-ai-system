from app.agent.agent import Agent
from app.agent.verification import CalculatorVerifier
from app.evaluation.evaluator import EvaluationCase, Evaluator
from app.llm.client import LLMClient
from app.llm.models import LLMResponse


class MockAlwaysCorrectLLM(LLMClient):
    """Mock LLM that always answers correctly for a fixed set of tasks."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    def generate(self, prompt: str) -> str:
        return "Mock response."

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        for task, answer in self.answers.items():
            if task in prompt:
                return LLMResponse(content=answer)

        return LLMResponse(content="Unknown.")


def test_evaluator_reports_full_success_rate():
    llm = MockAlwaysCorrectLLM(
        {
            "Calculate 25 * 4": "The answer is 100.",
            "Calculate 10 / 4": "The answer is 2.5.",
        }
    )

    agent = Agent(
        llm_client=llm,
        verifier=CalculatorVerifier(),
    )

    cases = [
        EvaluationCase(
            name="mult",
            task="Calculate 25 * 4",
            tools=[],
            expects_tool_call=False,
        ),
        EvaluationCase(
            name="div",
            task="Calculate 10 / 4",
            tools=[],
            expects_tool_call=False,
        ),
    ]

    evaluator = Evaluator()

    result = evaluator.run(agent, cases)

    assert result.total_cases == 2
    assert result.passed_cases == 2
    assert result.success_rate == 1.0


def test_evaluator_reports_partial_failure():
    llm = MockAlwaysCorrectLLM(
        {
            "Calculate 25 * 4": "The answer is 999.",
        }
    )

    agent = Agent(
        llm_client=llm,
        verifier=CalculatorVerifier(),
    )

    cases = [
        EvaluationCase(
            name="mult",
            task="Calculate 25 * 4",
            tools=[],
            expects_tool_call=False,
        ),
    ]

    evaluator = Evaluator()

    result = evaluator.run(agent, cases, max_retries=0)

    assert result.total_cases == 1
    assert result.passed_cases == 0
    assert result.success_rate == 0.0