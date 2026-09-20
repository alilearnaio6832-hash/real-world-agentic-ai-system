from app.agent.agent import Agent
from app.agent.verification import CalculatorVerifier
from app.llm.client import LLMClient
from app.llm.models import LLMResponse


class MockRepeatingLLM(LLMClient):
    """Mock LLM that always gives the same wrong answer."""

    def __init__(self) -> None:
        self.received_prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        return "Mock response."

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        self.received_prompts.append(prompt)

        return LLMResponse(content="The answer is 101.")


class MockEventuallyCorrectLLM(LLMClient):
    """Mock LLM that repeats a wrong answer once, then self-corrects."""

    def __init__(self) -> None:
        self.received_prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        return "Mock response."

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        self.received_prompts.append(prompt)

        if len(self.received_prompts) < 3:
            return LLMResponse(content="The answer is 101.")

        return LLMResponse(content="The answer is 100.")


def test_retry_feedback_flags_repeated_wrong_answer():
    llm = MockRepeatingLLM()
    verifier = CalculatorVerifier()

    agent = Agent(
        llm_client=llm,
        verifier=verifier,
    )

    result = agent.run_with_recovery(
        "Calculate 25 * 4",
        max_retries=2,
    )

    assert len(llm.received_prompts) == 3
    assert "already tried" not in llm.received_prompts[0]
    assert "already tried" in llm.received_prompts[1]
    assert "already tried" in llm.received_prompts[2]
    assert result.verification.passed is False


def test_retry_feedback_lists_all_previous_attempts():
    llm = MockEventuallyCorrectLLM()
    verifier = CalculatorVerifier()

    agent = Agent(
        llm_client=llm,
        verifier=verifier,
    )

    result = agent.run_with_recovery(
        "Calculate 25 * 4",
        max_retries=2,
    )

    assert len(llm.received_prompts) == 3
    assert result.verification.passed is True
    assert result.content == "The answer is 100."
    # Third prompt must reference the repeated wrong answer from
    # both previous attempts.
    assert llm.received_prompts[2].count("101") >= 2