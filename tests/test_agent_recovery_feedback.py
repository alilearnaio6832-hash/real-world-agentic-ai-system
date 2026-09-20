from app.agent.agent import Agent
from app.agent.verification import CalculatorVerifier
from app.llm.client import LLMClient
from app.llm.models import LLMResponse


class MockFeedbackAwareLLM(LLMClient):
    """
    Mock LLM that records every prompt it receives and gives a
    wrong answer unless the prompt contains feedback from a
    previous failed attempt.
    """

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

        if "Expected" in prompt:
            return LLMResponse(content="The answer is 100.")

        return LLMResponse(content="The answer is 101.")


def test_recovery_includes_previous_failure_reason_in_retry():
    llm = MockFeedbackAwareLLM()
    verifier = CalculatorVerifier()

    agent = Agent(
        llm_client=llm,
        verifier=verifier,
    )

    result = agent.run_with_recovery(
        "Calculate 25 * 4",
        max_retries=1,
    )

    assert len(llm.received_prompts) == 2
    assert "Expected" not in llm.received_prompts[0]
    assert "Expected" in llm.received_prompts[1]
    assert result.content == "The answer is 100."
    assert result.verification.passed is True


def test_recovery_first_attempt_has_no_feedback():
    llm = MockFeedbackAwareLLM()
    verifier = CalculatorVerifier()

    agent = Agent(
        llm_client=llm,
        verifier=verifier,
    )

    agent.run_with_recovery(
        "Calculate 25 * 4",
        max_retries=1,
    )

    assert llm.received_prompts[0] == "Calculate 25 * 4"