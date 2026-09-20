from app.agent.agent import Agent
from app.agent.verification import CalculatorVerifier
from app.llm.client import LLMClient
from app.llm.models import LLMResponse


class MockWrongThenRightLLM(LLMClient):
    """Mock LLM that answers incorrectly first, then correctly."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        return "Mock response."

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        self.call_count += 1

        if self.call_count == 1:
            return LLMResponse(content="The answer is 101.")

        return LLMResponse(content="The answer is 100.")


def test_agent_recovers_from_wrong_calculation():
    llm = MockWrongThenRightLLM()
    verifier = CalculatorVerifier()

    agent = Agent(
        llm_client=llm,
        verifier=verifier,
    )

    result = agent.run_with_recovery(
        "Calculate 25 * 4",
        max_retries=1,
    )

    assert llm.call_count == 2
    assert result.content == "The answer is 100."
    assert result.verification.passed is True


def test_agent_reports_failure_after_exhausting_retries():
    llm = MockWrongThenRightLLM()
    verifier = CalculatorVerifier()

    agent = Agent(
        llm_client=llm,
        verifier=verifier,
    )

    result = agent.run_with_recovery(
        "Calculate 25 * 4",
        max_retries=0,
    )

    assert llm.call_count == 1
    assert result.content == "The answer is 101."
    assert result.verification.passed is False