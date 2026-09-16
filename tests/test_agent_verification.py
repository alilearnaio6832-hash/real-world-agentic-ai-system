from app.agent.agent import Agent
from app.agent.verification import (
    VerificationResult,
    Verifier,
)
from app.llm.client import LLMClient
from app.llm.models import LLMResponse


class MockLLM(LLMClient):
    """Mock LLM that returns a final answer immediately."""

    def generate(self, prompt: str) -> str:
        return "The answer is 100."

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        return LLMResponse(
            content="The answer is 100."
        )


class MockVerifier(Verifier):
    """Mock verifier for Agent integration tests."""

    def __init__(self, passed: bool) -> None:
        self.passed = passed
        self.call_count = 0

    def verify(
        self,
        task: str,
        result: str,
    ) -> VerificationResult:
        self.call_count += 1

        if self.passed:
            return VerificationResult(
                passed=True,
                reason="Result verified successfully.",
            )

        return VerificationResult(
            passed=False,
            reason="Result failed verification.",
        )


def test_agent_can_use_verifier():
    verifier = MockVerifier(passed=True)

    agent = Agent(
        llm_client=MockLLM(),
        verifier=verifier,
    )

    result = agent.run_with_verification(
        "Calculate 25 * 4"
    )

    assert result.content == "The answer is 100."
    assert result.verification is not None
    assert result.verification.passed is True
    assert result.verification.status == "PASS"
    assert verifier.call_count == 1


def test_agent_can_receive_failed_verification():
    verifier = MockVerifier(passed=False)

    agent = Agent(
        llm_client=MockLLM(),
        verifier=verifier,
    )

    result = agent.run_with_verification(
        "Calculate 25 * 4"
    )

    assert result.content == "The answer is 100."
    assert result.verification is not None
    assert result.verification.passed is False
    assert result.verification.status == "FAIL"
    assert verifier.call_count == 1