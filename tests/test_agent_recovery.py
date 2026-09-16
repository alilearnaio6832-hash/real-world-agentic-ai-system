from app.agent.agent import Agent
from app.agent.verification import (
    VerificationResult,
    Verifier,
)
from app.llm.client import LLMClient
from app.llm.models import LLMResponse


class MockRecoveryLLM(LLMClient):
    """Mock LLM that produces a bad result first and a good result second."""

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
            return LLMResponse(
                content="Wrong answer."
            )

        return LLMResponse(
            content="Correct answer."
        )


class MockRecoveryVerifier(Verifier):
    """Mock verifier that fails the first result and passes the second."""

    def __init__(self) -> None:
        self.call_count = 0

    def verify(
        self,
        task: str,
        result: str,
    ) -> VerificationResult:
        self.call_count += 1

        if self.call_count == 1:
            return VerificationResult(
                passed=False,
                reason="First result is incorrect.",
            )

        return VerificationResult(
            passed=True,
            reason="Second result is correct.",
        )


def test_agent_recovers_after_verification_failure():
    llm = MockRecoveryLLM()
    verifier = MockRecoveryVerifier()

    agent = Agent(
        llm_client=llm,
        verifier=verifier,
    )

    result = agent.run_with_recovery(
        "Solve the task.",
        max_retries=1,
    )

    assert result.content == "Correct answer."
    assert result.verification.passed is True
    assert result.verification.status == "PASS"
    assert llm.call_count == 2
    assert verifier.call_count == 2