from app.agent.agent import Agent
from app.agent.verification import VerificationResult, Verifier
from app.llm.client import LLMClient
from app.llm.models import LLMResponse, ToolCall
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class MockToolCallingLLM(LLMClient):
    """Mock LLM that first requests a calculator call, then answers."""

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
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="calculator",
                        arguments={"expression": "25 * 4"},
                    )
                ],
            )

        return LLMResponse(content="The answer is 100.")


class MockPassVerifier(Verifier):
    """Mock verifier that always passes."""

    def __init__(self) -> None:
        self.call_count = 0

    def verify(
        self,
        task: str,
        result: str,
    ) -> VerificationResult:
        self.call_count += 1

        return VerificationResult(
            passed=True,
            reason="Result verified successfully.",
        )


def test_verification_runs_after_real_tool_execution():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    llm = MockToolCallingLLM()
    verifier = MockPassVerifier()

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
        verifier=verifier,
    )

    result = agent.run_with_verification(
        "Calculate 25 * 4",
        tools=[{"name": "calculator"}],
    )

    assert llm.call_count == 2
    assert result.content == "The answer is 100."
    assert result.verification.passed is True
    assert verifier.call_count == 1