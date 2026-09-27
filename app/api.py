from fastapi import Depends, FastAPI
from pydantic import BaseModel

from app.agent.agent import Agent
from app.agent.verification import CalculatorVerifier
from app.llm.ollama import OllamaLLMClient
from app.tools.calculator import CalculatorTool
from app.tools.text_analyzer import TextAnalyzerTool
from app.tools.registry import ToolRegistry

app = FastAPI(
    title="Real-World Agentic AI System",
    description=(
        "API for running tasks through a verifiable, recoverable "
        "agentic execution loop."
    ),
)


def get_agent() -> Agent:
    """
    Construct an Agent with the default LLM, tools, and verifier.

    Exposed as a FastAPI dependency so tests can override it with a
    mock Agent instead of hitting a real LLM.
    """

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TextAnalyzerTool())

    return Agent(
        llm_client=OllamaLLMClient(),
        tool_registry=registry,
        verifier=CalculatorVerifier(),
    )


class RunTaskRequest(BaseModel):
    """Request body for running a task."""

    task: str
    max_retries: int = 1


class ExecutionStateResponse(BaseModel):
    """Execution metrics for one run."""

    iterations: int
    tool_calls_made: int
    retries: int
    tool_retries: int
    empty_response_retries: int


class RunTaskResponse(BaseModel):
    """Response body for a completed task run."""

    content: str
    verification_passed: bool
    verification_reason: str
    state: ExecutionStateResponse


@app.post("/run", response_model=RunTaskResponse)
def run_task(
    request: RunTaskRequest,
    agent: Agent = Depends(get_agent),
) -> RunTaskResponse:
    """
    Run a task through the full execution loop with verification
    and recovery, and return the result along with execution metrics.
    """

    result = agent.run_with_recovery(
        request.task,
        max_retries=request.max_retries,
    )

    return RunTaskResponse(
        content=result.content,
        verification_passed=result.verification.passed,
        verification_reason=result.verification.reason,
        state=ExecutionStateResponse(
            iterations=result.state.iterations,
            tool_calls_made=result.state.tool_calls_made,
            retries=result.state.retries,
            tool_retries=result.state.tool_retries,
            empty_response_retries=result.state.empty_response_retries,
        ),
    )


@app.get("/health")
def health_check() -> dict:
    """Simple liveness check."""

    return {"status": "ok"}