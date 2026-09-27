from fastapi.testclient import TestClient

from app.agent.agent import Agent, VerifiedAgentResult
from app.agent.state import ExecutionState
from app.agent.verification import VerificationResult
from app.api import app, get_agent
from app.llm.models import LLMResponse


class MockAgent:
    """Mock Agent for API tests, avoiding real LLM calls."""

    def run_with_recovery(self, task: str, max_retries: int = 1):
        state = ExecutionState(iterations=1)

        return VerifiedAgentResult(
            response=LLMResponse(content="100"),
            verification=VerificationResult(
                passed=True,
                reason="Result is correct.",
            ),
            state=state,
        )


def _override_get_agent():
    return MockAgent()


client = TestClient(app)


def test_run_endpoint_returns_successful_result():
    app.dependency_overrides[get_agent] = _override_get_agent

    response = client.post(
        "/run",
        json={"task": "Calculate 25 * 4"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()
    assert data["content"] == "100"
    assert data["verification_passed"] is True
    assert data["state"]["iterations"] == 1


def test_run_endpoint_rejects_missing_task():
    response = client.post("/run", json={})

    assert response.status_code == 422


def test_health_check_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}