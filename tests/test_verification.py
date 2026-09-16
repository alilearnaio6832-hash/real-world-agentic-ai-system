from app.agent.verification import Verifier, VerificationResult


class MockVerifier(Verifier):
    """Simple verifier implementation for testing the interface."""

    def verify(
        self,
        task: str,
        result: str,
    ) -> VerificationResult:
        if result.strip():
            return VerificationResult(
                passed=True,
                reason="Result is not empty.",
            )

        return VerificationResult(
            passed=False,
            reason="Result is empty.",
        )


def test_verification_result_pass():
    result = VerificationResult(
        passed=True,
        reason="Result is correct.",
    )

    assert result.passed is True
    assert result.reason == "Result is correct."
    assert result.status == "PASS"


def test_verification_result_fail():
    result = VerificationResult(
        passed=False,
        reason="Result is incorrect.",
    )

    assert result.passed is False
    assert result.reason == "Result is incorrect."
    assert result.status == "FAIL"


def test_verifier_returns_pass():
    verifier = MockVerifier()

    result = verifier.verify(
        task="Calculate 25 * 4",
        result="100",
    )

    assert result.passed is True
    assert result.status == "PASS"


def test_verifier_returns_fail():
    verifier = MockVerifier()

    result = verifier.verify(
        task="Calculate 25 * 4",
        result="",
    )

    assert result.passed is False
    assert result.status == "FAIL"