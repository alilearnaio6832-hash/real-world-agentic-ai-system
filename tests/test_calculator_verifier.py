from app.agent.verification import (
    CalculatorVerifier,
    VerificationResult,
)


def test_calculator_verifier_passes_correct_result():
    verifier = CalculatorVerifier()

    result = verifier.verify(
        task="Calculate 25 * 4",
        result="100",
    )

    assert isinstance(result, VerificationResult)
    assert result.passed is True
    assert result.status == "PASS"


def test_calculator_verifier_fails_incorrect_result():
    verifier = CalculatorVerifier()

    result = verifier.verify(
        task="Calculate 25 * 4",
        result="90",
    )

    assert result.passed is False
    assert result.status == "FAIL"


def test_calculator_verifier_fails_when_result_is_empty():
    verifier = CalculatorVerifier()

    result = verifier.verify(
        task="Calculate 25 * 4",
        result="",
    )

    assert result.passed is False
    assert result.status == "FAIL"


def test_calculator_verifier_handles_decimal_calculation():
    verifier = CalculatorVerifier()

    result = verifier.verify(
        task="Calculate 10 / 4",
        result="2.5",
    )

    assert result.passed is True


def test_calculator_verifier_rejects_wrong_decimal_result():
    verifier = CalculatorVerifier()

    result = verifier.verify(
        task="Calculate 10 / 4",
        result="2.4",
    )

    assert result.passed is False