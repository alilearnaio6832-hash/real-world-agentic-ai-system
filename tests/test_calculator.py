import pytest

from app.tools.calculator import CalculatorTool


def test_calculator_addition():
    calculator = CalculatorTool()

    assert calculator.run("2 + 3") == "5"


def test_calculator_multiplication():
    calculator = CalculatorTool()

    assert calculator.run("6 * 7") == "42"


def test_calculator_parentheses():
    calculator = CalculatorTool()

    assert calculator.run("(10 + 5) * 2") == "30"


def test_calculator_division():
    calculator = CalculatorTool()

    assert calculator.run("10 / 4") == "2.5"


def test_calculator_negative_number():
    calculator = CalculatorTool()

    assert calculator.run("-5 + 2") == "-3"


def test_calculator_rejects_empty_input():
    calculator = CalculatorTool()

    with pytest.raises(
        ValueError,
        match="Calculator input cannot be empty.",
    ):
        calculator.run("")


def test_calculator_rejects_invalid_expression():
    calculator = CalculatorTool()

    with pytest.raises(
        ValueError,
        match="Invalid calculator expression",
    ):
        calculator.run("2 + hello")


def test_calculator_rejects_unsupported_expression():
    calculator = CalculatorTool()

    with pytest.raises(
        ValueError,
        match="Invalid calculator expression",
    ):
        calculator.run("print(123)")


def test_calculator_rejects_division_by_zero():
    calculator = CalculatorTool()

    with pytest.raises(
        ValueError,
        match="Invalid calculator expression",
    ):
        calculator.run("10 / 0")