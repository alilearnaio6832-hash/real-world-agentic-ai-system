from app.evaluation.evaluator import EvaluationCase


def build_calculator_cases() -> list[EvaluationCase]:
    """A small benchmark set of arithmetic tasks for evaluation."""

    return [
        EvaluationCase(
            name="simple_multiplication",
            task="Calculate 25 * 4",
            tools=[],
            expects_tool_call=False,
        ),
        EvaluationCase(
            name="simple_division",
            task="Calculate 10 / 4",
            tools=[],
            expects_tool_call=False,
        ),
        EvaluationCase(
            name="negative_addition",
            task="Calculate -5 + 12",
            tools=[],
            expects_tool_call=False,
        ),
        EvaluationCase(
            name="parentheses_expression",
            task="Calculate (10 + 5) * 3",
            tools=[],
            expects_tool_call=False,
        ),
    ]