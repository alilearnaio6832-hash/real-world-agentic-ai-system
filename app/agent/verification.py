import ast
import operator
import re
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Represents the result of verifying an agent response."""

    passed: bool
    reason: str

    @property
    def status(self) -> str:
        """Return the verification status."""

        return "PASS" if self.passed else "FAIL"


class Verifier:
    """Base interface for verifying agent results."""

    def verify(
        self,
        task: str,
        result: str,
    ) -> VerificationResult:
        """
        Verify an agent result against the original task.

        Args:
            task: Original user task.
            result: Agent-generated result.

        Returns:
            VerificationResult containing PASS/FAIL status
            and an explanation.
        """

        raise NotImplementedError


class CalculatorVerifier(Verifier):
    """Rule-based verifier for basic arithmetic tasks."""

    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def verify(
        self,
        task: str,
        result: str,
    ) -> VerificationResult:
        """Verify whether a calculator result is mathematically correct."""

        if not task.strip():
            return VerificationResult(
                passed=False,
                reason="Task is empty.",
            )

        if not result.strip():
            return VerificationResult(
                passed=False,
                reason="Result is empty.",
            )

        expression = self._extract_expression(task)

        if expression is None:
            return VerificationResult(
                passed=False,
                reason="Could not extract an arithmetic expression from task.",
            )

        expected = self._evaluate_expression(expression)

        if expected is None:
            return VerificationResult(
                passed=False,
                reason="Could not evaluate the arithmetic expression.",
            )

        actual = self._extract_numeric_result(result)

        if actual is None:
            return VerificationResult(
                passed=False,
                reason="Could not extract a numeric result.",
            )

        if self._numbers_equal(expected, actual):
            return VerificationResult(
                passed=True,
                reason=(
                    f"Result is correct. "
                    f"Expected {expected}, received {actual}."
                ),
            )

        return VerificationResult(
            passed=False,
            reason=(
                f"Result is incorrect. "
                f"Expected {expected}, received {actual}."
            ),
        )

    def _extract_expression(
        self,
        task: str,
    ) -> str | None:
        """Extract a basic arithmetic expression from a task."""

        match = re.search(
            r"(?P<expression>"
            r"-?\d+(?:\.\d+)?"
            r"(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+"
            r")",
            task,
        )

        if match is None:
            return None

        return match.group("expression")

    def _evaluate_expression(
        self,
        expression: str,
    ) -> float | int | None:
        """Safely evaluate a basic arithmetic expression."""

        try:
            tree = ast.parse(
                expression,
                mode="eval",
            )

            return self._evaluate_node(tree.body)

        except (
            SyntaxError,
            ValueError,
            TypeError,
            ZeroDivisionError,
        ):
            return None

    def _evaluate_node(
        self,
        node: ast.AST,
    ) -> float | int:
        """Recursively evaluate allowed AST nodes."""

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(
                "Only numeric values are allowed."
            )

        if isinstance(node, ast.BinOp):
            operation = self._operators.get(type(node.op))

            if operation is None:
                raise ValueError(
                    "Unsupported arithmetic operation."
                )

            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):
            operation = self._operators.get(type(node.op))

            if operation is None:
                raise ValueError(
                    "Unsupported unary operation."
                )

            operand = self._evaluate_node(node.operand)

            return operation(operand)

        raise ValueError(
            "Only basic arithmetic expressions are allowed."
        )

    def _extract_numeric_result(
        self,
        result: str,
    ) -> float | None:
        """Extract the first numeric value from an agent result."""

        match = re.search(
            r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)",
            result,
        )

        if match is None:
            return None

        try:
            return float(match.group())
        except ValueError:
            return None

    def _numbers_equal(
        self,
        expected: float | int,
        actual: float,
        tolerance: float = 1e-9,
    ) -> bool:
        """Compare numeric values with a small floating-point tolerance."""

        return abs(float(expected) - actual) <= tolerance