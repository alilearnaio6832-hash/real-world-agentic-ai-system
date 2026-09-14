import ast
import operator

from app.tools.base import Tool


class CalculatorTool(Tool):
    """Safe calculator for basic arithmetic expressions."""

    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Calculate basic arithmetic expressions."

    def run(self, tool_input: str) -> str:
        """Evaluate a safe arithmetic expression."""

        if not tool_input.strip():
            raise ValueError("Calculator input cannot be empty.")

        try:
            tree = ast.parse(
                tool_input,
                mode="eval",
            )

            result = self._evaluate(tree.body)

        except (
            SyntaxError,
            ValueError,
            TypeError,
            ZeroDivisionError,
        ) as exc:
            raise ValueError(
                f"Invalid calculator expression: {tool_input}"
            ) from exc

        return str(result)

    def _evaluate(self, node: ast.AST) -> float | int:
        """Recursively evaluate allowed AST nodes."""

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Only numeric values are allowed.")

        if isinstance(node, ast.BinOp):
            operation = self._operators.get(type(node.op))

            if operation is None:
                raise ValueError(
                    "Unsupported arithmetic operation."
                )

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):
            operation = self._operators.get(type(node.op))

            if operation is None:
                raise ValueError(
                    "Unsupported unary operation."
                )

            operand = self._evaluate(node.operand)

            return operation(operand)

        raise ValueError(
            "Only basic arithmetic expressions are allowed."
        )