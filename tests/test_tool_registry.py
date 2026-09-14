import pytest

from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


def test_registry_registers_tool():
    registry = ToolRegistry()
    calculator = CalculatorTool()

    registry.register(calculator)

    assert registry.list_tools() == ["calculator"]


def test_registry_gets_tool():
    registry = ToolRegistry()
    calculator = CalculatorTool()

    registry.register(calculator)

    result = registry.get("calculator")

    assert result is calculator


def test_registry_runs_tool():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    result = registry.run("calculator", "10 + 5")

    assert result == "15"


def test_registry_rejects_duplicate_tool():
    registry = ToolRegistry()

    registry.register(CalculatorTool())

    with pytest.raises(
        ValueError,
        match="Tool already registered: calculator",
    ):
        registry.register(CalculatorTool())


def test_registry_rejects_unknown_tool():
    registry = ToolRegistry()

    with pytest.raises(
        KeyError,
        match="Tool not found: calculator",
    ):
        registry.get("calculator")