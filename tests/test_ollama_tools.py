from app.llm.ollama import OllamaLLMClient


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calculate basic arithmetic expressions.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to calculate",
                }
            },
            "required": ["expression"],
        },
    },
}


def test_ollama_generate_with_tools():
    client = OllamaLLMClient()

    response = client.generate_with_tools(
        "Calculate 25 * 4",
        [CALCULATOR_TOOL],
    )

    assert response.has_tool_calls
    assert len(response.tool_calls) >= 1

    tool_call = response.tool_calls[0]

    assert tool_call.name == "calculator"
    assert tool_call.arguments["expression"] == "25 * 4"