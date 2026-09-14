from app.llm.ollama import OllamaLLMClient


def test_ollama_generate():
    client = OllamaLLMClient()

    response = client.generate("Reply with exactly: TEST_OK")

    assert isinstance(response, str)
    assert response.strip()