import json
from urllib import error, request

from app.config import settings
from app.llm.client import LLMClient, LLMError


class OllamaLLMClient(LLMClient):
    """LLM client implementation for Ollama."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = (
            base_url or settings.OLLAMA_BASE_URL
        ).rstrip("/")

    def generate(self, prompt: str) -> str:
        """Generate a text response using Ollama."""

        if not prompt.strip():
            raise LLMError("Prompt cannot be empty.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=120) as response:
                response_data = json.loads(
                    response.read().decode("utf-8")
                )

        except error.URLError as exc:
            raise LLMError(
                f"Could not connect to Ollama: {exc}"
            ) from exc

        except json.JSONDecodeError as exc:
            raise LLMError(
                "Ollama returned invalid JSON."
            ) from exc

        response_text = response_data.get("response")

        if not isinstance(response_text, str):
            raise LLMError(
                "Ollama response does not contain valid text."
            )

        return response_text