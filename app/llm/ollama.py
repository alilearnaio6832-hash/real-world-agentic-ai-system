import json
from urllib import error, request

from app.config import settings
from app.llm.client import LLMClient, LLMError
from app.llm.models import LLMResponse, ToolCall

_DEFAULT_MAX_OUTPUT_TOKENS = 8192


class OllamaLLMClient(LLMClient):
    """LLM client implementation for Ollama."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = (
            base_url or settings.OLLAMA_BASE_URL
        ).rstrip("/")
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: str) -> str:
        """Generate a text response using Ollama."""

        if not prompt.strip():
            raise LLMError("Prompt cannot be empty.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_output_tokens,
            },
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

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        """Generate a response using Ollama tool calling."""

        if not prompt.strip():
            raise LLMError("Prompt cannot be empty.")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "tools": tools,
            "stream": False,
            "options": {
                "num_predict": self.max_output_tokens,
            },
        }

        data = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url=f"{self.base_url}/api/chat",
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

        message = response_data.get("message", {})

        if not isinstance(message, dict):
            raise LLMError(
                "Ollama response contains an invalid message."
            )

        content = message.get("content", "")

        if not isinstance(content, str):
            raise LLMError(
                "Ollama response contains invalid content."
            )

        raw_tool_calls = message.get("tool_calls", [])

        if not isinstance(raw_tool_calls, list):
            raise LLMError(
                "Ollama response contains invalid tool calls."
            )

        tool_calls: list[ToolCall] = []
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                raise LLMError(
                    "Ollama returned an invalid tool call."
                )

            function = raw_call.get("function", {})

            if not isinstance(function, dict):
                raise LLMError(
                    "Ollama tool call contains invalid function data."
                )

            call_id = raw_call.get("id", "")
            name = function.get("name")
            arguments = function.get("arguments", {})

            if not isinstance(call_id, str):
                raise LLMError(
                    "Tool call ID must be a string."
                )

            if not isinstance(name, str):
                raise LLMError(
                    "Tool call name must be a string."
                )

            if not isinstance(arguments, dict):
                raise LLMError(
                    "Tool call arguments must be an object."
                )

            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
        )