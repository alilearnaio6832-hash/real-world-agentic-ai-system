from app.llm.client import LLMClient


class Agent:
    """Core agent responsible for interacting with an LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def run(self, task: str) -> str:
        """Execute a task using the configured LLM."""

        if not task.strip():
            raise ValueError("Task cannot be empty.")

        return self.llm_client.generate(task)