import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Application configuration loaded from environment variables."""

    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_NAME: str = os.getenv("APP_NAME", "agentic-ai-system")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")


settings = Settings()