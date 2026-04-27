"""Application configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Server and LLM provider configuration.

    All values can be overridden with environment variables (or a ``.env`` file
    loaded by ``python-dotenv``).
    """

    # --- LLM provider ---
    llm_provider: str = "cerebras"
    cerebras_api_key: str = ""
    cerebras_model: str = "qwen-3-235b-a22b-instruct-2507"
    openjustice_api_key: str = ""
    openjustice_model: str = "gpt-5.4-nano"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout_seconds: int = 120

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
