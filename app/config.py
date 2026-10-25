"""Configuration settings for the Autonomous Document Agent."""

from dataclasses import dataclass
import os
from typing import Mapping

PROVIDER_DEFAULTS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1:8b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-1.5-flash",
    },
}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration settings."""

    provider: str = "groq"
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    api_key: str | None = None
    output_dir: str = "output"
    llm_timeout: float = 90.0
    llm_http_retries: int = 4
    json_retries: int = 2
    tool_retries: int = 2
    max_tasks: int = 10
    max_reflection_rounds: int = 1
    quality_threshold: int = 7

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build Settings from an environment mapping."""
        if env is None:
            env = os.environ

        provider = env.get("LLM_PROVIDER", "groq").strip().lower()
        base_url = env.get("LLM_BASE_URL", "").strip() or None
        model = env.get("LLM_MODEL", "").strip() or None

        if provider in PROVIDER_DEFAULTS:
            default_info = PROVIDER_DEFAULTS[provider]
            if not base_url:
                base_url = default_info["base_url"]
            if not model:
                model = default_info["model"]
        else:
            if not base_url:
                raise ValueError(
                    f"Unknown provider '{provider}' and LLM_BASE_URL is not configured."
                )
            if not model:
                model = "default-model"

        # Determine API key based on provider
        api_key = env.get("LLM_API_KEY")
        if not api_key:
            if provider == "groq":
                api_key = env.get("GROQ_API_KEY")
            elif provider == "gemini":
                api_key = env.get("GEMINI_API_KEY")

        output_dir = env.get("OUTPUT_DIR", "output")
        llm_timeout = float(env.get("LLM_TIMEOUT", "90"))
        llm_http_retries = int(env.get("LLM_HTTP_RETRIES", "4"))
        json_retries = int(env.get("JSON_RETRIES", "2"))
        tool_retries = int(env.get("TOOL_RETRIES", "2"))
        max_tasks = int(env.get("MAX_TASKS", "10"))
        max_reflection_rounds = int(env.get("MAX_REFLECTION_ROUNDS", "1"))
        quality_threshold = int(env.get("QUALITY_THRESHOLD", "7"))

        return cls(
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key if api_key else None,
            output_dir=output_dir,
            llm_timeout=llm_timeout,
            llm_http_retries=llm_http_retries,
            json_retries=json_retries,
            tool_retries=tool_retries,
            max_tasks=max_tasks,
            max_reflection_rounds=max_reflection_rounds,
            quality_threshold=quality_threshold,
        )
