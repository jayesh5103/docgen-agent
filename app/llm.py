"""LLM integration: protocol, OpenAI-compatible HTTP client, retry handling, and JSON parsing."""

import json
import time
from typing import Any, Callable, Protocol, TypeVar
import httpx
from pydantic import BaseModel

from app.config import Settings
from app.errors import LLMOutputError, LLMUnavailableError

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    """Protocol for LLM chat interactions."""

    def chat(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        """Send chat messages to LLM and return raw response string."""
        ...


class CountingLLM:
    """Wraps an LLM instance to track the number of invocations per request."""

    def __init__(self, inner: LLM):
        self.inner = inner
        self.call_count = 0

    def chat(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        self.call_count += 1
        return self.inner.chat(system, user, temperature=temperature)


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from raw LLM text, stripping code fences or prose wrappers."""
    stripped = text.strip()

    # Attempt direct JSON parse
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            raise ValueError("Parsed JSON is an array, expected an object.")
    except json.JSONDecodeError:
        pass

    # Find the outermost curly braces
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = stripped[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Extracted JSON is not an object.")
        except json.JSONDecodeError as err:
            raise ValueError(f"Extracted substring is not valid JSON: {err}")

    raise ValueError("No valid JSON object found in response.")


def complete_json(
    llm: LLM,
    step: str,
    system: str,
    user: str,
    model_cls: type[T],
    retries: int = 2,
) -> T:
    """Call LLM, extract JSON, and validate against Pydantic model with retry on error."""
    current_user = user
    last_error = ""

    for attempt in range(1 + retries):
        try:
            raw_reply = llm.chat(system, current_user)
            parsed_data = extract_json(raw_reply)
            return model_cls.model_validate(parsed_data)
        except LLMUnavailableError:
            raise
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                current_user = (
                    f"{user}\n\n"
                    f"PREVIOUS_ATTEMPT_VALIDATION_ERROR: {last_error}\n"
                    f"Please fix the validation error and reply again with ONLY one valid JSON object."
                )

    raise LLMOutputError(step=step, reason=last_error)


class OpenAICompatLLM:
    """Direct HTTP client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 90.0,
        max_retries: int = 4,
        json_mode: bool = True,
        max_tokens: int = 4096,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        self._external_client = client is not None
        self.client = client or httpx.Client(timeout=timeout)
        self.sleep = sleep

    def chat(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        attempts = 0
        max_attempts = self.max_retries + 1
        dropped_response_format = False

        while attempts < max_attempts:
            attempts += 1
            try:
                response = self.client.post(url, json=payload, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise LLMOutputError("chat", "No choices returned in LLM response.")
                    return choices[0]["message"]["content"]

                # 401 / 403: auth failure -> do not retry
                if response.status_code in (401, 403):
                    raise LLMUnavailableError(
                        f"LLM authentication failed ({response.status_code}): {response.text}. "
                        "Please check your API key."
                    )

                # 400: if mentioning response_format or json validation failure, drop it and retry once
                if response.status_code == 400 and not dropped_response_format:
                    error_text = response.text.lower()
                    if (
                        "response_format" in error_text
                        or "json_validate_failed" in error_text
                        or "failed to generate json" in error_text
                        or "failed to validate json" in error_text
                    ):
                        dropped_response_format = True
                        payload.pop("response_format", None)
                        continue

                # 429 (rate limits) or 5xx server errors
                if response.status_code == 429 or response.status_code in (500, 502, 503, 504):
                    if attempts >= max_attempts:
                        raise LLMUnavailableError(
                            f"LLM error {response.status_code} after {attempts} attempts: {response.text}"
                        )
                    retry_after = response.headers.get("Retry-After")
                    wait_time: float | None = None
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            pass
                    if wait_time is None:
                        wait_time = min(30.0, 1.0 * (2 ** (attempts - 1)))
                    else:
                        wait_time = min(30.0, wait_time)

                    self.sleep(wait_time)
                    continue

                # Any other unexpected HTTP status
                raise LLMUnavailableError(
                    f"LLM request failed with status {response.status_code}: {response.text}"
                )

            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempts >= max_attempts:
                    raise LLMUnavailableError(
                        f"LLM connection error after {attempts} attempts: {exc}"
                    )
                wait_time = min(30.0, 1.0 * (2 ** (attempts - 1)))
                self.sleep(wait_time)
                continue

        raise LLMUnavailableError(f"LLM retries exhausted after {attempts} attempts.")


def build_llm(settings: Settings) -> OpenAICompatLLM:
    """Create an OpenAICompatLLM from application settings."""
    if settings.provider in ("groq", "gemini") and not settings.api_key:
        raise LLMUnavailableError(
            f"API key is required for provider '{settings.provider}'. "
            "Please set LLM_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY."
        )

    return OpenAICompatLLM(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_http_retries,
        json_mode=True,
    )
