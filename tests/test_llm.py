"""Tests for LLM layer (L1 - L8)."""

import json
import httpx
import pytest
from pydantic import BaseModel

from app.errors import LLMOutputError, LLMUnavailableError
from app.llm import OpenAICompatLLM, complete_json, extract_json


class SampleModel(BaseModel):
    name: str
    count: int


class MockLLM:
    """Simple in-memory fake LLM for testing complete_json."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    def chat(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        self.prompts.append((system, user))
        if self.responses:
            return self.responses.pop(0)
        return "{}"


def test_l1_extract_json():
    """L1 extract_json handles plain, fenced, prose-wrapped, and rejects no-JSON / arrays."""
    # Plain
    assert extract_json('{"name": "test"}') == {"name": "test"}

    # Fenced with markdown
    fenced = "```json\n{\"name\": \"test\"}\n```"
    assert extract_json(fenced) == {"name": "test"}

    # Prose wrapped
    prose = "Here is the result:\n{\"name\": \"test\"}\nHope this helps!"
    assert extract_json(prose) == {"name": "test"}

    # Rejects no-JSON
    with pytest.raises(ValueError, match="No valid JSON"):
        extract_json("Just plain text with no braces.")

    # Rejects array
    with pytest.raises(ValueError, match="array"):
        extract_json("[1, 2, 3]")


def test_l2_complete_json_retries_on_invalid_json():
    """L2 complete_json retries on invalid JSON; the second prompt contains the validation error text."""
    llm = MockLLM([
        "invalid json text",
        '{"name": "Valid", "count": 5}',
    ])
    result = complete_json(
        llm=llm,
        step="test_step",
        system="sys prompt",
        user="user prompt",
        model_cls=SampleModel,
        retries=2,
    )
    assert result.name == "Valid"
    assert result.count == 5
    assert len(llm.prompts) == 2
    # Verify second prompt contains previous error text
    second_user = llm.prompts[1][1]
    assert "PREVIOUS_ATTEMPT_VALIDATION_ERROR" in second_user


def test_l3_complete_json_exhaustion():
    """L3 complete_json raises LLMOutputError after retries are exhausted."""
    llm = MockLLM([
        "invalid",
        "also invalid",
        "still invalid",
    ])
    with pytest.raises(LLMOutputError) as exc_info:
        complete_json(
            llm=llm,
            step="plan_step",
            system="sys",
            user="user",
            model_cls=SampleModel,
            retries=2,
        )
    assert exc_info.value.step == "plan_step"


def test_l4_http_429_retry_after():
    """L4 HTTP 429 with Retry-After: 2 triggers sleep(2) via injected sleep, then succeeds."""
    slept: list[float] = []

    def mock_sleep(seconds: float):
        slept.append(seconds)

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, text="Too Many Requests")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{\"success\": true}"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAICompatLLM(
        base_url="https://api.mock.test/v1",
        api_key="test-key",
        client=client,
        sleep=mock_sleep,
    )
    reply = llm.chat("sys", "user")
    assert reply == "{\"success\": true}"
    assert call_count == 2
    assert slept == [2.0]


def test_l5_http_400_response_format():
    """L5 HTTP 400 mentioning response_format is retried once without it."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.read())
        if "response_format" in body:
            return httpx.Response(400, text="Unsupported parameter 'response_format'")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{\"ok\": true}"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAICompatLLM(
        base_url="https://api.mock.test/v1",
        client=client,
        json_mode=True,
    )
    reply = llm.chat("sys", "user")
    assert reply == "{\"ok\": true}"
    assert len(requests) == 2
    # Verify first request had response_format and second did not
    assert "response_format" in json.loads(requests[0].read())
    assert "response_format" not in json.loads(requests[1].read())


def test_l6_http_401_no_retry():
    """L6 HTTP 401 raises LLMUnavailableError with no retry (exactly one request made)."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="Unauthorized: invalid api key")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAICompatLLM(
        base_url="https://api.mock.test/v1",
        api_key="bad-key",
        client=client,
    )
    with pytest.raises(LLMUnavailableError, match="authentication failed"):
        llm.chat("sys", "user")
    assert calls == 1


def test_l7_persistent_503_exhaustion():
    """L7 persistent 503 raises LLMUnavailableError after max_retries + 1 requests."""
    calls = 0
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="Service Unavailable")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAICompatLLM(
        base_url="https://api.mock.test/v1",
        max_retries=3,
        client=client,
        sleep=lambda s: slept.append(s),
    )
    with pytest.raises(LLMUnavailableError, match="LLM error 503"):
        llm.chat("sys", "user")
    # 1 initial + 3 retries = 4 requests
    assert calls == 4
    assert len(slept) == 3


def test_l8_payload_format_and_bearer_header():
    """L8 request payload includes response_format when json_mode=True and the bearer header when a key is set."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{\"done\": true}"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAICompatLLM(
        base_url="https://api.mock.test/v1",
        api_key="secret-bearer-token",
        model="test-model",
        json_mode=True,
        client=client,
    )
    llm.chat("sys prompt", "user query")
    assert len(captured) == 1
    req = captured[0]
    assert req.headers.get("Authorization") == "Bearer secret-bearer-token"
    body = json.loads(req.read())
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == "test-model"
