"""Tests for FastAPI service endpoints (API1 - API7)."""

import base64
from datetime import date
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.agent import DocumentAgent
from app.config import Settings
from app.errors import AgentError, LLMOutputError, LLMUnavailableError
from app.main import app, get_agent, get_settings
from tests.fakes import FROZEN_DATE, ScriptedLLM


@pytest.fixture
def api_client(tmp_path: Path):
    test_settings = Settings(output_dir=str(tmp_path))
    fake_llm = ScriptedLLM()
    test_agent = DocumentAgent(
        llm=fake_llm,
        settings=test_settings,
        today=lambda: FROZEN_DATE,
    )

    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_agent] = lambda: test_agent

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_api1_post_agent_and_download(api_client: TestClient):
    """API1 POST /agent returns 200 with all response fields; GET download_url returns 200, docx content type, body starts with PK."""
    payload = {
        "request": "Create a project proposal for building a customer support chatbot.",
        "include_document_base64": False,
    }
    resp = api_client.post("/agent", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "request_id" in data
    assert data["status"] in ("completed", "completed_with_warnings")
    assert "message" in data
    assert "plan" in data
    assert "reflection" in data
    assert "document" in data
    assert "metrics" in data
    assert "trace" in data

    download_url = data["document"]["download_url"]
    doc_resp = api_client.get(download_url)
    assert doc_resp.status_code == 200
    assert "wordprocessingml.document" in doc_resp.headers["content-type"]
    assert doc_resp.content.startswith(b"PK")


def test_api2_base64_document_delivery(api_client: TestClient):
    """API2 include_document_base64=true returns base64 that decodes to a valid docx."""
    payload = {
        "request": "Create a project proposal for an enterprise chatbot solution.",
        "include_document_base64": True,
    }
    resp = api_client.post("/agent", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    b64_str = data["document"]["base64"]
    assert b64_str is not None
    decoded = base64.b64decode(b64_str)
    assert decoded.startswith(b"PK")


def test_api3_validation_errors_422(api_client: TestClient):
    """API3 Missing field, short, too long, wrong type all return 422."""
    # Missing field
    assert api_client.post("/agent", json={}).status_code == 422

    # Too short (< 10 chars)
    assert api_client.post("/agent", json={"request": "short"}).status_code == 422

    # Too long (> 4000 chars)
    assert api_client.post("/agent", json={"request": "a" * 4001}).status_code == 422

    # Wrong type
    assert api_client.post("/agent", json={"request": 12345}).status_code == 422


def test_api4_rejected_plan_422(tmp_path: Path):
    """API4 Rejected plan returns 422 with status='rejected'."""
    rejected_plan = json.dumps({
        "can_fulfil": False,
        "reason": "Request cannot be fulfilled because it is out of scope.",
        "doc_type": "rejected",
        "tasks": [],
    })
    fake_llm = ScriptedLLM({"plan": [rejected_plan]})
    test_agent = DocumentAgent(llm=fake_llm, settings=Settings(output_dir=str(tmp_path)), today=lambda: FROZEN_DATE)

    app.dependency_overrides[get_agent] = lambda: test_agent
    client = TestClient(app)

    resp = client.post("/agent", json={"request": "Make something out of scope."})
    assert resp.status_code == 422
    data = resp.json()
    assert data["status"] == "rejected"
    assert "out of scope" in data["message"]
    app.dependency_overrides.clear()


def test_api5_exception_handlers_502_503(tmp_path: Path):
    """API5 LLMUnavailableError returns 503; LLMOutputError returns 502; both with detail."""
    # 503 on LLMUnavailableError
    def raise_unavailable(user):
        raise LLMUnavailableError("Groq service rate limit exhausted.")

    fake_unavail = ScriptedLLM({"plan": [raise_unavailable]})
    agent_unavail = DocumentAgent(llm=fake_unavail, settings=Settings(output_dir=str(tmp_path)), today=lambda: FROZEN_DATE)

    app.dependency_overrides[get_agent] = lambda: agent_unavail
    client = TestClient(app)

    resp_503 = client.post("/agent", json={"request": "Create project proposal."})
    assert resp_503.status_code == 503
    assert "rate limit exhausted" in resp_503.json()["detail"]

    # 502 on LLMOutputError
    def raise_output_err(user):
        raise LLMOutputError("plan", "Unrecoverable JSON syntax error.")

    fake_output_err = ScriptedLLM({"plan": [raise_output_err]})
    agent_output_err = DocumentAgent(llm=fake_output_err, settings=Settings(output_dir=str(tmp_path)), today=lambda: FROZEN_DATE)

    app.dependency_overrides[get_agent] = lambda: agent_output_err
    resp_502 = client.post("/agent", json={"request": "Create project proposal."})
    assert resp_502.status_code == 502
    assert "JSON syntax error" in resp_502.json()["detail"]

    app.dependency_overrides.clear()


def test_api6_document_download_security_404(api_client: TestClient):
    """API6 GET /documents/not-a-valid-id, ../../etc/passwd, and a valid-but-unknown id all return 404."""
    # Invalid ID pattern
    assert api_client.get("/documents/not-a-valid-id").status_code == 404

    # Directory traversal attempt
    assert api_client.get("/documents/../../etc/passwd").status_code == 404

    # Valid hex format but non-existent file
    unknown_id = "0123456789abcdef0123456789abcdef"
    assert api_client.get(f"/documents/{unknown_id}").status_code == 404


def test_api7_health_and_tools(api_client: TestClient):
    """API7 GET /health reports provider and model; GET /tools lists the 5 tools."""
    # Health
    health_resp = api_client.get("/health")
    assert health_resp.status_code == 200
    h_data = health_resp.json()
    assert h_data["provider"] == "groq"
    assert h_data["model"] == "llama-3.3-70b-versatile"

    # Tools
    tools_resp = api_client.get("/tools")
    assert tools_resp.status_code == 200
    t_data = tools_resp.json()
    assert len(t_data["tools"]) == 5
    tool_names = [t["name"] for t in t_data["tools"]]
    assert "get_current_date" in tool_names
    assert "build_timeline" in tool_names
    assert "build_budget" in tool_names
    assert "build_risk_register" in tool_names
    assert "get_mock_data" in tool_names
