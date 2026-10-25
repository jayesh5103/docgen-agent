"""Tests for Pydantic schemas (S1 - S5)."""

import pytest
from pydantic import ValidationError
from app.schemas import (
    AgentRequest,
    CriticReply,
    PlanTask,
    Section,
    Table,
)


def test_s1_table_coercion():
    """S1 Table coerces numeric cells to strings, pads/truncates rows, maps dict rows by header."""
    # Numeric cells & padding/truncation
    data = {
        "headers": ["ColA", "ColB", "ColC"],
        "rows": [
            [1, 2.5, 3],            # numeric cells
            ["X", "Y"],             # needs padding to 3
            ["1", "2", "3", "extra"], # needs truncation to 3
        ],
    }
    table = Table.model_validate(data)
    assert table.rows[0] == ["1", "2.5", "3"]
    assert table.rows[1] == ["X", "Y", ""]
    assert table.rows[2] == ["1", "2", "3"]

    # Mapping dict rows by header
    dict_data = {
        "headers": ["Name", "Score"],
        "rows": [
            {"Name": "Alice", "Score": 95, "Ignore": "me"},
            {"Score": 88},  # missing Name
        ],
    }
    dict_table = Table.model_validate(dict_data)
    assert dict_table.rows[0] == ["Alice", "95"]
    assert dict_table.rows[1] == ["", "88"]


def test_s2_section_coercion():
    """S2 Section accepts a string for paragraphs, list-of-dict bullets, empty-dict table becomes None."""
    data = {
        "heading": "Introduction",
        "paragraphs": "A single paragraph string.",
        "bullets": [
            {"key": "Item 1", "val": "Detail 1"},
            {"item": "Item 2"},
        ],
        "table": {},
    }
    sec = Section.model_validate(data)
    assert sec.paragraphs == ["A single paragraph string."]
    assert sec.bullets == ["Item 1 - Detail 1", "Item 2"]
    assert sec.table is None
    assert sec.word_count() > 0
    assert "## Introduction" in sec.as_text()


def test_s3_plan_task_coercion():
    """S3 PlanTask.tool 'none'/'null'/'' becomes None; string depends_on becomes a list."""
    t1 = PlanTask.model_validate({"id": "t1", "title": "T1", "tool": "none", "depends_on": "t0"})
    assert t1.tool is None
    assert t1.depends_on == ["t0"]

    t2 = PlanTask.model_validate({"id": "t2", "title": "T2", "tool": "null", "depends_on": ""})
    assert t2.tool is None
    assert t2.depends_on == []

    t3 = PlanTask.model_validate({"id": "t3", "title": "T3", "tool": "", "depends_on": ["t1", "t2"]})
    assert t3.tool is None
    assert t3.depends_on == ["t1", "t2"]

    t4 = PlanTask.model_validate({"id": "t4", "title": "T4", "tool": "build_timeline"})
    assert t4.tool == "build_timeline"


def test_s4_agent_request_validation():
    """S4 AgentRequest rejects empty, whitespace-only, 9-char, digits-only, and 4001-char input; accepts a normal sentence."""
    # Valid normal sentence
    req = AgentRequest.model_validate({"request": "Please create a comprehensive proposal for our client."})
    assert "proposal" in req.request

    # Empty
    with pytest.raises(ValidationError):
        AgentRequest.model_validate({"request": ""})

    # Whitespace-only
    with pytest.raises(ValidationError):
        AgentRequest.model_validate({"request": "          "})

    # 9-character
    with pytest.raises(ValidationError):
        AgentRequest.model_validate({"request": "12345678a"})

    # Digits-only (>= 10 chars but no letters)
    with pytest.raises(ValidationError):
        AgentRequest.model_validate({"request": "1234567890123"})

    # 4001-character
    with pytest.raises(ValidationError):
        AgentRequest.model_validate({"request": "a" * 4001})


def test_s5_critic_reply_clamping():
    """S5 CriticReply clamps score 0 to 1 and 15 to 10; accepts '8' and 7.6."""
    c0 = CriticReply.model_validate({"score": 0, "issues": []})
    assert c0.score == 1

    c15 = CriticReply.model_validate({"score": 15, "issues": []})
    assert c15.score == 10

    c_str = CriticReply.model_validate({"score": "8", "issues": []})
    assert c_str.score == 8

    c_float = CriticReply.model_validate({"score": 7.6, "issues": []})
    assert c_float.score == 8
