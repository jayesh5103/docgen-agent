"""Tests for deterministic tools (T1 - T8)."""

from datetime import date
import pytest
from app.errors import ToolError
from app.tools import default_registry


@pytest.fixture
def registry():
    """Registry with frozen reference date 2026-10-23."""
    frozen_date = date(2026, 10, 23)
    return default_registry(today=lambda: frozen_date)


def test_t1_timeline_sequential(registry):
    """T1 timeline from 2026-11-02 with phases 2 w and 4 w gives 2026-11-02 to 2026-11-15 and 2026-11-16 to 2026-12-13."""
    tool = registry.get("build_timeline")
    result = tool.run({
        "start_date": "2026-11-02",
        "phases": [
            {"name": "Phase 1", "duration_weeks": 2, "parallel_with_previous": False},
            {"name": "Phase 2", "duration_weeks": 4, "parallel_with_previous": False},
        ],
    })
    rows = result.table.rows
    assert len(rows) == 2
    # Phase 1: 2026-11-02 to 2026-11-15
    assert rows[0][1] == "2026-11-02"
    assert rows[0][2] == "2026-11-15"
    assert rows[0][3] == "2"

    # Phase 2: 2026-11-16 to 2026-12-13
    assert rows[1][1] == "2026-11-16"
    assert rows[1][2] == "2026-12-13"
    assert rows[1][3] == "4"


def test_t2_timeline_parallel(registry):
    """T2 parallel_with_previous starts on the previous phase's start; the next sequential phase starts after the latest end."""
    tool = registry.get("build_timeline")
    result = tool.run({
        "start_date": "2026-11-02",
        "phases": [
            {"name": "Phase 1", "duration_weeks": 4, "parallel_with_previous": False},  # Nov 2 to Nov 29
            {"name": "Phase 2 (parallel)", "duration_weeks": 2, "parallel_with_previous": True}, # Nov 2 to Nov 15
            {"name": "Phase 3 (seq)", "duration_weeks": 2, "parallel_with_previous": False}, # Nov 30 to Dec 13
        ],
    })
    rows = result.table.rows
    assert len(rows) == 3
    # Phase 1
    assert rows[0][1] == "2026-11-02"
    assert rows[0][2] == "2026-11-29"

    # Phase 2 starts on previous phase's start (2026-11-02)
    assert rows[1][1] == "2026-11-02"
    assert rows[1][2] == "2026-11-15"

    # Phase 3 starts after latest end (Nov 29 + 1 day = 2026-11-30)
    assert rows[2][1] == "2026-11-30"
    assert rows[2][2] == "2026-12-13"


def test_t3_timeline_deadline_flags(registry):
    """T3 deadline exceeded gives a flag containing the number of days over; deadline met gives no flag."""
    tool = registry.get("build_timeline")

    # Exceeded deadline
    res_exceeded = tool.run({
        "start_date": "2026-11-02",
        "deadline": "2026-11-10",  # earlier than 2 weeks end
        "phases": [
            {"name": "Phase 1", "duration_weeks": 2}, # ends 2026-11-15 (5 days over)
        ],
    })
    assert len(res_exceeded.flags) == 1
    assert "5 days" in res_exceeded.flags[0]

    # Met deadline
    res_met = tool.run({
        "start_date": "2026-11-02",
        "deadline": "2026-11-20",
        "phases": [
            {"name": "Phase 1", "duration_weeks": 2},
        ],
    })
    assert len(res_met.flags) == 0
    assert "meeting the deadline" in res_met.summary


def test_t4_budget_calculation_and_cap(registry):
    """T4 budget: exact subtotal, contingency and total for a fixed input; cap exceeded gives a flag with the overage amount; Total row present."""
    tool = registry.get("build_budget")
    result = tool.run({
        "currency": "USD",
        "contingency_pct": 10.0,
        "budget_cap": 1000.0,
        "items": [
            {"name": "Item A", "quantity": 10, "unit_cost": 50.0},  # 500
            {"name": "Item B", "quantity": 5, "unit_cost": 100.0},  # 500
        ],
    })
    # Subtotal = 1000, Contingency 10% = 100, Total = 1100
    assert result.data["subtotal"] == 1000.0
    assert result.data["contingency"] == 100.0
    assert result.data["total"] == 1100.0

    # Total row present
    table_rows = result.table.rows
    last_row = table_rows[-1]
    assert last_row[0] == "Total"
    assert "1,100.00" in last_row[3]

    # Cap exceeded flag
    assert len(result.flags) == 1
    assert "100.00" in result.flags[0]
    assert "10.0%" in result.flags[0]


def test_t5_risk_register_boundaries_and_sorting(registry):
    """T5 risk register sorted descending by score; ratings at boundaries 7/8 and 14/15; one flag per High risk."""
    tool = registry.get("build_risk_register")
    valid_risks = [
        {"risk": "Risk Low (6)", "likelihood": 2, "impact": 3, "mitigation": "Mitigate Low"},   # 6 -> Low
        {"risk": "Risk High (20)", "likelihood": 4, "impact": 5, "mitigation": "Mitigate High"}, # 20 -> High
        {"risk": "Risk Med (8)", "likelihood": 2, "impact": 4, "mitigation": "Mitigate Med"},    # 8 -> Medium (boundary)
        {"risk": "Risk High (15)", "likelihood": 3, "impact": 5, "mitigation": "Mitigate 15"},   # 15 -> High (boundary)
        {"risk": "Risk Med (12)", "likelihood": 3, "impact": 4, "mitigation": "Mitigate 12"},   # 12 -> Medium
    ]
    res = tool.run({"risks": valid_risks})
    rows = res.table.rows

    # Verify descending sort: scores 20, 15, 12, 8, 6
    scores = [int(r[3]) for r in rows]
    assert scores == [20, 15, 12, 8, 6]

    ratings = [r[4] for r in rows]
    assert ratings == ["High", "High", "Medium", "Medium", "Low"]

    # One flag per High risk (2 high risks)
    assert len(res.flags) == 2
    assert "High risk 'Risk High (20)'" in res.flags[0]
    assert "High risk 'Risk High (15)'" in res.flags[1]


def test_t6_invalid_args_readable_error(registry):
    """T6 invalid args ('next Monday' as date, negative quantity, empty phases) raise ToolError with a readable message naming the field."""
    # 'next Monday' as date
    timeline_tool = registry.get("build_timeline")
    with pytest.raises(ToolError) as exc_date:
        timeline_tool.run({
            "start_date": "next Monday",
            "phases": [{"name": "P1", "duration_weeks": 2}],
        })
    assert "start_date" in str(exc_date.value)

    # Empty phases
    with pytest.raises(ToolError) as exc_phases:
        timeline_tool.run({
            "start_date": "2026-11-02",
            "phases": [],
        })
    assert "phases" in str(exc_phases.value)

    # Negative quantity in budget
    budget_tool = registry.get("build_budget")
    with pytest.raises(ToolError) as exc_budget:
        budget_tool.run({
            "currency": "USD",
            "items": [{"name": "Item", "quantity": -5, "unit_cost": 10.0}],
        })
    assert "quantity" in str(exc_budget.value)


def test_t7_get_mock_data(registry):
    """T7 get_mock_data unknown topic raises ToolError listing valid topics; valid topic summary contains 'MOCK DATA'."""
    tool = registry.get("get_mock_data")

    # Unknown topic
    with pytest.raises(ToolError) as exc_info:
        tool.run({"topic": "nonexistent_topic"})
    err_msg = str(exc_info.value)
    assert "company_profile" in err_msg
    assert "team_roster" in err_msg
    assert "sales_kpis" in err_msg
    assert "it_landscape" in err_msg

    # Valid topic
    res = tool.run({"topic": "company_profile"})
    assert "MOCK DATA" in res.summary
    assert res.table is not None
    assert len(res.table.rows) > 0


def test_t8_tool_registry_describe(registry):
    """T8 ToolRegistry.describe() mentions every tool name."""
    description = registry.describe()
    for name in [
        "get_current_date",
        "build_timeline",
        "build_budget",
        "build_risk_register",
        "get_mock_data",
    ]:
        assert f"`{name}`" in description
