"""Tests for planner normalization and execution order (P1 - P4)."""

from app.planner import execution_order, normalize_plan
from app.schemas import Plan, PlanTask


def test_p1_normalize_plan_corrections():
    """P1 duplicate ids renamed, unknown deps dropped, self-dependency dropped, unknown tool nulled, each with a note."""
    plan = Plan(
        doc_type="proposal",
        title="Sample Plan",
        audience="Leadership",
        tasks=[
            PlanTask(id="dup", title="Task 1", tool="build_timeline", depends_on=["ghost_id"]),
            PlanTask(id="dup", title="Task 2", tool="fake_tool_xyz", depends_on=["dup"]),
            PlanTask(id="t3", title="Task 3", tool=None, depends_on=["t3", "t3"]),
        ],
    )
    valid_tools = ["build_timeline", "build_budget"]
    normalized, notes = normalize_plan(plan, tool_names=valid_tools, max_tasks=10)

    # Duplicate IDs were re-assigned
    assert normalized.tasks[0].id == "t1"
    assert normalized.tasks[1].id == "t2"
    assert normalized.tasks[2].id == "t3"

    # Unknown tool was nulled
    assert normalized.tasks[1].tool is None

    # Self dependency was dropped
    assert "t3" not in normalized.tasks[2].depends_on

    # Unknown dependency was dropped
    assert "ghost_id" not in normalized.tasks[0].depends_on

    # Notes generated for each correction
    note_text = " ".join(notes)
    assert "ID set to" in note_text
    assert "unknown tool" in note_text
    assert "self-dependency" in note_text
    assert "unknown dependency" in note_text


def test_p2_truncate_max_tasks():
    """P2 more than max_tasks is truncated."""
    tasks = [PlanTask(id=f"t{i}", title=f"Task {i}") for i in range(12)]
    plan = Plan(doc_type="plan", title="Big Plan", audience="Team", tasks=tasks)

    normalized, notes = normalize_plan(plan, tool_names=[], max_tasks=5)
    assert len(normalized.tasks) == 5
    assert any("truncated" in n.lower() for n in notes)


def test_p3_execution_order_dependencies():
    """P3 execution_order: a first-in-plan summary that depends on all others runs last; independent tasks keep plan order."""
    t_sum = PlanTask(id="t_sum", title="Executive Summary", depends_on=["t_tech", "t_budget"])
    t_tech = PlanTask(id="t_tech", title="Technical Architecture", depends_on=[])
    t_budget = PlanTask(id="t_budget", title="Budget Estimate", depends_on=[])

    tasks = [t_sum, t_tech, t_budget]
    ordered, had_cycle = execution_order(tasks)

    assert not had_cycle
    assert [t.id for t in ordered] == ["t_tech", "t_budget", "t_sum"]


def test_p4_execution_order_cycle():
    """P4 a dependency cycle terminates, had_cycle=True, every task appears exactly once."""
    t1 = PlanTask(id="t1", title="Task 1", depends_on=["t2"])
    t2 = PlanTask(id="t2", title="Task 2", depends_on=["t1"])
    t3 = PlanTask(id="t3", title="Task 3", depends_on=["t1"])

    tasks = [t1, t2, t3]
    ordered, had_cycle = execution_order(tasks)

    assert had_cycle is True
    assert len(ordered) == 3
    assert set(t.id for t in ordered) == {"t1", "t2", "t3"}
