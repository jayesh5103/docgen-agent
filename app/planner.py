"""Plan normalization, cycle detection, and topological execution ordering."""

from app.schemas import Plan, PlanTask


def normalize_plan(
    plan: Plan,
    tool_names: list[str],
    max_tasks: int = 10,
) -> tuple[Plan, list[str]]:
    """Normalize plan tasks: cap count, fix IDs, strip bad tools and invalid dependencies."""
    notes: list[str] = []

    tasks = [t.model_copy() for t in plan.tasks]
    if len(tasks) > max_tasks:
        notes.append(f"Plan truncated from {len(tasks)} tasks to {max_tasks} tasks.")
        tasks = tasks[:max_tasks]

    # Check for empty or duplicate IDs
    seen_ids: set[str] = set()
    has_bad_ids = False
    for t in tasks:
        if not t.id or t.id in seen_ids:
            has_bad_ids = True
            break
        seen_ids.add(t.id)

    normalized_tasks: list[PlanTask] = []
    if has_bad_ids:
        # Regenerate all IDs as t1, t2, ...
        old_to_new: dict[str, str] = {}
        for idx, t in enumerate(tasks):
            new_id = f"t{idx + 1}"
            if t.id and t.id not in old_to_new:
                old_to_new[t.id] = new_id
            notes.append(f"Task '{t.title or new_id}' ID set to '{new_id}'.")
            task_copy = t.model_copy(update={"id": new_id})
            normalized_tasks.append(task_copy)

        # Remap dependencies: remap known IDs to new IDs, keep unknown IDs so dependency cleaner catches them
        for t in normalized_tasks:
            t.depends_on = [old_to_new.get(dep, dep) for dep in t.depends_on]
    else:
        normalized_tasks = [t.model_copy() for t in tasks]

    # Clean unknown tools
    for t in normalized_tasks:
        if t.tool and t.tool not in tool_names:
            notes.append(f"Task '{t.id}' had unknown tool '{t.tool}'; changed to None.")
            t.tool = None

    # Clean dependencies: remove self-dependencies, unknown IDs, and duplicates
    all_valid_ids = {t.id for t in normalized_tasks}
    for t in normalized_tasks:
        cleaned_deps: list[str] = []
        seen_deps: set[str] = set()
        for dep in t.depends_on:
            if dep == t.id:
                notes.append(f"Task '{t.id}' had self-dependency; dropped.")
            elif dep not in all_valid_ids:
                notes.append(f"Task '{t.id}' had unknown dependency '{dep}'; dropped.")
            elif dep in seen_deps:
                notes.append(f"Task '{t.id}' had duplicate dependency '{dep}'; dropped.")
            else:
                seen_deps.add(dep)
                cleaned_deps.append(dep)
        t.depends_on = cleaned_deps

    updated_plan = plan.model_copy(update={"tasks": normalized_tasks})
    return updated_plan, notes


def execution_order(tasks: list[PlanTask]) -> tuple[list[PlanTask], bool]:
    """Determine execution order following dependencies while preserving plan order when possible."""
    remaining = list(tasks)
    completed_ids: set[str] = set()
    ordered: list[PlanTask] = []
    had_cycle = False

    while remaining:
        ready_idx = -1
        for i, task in enumerate(remaining):
            if all(dep in completed_ids for dep in task.depends_on):
                ready_idx = i
                break

        if ready_idx != -1:
            ready_task = remaining.pop(ready_idx)
            ordered.append(ready_task)
            completed_ids.add(ready_task.id)
        else:
            # Dependency cycle or deadlock: break by taking the first remaining task in plan order
            had_cycle = True
            forced_task = remaining.pop(0)
            ordered.append(forced_task)
            completed_ids.add(forced_task.id)

    return ordered, had_cycle
