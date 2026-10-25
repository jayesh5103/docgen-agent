"""Autonomous Document Agent orchestrator."""

import base64
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import re
import time
from typing import Any, Callable
import uuid

from app.config import Settings
from app.docx_builder import build_docx
from app.errors import AgentError, LLMOutputError, LLMUnavailableError, ToolError
from app.llm import CountingLLM, LLM, complete_json
from app.planner import execution_order, normalize_plan
from app.prompts import (
    build_plan_system_prompt,
    build_plan_user_prompt,
    build_reflect_system_prompt,
    build_reflect_user_prompt,
    build_revise_section_system_prompt,
    build_revise_section_user_prompt,
    build_tool_args_system_prompt,
    build_tool_args_user_prompt,
    build_write_section_system_prompt,
    build_write_section_user_prompt,
)
from app.reflection import format_for_review, lint_document
from app.schemas import (
    AgentRequest,
    AgentResponse,
    CriticReply,
    DocumentInfo,
    DocumentSpec,
    Issue,
    Metrics,
    Plan,
    PlanResponse,
    PlanTask,
    ReflectionReport,
    Section,
    SectionDraft,
    Table,
    TaskRecord,
    ToolArgsReply,
)
from app.tools import ToolRegistry, default_registry


@dataclass
class RunState:
    """Scoped per-request state to maintain thread safety across concurrent requests."""
    request_id: str
    counting_llm: CountingLLM
    records: dict[str, TaskRecord] = field(default_factory=dict)
    sections: dict[str, SectionDraft] = field(default_factory=dict)
    original_tool_tables: dict[str, Table | None] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


def _slugify(text: str) -> str:
    """Produce filesystem-safe kebab-case slug from title."""
    cleaned = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "-", cleaned).strip("-")
    return slug or "document"


class DocumentAgent:
    """Hand-written orchestrator coordinating planning, tool execution, writing, reflection, and rendering."""

    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry | None = None,
        settings: Settings | None = None,
        builder: Any = build_docx,
        today: Any = date.today,
    ):
        self.llm = llm
        self.today = today
        self.tools = tools or default_registry(today=self.today)
        self.settings = settings or Settings()
        self.builder = builder

    def run(self, request_input: str | AgentRequest) -> AgentResponse:
        """Execute autonomous document generation pipeline for a single business request."""
        # Normalize request input
        if isinstance(request_input, str):
            request_dto = AgentRequest(request=request_input)
        else:
            request_dto = request_input

        # Initialize isolated per-request state
        request_id = uuid.uuid4().hex
        counting_llm = CountingLLM(self.llm)
        state = RunState(request_id=request_id, counting_llm=counting_llm)

        # ---------------------------------------------------------------------
        # 1. PLAN & NORMALIZE
        # ---------------------------------------------------------------------
        state.trace.append("Beginning autonomous planning step.")
        plan_sys = build_plan_system_prompt(self.tools.describe(), self.settings.max_tasks)
        plan_user = build_plan_user_prompt(request_dto.request, self.today())

        plan = complete_json(
            counting_llm,
            "plan",
            plan_sys,
            plan_user,
            Plan,
            retries=self.settings.json_retries,
        )

        plan, norm_notes = normalize_plan(plan, self.tools.names(), self.settings.max_tasks)
        for note in norm_notes:
            state.trace.append(f"Planner note: {note}")

        # Check if request cannot be fulfilled
        if not plan.can_fulfil:
            state.trace.append(f"Request rejected by planner: {plan.reason}")
            elapsed = round(time.time() - state.start_time, 2)
            return AgentResponse(
                request_id=state.request_id,
                status="rejected",
                message=plan.reason or "Request cannot be fulfilled.",
                plan=PlanResponse(
                    title=plan.title or "Rejected Request",
                    doc_type=plan.doc_type,
                    audience=plan.audience,
                    assumptions=plan.assumptions,
                    open_questions=plan.open_questions,
                    tasks=[],
                ),
                reflection=None,
                document=None,
                warnings=state.warnings,
                trace=state.trace,
                metrics=Metrics(llm_calls=counting_llm.call_count, elapsed_seconds=elapsed),
            )

        # Initialize TaskRecords in plan order
        for t in plan.tasks:
            state.records[t.id] = TaskRecord(
                id=t.id,
                title=t.title,
                description=t.description,
                tool=t.tool,
                depends_on=t.depends_on,
                status="pending",
            )

        # ---------------------------------------------------------------------
        # 2. EXECUTION IN DEPENDENCY ORDER
        # ---------------------------------------------------------------------
        ordered_tasks, had_cycle = execution_order(plan.tasks)
        if had_cycle:
            state.warnings.append("Dependency cycle detected during planning; cycle was broken deterministically.")
            state.trace.append("Dependency cycle broken.")

        for task in ordered_tasks:
            record = state.records[task.id]
            record.status = "running"

            if task.tool:
                self._execute_tool_task(state, task, request_dto.request, plan.assumptions)
            else:
                draft = self._execute_write_task(state, task, request_dto.request, plan.assumptions)
                if draft:
                    state.sections[task.id] = draft

        # ---------------------------------------------------------------------
        # 3. ASSEMBLE SECTIONS IN PLAN ORDER
        # ---------------------------------------------------------------------
        assembled_sections: list[SectionDraft] = []
        for t in plan.tasks:
            if t.id in state.sections:
                assembled_sections.append(state.sections[t.id])

        if not assembled_sections:
            raise AgentError("Failed to produce any document sections.")

        spec = DocumentSpec(
            title=plan.title,
            doc_type=plan.doc_type,
            audience=plan.audience,
            generated_on=self.today(),
            assumptions=plan.assumptions,
            open_questions=plan.open_questions,
            sections=assembled_sections,
        )

        # ---------------------------------------------------------------------
        # 4. REFLECT & SELF-CORRECT
        # ---------------------------------------------------------------------
        state.trace.append("Entering reflection and self-correction loop.")
        reflection_report = self._reflect_and_revise(state, spec, request_dto.request)

        # ---------------------------------------------------------------------
        # 5. RENDER TO DOCX
        # ---------------------------------------------------------------------
        out_dir = Path(self.settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_slugify(spec.title)}-{state.request_id}.docx"
        target_docx_path = out_dir / filename

        self.builder(spec, target_docx_path)
        file_size = target_docx_path.stat().st_size

        doc_base64: str | None = None
        if request_dto.include_document_base64:
            doc_base64 = base64.b64encode(target_docx_path.read_bytes()).decode("utf-8")

        doc_info = DocumentInfo(
            filename=filename,
            download_url=f"/documents/{state.request_id}",
            size_bytes=file_size,
            base64=doc_base64,
        )

        # ---------------------------------------------------------------------
        # 6. BUILD SUMMARY MESSAGE & RESPOND
        # ---------------------------------------------------------------------
        total_tasks = len(plan.tasks)
        total_sections = len(spec.sections)
        msg_parts = [
            f"Created '{spec.title}' ({spec.doc_type}) with {total_sections} sections from {total_tasks} tasks."
        ]
        if spec.assumptions:
            msg_parts.append(
                f"{len(spec.assumptions)} assumptions were made and are listed in the document."
            )
        if state.flags:
            msg_parts.append("Tool alerts: " + " | ".join(state.flags))
        if reflection_report:
            score_txt = f"Self-review {reflection_report.final_score}/10"
            rev_count = len(reflection_report.revisions)
            rev_txt = f" after {rev_count} revision{'s' if rev_count != 1 else ''}" if rev_count > 0 else ""
            msg_parts.append(f"{score_txt}{rev_txt}.")
        if state.warnings:
            msg_parts.append(f"Completed with {len(state.warnings)} warning(s).")

        summary_message = " ".join(msg_parts)

        final_status = "completed" if not state.warnings else "completed_with_warnings"
        elapsed_seconds = round(time.time() - state.start_time, 2)

        ordered_records = [state.records[t.id] for t in plan.tasks if t.id in state.records]
        plan_response = PlanResponse(
            title=plan.title,
            doc_type=plan.doc_type,
            audience=plan.audience,
            assumptions=plan.assumptions,
            open_questions=plan.open_questions,
            tasks=ordered_records,
        )

        return AgentResponse(
            request_id=state.request_id,
            status=final_status,
            message=summary_message,
            plan=plan_response,
            reflection=reflection_report,
            document=doc_info,
            warnings=state.warnings,
            trace=state.trace,
            metrics=Metrics(llm_calls=counting_llm.call_count, elapsed_seconds=elapsed_seconds),
        )

    # -------------------------------------------------------------------------
    # Internal Task Execution Methods
    # -------------------------------------------------------------------------

    def _execute_tool_task(
        self,
        state: RunState,
        task: PlanTask,
        request: str,
        assumptions: list[str],
    ) -> None:
        """Run tool task with argument generation retry loop and graceful write-task fallback."""
        record = state.records[task.id]
        tool_def = self.tools.get(task.tool or "")

        if not tool_def:
            record.status = "failed"
            record.error = f"Tool '{task.tool}' is not available."
            state.warnings.append(f"Task '{task.id}' references unavailable tool '{task.tool}'.")
            return

        success = False
        last_error: str | None = None
        prev_args: dict | None = None
        max_attempts = 1 + self.settings.tool_retries

        completed_summaries = [
            f"{s.heading}: {s.paragraphs[0][:120]}..."
            for s in state.sections.values()
            if s.paragraphs
        ]

        for attempt in range(1, max_attempts + 1):
            record.attempts = attempt
            state.trace.append(f"Task '{task.id}' attempt {attempt} generating arguments for '{task.tool}'.")

            tool_prompt = build_tool_args_user_prompt(
                task_title=task.title,
                tool_name=task.tool or "",
                tool_desc=tool_def.description,
                example_args=tool_def.example,
                user_request=request,
                assumptions=assumptions,
                completed_summaries=completed_summaries,
                today=self.today(),
                previous_attempt_error=last_error,
                previous_arguments=prev_args,
            )

            try:
                args_reply = complete_json(
                    state.counting_llm,
                    "tool_args",
                    build_tool_args_system_prompt(),
                    tool_prompt,
                    ToolArgsReply,
                    retries=0,
                )
                prev_args = args_reply.arguments
                result = tool_def.run(args_reply.arguments)

                # Tool run succeeded!
                record.status = "done"
                record.summary = result.summary
                state.flags.extend(result.flags)

                bullets = [f"Attention: {f}" for f in result.flags]
                draft = SectionDraft(
                    id=task.id,
                    heading=task.title,
                    paragraphs=[result.summary],
                    bullets=bullets,
                    table=result.table,
                    source="tool",
                )
                state.sections[task.id] = draft
                state.original_tool_tables[task.id] = result.table
                state.trace.append(f"Task '{task.id}' tool execution succeeded.")
                success = True
                break

            except ToolError as exc:
                last_error = str(exc)
                state.trace.append(f"{task.id} args rejected ({last_error}), retrying")
            except LLMOutputError as exc:
                last_error = str(exc)
                state.trace.append(f"{task.id} args generation failed ({last_error}), retrying")

        if not success:
            # Fallback to write task
            record.used_fallback = True
            warn_msg = f"Task '{task.id}' ({task.title}) tool '{task.tool}' failed; used fallback."
            state.warnings.append(warn_msg)
            state.trace.append(f"Task '{task.id}' switching to LLM writing fallback.")

            draft = self._execute_write_task(state, task, request, assumptions, is_fallback=True)
            if draft:
                state.sections[task.id] = draft
                record.status = "done"
                record.summary = "Generated using fallback writing."
            else:
                record.status = "failed"
                record.error = last_error

    def _execute_write_task(
        self,
        state: RunState,
        task: PlanTask,
        request: str,
        assumptions: list[str],
        is_fallback: bool = False,
    ) -> SectionDraft | None:
        """Run standard write task with contextual digest injection."""
        record = state.records[task.id]
        if not is_fallback:
            record.attempts += 1

        depended = [state.sections[d] for d in task.depends_on if d in state.sections]
        digests = [
            f"{s.id}: {s.heading} ({s.word_count()} words)"
            for s in state.sections.values()
        ]

        sys_prompt = build_write_section_system_prompt()
        user_prompt = build_write_section_user_prompt(
            section_id=task.id,
            task_title=task.title,
            task_desc=task.description or task.title,
            user_request=request,
            assumptions=assumptions,
            depended_sections=depended,
            other_section_digests=digests,
            tool_flags=state.flags,
            today=self.today(),
        )

        try:
            sec = complete_json(
                state.counting_llm,
                "write_section",
                sys_prompt,
                user_prompt,
                Section,
                retries=self.settings.json_retries,
            )
            heading = sec.heading.strip() or task.title
            draft = SectionDraft(
                id=task.id,
                heading=heading,
                paragraphs=sec.paragraphs,
                bullets=sec.bullets,
                table=sec.table,
                source="llm",
            )
            record.status = "done"
            record.summary = f"Written section with {draft.word_count()} words."
            state.trace.append(f"Task '{task.id}' completed.")
            return draft

        except LLMOutputError as exc:
            record.status = "failed"
            record.error = str(exc)
            warn_msg = f"Write task '{task.id}' ({task.title}) failed: {exc}"
            state.warnings.append(warn_msg)
            state.trace.append(f"Task '{task.id}' failed.")
            return None

    # -------------------------------------------------------------------------
    # Reflection & Revision Engine
    # -------------------------------------------------------------------------

    def _reflect_and_revise(
        self,
        state: RunState,
        spec: DocumentSpec,
        request: str,
    ) -> ReflectionReport:
        """Perform multi-round reflection, deterministic linting, and targeted section fixes."""
        report = ReflectionReport()
        initial_recorded = False

        for round_idx in range(self.settings.max_reflection_rounds + 1):
            report.rounds = round_idx
            lint_issues = lint_document(spec)
            doc_text = format_for_review(spec)

            critic_issues: list[Issue] = []
            critic_score = 10

            try:
                sys_prompt = build_reflect_system_prompt()
                user_prompt = build_reflect_user_prompt(
                    user_request=request,
                    assumptions=spec.assumptions,
                    tool_flags=state.flags,
                    document_text=doc_text,
                    today=self.today(),
                )
                critic_reply = complete_json(
                    state.counting_llm,
                    "reflect",
                    sys_prompt,
                    user_prompt,
                    CriticReply,
                    retries=self.settings.json_retries,
                )
                critic_score = critic_reply.score
                valid_ids = {s.id for s in spec.sections}

                for iss in critic_reply.issues:
                    # Coerce unknown section IDs to kind="missing"
                    if iss.section_id and iss.section_id not in valid_ids:
                        critic_issues.append(
                            Issue(
                                section_id=None,
                                kind="missing",
                                problem=iss.problem,
                                fix=iss.fix,
                                source="critic",
                            )
                        )
                    else:
                        critic_issues.append(iss)

            except LLMOutputError as exc:
                report.critic_error = str(exc)
                state.warnings.append(f"Critic call failed: {exc}; continuing with lint checks only.")
                state.trace.append(f"Critic call failed: {exc}")
                critic_score = 7

            all_issues = lint_issues + critic_issues

            if not initial_recorded:
                report.initial_score = critic_score
                report.issues_found = len(all_issues)
                report.issues = [f"[{i.kind}] {i.problem}" for i in all_issues]
                initial_recorded = True

            report.final_score = critic_score

            # Stop if clean and quality threshold met
            if not all_issues and critic_score >= self.settings.quality_threshold:
                state.trace.append(f"Self-check passed with score {critic_score}/10.")
                break

            # Stop if reached maximum rounds
            if round_idx == self.settings.max_reflection_rounds:
                report.remaining_issues = [f"[{i.kind}] {i.problem}" for i in all_issues]
                if report.remaining_issues:
                    state.warnings.append(
                        f"Reflection concluded with {len(report.remaining_issues)} remaining issue(s)."
                    )
                state.trace.append("Reflection reached maximum configured rounds.")
                break

            # Apply up to 3 fixes
            fixes_applied = 0
            fix_by_section: dict[str, list[Issue]] = {}
            missing_issues: list[Issue] = []

            for iss in all_issues:
                if iss.kind == "fix" and iss.section_id:
                    fix_by_section.setdefault(iss.section_id, []).append(iss)
                else:
                    missing_issues.append(iss)

            # 1. Section revisions
            for sec_id, sec_issues in fix_by_section.items():
                if fixes_applied >= 3:
                    break
                target_sec = next((s for s in spec.sections if s.id == sec_id), None)
                if not target_sec:
                    continue

                problems_text = "; ".join(i.problem for i in sec_issues)
                sys_prompt = build_revise_section_system_prompt()
                user_prompt = build_revise_section_user_prompt(
                    section_id=sec_id,
                    current_heading=target_sec.heading,
                    current_section_json=target_sec.model_dump_json(),
                    issues_text=problems_text,
                    today=self.today(),
                )

                try:
                    revised = complete_json(
                        state.counting_llm,
                        "revise_section",
                        sys_prompt,
                        user_prompt,
                        Section,
                        retries=self.settings.json_retries,
                    )
                    # Maintain original heading
                    revised.heading = target_sec.heading
                    # Restore authoritative table for tool sections
                    if target_sec.source == "tool":
                        revised.table = state.original_tool_tables.get(sec_id)

                    new_draft = SectionDraft(
                        id=sec_id,
                        heading=target_sec.heading,
                        paragraphs=revised.paragraphs,
                        bullets=revised.bullets,
                        table=revised.table,
                        source=target_sec.source,
                    )

                    idx = next(i for i, s in enumerate(spec.sections) if s.id == sec_id)
                    spec.sections[idx] = new_draft
                    state.sections[sec_id] = new_draft

                    rev_note = f"Revised '{target_sec.heading}': addressed {problems_text[:70]}"
                    report.revisions.append(rev_note)
                    state.trace.append(rev_note)
                    fixes_applied += 1

                except LLMOutputError as exc:
                    state.warnings.append(f"Failed to revise '{target_sec.heading}': {exc}")

            # 2. Missing sections
            missing_idx = 1
            for miss in missing_issues:
                if fixes_applied >= 3:
                    break
                new_sec_id = f"x{missing_idx}"
                missing_idx += 1
                new_heading = miss.problem[:40].strip() or "Additional Requirements"

                sys_prompt = build_write_section_system_prompt()
                user_prompt = build_write_section_user_prompt(
                    section_id=new_sec_id,
                    task_title=new_heading,
                    task_desc=f"Address missing requirement: {miss.problem}. Guidance: {miss.fix}",
                    user_request=request,
                    assumptions=spec.assumptions,
                    depended_sections=spec.sections,
                    other_section_digests=[],
                    tool_flags=state.flags,
                    today=self.today(),
                )

                try:
                    sec = complete_json(
                        state.counting_llm,
                        "write_section",
                        sys_prompt,
                        user_prompt,
                        Section,
                        retries=self.settings.json_retries,
                    )
                    new_draft = SectionDraft(
                        id=new_sec_id,
                        heading=sec.heading or new_heading,
                        paragraphs=sec.paragraphs,
                        bullets=sec.bullets,
                        table=sec.table,
                        source="llm",
                    )
                    spec.sections.append(new_draft)
                    state.sections[new_sec_id] = new_draft

                    rev_note = f"Added missing section '{new_draft.heading}': {miss.problem[:70]}"
                    report.revisions.append(rev_note)
                    state.trace.append(rev_note)
                    fixes_applied += 1

                except LLMOutputError as exc:
                    state.warnings.append(f"Failed to add section for '{miss.problem}': {exc}")

        return report
