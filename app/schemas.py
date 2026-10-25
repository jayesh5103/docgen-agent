"""Pydantic schemas with robust coercion for LLM outputs and API requests/responses."""

from datetime import date
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class AgentRequest(BaseModel):
    """User request for document generation."""

    request: str
    include_document_base64: bool = False

    @field_validator("request")
    @classmethod
    def validate_request(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError("Request must be at least 10 characters.")
        if len(stripped) > 4000:
            raise ValueError("Request cannot exceed 4000 characters.")
        if not any(c.isalpha() for c in stripped):
            raise ValueError("Request must contain at least one letter.")
        return stripped


class Table(BaseModel):
    """Structured table for sections."""

    caption: str | None = None
    headers: list[str]
    rows: list[list[str]]

    @model_validator(mode="before")
    @classmethod
    def coerce_table(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_headers = data.get("headers", [])
        if not raw_headers or not isinstance(raw_headers, list) or len(raw_headers) < 1:
            raise ValueError("Table must contain at least one header column.")

        headers = [str(h) for h in raw_headers]
        num_headers = len(headers)

        raw_rows = data.get("rows", [])
        coerced_rows: list[list[str]] = []

        if isinstance(raw_rows, list):
            for row in raw_rows:
                if isinstance(row, dict):
                    # Map dict row by header name
                    mapped = ["" if row.get(h) is None else str(row.get(h)) for h in headers]
                    coerced_rows.append(mapped)
                elif isinstance(row, (list, tuple)):
                    row_cells = ["" if cell is None else str(cell) for cell in row]
                    # Pad or truncate to match header length
                    if len(row_cells) < num_headers:
                        row_cells = row_cells + [""] * (num_headers - len(row_cells))
                    elif len(row_cells) > num_headers:
                        row_cells = row_cells[:num_headers]
                    coerced_rows.append(row_cells)
                elif row is not None:
                    # Single value in row
                    cell_val = str(row)
                    coerced_rows.append([cell_val] + [""] * (num_headers - 1))

        return {
            "caption": data.get("caption"),
            "headers": headers,
            "rows": coerced_rows,
        }


class Section(BaseModel):
    """Document section representation."""

    heading: str
    paragraphs: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    table: Table | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_section(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        res = dict(data)

        # Paragraphs coercion
        raw_paras = data.get("paragraphs")
        if isinstance(raw_paras, str):
            paragraphs = [raw_paras]
        elif isinstance(raw_paras, list):
            paragraphs = [str(p) for p in raw_paras if p is not None]
        else:
            paragraphs = []

        # Bullets coercion (support strings or list of dicts)
        raw_bullets = data.get("bullets")
        bullets: list[str] = []
        if isinstance(raw_bullets, str):
            bullets = [raw_bullets]
        elif isinstance(raw_bullets, list):
            for item in raw_bullets:
                if isinstance(item, dict):
                    joined = " - ".join(str(v) for v in item.values() if v is not None)
                    bullets.append(joined)
                elif item is not None:
                    bullets.append(str(item))

        # Table coercion: empty dict or empty headers/rows treated as None
        table = data.get("table")
        if isinstance(table, dict):
            if not table or (not table.get("headers") and not table.get("rows")):
                table = None
        elif table is not None and not isinstance(table, Table):
            table = None

        res["heading"] = str(data.get("heading", ""))
        res["paragraphs"] = paragraphs
        res["bullets"] = bullets
        res["table"] = table
        return res

    def as_text(self) -> str:
        """Render section as markdown-like text for prompts and linting."""
        lines = [f"## {self.heading}"]
        for p in self.paragraphs:
            lines.append(p)
        for b in self.bullets:
            lines.append(f"* {b}")
        if self.table:
            if self.table.caption:
                lines.append(f"Table: {self.table.caption}")
            lines.append(" | ".join(self.table.headers))
            for row in self.table.rows:
                lines.append(" | ".join(row))
        return "\n".join(lines)

    def word_count(self) -> int:
        """Count total words in headings, paragraphs, bullets, and table cells."""
        parts = [self.heading] + self.paragraphs + self.bullets
        if self.table:
            parts.extend(self.table.headers)
            for row in self.table.rows:
                parts.extend(row)
        text = " ".join(parts)
        return len(text.split())


class SectionDraft(Section):
    """Section with tracking metadata."""

    id: str
    source: Literal["llm", "tool"] = "llm"


class PlanTask(BaseModel):
    """Individual task in an execution plan."""

    id: str
    title: str
    description: str = ""
    tool: str | None = None
    depends_on: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_plan_task(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        tool = data.get("tool")
        if isinstance(tool, str):
            cleaned = tool.strip()
            if cleaned.lower() in ("", "none", "null"):
                tool = None
            else:
                tool = cleaned
        elif tool is not None:
            tool = str(tool).strip()

        deps = data.get("depends_on")
        if isinstance(deps, str):
            deps = [deps.strip()] if deps.strip() else []
        elif isinstance(deps, list):
            deps = [str(d).strip() for d in deps if d is not None and str(d).strip()]
        else:
            deps = []

        return {
            "id": str(data.get("id", "")).strip(),
            "title": str(data.get("title", "")).strip(),
            "description": str(data.get("description", "")).strip(),
            "tool": tool,
            "depends_on": deps,
        }


class Plan(BaseModel):
    """Autonomous execution plan generated by LLM."""

    can_fulfil: bool = True
    reason: str | None = None
    doc_type: str = "document"
    title: str = ""
    audience: str = ""
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    tasks: list[PlanTask] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_plan(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        doc_type = data.get("doc_type", "document")
        if isinstance(doc_type, str):
            doc_type = (
                doc_type.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

        assumptions = data.get("assumptions", [])
        if isinstance(assumptions, str):
            assumptions = [assumptions]
        elif isinstance(assumptions, list):
            assumptions = [str(a) for a in assumptions]

        open_questions = data.get("open_questions", [])
        if isinstance(open_questions, str):
            open_questions = [open_questions]
        elif isinstance(open_questions, list):
            open_questions = [str(q) for q in open_questions]

        return {
            "can_fulfil": bool(data.get("can_fulfil", True)),
            "reason": data.get("reason"),
            "doc_type": doc_type or "document",
            "title": str(data.get("title", "")).strip(),
            "audience": str(data.get("audience", "")).strip(),
            "assumptions": assumptions,
            "open_questions": open_questions,
            "tasks": data.get("tasks", []),
        }

    @model_validator(mode="after")
    def validate_plan(self) -> "Plan":
        if self.can_fulfil and not self.tasks:
            raise ValueError("A plan with can_fulfil=True must contain at least one task.")
        return self


class ToolResult(BaseModel):
    """Output from executing a tool."""

    summary: str
    table: Table | None = None
    flags: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


class ToolArgsReply(BaseModel):
    """Tool arguments produced by LLM."""

    arguments: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def coerce_arguments(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "arguments" not in data:
                return {"arguments": data}
            if not isinstance(data["arguments"], dict):
                return {"arguments": {}}
        return data


class TaskRecord(BaseModel):
    """Runtime tracking record for an execution task."""

    id: str
    title: str
    description: str = ""
    tool: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "failed"] = "pending"
    attempts: int = 0
    used_fallback: bool = False
    summary: str | None = None
    error: str | None = None


class Issue(BaseModel):
    """Identified issue during reflection or linting."""

    section_id: str | None = None
    kind: Literal["fix", "missing"] = "fix"
    problem: str
    fix: str
    source: Literal["lint", "critic"] = "critic"


class CriticReply(BaseModel):
    """Output from reflection critic LLM call."""

    score: int = 10
    issues: list[Issue] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_critic(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_score = data.get("score", 10)
        try:
            if isinstance(raw_score, (int, float, str)):
                score_val = int(round(float(raw_score)))
            else:
                score_val = 10
        except Exception:
            score_val = 10

        # Clamp between 1 and 10
        clamped_score = max(1, min(10, score_val))

        raw_issues = data.get("issues", [])
        coerced_issues: list[dict[str, Any]] = []
        if isinstance(raw_issues, list):
            for iss in raw_issues:
                if isinstance(iss, dict):
                    sec_id = iss.get("section_id")
                    if isinstance(sec_id, str):
                        sec_id = sec_id.strip() or None
                    kind = str(iss.get("kind", "fix")).lower()
                    if kind not in ("fix", "missing"):
                        kind = "fix"
                    coerced_issues.append({
                        "section_id": sec_id,
                        "kind": kind,
                        "problem": str(iss.get("problem", "")),
                        "fix": str(iss.get("fix", "")),
                        "source": iss.get("source", "critic"),
                    })

        return {
            "score": clamped_score,
            "issues": coerced_issues,
        }


class ReflectionReport(BaseModel):
    """Summary of the reflection and self-correction process."""

    initial_score: int | None = None
    final_score: int | None = None
    issues_found: int = 0
    issues: list[str] = Field(default_factory=list)
    revisions: list[str] = Field(default_factory=list)
    rounds: int = 0
    remaining_issues: list[str] = Field(default_factory=list)
    critic_error: str | None = None


class DocumentSpec(BaseModel):
    """Complete document specification assembled for rendering."""

    title: str
    doc_type: str
    audience: str
    generated_on: date
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    sections: list[SectionDraft] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Metadata regarding the generated .docx file."""

    filename: str
    download_url: str
    size_bytes: int
    base64: str | None = None


class Metrics(BaseModel):
    """Execution performance metrics."""

    llm_calls: int = 0
    elapsed_seconds: float = 0.0


class PlanResponse(BaseModel):
    """Plan information presented in the API response."""

    title: str
    doc_type: str
    audience: str
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    tasks: list[TaskRecord] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Top-level response returned by POST /agent."""

    request_id: str
    status: Literal["completed", "completed_with_warnings", "rejected"]
    message: str
    plan: PlanResponse | None = None
    reflection: ReflectionReport | None = None
    document: DocumentInfo | None = None
    warnings: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
