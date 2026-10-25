"""Deterministic calculation and retrieval tools with schema validation."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable
from pydantic import BaseModel, Field, ValidationError

from app.errors import ToolError
from app.schemas import Table, ToolResult


# -----------------------------------------------------------------------------
# Tool Arguments Models
# -----------------------------------------------------------------------------

class GetCurrentDateArgs(BaseModel):
    """Arguments for get_current_date tool."""
    pass


class PhaseSpec(BaseModel):
    """Specification for a project phase."""
    name: str = Field(min_length=1)
    duration_weeks: int = Field(gt=0)
    parallel_with_previous: bool = False


class BuildTimelineArgs(BaseModel):
    """Arguments for build_timeline tool."""
    start_date: date | None = None
    deadline: date | None = None
    phases: list[PhaseSpec] = Field(min_length=1)


class BudgetItem(BaseModel):
    """Single line item in a project budget."""
    name: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit_cost: float = Field(ge=0)


class BuildBudgetArgs(BaseModel):
    """Arguments for build_budget tool."""
    currency: str = "USD"
    items: list[BudgetItem] = Field(min_length=1)
    contingency_pct: float = Field(default=10.0, ge=0.0, le=50.0)
    budget_cap: float | None = None


class RiskItem(BaseModel):
    """Single item in a risk register."""
    risk: str = Field(min_length=1)
    likelihood: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    mitigation: str = Field(min_length=1)


class BuildRiskRegisterArgs(BaseModel):
    """Arguments for build_risk_register tool."""
    risks: list[RiskItem] = Field(min_length=1)


class GetMockDataArgs(BaseModel):
    """Arguments for get_mock_data tool."""
    topic: str


# -----------------------------------------------------------------------------
# Tool Class and Execution
# -----------------------------------------------------------------------------

@dataclass
class Tool:
    """Tool definition holding metadata, schema, and execution function."""
    name: str
    description: str
    args_model: type[BaseModel]
    example: dict[str, Any]
    func: Callable[[Any], ToolResult]

    def run(self, raw_args: dict[str, Any]) -> ToolResult:
        """Validate raw arguments and execute tool."""
        try:
            validated = self.args_model.model_validate(raw_args)
        except ValidationError as exc:
            errors = []
            for err in exc.errors():
                loc = ".".join(str(p) for p in err.get("loc", []))
                msg = err.get("msg", "")
                if loc:
                    errors.append(f"{loc}: {msg}")
                else:
                    errors.append(msg)
            raise ToolError("; ".join(errors)) from exc
        except Exception as exc:
            raise ToolError(f"Invalid arguments: {exc}") from exc

        try:
            return self.func(validated)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Tool execution failed: {exc}") from exc


class ToolRegistry:
    """Registry managing available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def describe(self) -> str:
        """Format tools description for system prompts."""
        blocks = []
        for tool in self._tools.values():
            block = (
                f"- Tool: `{tool.name}`\n"
                f"  Description: {tool.description}\n"
                f"  Example Arguments: {tool.example}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)


# -----------------------------------------------------------------------------
# Tool Implementations
# -----------------------------------------------------------------------------

def _make_get_current_date(today_fn: Callable[[], date]) -> Tool:
    def execute(_args: GetCurrentDateArgs) -> ToolResult:
        today_val = today_fn()
        iso = today_val.isoformat()
        return ToolResult(
            summary=f"Current date is {iso}.",
            table=None,
            flags=[],
            data={"current_date": iso},
        )

    return Tool(
        name="get_current_date",
        description="Returns the current reference date in ISO format.",
        args_model=GetCurrentDateArgs,
        example={},
        func=execute,
    )


def _make_build_timeline(today_fn: Callable[[], date]) -> Tool:
    def execute(args: BuildTimelineArgs) -> ToolResult:
        start_date = args.start_date or today_fn()
        phases_data = []

        for i, phase in enumerate(args.phases):
            duration_days = phase.duration_weeks * 7
            if i == 0 or not phase.parallel_with_previous:
                # Sequential phase starts after latest end date of previous phases
                if phases_data:
                    latest_end = max(p["end"] for p in phases_data)
                    p_start = latest_end + timedelta(days=1)
                else:
                    p_start = start_date
            else:
                # Parallel phase starts on previous phase's start date
                prev_start = phases_data[-1]["start"]
                p_start = prev_start

            p_end = p_start + timedelta(days=duration_days - 1)
            phases_data.append({
                "name": phase.name,
                "start": p_start,
                "end": p_end,
                "weeks": phase.duration_weeks,
            })

        project_start = min(p["start"] for p in phases_data)
        project_end = max(p["end"] for p in phases_data)
        total_days = (project_end - project_start).days + 1
        total_weeks = (total_days + 6) // 7

        table_rows = [
            [
                p["name"],
                p["start"].isoformat(),
                p["end"].isoformat(),
                str(p["weeks"]),
            ]
            for p in phases_data
        ]

        flags = []
        if args.deadline:
            if project_end > args.deadline:
                days_over = (project_end - args.deadline).days
                flags.append(
                    f"Timeline exceeds deadline of {args.deadline.isoformat()} by {days_over} days "
                    f"(scheduled end: {project_end.isoformat()})."
                )
                summary = (
                    f"Project runs {project_start.isoformat()} to {project_end.isoformat()} ({total_weeks} weeks), "
                    f"exceeding deadline by {days_over} days."
                )
            else:
                summary = (
                    f"Project runs {project_start.isoformat()} to {project_end.isoformat()} ({total_weeks} weeks), "
                    f"successfully meeting the deadline of {args.deadline.isoformat()}."
                )
        else:
            summary = (
                f"Project runs {project_start.isoformat()} to {project_end.isoformat()} ({total_weeks} weeks)."
            )

        table = Table(
            caption="Project Delivery Schedule",
            headers=["Phase", "Start", "End", "Weeks"],
            rows=table_rows,
        )

        return ToolResult(
            summary=summary,
            table=table,
            flags=flags,
            data={
                "project_start": project_start.isoformat(),
                "project_end": project_end.isoformat(),
                "total_weeks": total_weeks,
            },
        )

    return Tool(
        name="build_timeline",
        description="Calculates project phase dates, duration, milestones, and flags deadline overruns.",
        args_model=BuildTimelineArgs,
        example={
            "start_date": "2026-11-02",
            "deadline": "2027-01-24",
            "phases": [
                {"name": "Design & Discovery", "duration_weeks": 2, "parallel_with_previous": False},
                {"name": "Core Implementation", "duration_weeks": 6, "parallel_with_previous": False},
            ],
        },
        func=execute,
    )


def _make_build_budget() -> Tool:
    def execute(args: BuildBudgetArgs) -> ToolResult:
        currency = args.currency.upper()
        line_items = []
        subtotal = 0.0

        for item in args.items:
            line_total = round(item.quantity * item.unit_cost, 2)
            subtotal += line_total
            line_items.append([
                item.name,
                f"{item.quantity:g}",
                f"{currency} {item.unit_cost:,.2f}",
                f"{currency} {line_total:,.2f}",
            ])

        subtotal = round(subtotal, 2)
        contingency = round(subtotal * (args.contingency_pct / 100.0), 2)
        total = round(subtotal + contingency, 2)

        rows = list(line_items)
        rows.append(["Subtotal", "", "", f"{currency} {subtotal:,.2f}"])
        rows.append([f"Contingency ({args.contingency_pct:g}%)", "", "", f"{currency} {contingency:,.2f}"])
        rows.append(["Total", "", "", f"{currency} {total:,.2f}"])

        flags = []
        if args.budget_cap is not None:
            if total > args.budget_cap:
                overage = round(total - args.budget_cap, 2)
                pct_over = round((overage / args.budget_cap) * 100.0, 1)
                flags.append(
                    f"Budget of {currency} {total:,.2f} exceeds cap of {currency} {args.budget_cap:,.2f} "
                    f"by {currency} {overage:,.2f} ({pct_over}% over)."
                )

        summary = f"Total estimated budget is {currency} {total:,.2f} including {currency} {contingency:,.2f} contingency."
        if flags:
            summary += f" Attention: {flags[0]}"

        table = Table(
            caption="Financial Budget and Cost Breakdown",
            headers=["Item", "Quantity", "Unit Cost", "Total"],
            rows=rows,
        )

        return ToolResult(
            summary=summary,
            table=table,
            flags=flags,
            data={
                "currency": currency,
                "subtotal": subtotal,
                "contingency": contingency,
                "total": total,
                "budget_cap": args.budget_cap,
            },
        )

    return Tool(
        name="build_budget",
        description="Calculates itemized project costs, contingency, subtotal, total, and flags budget cap overages.",
        args_model=BuildBudgetArgs,
        example={
            "currency": "USD",
            "contingency_pct": 10.0,
            "budget_cap": 50000.0,
            "items": [
                {"name": "Senior Cloud Engineer", "quantity": 160, "unit_cost": 150.0},
                {"name": "Cloud Infrastructure", "quantity": 1, "unit_cost": 12000.0},
            ],
        },
        func=execute,
    )


def _make_build_risk_register() -> Tool:
    def execute(args: BuildRiskRegisterArgs) -> ToolResult:
        scored_risks = []
        high_risks = []
        med_risks = []
        low_risks = []

        for r in args.risks:
            score = r.likelihood * r.impact
            if score >= 15:
                rating = "High"
                high_risks.append(r)
            elif score >= 8:
                rating = "Medium"
                med_risks.append(r)
            else:
                rating = "Low"
                low_risks.append(r)

            scored_risks.append({
                "risk": r.risk,
                "likelihood": r.likelihood,
                "impact": r.impact,
                "score": score,
                "rating": rating,
                "mitigation": r.mitigation,
            })

        # Sort descending by score
        scored_risks.sort(key=lambda x: x["score"], reverse=True)

        flags = [
            f"High risk '{r['risk']}' (score {r['score']}): {r['mitigation']}"
            for r in scored_risks
            if r["rating"] == "High"
        ]

        rows = [
            [
                item["risk"],
                str(item["likelihood"]),
                str(item["impact"]),
                str(item["score"]),
                item["rating"],
                item["mitigation"],
            ]
            for item in scored_risks
        ]

        summary = (
            f"Identified {len(scored_risks)} risks "
            f"({len(high_risks)} High, {len(med_risks)} Medium, {len(low_risks)} Low). "
            f"Top risk: {scored_risks[0]['risk']} (Score {scored_risks[0]['score']})."
        )

        table = Table(
            caption="Risk Assessment Matrix",
            headers=["Risk Description", "Likelihood", "Impact", "Score", "Rating", "Mitigation Strategy"],
            rows=rows,
        )

        return ToolResult(
            summary=summary,
            table=table,
            flags=flags,
            data={
                "risk_count": len(scored_risks),
                "high_count": len(high_risks),
                "medium_count": len(med_risks),
                "low_count": len(low_risks),
            },
        )

    return Tool(
        name="build_risk_register",
        description="Assesses risks by likelihood and impact (1-5), classifies ratings (High/Medium/Low), and flags High risks.",
        args_model=BuildRiskRegisterArgs,
        example={
            "risks": [
                {
                    "risk": "Legacy data schema inconsistencies",
                    "likelihood": 4,
                    "impact": 4,
                    "mitigation": "Run automated data audit scripts during Week 2",
                }
            ]
        },
        func=execute,
    )


MOCK_TOPICS = {
    "company_profile": {
        "headers": ["Attribute", "Value"],
        "rows": [
            ["Company Name", "RetailGenius Inc."],
            ["Founded", "2018"],
            ["Headquarters", "Chicago, IL"],
            ["Employees", "450"],
            ["Primary Market", "E-Commerce"],
        ],
    },
    "team_roster": {
        "headers": ["Role", "Lead", "Allocation", "Location"],
        "rows": [
            ["Project Sponsor", "VP of Operations", "10%", "US HQ"],
            ["Tech Lead", "Principal Architect", "100%", "Remote"],
            ["ML Engineer", "Senior Data Scientist", "100%", "Remote"],
            ["QA Engineer", "Senior QA Specialist", "50%", "EU"],
        ],
    },
    "sales_kpis": {
        "headers": ["Metric", "Current", "Target", "YoY Growth"],
        "rows": [
            ["Annual Recurring Revenue", "$12.4M", "$18.0M", "+28%"],
            ["Customer Support Tickets", "45,000 / mo", "20,000 / mo", "-55% target"],
            ["Average Resolution Time", "4.2 hours", "1.5 hours", "-64% target"],
            ["CSAT Score", "3.8 / 5.0", "4.5 / 5.0", "+18%"],
        ],
    },
    "it_landscape": {
        "headers": ["System", "Technology", "Hosting", "Status"],
        "rows": [
            ["Core ERP", "SAP ECC 6.0", "On-Premises Datacenter", "Legacy"],
            ["CRM", "Salesforce Service Cloud", "SaaS Cloud", "Current"],
            ["Inventory DB", "PostgreSQL 14", "AWS us-east-1", "Current"],
            ["Support Ticketing", "Zendesk", "SaaS Cloud", "Migration Target"],
        ],
    },
}


def _make_get_mock_data() -> Tool:
    def execute(args: GetMockDataArgs) -> ToolResult:
        normalized = args.topic.strip().lower()
        if normalized not in MOCK_TOPICS:
            valid_list = ", ".join(MOCK_TOPICS.keys())
            raise ToolError(f"Unknown topic '{args.topic}'. Valid topics are: {valid_list}")

        mock_entry = MOCK_TOPICS[normalized]
        table = Table(
            caption=f"Reference Data: {normalized.replace('_', ' ').title()}",
            headers=mock_entry["headers"],
            rows=mock_entry["rows"],
        )
        return ToolResult(
            summary=f"MOCK DATA: Retrieved internal benchmark dataset for '{normalized}'.",
            table=table,
            flags=[],
            data={"topic": normalized, "is_mock": True},
        )

    return Tool(
        name="get_mock_data",
        description="Retrieves mock company dataset for topics: company_profile, team_roster, sales_kpis, it_landscape.",
        args_model=GetMockDataArgs,
        example={"topic": "company_profile"},
        func=execute,
    )


# -----------------------------------------------------------------------------
# Default Registry Factory
# -----------------------------------------------------------------------------

def default_registry(today: Callable[[], date] = date.today) -> ToolRegistry:
    """Build default registry populated with all 5 operational tools."""
    registry = ToolRegistry()
    registry.register(_make_get_current_date(today))
    registry.register(_make_build_timeline(today))
    registry.register(_make_build_budget())
    registry.register(_make_build_risk_register())
    registry.register(_make_get_mock_data())
    return registry
