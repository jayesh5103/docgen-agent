"""Scripted fake LLM and plan fixtures for deterministic testing."""

import json
import re
from datetime import date
from typing import Any, Callable

FROZEN_DATE = date(2026, 10, 23)

STANDARD_PLAN = {
    "can_fulfil": True,
    "reason": None,
    "doc_type": "proposal",
    "title": "Customer Support Chatbot Proposal",
    "audience": "Executive Leadership",
    "assumptions": [
        "Current support volume is 45,000 tickets monthly",
        "Integration target is Zendesk and core CRM",
        "Deployment target is cloud SaaS infrastructure",
    ],
    "open_questions": [
        "What is the exact target launch date?",
        "Are multi-language capabilities required in Phase 1?",
    ],
    "tasks": [
        {
            "id": "t1",
            "title": "Executive Summary",
            "description": "High-level overview of objectives and strategic value",
            "tool": None,
            "depends_on": ["t2", "t3", "t4", "t5"],
        },
        {
            "id": "t2",
            "title": "Scope and Objectives",
            "description": "Functional boundaries, conversational capabilities, and integration requirements",
            "tool": None,
            "depends_on": [],
        },
        {
            "id": "t3",
            "title": "Delivery Timeline",
            "description": "Phase schedules and implementation roadmap",
            "tool": "build_timeline",
            "depends_on": [],
        },
        {
            "id": "t4",
            "title": "Budget Estimate",
            "description": "Cost breakdown for engineering, infrastructure, and licensing",
            "tool": "build_budget",
            "depends_on": [],
        },
        {
            "id": "t5",
            "title": "Key Risks and Mitigations",
            "description": "Risk register covering data accuracy and security",
            "tool": "build_risk_register",
            "depends_on": [],
        },
    ],
}

COMPLEX_PLAN = {
    "can_fulfil": True,
    "reason": None,
    "doc_type": "proposal",
    "title": "Strategic Cloud Migration Proposal",
    "audience": "Board of Directors",
    "assumptions": [
        "Existing on-premises infrastructure spans 3 core systems",
        "Migration kickoff is scheduled for 2026-11-02",
        "Core IT staff is available for parallel workload transition",
    ],
    "open_questions": [
        "Can multi-region redundancy be deferred to Phase 2 to meet budget?",
        "Will the board approve an accelerated 6-week cutover window?",
    ],
    "tasks": [
        {
            "id": "t1",
            "title": "Executive Summary",
            "description": "Executive synthesis of cloud modernization initiative",
            "tool": None,
            "depends_on": ["t2", "t3", "t4", "t5", "t6", "t7"],
        },
        {
            "id": "t2",
            "title": "Delivery Timeline",
            "description": "Accelerated 6-week timeline analysis",
            "tool": "build_timeline",
            "depends_on": [],
        },
        {
            "id": "t3",
            "title": "Budget Breakdown and Cost Constraints",
            "description": "Evaluation of total migration cost against $50,000 cap",
            "tool": "build_budget",
            "depends_on": [],
        },
        {
            "id": "t4",
            "title": "Conflict Analysis and Recommended Phasing",
            "description": "Trade-offs and resolution of 6-week and $50k budget conflict with multi-region scope",
            "tool": None,
            "depends_on": ["t2", "t3"],
        },
        {
            "id": "t5",
            "title": "Team Responsibilities Matrix",
            "description": "Operational duties across DevOps, Security, and App Support",
            "tool": None,
            "depends_on": [],
        },
        {
            "id": "t6",
            "title": "Scope Pushback and Non-Negotiable Boundaries",
            "description": "Detailed justification of features to push back on",
            "tool": None,
            "depends_on": ["t4"],
        },
        {
            "id": "t7",
            "title": "Migration Risk Register",
            "description": "High availability and downtime risk evaluation",
            "tool": "build_risk_register",
            "depends_on": [],
        },
    ],
}


class ScriptedLLM:
    """Fake LLM that dispatches canned responses based on [[STEP:tag]] and records calls."""

    def __init__(self, script: dict[str, list[str | Callable[[str], str]]] | None = None):
        self.script: dict[str, list[Any]] = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[tuple[str, str]] = []
        self.system_prompts: list[str] = []

    def chat(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        # Extract [[STEP:name]]
        step_match = re.search(r"\[\[STEP:([a-zA-Z0-9_]+)\]\]", system)
        step = step_match.group(1) if step_match else "unknown"

        self.calls.append((step, user))
        self.system_prompts.append(system)

        if step in self.script and self.script[step]:
            queue = self.script[step]
            # Consume item; if only 1 remains, keep repeating it
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if callable(item):
                return item(user)
            return str(item)

        # Default scripted responses per step
        return self._default_response(step, user)

    def _default_response(self, step: str, user: str) -> str:
        if step == "plan":
            return json.dumps(STANDARD_PLAN)

        if step == "tool_args":
            tool_match = re.search(r"TOOL:\s*([a-zA-Z0-9_]+)", user)
            tool_name = tool_match.group(1).strip() if tool_match else ""

            if tool_name == "get_current_date":
                return json.dumps({"arguments": {}})
            elif tool_name == "build_timeline":
                return json.dumps({
                    "arguments": {
                        "start_date": "2026-11-02",
                        "phases": [
                            {"name": "Discovery & Architecture", "duration_weeks": 2, "parallel_with_previous": False},
                            {"name": "Core Implementation", "duration_weeks": 4, "parallel_with_previous": False},
                        ],
                    }
                })
            elif tool_name == "build_budget":
                return json.dumps({
                    "arguments": {
                        "currency": "USD",
                        "contingency_pct": 10.0,
                        "items": [
                            {"name": "Senior Software Architect", "quantity": 80, "unit_cost": 150.0},
                            {"name": "Cloud Infrastructure Setup", "quantity": 1, "unit_cost": 8000.0},
                        ],
                    }
                })
            elif tool_name == "build_risk_register":
                return json.dumps({
                    "arguments": {
                        "risks": [
                            {
                                "risk": "Data schema incompatibility",
                                "likelihood": 3,
                                "impact": 4,
                                "mitigation": "Perform automated schema validation in sprint 1",
                            }
                        ]
                    }
                })
            elif tool_name == "get_mock_data":
                return json.dumps({"arguments": {"topic": "company_profile"}})
            else:
                return json.dumps({"arguments": {}})

        if step in ("write_section", "revise_section"):
            title_match = re.search(r"(?:TASK_TITLE|CURRENT_SECTION_HEADING):\s*(.+)", user)
            heading = title_match.group(1).strip() if title_match else "Business Overview"

            return json.dumps({
                "heading": heading,
                "paragraphs": [
                    f"This section delivers comprehensive analysis and strategic recommendations for {heading}. "
                    "The proposed approach aligns strictly with stated enterprise requirements, ensuring operational continuity, "
                    "robust system integration, and clear risk mitigation across all organizational units. "
                    "Key deliverables and milestones are scheduled to ensure quality verification at every step."
                ],
                "bullets": [
                    "Establish standardized governance and clear accountability across all participating teams",
                    "Implement continuous monitoring to validate performance and milestone completion",
                ],
                "table": None,
            })

        if step == "reflect":
            return json.dumps({"score": 9, "issues": []})

        return "{}"
