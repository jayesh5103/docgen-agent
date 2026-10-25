"""Live smoke test executing against real configured LLM provider (opt-in via RUN_LIVE=1)."""

import json
import os
from pathlib import Path
import pytest
from app.agent import DocumentAgent
from app.config import Settings
from app.llm import build_llm
from app.reflection import PLACEHOLDER_PATTERN
from app.tools import default_registry

RUN_LIVE = os.environ.get("RUN_LIVE", "0").strip() == "1"


@pytest.mark.skipif(not RUN_LIVE, reason="Live LLM tests are opt-in (set RUN_LIVE=1 to execute).")
def test_live_smoke_both_inputs(tmp_path: Path):
    """Run both demo inputs against configured live LLM and verify core structural invariants."""
    settings = Settings.from_env()
    # Direct output to temporary directory
    settings = Settings(
        provider=settings.provider,
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
        output_dir=str(tmp_path),
        llm_timeout=settings.llm_timeout,
        llm_http_retries=settings.llm_http_retries,
        json_retries=settings.json_retries,
        tool_retries=settings.tool_retries,
        max_tasks=settings.max_tasks,
        max_reflection_rounds=settings.max_reflection_rounds,
        quality_threshold=settings.quality_threshold,
    )

    llm = build_llm(settings)
    tools = default_registry()
    agent = DocumentAgent(llm=llm, tools=tools, settings=settings)

    samples_path = Path(__file__).resolve().parent.parent / "samples" / "test_inputs.json"
    with open(samples_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        req = case["request"]
        response = agent.run(req)

        # 1. Successful execution
        assert response.status in ("completed", "completed_with_warnings")

        # 2. Document file exists and is non-empty
        assert response.document is not None
        docx_path = tmp_path / response.document.filename
        assert docx_path.exists()
        assert docx_path.stat().st_size > 5000

        # 3. Structural invariant: at least 4 plan tasks
        assert response.plan is not None
        assert len(response.plan.tasks) >= 4

        # 4. Every plan task has a terminal status
        for task in response.plan.tasks:
            assert task.status in ("done", "failed")

        # 5. Reflection block is populated
        assert response.reflection is not None
        assert response.reflection.final_score is not None
        assert response.reflection.final_score >= 1

        # 6. Reopen generated docx and assert no placeholder text
        from docx import Document
        doc = Document(str(docx_path))
        for p in doc.paragraphs:
            assert not PLACEHOLDER_PATTERN.search(p.text)
