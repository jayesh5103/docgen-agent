"""Tests for deterministic reflection and review formatting (R1 - R2)."""

from datetime import date
from app.reflection import format_for_review, lint_document
from app.schemas import DocumentSpec, SectionDraft, Table


def test_r1_lint_document_issues():
    """R1 lint flags empty section, thin section, TBD, [insert date], lorem ipsum, duplicate heading; a clean document yields no issues."""
    spec_with_issues = DocumentSpec(
        title="Audit Test",
        doc_type="report",
        audience="Auditors",
        generated_on=date(2026, 10, 23),
        sections=[
            # 1. Empty section
            SectionDraft(id="t1", heading="Empty Section"),
            # 2. Thin LLM section (< 25 words, no table)
            SectionDraft(id="t2", heading="Thin Section", paragraphs=["Only seven words here in this draft."]),
            # 3. TBD placeholder
            SectionDraft(
                id="t3",
                heading="TBD Section",
                paragraphs=["This budget is TBD pending approval from finance executive team over the next quarter."],
            ),
            # 4. [insert date] placeholder
            SectionDraft(
                id="t4",
                heading="Date Section",
                paragraphs=["Kickoff will occur on [insert date] once resources are staffed across departments."],
            ),
            # 5. lorem ipsum placeholder
            SectionDraft(
                id="t5",
                heading="Latin Section",
                paragraphs=["Lorem ipsum dolor sit amet, consectetur adipiscing elit sed do eiusmod tempor."],
            ),
            # 6. Duplicate heading
            SectionDraft(
                id="t6",
                heading="Date Section",
                paragraphs=["Another substantive section with sufficient words to pass the minimum word count check easily."],
            ),
        ],
    )

    issues = lint_document(spec_with_issues)
    problems = " ".join(iss.problem for iss in issues)

    assert "completely empty" in problems
    assert "too thin" in problems
    assert "TBD" in problems
    assert "[insert date]" in problems
    assert "Lorem ipsum" in problems
    assert "Duplicate section heading" in problems

    # Clean document yields no issues
    clean_spec = DocumentSpec(
        title="Clean Document",
        doc_type="proposal",
        audience="Leadership",
        generated_on=date(2026, 10, 23),
        sections=[
            SectionDraft(
                id="t1",
                heading="Executive Summary",
                paragraphs=[
                    "This comprehensive proposal details the strategic initiative to modernize customer support "
                    "infrastructure using machine learning techniques. All prerequisites and organizational readiness "
                    "metrics have been validated for deployment."
                ],
            ),
            SectionDraft(
                id="t2",
                heading="Cost Model",
                table=Table(headers=["Role", "Rate"], rows=[["Engineer", "$150"]]),
            ),
        ],
    )
    clean_issues = lint_document(clean_spec)
    assert len(clean_issues) == 0


def test_r2_format_for_review():
    """R2 format_for_review includes section ids and respects max_chars."""
    spec = DocumentSpec(
        title="Review Spec",
        doc_type="proposal",
        audience="Reviewers",
        generated_on=date(2026, 10, 23),
        sections=[
            SectionDraft(
                id="sec_1",
                heading="Introduction",
                paragraphs=["First paragraph text."],
                bullets=["Bullet point A"],
            ),
            SectionDraft(
                id="sec_2",
                heading="Data Overview",
                table=Table(
                    headers=["Metric", "Value"],
                    rows=[[f"Key {i}", f"Val {i}"] for i in range(10)],
                ),
            ),
        ],
    )

    formatted = format_for_review(spec, max_chars=100)
    assert "[sec_1]" in formatted
    assert "Introduction" in formatted
    assert len(formatted) <= 150  # truncated with truncation notice
    assert "[Draft truncated" in formatted
