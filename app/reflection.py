"""Deterministic linting and document formatting for self-review."""

import re
from app.schemas import DocumentSpec, Issue

PLACEHOLDER_PATTERN = re.compile(
    r"\b(TBD|TBC|XXX|to\s+be\s+determined)\b|lorem\s+ipsum|\[insert\b[^\]]*\]|\[placeholder\b[^\]]*\]",
    re.IGNORECASE,
)


def lint_document(spec: DocumentSpec) -> list[Issue]:
    """Deterministically lint draft document for empty/thin sections, placeholders, and duplicate headings."""
    issues: list[Issue] = []
    seen_headings: set[str] = set()

    for draft in spec.sections:
        # Check for empty section
        has_content = bool(draft.paragraphs or draft.bullets or (draft.table and draft.table.headers))
        if not has_content:
            issues.append(
                Issue(
                    section_id=draft.id,
                    kind="fix",
                    problem=f"Section '{draft.heading}' is completely empty.",
                    fix="Provide substantive content for this section.",
                    source="lint",
                )
            )
            continue

        # Check for thin LLM section (< 25 words with no table)
        if draft.source == "llm" and not draft.table and draft.word_count() < 25:
            issues.append(
                Issue(
                    section_id=draft.id,
                    kind="fix",
                    problem=f"Section '{draft.heading}' is too thin ({draft.word_count()} words; minimum 25).",
                    fix="Expand the section with concrete domain analysis and details.",
                    source="lint",
                )
            )

        # Check for placeholder patterns
        full_text = draft.as_text()
        match = PLACEHOLDER_PATTERN.search(full_text)
        if match:
            matched_token = match.group(0)
            issues.append(
                Issue(
                    section_id=draft.id,
                    kind="fix",
                    problem=f"Placeholder text '{matched_token}' detected in section '{draft.heading}'.",
                    fix="Replace placeholder text with specific, validated information.",
                    source="lint",
                )
            )

        # Check for duplicate headings (case-insensitive, cleaned)
        normalized_heading = re.sub(r"^(\d+[\.\)]|\#+|\*|\-)\s*", "", draft.heading).strip().lower()
        if normalized_heading in seen_headings:
            issues.append(
                Issue(
                    section_id=draft.id,
                    kind="fix",
                    problem=f"Duplicate section heading '{draft.heading}'.",
                    fix="Give this section a distinct, descriptive heading.",
                    source="lint",
                )
            )
        else:
            seen_headings.add(normalized_heading)

    return issues


def format_for_review(spec: DocumentSpec, max_chars: int = 12000) -> str:
    """Format document draft into a concise text representation for the LLM reviewer."""
    blocks: list[str] = []

    for s in spec.sections:
        lines = [f"### [{s.id}] {s.heading}"]
        for p in s.paragraphs:
            lines.append(p)
        for b in s.bullets:
            lines.append(f"* {b}")
        if s.table and s.table.headers:
            if s.table.caption:
                lines.append(f"Table: {s.table.caption}")
            lines.append(" | ".join(s.table.headers))
            for row in s.table.rows:
                lines.append(" | ".join(row))
        blocks.append("\n".join(lines))

    full_text = "\n\n".join(blocks)
    if len(full_text) > max_chars:
        return full_text[:max_chars] + "\n... [Draft truncated for review length]"
    return full_text
