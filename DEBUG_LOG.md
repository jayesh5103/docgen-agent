# Autonomous Document Agent - Debug Log

This document records real failures, edge cases, and unexpected behaviors encountered during the implementation and testing of the Autonomous Document Agent.

### 1: Risk Register Schema Range Validation Error

- **Symptom**: `test_t5_risk_register_boundaries_and_sorting` failed with `ToolError: risks.0.impact: Input should be less than or equal to 5`.
- **Root Cause**: An initial test draft included test items with `impact: 7`. The `RiskItem` schema strictly restricts `likelihood` and `impact` to `[1, 5]`. Attempting to simulate boundary score 7 directly by setting an impact of 7 was rejected by Pydantic's `le=5` validator.
- **Fix**: Replaced the test data with mathematically valid combinations within `[1, 5]` (e.g. `2 * 4 = 8` for the Medium threshold and `3 * 5 = 15` for the High threshold) and cleaned up the stale preliminary block.
- **Test Added**: `tests/test_tools.py::test_t5_risk_register_boundaries_and_sorting` verifies the exact rating boundaries (High >= 15, Medium >= 8, Low < 8), score ordering, and flag generation.

### 2: Premature Dropping of Unknown Dependencies During Task ID Regeneration

- **Symptom**: `test_p1_normalize_plan_corrections` failed because the trace note for dropping unknown dependencies (`"unknown dependency"`) was missing.
- **Root Cause**: When duplicate IDs necessitated regenerating IDs (`t1`, `t2`, ...), the dependency remapping logic filtered dependencies using `if dep in old_to_new: remapped_deps.append(old_to_new[dep])`. This silently filtered out unknown dependencies like `ghost_id` before the downstream dependency-validation loop could detect them and append the audit note to `notes`. Additionally, duplicate keys in `old_to_new` overwrote previous entries, turning downstream dependencies into accidental self-dependencies.
- **Fix**: Preserved the first mapping of an ID when building `old_to_new`, and used `old_to_new.get(dep, dep)` during remapping so unrecognized dependencies are preserved for the explicit dependency validation and logging stage.
- **Test Added**: `tests/test_planner.py::test_p1_normalize_plan_corrections` tests that duplicate IDs are reassigned, unknown dependencies are detected and logged, self-dependencies are eliminated, and unknown tools are nulled.

### 3: Pydantic Subclass Field Dropping in Section Before-Validator

- **Symptom**: `SectionDraft(id="t1", ...)` raised `ValidationError: id: Field required`.
- **Root Cause**: `Section` implemented a `@model_validator(mode="before")` that returned a new dictionary containing only the parent fields (`heading`, `paragraphs`, `bullets`, `table`). When `SectionDraft` inherited from `Section`, this validator executed and stripped subclass-specific attributes (`id` and `source`).
- **Fix**: Updated `Section.coerce_section` to copy the incoming dictionary (`res = dict(data)`) and update existing keys in place rather than returning an isolated new dictionary.
- **Test Added**: `tests/test_docx_builder.py` tests instantiate `SectionDraft` and build full documents without validation failures.

### 4: Composite Markdown and Numeric Heading Prefix Stripping

- **Symptom**: `test_d4_strip_existing_numbering` failed because a heading with both markdown hashes and numbers (e.g., `"##2. Technical Approach"`) retained the second prefix after single-pass regex substitution, rendering as `"2. 2. Technical Approach"`.
- **Root Cause**: A single regex substitution `re.sub(r"^(\d+[\.\)]|\#+|\*|\-)\s*", "", heading)` only strips the first matched prefix token and stops, failing on compound prefixes like `"## 2."`.
- **Fix**: Replaced single-pass substitution with a loop that strips all leading hashes, bullets, digits, parentheses, and colon tokens until none remain.
- **Test Added**: `tests/test_docx_builder.py::test_d4_strip_existing_numbering` verifies that compound prefixes (`"## 2. Technical Approach"`, `"1. Scope"`, `"3) Terms"`) are cleanly normalized without duplicating numbers.

### 5: Masking of Fatal LLMUnavailableError by complete_json

- **Symptom**: `test_a10_llm_unavailable_propagates` failed because `LLMUnavailableError` was converted into `LLMOutputError`.
- **Root Cause**: `complete_json` wrapped LLM invocations in a generic `except Exception as exc:` block to handle model formatting/validation errors. When the fake LLM simulated an unreachable backend or auth failure by raising `LLMUnavailableError`, this was intercepted by the retry handler and converted into `LLMOutputError` upon retry exhaustion.
- **Fix**: Added an explicit `except LLMUnavailableError: raise` block in `complete_json` prior to the generic exception handler, ensuring fatal provider errors propagate up immediately as designed.
- **Test Added**: `tests/test_agent.py::test_a10_llm_unavailable_propagates` verifies that `LLMUnavailableError` propagates directly to the caller.
