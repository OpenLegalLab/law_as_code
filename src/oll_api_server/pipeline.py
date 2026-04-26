"""Core orchestration — chains classification, code generation, and execution."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

import dspy

from case_classifier.classifier import CaseClassifier
from case_classifier.law_index import (
    format_law_index_for_prompt,
    get_all_laws,
    get_law_by_reference,
)

from openfisca_switzerland import CountryTaxBenefitSystem

from oll_law_as_code.runner import run_variables

from oll_api_server.i18n import (
    build_language_prefix,
    get_section_headers,
    translate_error,
)
from oll_api_server.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    CodeGenerationItem,
    ExecutionResult,
)

log = logging.getLogger(__name__)

# Module-level caches populated at startup
_law_cache: list[dict] = []
_law_index_prompt: str = ""
_tbs: CountryTaxBenefitSystem | None = None


def _get_tbs() -> CountryTaxBenefitSystem:
    """Return a cached TaxBenefitSystem instance (created once)."""
    global _tbs  # noqa: PLW0603
    if _tbs is None:
        _tbs = CountryTaxBenefitSystem()
    return _tbs


def load_law_index() -> None:
    """Load the law index into the module-level cache (called once at startup)."""
    global _law_cache, _law_index_prompt  # noqa: PLW0603
    _law_cache = get_all_laws()
    _law_index_prompt = format_law_index_for_prompt(_law_cache)
    log.info("Loaded %d laws into cache", len(_law_cache))


def get_cached_laws() -> list[dict]:
    """Return the cached law index."""
    return _law_cache


def get_cached_law_index_prompt() -> str:
    """Return the cached formatted law index for LLM prompts."""
    return _law_index_prompt


# ---------------------------------------------------------------------------
# Parsing helpers (moved from case_classifier main.py)
# ---------------------------------------------------------------------------


def parse_applicable_laws(raw: str) -> list[str]:
    """Parse the applicable_laws output from Stage 1 into a list of strings."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    matches = re.findall(r"['\"]([^'\"]+)['\"]", raw)
    return matches


def parse_input_parameters(raw: str) -> dict:
    """Parse the input_parameters JSON output from Stage 1."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


# ---------------------------------------------------------------------------
# Pipeline stages (sync, will be wrapped in asyncio.to_thread)
# ---------------------------------------------------------------------------


def _run_classification(case_description: str, language: str = "en") -> dspy.Prediction:
    """Stage 1: Classify the case against the law index."""
    prefix = build_language_prefix(language)
    classifier = CaseClassifier()
    return classifier(
        case_description=prefix + case_description,
        law_index=_law_index_prompt,
    )


def _lookup_variables(applicable_law_refs: list[str]) -> list[dict]:
    """Stage 2: Look up pre-written OpenFisca variable names from the law index."""
    results: list[dict] = []
    for ref in applicable_law_refs:
        law = get_law_by_reference(ref, _law_cache)
        if law and law.get("openfisca_variable"):
            results.append({
                "article_reference": ref,
                "openfisca_variable": law["openfisca_variable"],
                "law_name": law.get("law_name", ""),
            })
        else:
            log.warning("No openfisca_variable mapping for %r", ref)
    return results


def _run_execution(variable_names: list[str], input_data: dict | None, period: str):
    """Stage 3: Execute pre-registered variables by name."""
    return run_variables(
        variable_names=variable_names,
        input_data=input_data,
        period=period,
    )


# ---------------------------------------------------------------------------
# Async pipeline entry point
# ---------------------------------------------------------------------------


async def run_pipeline(request: ClassifyRequest) -> ClassifyResponse:
    """Run the full classify → generate → execute pipeline."""
    start = time.monotonic()
    stages: list[str] = []
    lang = request.language

    # --- Stage 1: Classification ---
    try:
        result = await asyncio.to_thread(
            _run_classification, request.case_description, lang
        )
    except Exception as exc:
        log.exception("Classification failed")
        return ClassifyResponse(
            error=translate_error("classification_failed", lang, detail=str(exc)),
            processing_time_seconds=time.monotonic() - start,
        )

    applicable_law_refs = parse_applicable_laws(result.applicable_laws)
    input_params = parse_input_parameters(result.input_parameters)
    filtered_reasoning = _filter_reasoning_by_applicable_laws(
        result.syllogistic_reasoning, applicable_law_refs
    )
    legal_reasoning = _extract_legal_reasoning(filtered_reasoning, lang)
    stages.append("classification")

    if not applicable_law_refs:
        return ClassifyResponse(
            syllogistic_reasoning=filtered_reasoning,
            legal_reasoning=legal_reasoning,
            applicable_laws=[],
            input_parameters=input_params,
            stages_completed=stages,
            processing_time_seconds=time.monotonic() - start,
            error=translate_error("no_applicable_laws", lang),
        )

    # --- Stage 2: Variable lookup ---
    variable_mappings = _lookup_variables(applicable_law_refs)

    code_items = [
        CodeGenerationItem(
            article_reference=m["article_reference"],
            openfisca_variable=m["openfisca_variable"],
            validation_passed=True,
        )
        for m in variable_mappings
    ]
    stages.append("code_generation")

    # --- Stage 3: Execution (optional) ---
    execution = None
    if request.execute:
        variable_names = [m["openfisca_variable"] for m in variable_mappings]

        if not variable_names:
            execution = ExecutionResult(
                success=False,
                error=translate_error("no_variable_mappings", lang),
                error_stage="pre_execution",
            )
        else:
            # Build input_data from parsed parameters
            input_data = _build_input_data(input_params, request.period)

            try:
                exec_result = await asyncio.to_thread(
                    _run_execution, variable_names, input_data, request.period
                )
                execution = ExecutionResult(
                    success=exec_result.success,
                    computed_values=exec_result.computed_values,
                    error=exec_result.error,
                    error_stage=exec_result.error_stage,
                )
            except Exception as exc:
                log.exception("Execution failed")
                execution = ExecutionResult(
                    success=False,
                    error=translate_error("execution_failed", lang, detail=str(exc)),
                    error_stage="runtime",
                )

        stages.append("execution")

    return ClassifyResponse(
        syllogistic_reasoning=filtered_reasoning,
        legal_reasoning=legal_reasoning,
        applicable_laws=applicable_law_refs,
        input_parameters=input_params,
        code_generation=code_items,
        execution=execution,
        stages_completed=stages,
        processing_time_seconds=round(time.monotonic() - start, 2),
    )


def _build_input_data(input_params: dict, period: str) -> dict | None:
    """Convert classifier input_parameters to OpenFisca simulation input format.

    Validates each parameter name against the TaxBenefitSystem and uses the
    variable's ``definition_period`` to choose the correct period key
    (``YYYY`` for YEAR variables, ``YYYY-MM`` for MONTH variables).

    Returns None if no parameters could be extracted (runner will use defaults).
    """
    if not input_params:
        return None

    tbs = _get_tbs()
    person_data: dict = {}
    year = period[:4]

    for _law_ref, params in input_params.items():
        if not isinstance(params, dict):
            continue
        for key, value in params.items():
            if key not in tbs.variables:
                log.warning("Skipping unknown variable %r (not in TBS)", key)
                continue
            var = tbs.variables[key]
            if str(var.definition_period).upper() == "YEAR":
                person_data[key] = {year: value}
            else:
                person_data[key] = {period: value}

    if not person_data:
        return None

    return {
        "persons": {"p1": person_data},
        "households": {"h1": {"parents": ["p1"]}},
    }


_ARTICLE_HEADER_RE = re.compile(r"\b([A-Z]{2,})\s+Art\.?\s+(\d+\w*)\b")
_BULLET_RE = re.compile(r"^(?:[-*+]|\d+[.)]) *")


def _filter_reasoning_by_applicable_laws(
    reasoning: str, applicable_laws: list[str]
) -> str:
    """Keep only reasoning sections for articles in *applicable_laws*.

    The LLM emits reasoning for all candidate articles. This splits the text
    into per-article sections and drops those whose header doesn't match any
    entry in *applicable_laws*. Any preamble before the first article header
    is preserved.
    """
    if not reasoning or not applicable_laws:
        return reasoning

    # Normalise applicable_laws to "ABBREV Art. NN" for matching
    norm_refs: set[str] = set()
    for ref in applicable_laws:
        m = re.search(r"([A-Z]{2,})\s+Art\.?\s*(\d+\w*)", ref)
        if m:
            norm_refs.add(f"{m.group(1)} {m.group(2)}")

    lines = reasoning.split("\n")
    sections: list[tuple[str | None, list[str]]] = []  # (key|None, lines)
    current_key: str | None = None
    current_lines: list[str] = []

    for line in lines:
        m = _ARTICLE_HEADER_RE.search(line)
        if m:
            # Save previous section
            sections.append((current_key, current_lines))
            current_key = f"{m.group(1)} {m.group(2)}"
            current_lines = [line]
        else:
            current_lines.append(line)

    sections.append((current_key, current_lines))

    kept: list[str] = []
    for key, sec_lines in sections:
        if key is None:
            # Preamble — always keep
            kept.extend(sec_lines)
        elif key in norm_refs:
            kept.extend(sec_lines)

    return "\n".join(kept).strip()


def _starts_with_any(text: str, candidates: list[str]) -> bool:
    """Check if *text* starts with any candidate (with optional ``**`` prefix)."""
    for c in candidates:
        if text.startswith(c) or text.startswith(f"**{c}"):
            return True
    return False


def _extract_legal_reasoning(syllogistic_reasoning: str, language: str = "en") -> str:
    """Extract minor premise and conclusion from syllogistic reasoning.

    Parses the structured syllogistic output to find lines labelled as
    "Minor Premise" and "Conclusion", then concatenates them into a
    flowing paragraph.  Recognises section headers in both English and
    the target *language*.
    """
    if not syllogistic_reasoning:
        return ""

    headers = get_section_headers(language)
    lines = syllogistic_reasoning.split("\n")
    parts: list[str] = []
    capture = False

    for line in lines:
        stripped = line.strip()
        # Strip bullet/list prefixes so "- Minor premise: ..." is recognised
        cleaned = _BULLET_RE.sub("", stripped)
        lower = cleaned.lower()

        # Article-reference header (e.g. "For **AHVG Art. 3...") → stop capture
        if re.match(r"(?:For\s+)?\**\s*[A-Z]{2,}\s+Art\.?\s+\d+", cleaned):
            capture = False
            continue

        # Detect section headers
        if _starts_with_any(lower, headers["minor_premise"]):
            capture = True
            after_colon = cleaned.split(":", 1)
            if len(after_colon) > 1 and after_colon[1].strip():
                text = after_colon[1].strip().strip("*").strip()
                if text:
                    parts.append(text)
            continue

        if _starts_with_any(lower, headers["conclusion"]):
            capture = True
            after_colon = cleaned.split(":", 1)
            if len(after_colon) > 1 and after_colon[1].strip():
                text = after_colon[1].strip().strip("*").strip()
                if text:
                    parts.append(text)
            continue

        # Stop capturing at the next section header or blank line separator
        if _starts_with_any(lower, headers["major_premise"]):
            capture = False
            continue
        if lower.startswith("---") or lower.startswith("==="):
            capture = False
            continue
        # New article header resets capture
        if lower.startswith("article") or lower.startswith("**article"):
            capture = False
            continue

        if capture and stripped:
            parts.append(stripped.strip("*").strip())

    return " ".join(parts)
