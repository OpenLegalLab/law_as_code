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

from oll_law_as_code.runner import run_variables, get_tbs, _HOUSEHOLD_ROLES

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
    MissingParameter,
)

log = logging.getLogger(__name__)

# Module-level caches populated at startup
_law_cache: list[dict] = []
_law_index_prompt: str = ""


def load_law_index() -> None:
    """Load the law index into the module-level cache (called once at startup)."""
    global _law_cache, _law_index_prompt  # noqa: PLW0603
    _law_cache = get_all_laws()
    _law_index_prompt = format_law_index_for_prompt(_law_cache)
    log.info("Loaded %d laws into cache", len(_law_cache))


def validate_law_index_variables() -> None:
    """Verify every openfisca_variable in the law index exists in its TBS."""
    missing = []
    for law in _law_cache:
        var = law.get("openfisca_variable")
        if not var:
            continue
        system = law.get("system", "openfisca_switzerland")
        tbs = get_tbs(system)
        if var not in tbs.variables:
            missing.append(f"{var} (not in {system})")
    if missing:
        log.warning("Missing variables (not yet in TBS):\n  %s", "\n  ".join(missing))
    else:
        log.info("All law index variables validated against TBS")


def get_cached_laws() -> list[dict]:
    """Return the cached law index."""
    return _law_cache


def get_cached_law_index_prompt() -> str:
    """Return the cached formatted law index for LLM prompts."""
    return _law_index_prompt


def get_augmentation_fields(applicable_law_refs: list[str]) -> list[dict]:
    """Return deduplicated input fields for the given law references.

    For each law, iterates its ``input_parameters`` and resolves the
    human-readable label from the TaxBenefitSystem variable when available,
    falling back to the YAML ``description``.

    No LLM calls — pure in-memory lookup.
    """
    seen: set[str] = set()
    fields: list[dict] = []

    for ref in applicable_law_refs:
        law = get_law_by_reference(ref, _law_cache)
        if law is None:
            log.warning("Unknown law reference %r — skipping", ref)
            continue

        system = law.get("system", "openfisca_switzerland")
        try:
            tbs = get_tbs(system)
        except ValueError:
            log.warning("Unknown TBS system %r for %r — skipping", system, ref)
            continue

        for param in law.get("input_parameters", []):
            if isinstance(param, str):
                name = param
                description = param
            else:
                name = param.get("name", "")
                description = param.get("description", name)
            if not name or name in seen:
                continue
            seen.add(name)

            tbs_var = tbs.variables.get(name)
            label = (
                tbs_var.label
                if tbs_var is not None and getattr(tbs_var, "label", None)
                else description
            )

            fields.append({"id": name, "full_name": label, "value": 0})

    return fields


def _compute_missing_parameters(
    applicable_law_refs: list[str],
    extracted_params: dict,
) -> list[MissingParameter]:
    """Identify required parameters that the LLM could not extract.

    For each applicable law, checks its ``input_parameters`` against the
    values extracted by the classifier.  A parameter is considered missing
    when it is absent or ``None`` in *extracted_params*.

    Deduplicates by parameter ``id`` (same variable needed by multiple laws
    keeps only the first occurrence).
    """
    seen: set[str] = set()
    missing: list[MissingParameter] = []

    for ref in applicable_law_refs:
        law = get_law_by_reference(ref, _law_cache)
        if law is None:
            continue

        law_extracted = extracted_params.get(ref, {})
        if not isinstance(law_extracted, dict):
            law_extracted = {}

        system = law.get("system", "openfisca_switzerland")
        try:
            tbs = get_tbs(system)
        except ValueError:
            tbs = None

        for param in law.get("input_parameters", []):
            # Handle both dict entries and plain string entries
            if isinstance(param, str):
                name = param
                description = param
                param_type = "str"
            else:
                name = param.get("name", "")
                description = param.get("description", name)
                param_type = param.get("type", "str")

            if not name or name in seen:
                continue

            # Parameter present and non-null → not missing
            if law_extracted.get(name) is not None:
                continue

            seen.add(name)

            # Resolve human-readable description from TBS label, fallback to YAML
            if tbs is not None:
                tbs_var = tbs.variables.get(name)
                if tbs_var is not None and getattr(tbs_var, "label", None):
                    description = tbs_var.label

            missing.append(MissingParameter(
                id=name,
                type=param_type,
                description=description,
                law_reference=ref,
            ))

    return missing


def _merge_known_parameters(
    input_params: dict,
    known_parameters: dict[str, object],
    applicable_law_refs: list[str],
) -> dict:
    """Merge client-provided *known_parameters* into LLM-extracted *input_params*.

    *known_parameters* is a flat dict keyed by OpenFisca variable name.  For
    each applicable law, if a required parameter name appears in
    *known_parameters* but is absent/None in the extracted params, inject it.
    """
    if not known_parameters:
        return input_params

    for ref in applicable_law_refs:
        law = get_law_by_reference(ref, _law_cache)
        if law is None:
            continue

        law_extracted = input_params.get(ref)
        if law_extracted is None:
            law_extracted = {}
            input_params[ref] = law_extracted
        elif not isinstance(law_extracted, dict):
            continue

        for param in law.get("input_parameters", []):
            if isinstance(param, str):
                name = param
            else:
                name = param.get("name", "")
            if not name:
                continue
            if name in known_parameters and law_extracted.get(name) is None:
                law_extracted[name] = known_parameters[name]

    return input_params


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
                "system": law.get("system", "openfisca_switzerland"),
            })
        else:
            log.warning("No openfisca_variable mapping for %r", ref)
    return results


def _run_execution(
    variable_names: list[str],
    input_data: dict | None,
    period: str,
    system: str = "openfisca_switzerland",
):
    """Stage 3: Execute pre-registered variables by name."""
    return run_variables(
        variable_names=variable_names,
        input_data=input_data,
        period=period,
        system=system,
        tbs=get_tbs(system),
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

    # Deterministic post-classification inference: fill obvious nulls from text
    input_params = _infer_parameters_from_text(
        request.case_description, input_params, applicable_law_refs
    )

    # Merge client-provided known_parameters into LLM-extracted values
    input_params = _merge_known_parameters(
        input_params, request.known_parameters, applicable_law_refs
    )

    # Compute missing parameters (before execution, may be recomputed after filtering)
    missing_params = _compute_missing_parameters(applicable_law_refs, input_params)

    if not applicable_law_refs:
        return ClassifyResponse(
            syllogistic_reasoning=filtered_reasoning,
            legal_reasoning=legal_reasoning,
            applicable_laws=[],
            input_parameters=input_params,
            missing_parameters=missing_params,
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
            # Group variable mappings by system for multi-system dispatch
            groups: dict[str, list[dict]] = {}
            for m in variable_mappings:
                sys_key = m.get("system", "openfisca_switzerland")
                groups.setdefault(sys_key, []).append(m)

            merged_computed: dict[str, float] = {}
            first_error = ""
            first_error_stage = ""
            all_success = True

            for sys_key, sys_mappings in groups.items():
                sys_var_names = [m["openfisca_variable"] for m in sys_mappings]
                input_data = _build_input_data(input_params, request.period, system=sys_key)

                try:
                    exec_result = await asyncio.to_thread(
                        _run_execution, sys_var_names, input_data, request.period, sys_key
                    )
                    merged_computed.update(exec_result.computed_values)
                    if not exec_result.success:
                        all_success = False
                        if not first_error:
                            first_error = exec_result.error
                            first_error_stage = exec_result.error_stage
                except Exception as exc:
                    log.exception("Execution failed for system %s", sys_key)
                    all_success = False
                    if not first_error:
                        first_error = translate_error("execution_failed", lang, detail=str(exc))
                        first_error_stage = "runtime"

            execution = ExecutionResult(
                success=all_success,
                computed_values=merged_computed,
                error=first_error,
                error_stage=first_error_stage,
            )

        stages.append("execution")

    # --- Stage 4: Filter inapplicable laws ---
    if execution and execution.success:
        (
            applicable_law_refs,
            code_items,
            input_params,
            execution,
            filtered_reasoning,
            legal_reasoning,
        ) = _filter_inapplicable_laws(
            applicable_law_refs,
            code_items,
            input_params,
            execution,
            result.syllogistic_reasoning,
            variable_mappings,
            lang,
        )
        # Recompute missing params so filtered-out laws don't contribute
        missing_params = _compute_missing_parameters(applicable_law_refs, input_params)

    return ClassifyResponse(
        syllogistic_reasoning=filtered_reasoning,
        legal_reasoning=legal_reasoning,
        applicable_laws=applicable_law_refs,
        input_parameters=input_params,
        code_generation=code_items,
        missing_parameters=missing_params,
        execution=execution,
        stages_completed=stages,
        processing_time_seconds=round(time.monotonic() - start, 2),
    )


def _is_numeric_string(s: str) -> bool:
    """Return True if *s* can be parsed as a number."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Post-classification inference — deterministically fill obvious nulls
# ---------------------------------------------------------------------------

# Maps: keyword in case text → (parameter_name, inferred_value)
_INFERENCE_RULES: list[tuple[list[str], str, str]] = [
    # wohnsitz_grundlage: "living in Bern" / "wohne in Bern" → elterlicher_wohnsitz (default for students)
    (["living in bern", "wohne in bern", "wohnhaft in bern", "live in bern", "berner", "kanton bern"],
     "wohnsitz_grundlage", "elterlicher_wohnsitz"),
    # staatsangehoerigkeit: assume Swiss if not specified and in Bern
    (["swiss", "schweizer", "schweizerin"], "staatsangehoerigkeit", "schweizer"),
    # ausbildungstyp: "university studies" / "studium" without "second" / "zweit" → erstausbildung
    (["university studies", "uni studies", "studium", "bachelor", "studies at university",
      "my studies", "höhere ausbildung", "my university"],
     "ausbildungstyp", "erstausbildung"),
    # ausbildungsstufe: university → tertiaerstufe
    (["university", "uni", "hochschule", "fachhochschule", "eth", "universität"],
     "ausbildungsstufe", "tertiaerstufe"),
    # ausbildungsstaette_anerkannt: university is always recognised
    (["university", "uni", "hochschule", "fachhochschule", "eth", "universität"],
     "ausbildungsstaette_anerkannt", True),
    # private_ausbildungsstaette: university is typically public
    (["university", "uni", "hochschule", "eth", "universität"],
     "private_ausbildungsstaette", False),
]

# Negative keywords that BLOCK an inference (e.g., "second degree" blocks erstausbildung)
_INFERENCE_BLOCKERS: dict[str, list[str]] = {
    "ausbildungstyp:erstausbildung": [
        "second degree", "zweite ausbildung", "zweitausbildung", "second university",
        "already have a degree", "bereits ein studium", "retraining", "umschulung",
    ],
}


def _infer_parameters_from_text(
    case_description: str,
    input_params: dict,
    applicable_law_refs: list[str],
) -> dict:
    """Fill obvious null parameters and correct impossible values using deterministic keyword matching.

    Phase 1: Correct values the LLM picked that contradict the case text.
    Phase 2: Fill null values that can be inferred from keywords.
    """
    lower_text = case_description.lower()

    for ref in applicable_law_refs:
        law_params = input_params.get(ref)
        if not isinstance(law_params, dict):
            continue

        # Phase 1: Correct impossible LLM values
        _correct_impossible_values(lower_text, law_params, ref)

        # Phase 2: Fill nulls from keywords
        for keywords, param_name, value in _INFERENCE_RULES:
            # Skip if already filled
            if law_params.get(param_name) is not None:
                continue

            # Check if any keyword matches
            matched = any(kw in lower_text for kw in keywords)
            if not matched:
                continue

            # Check blockers
            blocker_key = f"{param_name}:{value}"
            blockers = _INFERENCE_BLOCKERS.get(blocker_key, [])
            if any(b in lower_text for b in blockers):
                continue

            law_params[param_name] = value
            log.info("Inferred %s=%r from case text for %s", param_name, value, ref)

    return input_params


# Corrections: (current_value, required_keywords_absent, replacement)
# If current value is X and NONE of the required keywords are present → replace with default
_CORRECTION_RULES: list[tuple[str, str, list[str], str]] = [
    # zweite_hochschulausbildung requires evidence of a prior completed degree
    ("ausbildungstyp", "zweite_hochschulausbildung",
     ["second degree", "second university", "already completed", "bereits abgeschlossen",
      "zweites studium", "zweite hochschul", "already have a degree"],
     "erstausbildung"),
    # zweitausbildung requires evidence of a completed prior training
    ("ausbildungstyp", "zweitausbildung",
     ["second training", "zweite ausbildung", "already completed", "bereits abgeschlossen",
      "second profession"],
     "erstausbildung"),
]


def _correct_impossible_values(lower_text: str, law_params: dict, ref: str) -> None:
    """Override LLM-extracted values that are logically impossible given the case text."""
    for param_name, wrong_value, evidence_keywords, replacement in _CORRECTION_RULES:
        current = law_params.get(param_name)
        if current != wrong_value:
            continue
        # If NONE of the evidence keywords are present, the LLM guessed wrong
        if not any(kw in lower_text for kw in evidence_keywords):
            log.info(
                "Corrected %s: %r → %r (no supporting evidence in text) for %s",
                param_name, wrong_value, replacement, ref,
            )
            law_params[param_name] = replacement


# Domain-specific keyword → enum key mappings for common LLM paraphrases
_ENUM_KEYWORD_MAP: dict[str, dict[str, str]] = {
    "ausbildungsstufe": {
        "university": "tertiaerstufe",
        "universität": "tertiaerstufe",
        "hochschule": "tertiaerstufe",
        "fachhochschule": "tertiaerstufe",
        "tertiary": "tertiaerstufe",
        "tertiär": "tertiaerstufe",
        "secondary": "sekundarstufe_ii",
        "gymnasium": "sekundarstufe_ii",
        "lehre": "sekundarstufe_ii",
        "apprenticeship": "sekundarstufe_ii",
    },
    "ausbildungstyp": {
        "university": "erstausbildung",
        "first degree": "erstausbildung",
        "bachelor": "erstausbildung",
        "master": "erstausbildung",
        "studies": "erstausbildung",
        "studium": "erstausbildung",
        "first": "erstausbildung",
        "second degree": "zweitausbildung",
        "retraining": "umschulung",
        "further education": "weiterbildung",
    },
    "wohnsitz_grundlage": {
        "bern": "elterlicher_wohnsitz",
        "parents": "elterlicher_wohnsitz",
        "eltern": "elterlicher_wohnsitz",
        "orphan": "elternlos",
        "abroad": "eltern_im_ausland",
        "independent": "finanziell_unabhaengig",
        "refugee": "fluechtling_staatenlos",
    },
    "staatsangehoerigkeit": {
        "swiss": "schweizer",
        "schweiz": "schweizer",
        "eu": "eu_efta_niederlassung",
        "efta": "eu_efta_niederlassung",
        "permit b": "aufenthaltsbewilligung_b",
        "b permit": "aufenthaltsbewilligung_b",
        "refugee": "fluechtling_staatenlos",
    },
}


def _fuzzy_match_enum(value: str, members: dict, variable_name: str = "") -> str | None:
    """Try to map a descriptive LLM string to a valid enum member name.

    Uses domain-specific keyword mappings first, then falls back to
    substring matching against member names.
    """
    lower = value.lower().strip()

    # Direct case-insensitive match
    for name in members:
        if lower == name.lower():
            return name

    # Domain-specific keyword mapping
    kw_map = _ENUM_KEYWORD_MAP.get(variable_name, {})
    for keyword, target in kw_map.items():
        if keyword in lower and target in members:
            return target

    # Check if any member name is contained in the value
    for name in members:
        if name.lower() in lower:
            return name

    # Check if the value is contained in any member name
    for name in members:
        if lower in name.lower():
            return name

    return None


def _build_input_data(
    input_params: dict,
    period: str,
    system: str = "openfisca_switzerland",
) -> dict | None:
    """Convert classifier input_parameters to OpenFisca simulation input format.

    Validates each parameter name against the TaxBenefitSystem and uses the
    variable's ``definition_period`` to choose the correct period key
    (``YYYY`` for YEAR variables, ``YYYY-MM`` for MONTH variables).

    Returns None if no parameters could be extracted (runner will use defaults).
    """
    if not input_params:
        return None

    tbs = get_tbs(system)
    person_data: dict = {}
    household_data: dict = {}
    year = period[:4]

    for _law_ref, params in input_params.items():
        if not isinstance(params, dict):
            continue
        for key, value in params.items():
            if value is None:
                continue
            if key not in tbs.variables:
                log.warning("Skipping unknown variable %r (not in TBS %s)", key, system)
                continue
            var = tbs.variables[key]

            # Handle Enum variables: accept valid enum member names
            if isinstance(value, str) and not _is_numeric_string(value):
                possible = getattr(var, "possible_values", None)
                if possible is not None and hasattr(possible, "__members__"):
                    if value in possible.__members__:
                        pass  # already a valid enum key
                    else:
                        # Try fuzzy matching for LLM-provided descriptive values
                        matched = _fuzzy_match_enum(value, possible.__members__, variable_name=key)
                        if matched:
                            log.info("Fuzzy-matched enum %r -> %r for %r", value, matched, key)
                            value = matched
                        else:
                            log.warning(
                                "Skipping invalid enum value for %r: %r (valid: %s)",
                                key, value, list(possible.__members__),
                            )
                            continue
                else:
                    # Not an Enum variable — skip non-numeric strings
                    log.warning("Skipping non-numeric value for %r: %r", key, value)
                    continue

            period_key = year if str(var.definition_period).upper() == "YEAR" else period

            # Household-level variables (e.g. eltern_anrechenbare_mittel)
            if str(var.entity.key) == "household":
                household_data[key] = {period_key: value}
            else:
                person_data[key] = {period_key: value}

    if not person_data and not household_data:
        return None

    role_key = _HOUSEHOLD_ROLES.get(system, "parents")
    result = {
        "persons": {"p1": person_data},
        "households": {"h1": {role_key: ["p1"], **household_data}},
    }
    return result


def _filter_inapplicable_laws(
    applicable_law_refs: list[str],
    code_items: list[CodeGenerationItem],
    input_params: dict,
    execution: ExecutionResult,
    raw_reasoning: str,
    variable_mappings: list[dict],
    language: str,
) -> tuple[list[str], list[CodeGenerationItem], dict, ExecutionResult, str, str]:
    """Remove laws whose boolean OpenFisca result evaluated to False.

    Only boolean variables are considered — a numeric variable returning 0
    does not mean "inapplicable" (e.g. ``ahv_employee_contribution`` or
    ``stipendium_betrag`` can legitimately be 0).

    Returns updated copies of all pipeline outputs.
    """
    if not execution or not execution.computed_values:
        return (applicable_law_refs, code_items, input_params, execution,
                raw_reasoning, _extract_legal_reasoning(raw_reasoning, language))

    # Build a map: openfisca_variable -> article_reference, system
    var_to_ref: dict[str, str] = {}
    var_to_system: dict[str, str] = {}
    for m in variable_mappings:
        var_to_ref[m["openfisca_variable"]] = m["article_reference"]
        var_to_system[m["openfisca_variable"]] = m.get("system", "openfisca_switzerland")

    # Identify laws to remove: only boolean variables that evaluated to False
    refs_to_remove: set[str] = set()
    for var_name, value in execution.computed_values.items():
        sys_key = var_to_system.get(var_name, "openfisca_switzerland")
        try:
            tbs = get_tbs(sys_key)
            var_def = tbs.variables.get(var_name)
            if var_def is None:
                continue
            if var_def.value_type is bool and value <= 0:
                ref = var_to_ref.get(var_name)
                if ref:
                    refs_to_remove.add(ref)
        except Exception:
            log.warning("Could not check value_type for %r", var_name)

    if not refs_to_remove:
        return (applicable_law_refs, code_items, input_params, execution,
                _filter_reasoning_by_applicable_laws(raw_reasoning, applicable_law_refs),
                _extract_legal_reasoning(
                    _filter_reasoning_by_applicable_laws(raw_reasoning, applicable_law_refs),
                    language,
                ))

    # Filter all outputs
    new_law_refs = [r for r in applicable_law_refs if r not in refs_to_remove]
    new_code_items = [c for c in code_items if c.article_reference not in refs_to_remove]
    new_input_params = {
        k: v for k, v in input_params.items() if k not in refs_to_remove
    }

    # Remove filtered variables from computed_values
    vars_to_remove = {
        var_name for var_name, ref in var_to_ref.items()
        if ref in refs_to_remove
    }
    new_computed = {
        k: v for k, v in execution.computed_values.items()
        if k not in vars_to_remove
    }
    new_execution = ExecutionResult(
        success=execution.success,
        computed_values=new_computed,
        error=execution.error,
        error_stage=execution.error_stage,
    )

    # Re-filter reasoning for remaining laws
    new_reasoning = _filter_reasoning_by_applicable_laws(raw_reasoning, new_law_refs)
    new_legal_reasoning = _extract_legal_reasoning(new_reasoning, language)

    # Edge case: all laws filtered out
    if not new_law_refs:
        new_execution = ExecutionResult(
            success=new_execution.success,
            computed_values=new_execution.computed_values,
            error="no_applicable_laws",
            error_stage=new_execution.error_stage,
        )

    return (new_law_refs, new_code_items, new_input_params, new_execution,
            new_reasoning, new_legal_reasoning)


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
