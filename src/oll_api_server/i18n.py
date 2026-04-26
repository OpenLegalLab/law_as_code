"""Internationalization — translations and language utilities."""

from __future__ import annotations

from typing import Literal

SupportedLanguage = Literal["en", "de", "fr", "it"]

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "it": "Italian",
}

# ---------------------------------------------------------------------------
# LLM language instruction
# ---------------------------------------------------------------------------

_LANGUAGE_INSTRUCTION = (
    "[LANGUAGE INSTRUCTION: Write all reasoning, analysis, and conclusions "
    "in {name} ({code}). "
    "You MUST keep the structural section labels exactly as: "
    '"Major Premise", "Minor Premise", and "Conclusion" in English. '
    "Only the content should be in {name}.]\n\n"
)


def build_language_prefix(language: str) -> str:
    """Return the instruction to prepend to case_description.

    Returns empty string for English (no instruction needed).
    """
    if language == "en":
        return ""
    name = LANGUAGE_NAMES.get(language, language)
    return _LANGUAGE_INSTRUCTION.format(name=name, code=language)


# ---------------------------------------------------------------------------
# Section header variants for _extract_legal_reasoning fallback
# ---------------------------------------------------------------------------

SECTION_HEADERS: dict[str, dict[str, list[str]]] = {
    "minor_premise": {
        "en": ["minor premise"],
        "de": ["untersatz", "nebensatz", "minor premise"],
        "fr": ["prémisse mineure", "premise mineure", "minor premise"],
        "it": ["premessa minore", "minor premise"],
    },
    "conclusion": {
        "en": ["conclusion"],
        "de": ["schlussfolgerung", "konklusion", "conclusion"],
        "fr": ["conclusion"],
        "it": ["conclusione", "conclusion"],
    },
    "major_premise": {
        "en": ["major premise"],
        "de": ["obersatz", "hauptsatz", "major premise"],
        "fr": ["prémisse majeure", "premise majeure", "major premise"],
        "it": ["premessa maggiore", "major premise"],
    },
}


def get_section_headers(language: str) -> dict[str, list[str]]:
    """Return recognised section header variants for *language*.

    Merges English defaults with the target language so the parser
    always recognises both.
    """
    merged: dict[str, list[str]] = {}
    for canonical, by_lang in SECTION_HEADERS.items():
        en = by_lang.get("en", [])
        target = by_lang.get(language, [])
        seen: set[str] = set()
        combined: list[str] = []
        for h in en + target:
            if h not in seen:
                combined.append(h)
                seen.add(h)
        merged[canonical] = combined
    return merged


# ---------------------------------------------------------------------------
# Error message translations
# ---------------------------------------------------------------------------

_ERRORS: dict[str, dict[str, str]] = {
    "classification_failed": {
        "en": "Classification failed: {detail}",
        "de": "Klassifizierung fehlgeschlagen: {detail}",
        "fr": "Échec de la classification\u00a0: {detail}",
        "it": "Classificazione fallita: {detail}",
    },
    "no_applicable_laws": {
        "en": "No applicable laws identified",
        "de": "Keine anwendbaren Gesetze identifiziert",
        "fr": "Aucune loi applicable identifiée",
        "it": "Nessuna legge applicabile identificata",
    },
    "no_variable_mappings": {
        "en": "No variable mappings found for applicable laws",
        "de": "Keine Variablen-Zuordnungen für anwendbare Gesetze gefunden",
        "fr": "Aucune correspondance de variable trouvée pour les lois applicables",
        "it": "Nessuna mappatura di variabili trovata per le leggi applicabili",
    },
    "execution_failed": {
        "en": "Execution failed: {detail}",
        "de": "Ausführung fehlgeschlagen: {detail}",
        "fr": "Échec de l'exécution\u00a0: {detail}",
        "it": "Esecuzione fallita: {detail}",
    },
}


def translate_error(key: str, language: str = "en", **kwargs: object) -> str:
    """Look up a translated error message.  Falls back to English."""
    messages = _ERRORS.get(key, {})
    template = messages.get(language) or messages.get("en", key)
    try:
        return template.format(**kwargs)
    except KeyError:
        return template
