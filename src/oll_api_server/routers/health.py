"""Health check and law catalogue endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from oll_api_server.pipeline import get_cached_laws

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Basic liveness check."""
    laws = get_cached_laws()
    return {
        "status": "ok",
        "laws_loaded": len(laws),
    }


@router.get("/laws")
async def list_laws(
    lang: Literal["en", "de", "fr", "it"] = Query(
        default="en", description="Response language (for future translated metadata)"
    ),
) -> list[dict]:
    """Return the loaded law catalogue."""
    laws = get_cached_laws()
    return [
        {
            "article_reference": law["article_reference"],
            "law_name": law["law_name"],
            "openfisca_variable": law["openfisca_variable"],
            "input_parameters": law.get("input_parameters", []),
            "language": lang,
        }
        for law in laws
    ]
