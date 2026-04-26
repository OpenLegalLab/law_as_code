"""Classification endpoint — POST /api/v1/classify."""

from __future__ import annotations

from fastapi import APIRouter

from oll_api_server.pipeline import run_pipeline
from oll_api_server.schemas import ClassifyRequest, ClassifyResponse

router = APIRouter(prefix="/api/v1", tags=["classify"])


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a legal case, generate OpenFisca code, and optionally execute it."""
    return await run_pipeline(request)
