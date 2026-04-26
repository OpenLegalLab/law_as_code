"""Augmentation endpoint — POST /api/v1/augment."""

from __future__ import annotations

from fastapi import APIRouter

from oll_api_server.pipeline import get_augmentation_fields
from oll_api_server.schemas import AugmentField, AugmentRequest

router = APIRouter(prefix="/api/v1", tags=["augment"])


@router.post("/augment", response_model=list[AugmentField])
async def augment(request: AugmentRequest) -> list[AugmentField]:
    """Return all required input fields for the given applicable laws."""
    raw = get_augmentation_fields(request.applicable_laws)
    return [AugmentField(**f) for f in raw]
