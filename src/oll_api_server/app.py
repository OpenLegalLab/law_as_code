"""FastAPI application with lifespan management."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import dspy
from dotenv import load_dotenv
from fastapi import FastAPI

from oll_api_server.config import settings
from oll_api_server.pipeline import load_law_index, validate_law_index_variables
from oll_law_as_code.runner import warm_tbs_cache
from oll_api_server.routers import augment, classify, health

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configure_lm() -> None:
    """Configure the DSPy language model based on settings."""
    from case_classifier.openjustice_lm import OpenJusticeLM

    if settings.llm_provider.lower() == "openjustice":
        lm = OpenJusticeLM(
            model=settings.openjustice_model,
            api_key=settings.openjustice_api_key,
        )
    else:
        lm = dspy.LM(
            model=f"cerebras/{settings.cerebras_model}",
            api_key=settings.cerebras_api_key,
        )
    dspy.configure(lm=lm)
    log.info("Configured LLM provider: %s", settings.llm_provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load .env, configure LM, preload law index."""
    load_dotenv(_PROJECT_ROOT / "oll_2026_law_as_code_case_classifier" / ".env")

    # Re-read settings after dotenv (env vars now populated)
    settings.cerebras_api_key = settings.cerebras_api_key or os.getenv("CEREBRAS_API_KEY", "")
    settings.openjustice_api_key = settings.openjustice_api_key or os.getenv("OPENJUSTICE_API_KEY", "")
    settings.llm_provider = settings.llm_provider or os.getenv("LLM_PROVIDER", "cerebras")

    _configure_lm()
    load_law_index()
    warm_tbs_cache()
    validate_law_index_variables()
    log.info("Server ready")

    yield


app = FastAPI(
    title="OLL Legal Case Classifier API",
    version="0.1.0",
    description="Classify legal cases, generate OpenFisca code, and execute it.",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(classify.router)
app.include_router(augment.router)
