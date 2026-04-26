"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Input payload for the classification endpoint."""

    case_description: str = Field(
        ..., min_length=1, description="Natural language description of the legal case"
    )
    execute: bool = Field(
        default=True, description="Whether to execute generated code after classification"
    )
    period: str = Field(
        default="2024-01", description="OpenFisca simulation period (YYYY-MM)"
    )
    language: Literal["en", "de", "fr", "it"] = Field(
        default="en",
        description="Response language for reasoning and error messages",
    )


class CodeGenerationItem(BaseModel):
    """Result of code generation for a single law article."""

    article_reference: str
    openfisca_variable: str = ""
    parameter_yaml: str = ""
    reasoning: str = ""
    validation_passed: bool = False
    attempts: int = 0
    error: str = ""


class ExecutionResult(BaseModel):
    """Result of executing generated OpenFisca code."""

    success: bool
    computed_values: dict[str, float] = Field(default_factory=dict)
    error: str = ""
    error_stage: str = ""


class ClassifyResponse(BaseModel):
    """Full pipeline response."""

    syllogistic_reasoning: str = ""
    legal_reasoning: str = ""
    applicable_laws: list[str] = Field(default_factory=list)
    input_parameters: dict[str, Any] = Field(default_factory=dict)
    code_generation: list[CodeGenerationItem] = Field(default_factory=list)
    execution: ExecutionResult | None = None
    processing_time_seconds: float = 0.0
    stages_completed: list[str] = Field(default_factory=list)
    error: str = ""
