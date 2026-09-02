"""Traceable evidence contracts for analytical tools."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import AwareDatetime, Field

from autonomous_analytics.models.kpi import AnalyticsModel


class Evidence(AnalyticsModel):
    evidence_id: str = Field(min_length=1, max_length=120)
    source_type: Literal["metric", "sql", "python", "resource", "relationship", "funnel"]
    question: str = Field(min_length=1, max_length=4_000)
    result_summary: dict[str, Any]
    sql: str | None = Field(default=None, max_length=50_000)
    confidence: float | None = Field(default=None, ge=0, le=1)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    source_reference: str | None = Field(default=None, max_length=2_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceValidation(AnalyticsModel):
    valid: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
