"""Structured results returned by autonomous-analytics tools."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from autonomous_analytics.models.evidence import Evidence
from autonomous_analytics.models.kpi import AnalyticsModel


class QueryResultSummary(AnalyticsModel):
    """A deliberately bounded result sample suitable for investigation state."""

    columns: list[str] = Field(default_factory=list, max_length=500)
    row_count: int = Field(default=0, ge=0)
    truncated: bool = False
    rows: list[dict[str, Any]] = Field(default_factory=list, max_length=500)


class ToolError(AnalyticsModel):
    code: str = Field(min_length=1, max_length=120)
    category: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=2_000)
    retryable: bool = False


class ToolResult(AnalyticsModel):
    success: bool
    evidence: Evidence | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> ToolResult:
        if self.success and (self.evidence is None or self.error is not None):
            raise ValueError(
                "a successful tool result requires evidence and cannot contain an error"
            )
        if not self.success and self.error is None:
            raise ValueError("a failed tool result requires a structured error")
        return self
