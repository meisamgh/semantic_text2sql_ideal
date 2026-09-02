"""Explicit and replayable investigation state."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from autonomous_analytics.models.evidence import Evidence
from autonomous_analytics.models.kpi import AnalyticsModel


class Hypothesis(AnalyticsModel):
    hypothesis_id: str = Field(min_length=1, max_length=120)
    statement: str = Field(min_length=1, max_length=4_000)
    status: Literal["proposed", "supported", "rejected", "unresolved"] = "proposed"
    confidence: float = Field(default=0, ge=0, le=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)


class InvestigationBudget(AnalyticsModel):
    max_sql_queries: int = Field(default=12, ge=0, le=100)
    max_python_calls: int = Field(default=5, ge=0, le=100)
    max_resource_searches: int = Field(default=5, ge=0, le=100)
    max_total_tool_calls: int = Field(default=25, ge=1, le=250)
    max_rows_per_query: int = Field(default=10_000, ge=1, le=1_000_000)


class InvestigationState(AnalyticsModel):
    case_id: str = Field(min_length=1, max_length=120)
    observation: str = Field(min_length=1, max_length=4_000)
    primary_metrics: list[str] = Field(min_length=1, max_length=50)
    business_context: dict[str, Any] = Field(default_factory=dict)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=100)
    rejected_hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=100)
    evidence: list[Evidence] = Field(default_factory=list, max_length=500)
    current_question: str | None = Field(default=None, max_length=4_000)
    affected_segments: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    confidence: float = Field(default=0, ge=0, le=1)
    sql_calls: int = Field(default=0, ge=0)
    python_calls: int = Field(default=0, ge=0)
    resource_calls: int = Field(default=0, ge=0)
    status: Literal["candidate", "investigating", "validating", "complete", "stopped"] = "candidate"
    stopping_reason: str | None = Field(default=None, max_length=1_000)
    budget: InvestigationBudget = Field(default_factory=InvestigationBudget)

    @property
    def total_tool_calls(self) -> int:
        return self.sql_calls + self.python_calls + self.resource_calls

    def can_call(self, tool: Literal["sql", "python", "resource"]) -> bool:
        if self.total_tool_calls >= self.budget.max_total_tool_calls:
            return False
        limits = {
            "sql": (self.sql_calls, self.budget.max_sql_queries),
            "python": (self.python_calls, self.budget.max_python_calls),
            "resource": (self.resource_calls, self.budget.max_resource_searches),
        }
        used, maximum = limits[tool]
        return used < maximum

    def record_call(self, tool: Literal["sql", "python", "resource"]) -> InvestigationState:
        if not self.can_call(tool):
            raise ValueError(f"{tool} investigation budget is exhausted")
        field = {"sql": "sql_calls", "python": "python_calls", "resource": "resource_calls"}[tool]
        return self.model_copy(update={field: getattr(self, field) + 1})

    def add_evidence(self, evidence: Evidence) -> InvestigationState:
        """Return a new state containing evidence without mutating the current state."""

        return self.model_copy(update={"evidence": [*self.evidence, evidence]})

    @model_validator(mode="after")
    def validate_hypothesis_partition(self) -> InvestigationState:
        active_ids = {item.hypothesis_id for item in self.hypotheses}
        rejected_ids = {item.hypothesis_id for item in self.rejected_hypotheses}
        if active_ids & rejected_ids:
            raise ValueError("a hypothesis cannot be active and rejected simultaneously")
        if any(item.status != "rejected" for item in self.rejected_hypotheses):
            raise ValueError("rejected_hypotheses must have status='rejected'")
        return self
