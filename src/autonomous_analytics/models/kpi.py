"""Governed KPI, relationship, observation, and snapshot contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class AnalyticsModel(BaseModel):
    """Strict base model for replayable analytics state."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Priority = Literal["none", "low", "medium", "high", "critical"]


class StakeholderPriority(AnalyticsModel):
    ceo: Priority = "none"
    finance: Priority = "none"
    marketing: Priority = "none"
    product: Priority = "none"
    operations: Priority = "none"


class ExpectedRelationship(AnalyticsModel):
    target: str = Field(min_length=1, max_length=120)
    direction: Literal["positive", "negative", "none"]
    lag_min_days: int = Field(default=0, ge=0, le=365)
    lag_max_days: int = Field(default=0, ge=0, le=365)
    expected_elasticity: float | None = None
    description: str | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def validate_lag(self) -> ExpectedRelationship:
        if self.lag_max_days < self.lag_min_days:
            raise ValueError("lag_max_days must be greater than or equal to lag_min_days")
        return self


class KPIDefinition(AnalyticsModel):
    name: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    description: str = Field(min_length=1, max_length=2_000)
    business_meaning: str | None = Field(default=None, max_length=2_000)
    business_area: str = Field(min_length=1, max_length=120)
    definition_reference: str = Field(min_length=1, max_length=2_000)
    grain: list[str] = Field(min_length=1, max_length=20)
    dimensions: list[str] = Field(default_factory=list, max_length=100)
    upstream_metrics: list[str] = Field(default_factory=list, max_length=100)
    downstream_metrics: list[str] = Field(default_factory=list, max_length=100)
    expected_relationships: list[ExpectedRelationship] = Field(default_factory=list, max_length=100)
    targets: dict[str, float] = Field(default_factory=dict)
    directionality: Literal["higher_is_better", "lower_is_better", "neutral"] = "neutral"
    business_importance: float = Field(default=0.5, ge=0, le=1)
    stakeholder_priority: StakeholderPriority = Field(default_factory=StakeholderPriority)
    owner: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_references(self) -> KPIDefinition:
        referenced = [*self.upstream_metrics, *self.downstream_metrics]
        referenced.extend(item.target for item in self.expected_relationships)
        if self.name in referenced:
            raise ValueError("a KPI cannot reference itself as an upstream/downstream metric")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("dimensions must be unique")
        return self


class KPIObservation(AnalyticsModel):
    timestamp: AwareDatetime
    value: float
    target: float | None = None
    complete: bool = True
    dimensions: dict[str, str] = Field(default_factory=dict)


class KPISnapshot(AnalyticsModel):
    metric: str
    as_of: AwareDatetime
    current: float
    change_1d: float | None = None
    change_7d: float | None = None
    change_30d: float | None = None
    rolling_mean: float | None = None
    rolling_std: float | None = Field(default=None, ge=0)
    robust_z_score: float | None = None
    ewma: float | None = None
    trend_slope: float | None = None
    trend_strength: float | None = Field(default=None, ge=0, le=1)
    change_point_score: float | None = Field(default=None, ge=0)
    anomaly_score: float = Field(default=0, ge=0, le=1)
    forecast_delta: float | None = None
    target_delta: float | None = None
    observation_count: int = Field(ge=1)
    latest_period_complete: bool = True
    method_version: str = "deterministic-v1"
