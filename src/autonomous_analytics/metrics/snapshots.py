"""Replaceable deterministic KPI snapshot calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timedelta
from statistics import fmean, median, pstdev
from typing import Protocol

from autonomous_analytics.models.kpi import KPIObservation, KPISnapshot


class SnapshotBuilder(Protocol):
    def build(self, metric: str, observations: Sequence[KPIObservation]) -> KPISnapshot: ...


class DeterministicSnapshotBuilder:
    """Small, dependency-free baseline whose algorithms can be replaced independently."""

    def __init__(self, *, rolling_days: int = 30, ewma_alpha: float = 0.3) -> None:
        if rolling_days < 2:
            raise ValueError("rolling_days must be at least 2")
        if not 0 < ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        self.rolling_days = rolling_days
        self.ewma_alpha = ewma_alpha

    def build(self, metric: str, observations: Sequence[KPIObservation]) -> KPISnapshot:
        if not metric:
            raise ValueError("metric must not be empty")
        if not observations:
            raise ValueError("at least one observation is required")
        ordered = sorted(observations, key=lambda item: item.timestamp)
        timestamps = [item.timestamp for item in ordered]
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("observation timestamps must be unique")

        latest = ordered[-1]
        window_start = latest.timestamp - timedelta(days=self.rolling_days)
        history = [item for item in ordered if item.timestamp >= window_start]
        history_values = [item.value for item in history]
        baseline_values = history_values[:-1]
        rolling_mean = fmean(baseline_values) if baseline_values else None
        rolling_std = pstdev(baseline_values) if len(baseline_values) >= 2 else None
        robust_z = _robust_z_score(latest.value, baseline_values)
        ewma_forecast = _ewma([item.value for item in ordered[:-1]], self.ewma_alpha)
        trend_slope, trend_strength = _linear_trend(history)
        change_point_score = _change_point_score(history_values)
        forecast_delta = _relative_change(latest.value, ewma_forecast)
        target = latest.target
        target_delta = _relative_change(latest.value, target)
        components = [
            min(abs(robust_z or 0) / 4, 1),
            min(change_point_score / 3, 1),
            min(abs(forecast_delta or 0), 1),
        ]
        anomaly_score = 1 - math.prod(1 - component for component in components)

        return KPISnapshot(
            metric=metric,
            as_of=latest.timestamp,
            current=latest.value,
            change_1d=_period_change(ordered, 1),
            change_7d=_period_change(ordered, 7),
            change_30d=_period_change(ordered, 30),
            rolling_mean=rolling_mean,
            rolling_std=rolling_std,
            robust_z_score=robust_z,
            ewma=ewma_forecast,
            trend_slope=trend_slope,
            trend_strength=trend_strength,
            change_point_score=change_point_score,
            anomaly_score=max(0, min(anomaly_score, 1)),
            forecast_delta=forecast_delta,
            target_delta=target_delta,
            observation_count=len(ordered),
            latest_period_complete=latest.complete,
        )


def _period_change(observations: Sequence[KPIObservation], days: int) -> float | None:
    latest = observations[-1]
    cutoff = latest.timestamp - timedelta(days=days)
    eligible = [item for item in observations[:-1] if item.timestamp <= cutoff]
    if not eligible:
        return None
    return _relative_change(latest.value, eligible[-1].value)


def _relative_change(value: float, baseline: float | None) -> float | None:
    if baseline is None or baseline == 0:
        return None
    return (value - baseline) / abs(baseline)


def _ewma(values: Sequence[float], alpha: float) -> float | None:
    if not values:
        return None
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return result


def _robust_z_score(value: float, baseline: Sequence[float]) -> float | None:
    if len(baseline) < 3:
        return None
    center = median(baseline)
    mad = median(abs(item - center) for item in baseline)
    if mad == 0:
        return 0 if value == center else math.copysign(10.0, value - center)
    return 0.6745 * (value - center) / mad


def _linear_trend(observations: Sequence[KPIObservation]) -> tuple[float | None, float | None]:
    if len(observations) < 2:
        return None, None
    origin = observations[0].timestamp
    x = [(item.timestamp - origin).total_seconds() / 86_400 for item in observations]
    y = [item.value for item in observations]
    x_mean = fmean(x)
    y_mean = fmean(y)
    denominator = sum((item - x_mean) ** 2 for item in x)
    if denominator == 0:
        return None, None
    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y, strict=True)) / denominator
    total = sum((item - y_mean) ** 2 for item in y)
    if total == 0:
        return slope, 1.0
    residual = sum(
        (yi - (y_mean + slope * (xi - x_mean))) ** 2 for xi, yi in zip(x, y, strict=True)
    )
    return slope, max(0, min(1 - residual / total, 1))


def _change_point_score(values: Sequence[float]) -> float:
    if len(values) < 6:
        return 0
    split = len(values) // 2
    before = values[:split]
    after = values[split:]
    scale = pstdev(values)
    if scale == 0:
        return 0
    return abs(fmean(after) - fmean(before)) / scale
