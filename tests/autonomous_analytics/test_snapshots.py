from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from autonomous_analytics.metrics.snapshots import DeterministicSnapshotBuilder
from autonomous_analytics.models.kpi import KPIObservation


def observations(values: list[float], *, target: float | None = None) -> list[KPIObservation]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        KPIObservation(
            timestamp=start + timedelta(days=index),
            value=value,
            target=target if index == len(values) - 1 else None,
        )
        for index, value in enumerate(values)
    ]


def test_snapshot_calculates_changes_trend_forecast_and_target() -> None:
    snapshot = DeterministicSnapshotBuilder(rolling_days=30, ewma_alpha=0.5).build(
        "signups",
        observations([100, 102, 104, 106, 108, 110, 112, 140], target=125),
    )

    assert snapshot.current == 140
    assert snapshot.change_1d == pytest.approx((140 - 112) / 112)
    assert snapshot.change_7d == pytest.approx(0.4)
    assert snapshot.change_30d is None
    assert snapshot.trend_slope is not None and snapshot.trend_slope > 0
    assert snapshot.trend_strength is not None
    assert snapshot.ewma is not None
    assert snapshot.forecast_delta == pytest.approx((140 - snapshot.ewma) / snapshot.ewma)
    assert snapshot.target_delta == pytest.approx(0.12)
    assert 0 < snapshot.anomaly_score <= 1


def test_snapshot_uses_robust_outlier_signal_and_tracks_completeness() -> None:
    series = observations([100, 100, 100, 100, 100, 100, 180])
    series[-1] = series[-1].model_copy(update={"complete": False})

    snapshot = DeterministicSnapshotBuilder().build("revenue", series)

    assert snapshot.robust_z_score == 10
    assert snapshot.anomaly_score > 0.9
    assert snapshot.latest_period_complete is False


def test_snapshot_handles_zero_baseline_and_short_history() -> None:
    snapshot = DeterministicSnapshotBuilder().build("orders", observations([0, 5]))

    assert snapshot.change_1d is None
    assert snapshot.robust_z_score is None
    assert snapshot.change_point_score == 0


def test_snapshot_rejects_empty_and_duplicate_timestamps() -> None:
    builder = DeterministicSnapshotBuilder()
    with pytest.raises(ValueError, match="at least one"):
        builder.build("revenue", [])
    duplicated = observations([1, 2])
    duplicated[1] = duplicated[1].model_copy(update={"timestamp": duplicated[0].timestamp})
    with pytest.raises(ValueError, match="unique"):
        builder.build("revenue", duplicated)


def test_observation_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone"):
        KPIObservation(timestamp=datetime(2026, 1, 1), value=1)
