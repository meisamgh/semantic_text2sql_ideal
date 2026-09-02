from __future__ import annotations

import json

import pytest

from autonomous_analytics.metrics.registry import KPIRegistry, KPIRegistryError
from autonomous_analytics.models.kpi import ExpectedRelationship, KPIDefinition


def metric(name: str, **updates: object) -> KPIDefinition:
    values: dict[str, object] = {
        "name": name,
        "description": f"Definition for {name}",
        "business_area": "growth",
        "definition_reference": f"metrics.{name}",
        "grain": ["day"],
    }
    values.update(updates)
    return KPIDefinition.model_validate(values)


def test_registry_validates_relationships_and_returns_related_metrics() -> None:
    signups = metric(
        "signups",
        downstream_metrics=["purchases"],
        expected_relationships=[
            ExpectedRelationship(
                target="purchases", direction="positive", lag_min_days=3, lag_max_days=10
            )
        ],
    )
    purchases = metric("purchases", upstream_metrics=["signups"])
    registry = KPIRegistry([signups, purchases])

    registry.validate_relationships()

    assert [item.name for item in registry.related("signups")] == ["purchases"]


def test_registry_rejects_duplicates_and_unknown_relationships() -> None:
    registry = KPIRegistry([metric("signups", downstream_metrics=["purchases"])])

    with pytest.raises(KPIRegistryError, match="already registered"):
        registry.register(metric("signups"))
    with pytest.raises(KPIRegistryError, match="signups->purchases"):
        registry.validate_relationships()


def test_registry_loads_strict_json(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps([metric("revenue").model_dump(mode="json")]), encoding="utf-8")

    registry = KPIRegistry.from_json(path)

    assert registry.get("revenue").definition_reference == "metrics.revenue"


def test_kpi_definition_rejects_invalid_lag_and_self_reference() -> None:
    with pytest.raises(ValueError, match="lag_max_days"):
        ExpectedRelationship(
            target="purchases", direction="positive", lag_min_days=10, lag_max_days=3
        )
    with pytest.raises(ValueError, match="cannot reference itself"):
        metric("signups", downstream_metrics=["signups"])
