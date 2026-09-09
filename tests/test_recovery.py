from __future__ import annotations

from semantic_text2sql.recovery import RecoveryCoordinator, RecoveryTools, recovery_feedback


def test_recovery_graph_uses_schema_tool_for_unknown_column(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    trace = RecoveryCoordinator(
        RecoveryTools(registry, "shop", schema, profile=None)
    ).investigate(
        question="Count orders by customer",
        failed_sql="SELECT missing FROM orders",
        failure_code="DATABASE_ERROR",
        failure_message="no such column: missing",
        allowed_tables=["orders"],
    )

    assert trace.failure_category == "schema"
    assert [call.tool for call in trace.tool_calls] == ["inspect_schema"]
    assert "orders" in recovery_feedback(trace)


def test_value_probe_is_select_only_and_bounded(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    values = RecoveryTools(registry, "shop", schema, profile=None).sample_values(
        "customers", "country", allowed_tables=["customers"]
    )

    assert values["columns"] == ["country"]
    assert sorted(row[0] for row in values["rows"]) == ["Germany", "Italy"]


def test_provider_failure_does_not_probe_database(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    trace = RecoveryCoordinator(
        RecoveryTools(registry, "shop", schema, profile=None)
    ).investigate(
        question="Count orders",
        failed_sql="",
        failure_code="MODEL_ERROR",
        failure_message="provider rate limit",
        allowed_tables=["orders"],
    )

    assert trace.failure_category == "provider"
    assert trace.tool_calls == []
    assert trace.requires_human_review
