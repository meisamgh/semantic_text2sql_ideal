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


def test_zero_result_recovery_inspects_every_filter(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    trace = RecoveryCoordinator(
        RecoveryTools(registry, "shop", schema, profile=None)
    ).investigate(
        question="List German completed orders above 10",
        failed_sql=(
            "SELECT o.order_id FROM orders o JOIN customers c "
            "ON c.customer_id = o.customer_id "
            "WHERE c.country = 'Germany' AND o.status = 'complete' AND o.amount > 10"
        ),
        failure_code="ZERO_RESULT",
        failure_message="Executed with zero rows",
        allowed_tables=["customers", "orders"],
        mode="ZERO_RESULT",
    )

    assert trace.mode == "ZERO_RESULT"
    assert [call.tool for call in trace.tool_calls] == [
        "inspect_schema",
        "inspect_column",
        "inspect_filters",
    ]
    filter_evidence = next(item for item in trace.evidence if item.startswith("All explicit"))
    assert "country = 'Germany'" in filter_evidence
    assert "status = 'complete'" in filter_evidence
    assert "amount > 10" in filter_evidence


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
