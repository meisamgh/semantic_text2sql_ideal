from __future__ import annotations

from semantic_text2sql.models import ColumnProfile, DatabaseProfile, ValueFrequency
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
        question="List German pending orders above 10",
        failed_sql=(
            "SELECT o.order_id FROM orders o JOIN customers c "
            "ON c.customer_id = o.customer_id "
            "WHERE c.country = 'Germany' AND o.status = 'pending' AND o.amount > 10"
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
        "probe_filter_counts",
    ]
    filter_evidence = next(item for item in trace.evidence if item.startswith("All explicit"))
    assert "country = 'Germany'" in filter_evidence
    assert "status = 'pending'" in filter_evidence
    assert "amount > 10" in filter_evidence
    assert len(trace.filter_checks) == 3
    assert all(item["status"] == "MATCH" for item in trace.filter_checks)
    assert trace.diagnosis_code == "FILTER_COMBINATION_EMPTY"
    assert trace.usage.llm_calls == 0
    assert trace.usage.token_usage.total_tokens == 0
    assert trace.usage.database_probe_count == 3
    assert trace.usage.latency_ms >= 0


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


def test_sampled_date_values_are_not_treated_as_complete_domain(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    profile = DatabaseProfile(
        db_id="shop",
        dialect="sqlite",
        profiled_at="2026-09-09T00:00:00Z",
        columns=[
            ColumnProfile(
                table="orders",
                column="amount",
                database_type="REAL",
                semantic_type="numeric",
                row_count=3,
                null_count=0,
                null_ratio=0,
                distinct_count=3,
                top_values=[ValueFrequency(value="100.0", count=1)],
            )
        ],
    )
    trace = RecoveryCoordinator(
        RecoveryTools(registry, "shop", schema, profile=profile)
    ).investigate(
        question="Find orders below 60",
        failed_sql="SELECT SUM(amount) FROM orders WHERE amount < 60",
        failure_code="NULL_RESULT",
        failure_message="Aggregate returned NULL",
        allowed_tables=["orders"],
        mode="NULL_RESULT",
    )

    assert trace.diagnosis_code != "FILTER_VALUE_NOT_FOUND"
