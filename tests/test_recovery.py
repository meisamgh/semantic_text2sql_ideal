from __future__ import annotations

from sqlglot import parse_one

from semantic_text2sql.models import ColumnProfile, DatabaseProfile, ValueFrequency
from semantic_text2sql.recovery import (
    RecoveryCoordinator,
    RecoveryTools,
    _no_match_explanation,
    _plain_filter,
    recovery_feedback,
)


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
    assert trace.usage.max_tool_calls == 6
    assert trace.usage.max_database_probes == 8
    assert trace.usage.max_recovery_ms == 8_000
    assert trace.usage.budget_exhausted is False


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


def test_recovery_caps_database_filter_probes(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    tools = RecoveryTools(registry, "shop", schema, profile=None)
    predicates = " AND ".join(f"amount >= {index}" for index in range(1, 10))

    trace = RecoveryCoordinator(tools).investigate(
        question="Apply several amount filters",
        failed_sql=f"SELECT order_id FROM orders WHERE {predicates}",
        failure_code="ZERO_RESULT",
        failure_message="Executed with zero rows",
        allowed_tables=["orders"],
        mode="ZERO_RESULT",
    )

    assert trace.usage.database_probe_count == 8
    assert trace.usage.budget_exhausted is True


def test_missing_identifier_is_explained_in_plain_language(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    trace = RecoveryCoordinator(
        RecoveryTools(registry, "shop", schema, profile=None)
    ).investigate(
        question="How much did customer 600000 spend?",
        failed_sql="SELECT SUM(amount) FROM orders WHERE customer_id = 600000",
        failure_code="NULL_RESULT",
        failure_message="Aggregate returned NULL",
        allowed_tables=["orders"],
        mode="NULL_RESULT",
    )

    assert trace.diagnosis_code == "FILTER_NO_MATCH"
    assert trace.filter_checks[0]["subject_kind"] == "identifier"
    assert trace.filter_checks[0]["no_match_explanation"] == (
        "There is no customer with ID 600000 in this database. Because the selected customer "
        "is unavailable, the requested result cannot be calculated."
    )


def test_compact_date_range_is_rendered_for_stakeholders() -> None:
    tree = parse_one("SELECT * FROM usage WHERE Date BETWEEN '204308' AND '204311'")
    predicate = tree.args["where"].this

    assert _plain_filter(predicate) == "Date must be from August 2043 through November 2043"
    assert _no_match_explanation(predicate, "date") == (
        "No records were found from August 2043 through November 2043."
    )
