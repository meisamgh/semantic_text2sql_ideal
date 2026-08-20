from __future__ import annotations

import sqlite3

from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.grain_validator import validate_grain_cardinality
from semantic_text2sql.models import (
    AggregationStage,
    MetricSelector,
    SemanticContract,
    ValidationResult,
)
from semantic_text2sql.profiling import profile_database


def _valid() -> ValidationResult:
    return ValidationResult(valid=True, code="SQL_VALID", message="valid")


def test_rejects_wrong_aggregation_stage_grain(registry) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    contract = SemanticContract(
        measures=["orders.amount"],
        aggregation_stages=[
            AggregationStage(
                name="customer_total",
                function="sum",
                input="orders.amount",
                group_by=["orders.customer_id"],
                output_grain=["orders.customer_id"],
            )
        ],
    )
    result = validate_grain_cardinality(
        "SELECT status, SUM(amount) FROM orders GROUP BY status",
        contract,
        profile,
        _valid(),
        dialect="sqlite",
    )

    assert result.code == "AGGREGATION_GRAIN_MISMATCH"


def test_accepts_matching_aggregation_stage_grain(registry) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    contract = SemanticContract(
        requires_grouping=True,
        grain=["orders.customer_id"],
        aggregation_stages=[
            AggregationStage(
                name="customer_total",
                function="sum",
                input="orders.amount",
                output_grain=["orders.customer_id"],
            )
        ],
    )
    result = validate_grain_cardinality(
        "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",
        contract,
        profile,
        _valid(),
        dialect="sqlite",
    )

    assert result.valid is True
    assert "grain:aggregation_stages" in result.semantic_checks
    assert "grain:final_output" in result.semantic_checks


def test_count_star_matches_count_of_non_null_primary_key(registry) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    contract = SemanticContract(
        aggregation_stages=[
            AggregationStage(
                name="german_count",
                function="count",
                input="customers.customer_id",
            ),
            AggregationStage(
                name="italian_count",
                function="count",
                input="customers.customer_id",
            ),
        ]
    )
    sql = (
        "SELECT CAST((SELECT COUNT(*) FROM customers WHERE country = 'Germany') AS REAL) / "
        "NULLIF((SELECT COUNT(*) FROM customers WHERE country = 'Italy'), 0) AS ratio"
    )

    result = validate_grain_cardinality(
        sql,
        contract,
        profile,
        _valid(),
        dialect="sqlite",
    )

    assert result.valid is True
    assert "grain:aggregation_stages" in result.semantic_checks


def test_rejects_incomplete_composite_join(tmp_path) -> None:  # type: ignore[no-untyped-def]
    directory = tmp_path / "composite"
    directory.mkdir()
    with sqlite3.connect(directory / "composite.sqlite") as connection:
        connection.executescript(
            """
            CREATE TABLE parent (a INTEGER, b INTEGER, PRIMARY KEY (a, b));
            CREATE TABLE child (
              id INTEGER PRIMARY KEY, a INTEGER, b INTEGER,
              FOREIGN KEY (a, b) REFERENCES parent (a, b)
            );
            INSERT INTO parent VALUES (1, 1);
            INSERT INTO child VALUES (1, 1, 1);
            """
        )
    registry = DatabaseRegistry(tmp_path)
    profile = profile_database(registry, "composite", "sqlite")
    result = validate_grain_cardinality(
        "SELECT COUNT(*) FROM parent p JOIN child c ON c.a = p.a",
        SemanticContract(),
        profile,
        _valid(),
        dialect="sqlite",
    )

    assert result.code == "COMPOSITE_JOIN_INCOMPLETE"


def test_rejects_distinct_as_aggregate_fanout_repair(registry) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    result = validate_grain_cardinality(
        "SELECT DISTINCT SUM(c.customer_id) FROM customers c "
        "JOIN orders o ON o.customer_id = c.customer_id",
        SemanticContract(),
        profile,
        _valid(),
        dialect="sqlite",
    )

    assert result.code == "DISTINCT_DUPLICATION_REPAIR"


def test_rejects_wrong_final_output_grain(registry) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    result = validate_grain_cardinality(
        "SELECT status, COUNT(*) FROM orders GROUP BY status",
        SemanticContract(requires_grouping=True, grain=["customers.country"]),
        profile,
        _valid(),
        dialect="sqlite",
    )

    assert result.code == "OUTPUT_GRAIN_MISMATCH"


def test_accepts_outer_ranking_over_cte_grouped_at_required_grain(registry) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    result = validate_grain_cardinality(
        "WITH customer_totals AS ("
        "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id"
        ") SELECT customer_id, total FROM customer_totals ORDER BY total DESC LIMIT 1",
        SemanticContract(requires_grouping=True, grain=["orders.customer_id"]),
        profile,
        _valid(),
        dialect="sqlite",
    )

    assert result.valid is True


def test_count_star_matches_unqualified_non_null_key_in_cte_scope(registry) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    contract = SemanticContract(
        aggregation_stages=[
            AggregationStage(
                name="customer_count",
                function="count",
                input="customer_id",
            )
        ]
    )
    result = validate_grain_cardinality(
        "WITH counts AS (SELECT COUNT(*) AS customer_count FROM customers) "
        "SELECT customer_count FROM counts",
        contract,
        profile,
        _valid(),
        dialect="sqlite",
    )

    assert result.valid is True


def test_rejects_ranking_at_wrong_partition_grain(registry) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    contract = SemanticContract(
        selectors=[
            MetricSelector(
                name="largest_order",
                function="argmax",
                metric="orders.amount",
                partition_by=["customers.country"],
            )
        ]
    )
    result = validate_grain_cardinality(
        "SELECT ROW_NUMBER() OVER (PARTITION BY o.status ORDER BY o.amount DESC) AS rn "
        "FROM orders o",
        contract,
        profile,
        _valid(),
        dialect="sqlite",
    )

    assert result.code == "RANKING_GRAIN_MISMATCH"
