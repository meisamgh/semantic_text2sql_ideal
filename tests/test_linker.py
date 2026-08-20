from __future__ import annotations

from semantic_text2sql.linker import (
    compact_interpreter_profile_context,
    compact_profile_context,
    select_schema,
)
from semantic_text2sql.models import DatabaseProfile, SchemaInfo


def test_table_first_column_second_preserves_join_keys() -> None:
    schema = SchemaInfo.model_validate(
        {
            "db_id": "sales",
            "tables": [
                {
                    "name": "customers",
                    "create_sql": "",
                    "columns": [
                        {"name": "customer_id", "data_type": "INTEGER", "primary_key": True},
                        {"name": "name", "data_type": "TEXT"},
                        {"name": "country", "data_type": "TEXT"},
                    ],
                },
                {
                    "name": "orders",
                    "create_sql": "",
                    "columns": [
                        {"name": "order_id", "data_type": "INTEGER", "primary_key": True},
                        {"name": "customer_id", "data_type": "INTEGER"},
                        {"name": "amount", "data_type": "REAL"},
                        {"name": "status", "data_type": "TEXT"},
                    ],
                },
                *[
                    {
                        "name": f"unrelated_{number}",
                        "create_sql": "",
                        "columns": [{"name": "value", "data_type": "TEXT"}],
                    }
                    for number in range(5)
                ],
            ],
            "relationships": [
                {
                    "from_table": "orders",
                    "from_column": "customer_id",
                    "to_table": "customers",
                    "to_column": "customer_id",
                }
            ],
        }
    )

    reduced, selection = select_schema(
        "total order amount by customer country",
        None,
        schema,
        max_tables=2,
        max_columns_per_table=1,
    )

    assert selection.tables == ["customers", "orders"]
    assert set(selection.columns["customers"]) >= {"customer_id", "country"}
    assert set(selection.columns["orders"]) >= {"order_id", "customer_id", "amount"}
    assert len(reduced.relationships) == 1


def test_compact_profile_context_only_contains_retrieved_columns() -> None:
    schema = SchemaInfo.model_validate(
        {
            "db_id": "shop",
            "tables": [
                {
                    "name": "orders",
                    "create_sql": "",
                    "columns": [{"name": "status", "data_type": "TEXT"}],
                }
            ],
        }
    )
    profile = DatabaseProfile.model_validate(
        {
            "db_id": "shop",
            "dialect": "sqlite",
            "profiled_at": "2026-01-01T00:00:00+00:00",
            "columns": [
                {
                    "table": "orders",
                    "column": "status",
                    "database_type": "TEXT",
                    "semantic_type": "categorical",
                    "description": "Current order status",
                    "aliases": ["state"],
                    "row_count": 3,
                    "null_count": 0,
                    "null_ratio": 0,
                    "distinct_count": 2,
                    "empty_string_count": 1,
                    "minimum": "complete",
                    "maximum": "pending",
                    "observed_format": "status-code",
                    "timezone": "UTC",
                    "median": 7.5,
                    "quantiles": {"q25": 7.0, "q75": 8.0},
                    "text_length_min": 7,
                    "text_length_median": 7.5,
                    "text_length_max": 8,
                    "detected_pattern": "lowercase-word",
                    "suspected_sentinels": [
                        {"value": "-", "confidence": 0.6, "reason": "documented missing"}
                    ],
                    "top_values": [
                        {"value": "complete", "count": 2},
                        {"value": "pending", "count": 1},
                    ],
                },
                {
                    "table": "customers",
                    "column": "email",
                    "database_type": "TEXT",
                    "semantic_type": "text",
                    "row_count": 3,
                    "null_count": 0,
                    "null_ratio": 0,
                },
            ],
        }
    )

    context = compact_profile_context(schema, profile)

    assert "orders.status" in context
    assert "complete (2)" in context
    assert "storage_type=TEXT" in context
    assert "description=" not in context
    assert "aliases=" not in context
    assert "timezone=" not in context
    assert "median=" not in context
    assert "quantiles=" not in context
    assert "sql_null=not_observed" in context
    assert "empty_strings=1" in context
    assert "suspected_missing_representations" in context
    assert "observed_format=status-code" in context
    assert "text_length=" not in context
    assert "detected_pattern=lowercase-word" in context
    assert "customers.email" not in context

    interpreter_context = compact_interpreter_profile_context(
        "Show orders by status", None, profile
    )
    assert "orders.status" in interpreter_context
    assert "description=Current order status" in interpreter_context
    assert "aliases=['state']" in interpreter_context
    assert "timezone=" not in interpreter_context
    assert "empty_strings=1" in interpreter_context
    assert "observed_format=status-code" in interpreter_context
    assert "text_length=" not in interpreter_context

    lean = compact_profile_context(schema, profile, "Count orders")
    assert "storage_type=TEXT" in lean
    assert "semantic_type=" not in lean
    assert "sql_null=" not in lean
    assert "empty_strings=" not in lean
    assert "suspected_missing_representations" not in lean
    assert "observed_format=" not in lean
    assert "top_values=" not in lean
    assert "text_length=" not in lean

    filtered = compact_profile_context(
        schema,
        profile,
        "Show pending orders with missing status values",
    )
    assert "pending (1)" in filtered
    assert "sql_null=not_observed" in filtered
    assert "empty_strings=1" in filtered
    assert "suspected_missing_representations" in filtered
