from __future__ import annotations

from semantic_text2sql.linker import select_schema
from semantic_text2sql.models import SchemaInfo


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


def test_semantic_required_columns_do_not_replace_primary_and_foreign_keys() -> None:
    schema = SchemaInfo.model_validate(
        {
            "db_id": "sales",
            "tables": [
                {
                    "name": "orders",
                    "create_sql": "",
                    "columns": [
                        {"name": "order_id", "data_type": "INTEGER", "primary_key": True},
                        {"name": "customer_id", "data_type": "INTEGER"},
                        {"name": "amount", "data_type": "REAL"},
                    ],
                },
                {
                    "name": "customers",
                    "create_sql": "",
                    "columns": [
                        {"name": "customer_id", "data_type": "INTEGER", "primary_key": True}
                    ],
                },
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

    _, selection = select_schema(
        "order amount",
        None,
        schema,
        required_tables=["orders"],
        required_columns=["orders.amount"],
    )

    assert set(selection.columns["orders"]) >= {"order_id", "customer_id", "amount"}
