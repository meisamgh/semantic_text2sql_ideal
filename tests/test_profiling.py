from __future__ import annotations

import sqlite3

from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.linker import cardinality_plan_context, compact_profile_context
from semantic_text2sql.models import SemanticContract
from semantic_text2sql.profiling import (
    ProfileStore,
    apply_bird_descriptions,
    profile_database,
)


def test_profile_store_loads_newer_relationship_metadata(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    path = tmp_path / "sqlite__compat.json"
    path.write_text(
        '{"db_id":"compat","dialect":"sqlite","profiled_at":"2026-01-01",'
        '"columns":[],"relationships":[{"parent_table":"left_table",'
        '"parent_column":"id","child_table":"right_table","child_column":"id",'
        '"type":"MANY_TO_MANY","parent_key_unique":false,"child_key_unique":false,'
        '"inferred":true,"confidence":0.8}]}'
    )

    profile = store.load("sqlite", "compat")

    assert profile is not None
    assert profile.relationships[0].type == "MANY_TO_MANY"
    assert profile.relationships[0].inferred is True
    assert profile.relationships[0].confidence == 0.8


def test_offline_profiler_captures_numeric_and_categorical_data(registry, tmp_path) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    by_column = {(item.table, item.column): item for item in profile.columns}

    amount = by_column[("orders", "amount")]
    status = by_column[("orders", "status")]
    customer_name = by_column[("customers", "name")]
    assert amount.semantic_type == "numeric"
    assert amount.minimum == "20.0"
    assert amount.maximum == "100.0"
    assert amount.median == 50.0
    assert [(item.value, item.count) for item in status.top_values] == [
        ("complete", 2),
        ("pending", 1),
    ]
    assert customer_name.semantic_type == "personal_name"
    assert customer_name.top_values == []
    assert customer_name.examples == []
    assert by_column[("customers", "country")].allowed_values == ["Germany", "Italy"]
    assert [item.model_dump() for item in profile.relationships] == [
        {
            "parent_table": "customers",
            "parent_column": "customer_id",
            "child_table": "orders",
            "child_column": "customer_id",
            "parent_columns": ["customer_id"],
            "child_columns": ["customer_id"],
                "type": "ONE_TO_MANY",
                "parent_key_unique": True,
                "child_key_unique": False,
                "inferred": False,
                "confidence": 1.0,
            }
    ]
    context = compact_profile_context(registry.inspect("shop"), profile)
    assert "RELATIONSHIP" not in context
    plan = cardinality_plan_context(
        registry.inspect("shop"),
        profile,
        SemanticContract(grain=["customers.customer_id"], measures=["customers.customer_id"]),
    )
    assert '"type":"ONE_TO_MANY"' in plan
    assert '"target_grain":["customers.customer_id"]' in plan
    store = ProfileStore(tmp_path / "profiles")
    path = store.save(profile)
    assert path.is_file()
    assert store.load("sqlite", "shop") == profile


def test_composite_relationship_is_profiled_as_one_key(tmp_path) -> None:  # type: ignore[no-untyped-def]
    directory = tmp_path / "composite"
    directory.mkdir()
    with sqlite3.connect(directory / "composite.sqlite") as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE parent (a INTEGER, b INTEGER, value REAL, PRIMARY KEY (a, b));
            CREATE TABLE child (
              id INTEGER PRIMARY KEY,
              a INTEGER,
              b INTEGER,
              FOREIGN KEY (a, b) REFERENCES parent (a, b)
            );
            INSERT INTO parent VALUES (1, 1, 10), (1, 2, 20);
            INSERT INTO child VALUES (1, 1, 1), (2, 1, 1), (3, 1, 2);
            """
        )
    profile = profile_database(DatabaseRegistry(tmp_path), "composite", "sqlite")

    assert len(profile.relationships) == 1
    relationship = profile.relationships[0]
    assert relationship.parent_columns == ["a", "b"]
    assert relationship.child_columns == ["a", "b"]
    assert relationship.type == "ONE_TO_MANY"

    store = ProfileStore(tmp_path / "profiles")
    path = store.save(profile)
    assert path.is_file()
    assert store.load("sqlite", "composite") == profile


def test_bird_descriptions_enrich_profiles(registry, tmp_path) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    description_dir = tmp_path / "database_description"
    description_dir.mkdir()
    (description_dir / "orders.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "amount,order value,total value of the order,number,unit: EUR\n"
    )

    enriched = apply_bird_descriptions(profile, description_dir)
    amount = next(
        item for item in enriched.columns if item.table == "orders" and item.column == "amount"
    )

    assert amount.description == "total value of the order; unit: EUR"
    assert amount.aliases == ["order value"]
    assert amount.semantic_type == "numeric"
