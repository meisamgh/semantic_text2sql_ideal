from __future__ import annotations

import os

import pytest

from semantic_text2sql.database import DatabaseError
from semantic_text2sql.models import SchemaInfo
from semantic_text2sql.postgres import PostgresRegistry
from semantic_text2sql.validator import validate_sql


def test_postgres_dialect_validator_accepts_grounded_select() -> None:
    schema = SchemaInfo.model_validate(
        {
            "db_id": "books_postgres",
            "dialect": "postgres",
            "tables": [
                {
                    "name": "books",
                    "create_sql": "CREATE TABLE books (title text)",
                    "columns": [{"name": "title", "data_type": "text", "primary_key": False}],
                }
            ],
        }
    )

    result = validate_sql(
        "SELECT title FROM books ORDER BY title NULLS LAST",
        schema,
        dialect="postgres",
    )

    assert result.valid is True


def test_validator_blocks_postgres_row_lock() -> None:
    schema = SchemaInfo.model_validate(
        {
            "db_id": "books_postgres",
            "dialect": "postgres",
            "tables": [
                {
                    "name": "books",
                    "create_sql": "CREATE TABLE books (title text)",
                    "columns": [{"name": "title", "data_type": "text", "primary_key": False}],
                }
            ],
        }
    )

    result = validate_sql("SELECT title FROM books FOR UPDATE", schema, dialect="postgres")

    assert result.valid is False
    assert result.code == "SQL_NOT_READ_ONLY"


@pytest.mark.skipif(not os.environ.get("POSTGRES_BOOKS_DSN"), reason="PostgreSQL DSN not set")
def test_postgres_runtime_role_is_read_only() -> None:
    registry = PostgresRegistry({"books_postgres": os.environ["POSTGRES_BOOKS_DSN"]})

    schema = registry.inspect("books_postgres")
    columns, rows, truncated = registry.execute(
        "books_postgres",
        "SELECT title FROM books ORDER BY title",
        max_rows=10,
    )

    assert schema.dialect == "postgres"
    assert {table.name for table in schema.tables} >= {"authors", "books", "reviews"}
    assert columns == ["title"]
    assert len(rows) == 4
    assert truncated is False
    with pytest.raises(DatabaseError):
        registry.execute(
            "books_postgres",
            "INSERT INTO categories(name) VALUES ('blocked') RETURNING name",
            max_rows=10,
        )
