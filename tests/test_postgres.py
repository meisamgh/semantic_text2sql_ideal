from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from semantic_text2sql.database import DatabaseError
from semantic_text2sql.models import SchemaInfo
from semantic_text2sql.postgres import PostgresRegistry, postgres_databases_from_environment
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


def test_postgres_allowlist_loads_multiple_databases_from_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(
        "TEXT2SQL_POSTGRES_DATABASES",
        '{"business":"postgresql://reader:secret@db/business",'
        '"finance":"postgresql://reader:secret@db/finance"}',
    )
    monkeypatch.setenv("POSTGRES_BOOKS_DSN", "postgresql://reader:secret@db/books")

    configured = postgres_databases_from_environment()

    assert set(configured) == {"business", "finance", "books_postgres"}


def test_postgres_allowlist_rejects_invalid_database_id(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(
        "TEXT2SQL_POSTGRES_DATABASES",
        '{"../../business":"postgresql://reader:secret@db/business"}',
    )

    with pytest.raises(ValueError, match="Unsupported PostgreSQL database ID"):
        postgres_databases_from_environment()


@pytest.mark.parametrize("seconds,setting", [(0, "1ms"), (.125, "125ms"), (10, "5000ms")])
def test_postgres_timeout_uses_bound_set_config(
    monkeypatch: pytest.MonkeyPatch, seconds: float, setting: str,
) -> None:
    calls: list[tuple[str, tuple[Any, ...]]] = []

    class Cursor:
        description = [SimpleNamespace(name="value")]

        def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> None:
            calls.append((sql, parameters))
            assert not sql.startswith("SET LOCAL statement_timeout = %s")

        def fetchmany(self, limit: int) -> list[tuple[int]]:
            return [(7,)]

    connection = SimpleNamespace(cursor=lambda: nullcontext(Cursor()))
    registry = PostgresRegistry({"test": "unused"})
    monkeypatch.setattr(registry, "connect", lambda db_id: nullcontext(connection))
    result = registry.execute("test", "SELECT %s::int AS value", max_rows=1,
                               timeout_seconds=seconds, parameters=(7,))
    assert calls == [("SELECT set_config('statement_timeout', %s, true)", (setting,)),
                     ("SELECT %s::int AS value", (7,))]
    assert result == (["value"], [[7]], False)


@pytest.mark.skipif(not os.environ.get("TEXT2SQL_TEST_POSTGRES_DSN"),
                    reason="Dedicated PostgreSQL integration DSN not set")
def test_live_postgres_bound_timeout_and_transaction_cleanup() -> None:
    import psycopg

    dsn = os.environ["TEXT2SQL_TEST_POSTGRES_DSN"]
    with psycopg.connect(dsn) as connection:
        original_timeout = connection.execute("SHOW statement_timeout").fetchone()
    registry = PostgresRegistry({"integration": dsn})
    columns, rows, truncated = registry.execute(
        "integration", "SELECT %s::int AS value, current_setting('statement_timeout') AS timeout, "
        "current_setting('transaction_read_only') AS readonly",
        max_rows=1, timeout_seconds=.125, parameters=(7,),
    )
    assert columns == ["value", "timeout", "readonly"]
    assert rows == [[7, "125ms", "on"]] and not truncated
    with pytest.raises(DatabaseError, match="statement timeout"):
        registry.execute("integration", "SELECT pg_sleep(0.2)", max_rows=1, timeout_seconds=.02)
    assert registry.execute("integration", "SELECT 1", max_rows=1)[1] == [[1]]
    with psycopg.connect(dsn) as connection:
        assert connection.execute("SHOW statement_timeout").fetchone() == original_timeout


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
