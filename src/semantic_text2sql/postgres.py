"""Allowlisted PostgreSQL inspection and read-only execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg import Connection

from semantic_text2sql.database import DatabaseError
from semantic_text2sql.models import (
    ColumnInfo,
    ForeignKeyInfo,
    SchemaInfo,
    TableInfo,
    ValidationResult,
)


class PostgresRegistry:
    def __init__(self, databases: dict[str, str]) -> None:
        self.databases = dict(databases)

    def configured_ids(self) -> list[str]:
        return sorted(self.databases)

    def _dsn(self, db_id: str) -> str:
        dsn = self.databases.get(db_id)
        if not dsn:
            raise DatabaseError(
                "DATABASE_NOT_FOUND",
                f"PostgreSQL database {db_id} is not configured.",
            )
        return dsn

    @contextmanager
    def connect(self, db_id: str) -> Iterator[Connection[Any]]:
        try:
            with psycopg.connect(self._dsn(db_id), autocommit=False) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute("SET LOCAL statement_timeout = '5s'")
                    cursor.execute("SET LOCAL lock_timeout = '1s'")
                    cursor.execute("SET LOCAL idle_in_transaction_session_timeout = '10s'")
                yield connection
                connection.rollback()
        except psycopg.Error as exc:
            raise DatabaseError("POSTGRES_CONNECTION_FAILED", str(exc)) from exc

    def inspect(self, db_id: str) -> SchemaInfo:
        with self.connect(db_id) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.table_name, c.column_name, c.data_type,
                       c.is_nullable = 'YES',
                       COALESCE(pk.column_name IS NOT NULL, false)
                FROM information_schema.columns c
                LEFT JOIN (
                    SELECT cls.relname AS table_name, attr.attname AS column_name
                    FROM pg_catalog.pg_index idx
                    JOIN pg_catalog.pg_class cls ON cls.oid = idx.indrelid
                    JOIN pg_catalog.pg_namespace ns ON ns.oid = cls.relnamespace
                    JOIN pg_catalog.pg_attribute attr
                      ON attr.attrelid = cls.oid
                     AND attr.attnum = ANY(idx.indkey)
                    WHERE idx.indisprimary AND ns.nspname = 'public'
                ) pk ON pk.table_name = c.table_name AND pk.column_name = c.column_name
                WHERE c.table_schema = 'public'
                ORDER BY c.table_name, c.ordinal_position
                """
            )
            raw_columns = cursor.fetchall()
            cursor.execute(
                """
                SELECT tc.constraint_name, kcu.table_name, kcu.column_name,
                       ccu.table_name, ccu.column_name, kcu.ordinal_position
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.constraint_schema = kcu.constraint_schema
                JOIN information_schema.referential_constraints rc
                  ON rc.constraint_name = tc.constraint_name
                 AND rc.constraint_schema = tc.constraint_schema
                JOIN information_schema.key_column_usage ccu
                  ON ccu.constraint_name = rc.unique_constraint_name
                 AND ccu.constraint_schema = rc.unique_constraint_schema
                 AND ccu.ordinal_position = kcu.position_in_unique_constraint
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public'
                ORDER BY tc.constraint_name, kcu.ordinal_position
                """
            )
            raw_relationships = cursor.fetchall()
        grouped: dict[str, list[ColumnInfo]] = {}
        for table, column, data_type, nullable, primary_key in raw_columns:
            grouped.setdefault(str(table), []).append(
                ColumnInfo(
                    name=str(column),
                    data_type=str(data_type),
                    primary_key=bool(primary_key),
                    nullable=bool(nullable),
                )
            )
        if len(grouped) > 100:
            raise DatabaseError("SCHEMA_TOO_LARGE", "Database has more than 100 public tables.")
        tables = [
            TableInfo(
                name=name,
                create_sql=(
                    f"CREATE TABLE {name} ("
                    + ", ".join(f"{column.name} {column.data_type}" for column in columns)
                    + ")"
                ),
                columns=columns,
            )
            for name, columns in grouped.items()
        ]
        relationships = [
            ForeignKeyInfo(
                from_table=str(from_table),
                from_column=str(from_column),
                to_table=str(to_table),
                to_column=str(to_column),
                constraint_id=str(constraint),
                ordinal=int(ordinal) - 1,
            )
            for (
                constraint,
                from_table,
                from_column,
                to_table,
                to_column,
                ordinal,
            ) in raw_relationships
        ]
        return SchemaInfo(
            db_id=db_id,
            dialect="postgres",
            tables=tables,
            relationships=relationships,
        )

    def explain(self, db_id: str, sql: str, validation: ValidationResult) -> ValidationResult:
        try:
            with self.connect(db_id) as connection, connection.cursor() as cursor:
                cursor.execute("EXPLAIN (FORMAT TEXT) " + sql)
                rows = cursor.fetchall()
        except DatabaseError as exc:
            return validation.model_copy(
                update={
                    "valid": False,
                    "code": "SQL_EXPLAIN_FAILED",
                    "message": f"PostgreSQL rejected the query: {exc}",
                }
            )
        return validation.model_copy(update={"explain_plan": [str(row[0]) for row in rows]})

    def execute(
        self,
        db_id: str,
        sql: str,
        *,
        max_rows: int,
        timeout_seconds: float = 5.0,
    ) -> tuple[list[str], list[list[Any]], bool]:
        del timeout_seconds
        try:
            with self.connect(db_id) as connection, connection.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchmany(max_rows + 1)
                columns = [item.name for item in cursor.description or ()]
        except (psycopg.Error, DatabaseError) as exc:
            raise DatabaseError("SQL_EXECUTION_FAILED", str(exc)) from exc
        return columns, [_json_row(row) for row in rows[:max_rows]], len(rows) > max_rows


def _json_row(row: tuple[Any, ...]) -> list[Any]:
    return [
        value if value is None or isinstance(value, str | int | float | bool) else str(value)
        for value in row
    ]
