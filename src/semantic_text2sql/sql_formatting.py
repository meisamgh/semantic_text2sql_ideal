"""Presentation-only SQL formatting helpers."""

from __future__ import annotations

import sqlglot


def format_sql_for_display(sql: str | None, dialect: str) -> str | None:
    """Pretty-print SQL without changing the statement used for execution."""
    if not sql:
        return None
    try:
        expression = sqlglot.parse_one(sql, read=dialect)
        return expression.sql(dialect=dialect, pretty=True)
    except (ValueError, sqlglot.errors.SqlglotError):
        # Accepted SQL should already parse. Keep API serialization resilient if an
        # upstream/provider error nevertheless leaves an unparseable statement.
        return sql
