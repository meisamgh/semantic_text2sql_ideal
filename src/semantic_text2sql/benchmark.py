"""BIRD-style bounded execution-result comparison."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from time import monotonic

from semantic_text2sql.database import DatabaseRegistry


@dataclass(frozen=True)
class ExecutionComparison:
    executable: bool
    equivalent: bool
    predicted_rows: int | None
    gold_rows: int | None
    error: str | None = None


def compare_sql(
    registry: DatabaseRegistry,
    db_id: str,
    predicted_sql: str,
    gold_sql: str,
    *,
    timeout_seconds: float = 30.0,
    max_rows: int = 10_000,
) -> ExecutionComparison:
    deadline = monotonic() + timeout_seconds
    try:
        with registry.connect(db_id) as connection:
            connection.set_progress_handler(lambda: 1 if monotonic() > deadline else 0, 1_000)
            predicted = _fetch(connection, predicted_sql, max_rows)
            gold = _fetch(connection, gold_sql, max_rows)
    except (sqlite3.Error, ValueError) as exc:
        return ExecutionComparison(False, False, None, None, f"{type(exc).__name__}: {exc}")
    return ExecutionComparison(
        executable=True,
        equivalent=set(predicted) == set(gold),
        predicted_rows=len(predicted),
        gold_rows=len(gold),
    )


def _fetch(connection: sqlite3.Connection, sql: str, max_rows: int) -> list[tuple[object, ...]]:
    rows = connection.execute(sql).fetchmany(max_rows + 1)
    if len(rows) > max_rows:
        raise ValueError(f"Query exceeded benchmark row limit {max_rows}.")
    return [tuple(_normal(value) for value in row) for row in rows]


def _normal(value: object) -> object:
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, bytes):
        return value.hex()
    return value
