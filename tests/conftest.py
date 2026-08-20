from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from semantic_text2sql.database import DatabaseRegistry


@pytest.fixture
def registry(tmp_path: Path) -> DatabaseRegistry:
    directory = tmp_path / "shop"
    directory.mkdir()
    with sqlite3.connect(directory / "shop.sqlite") as connection:
        connection.executescript(
            """
            CREATE TABLE customers (
              customer_id INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              country TEXT NOT NULL
            );
            CREATE TABLE orders (
              order_id INTEGER PRIMARY KEY,
              customer_id INTEGER NOT NULL REFERENCES customers,
              amount REAL NOT NULL,
              status TEXT NOT NULL
            );
            INSERT INTO customers VALUES (1, 'Anna', 'Germany'), (2, 'Luca', 'Italy');
            INSERT INTO orders VALUES
              (10, 1, 100.0, 'complete'),
              (11, 1, 50.0, 'complete'),
              (12, 2, 20.0, 'pending');
            """
        )
    return DatabaseRegistry(tmp_path)
