#!/usr/bin/env python3
"""Create the local books demonstration database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    directory = args.root / "books"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "books.sqlite"
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE authors (
                author_id INTEGER PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                country TEXT
            );
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                author_id INTEGER NOT NULL REFERENCES authors(author_id),
                category TEXT NOT NULL,
                publication_date TEXT,
                retail_price REAL,
                description TEXT
            );
            INSERT INTO authors VALUES
              (1, 'George', 'Orwell', 'UK'),
              (2, 'Ursula', 'Le Guin', 'USA'),
              (3, 'Terry', 'Pratchett', 'UK');
            INSERT INTO books VALUES
              (1, '1984', 1, 'Dystopian', '1949-06-08', 12.5,
               'A totalitarian society governed through surveillance.'),
              (2, 'Animal Farm', 1, 'Satire', '1945-08-17', 9.5,
               'A political allegory about power and revolution.'),
              (3, 'The Dispossessed', 2, 'Science Fiction', '1974-05-01', 14.0,
               'An exploration of anarchism, freedom, and social organization.'),
              (4, 'Guards! Guards!', 3, 'Fantasy', '1989-07-01', 11.0,
               'Comic fantasy involving dragons and a city watch.');
            """
        )
    print(path.resolve())


if __name__ == "__main__":
    main()
