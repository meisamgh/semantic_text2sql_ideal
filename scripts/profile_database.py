#!/usr/bin/env python3
"""Create a cached offline column profile for one allowlisted database."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.postgres import PostgresRegistry
from semantic_text2sql.profiling import (
    ProfileStore,
    apply_bird_descriptions,
    profile_database,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialect", choices=("sqlite", "postgres"), required=True)
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("profiles"))
    parser.add_argument("--bird-description-dir", type=Path)
    args = parser.parse_args()
    database: DatabaseRegistry | PostgresRegistry
    if args.dialect == "sqlite":
        database = DatabaseRegistry(Path(os.environ.get("TEXT2SQL_DATABASE_ROOT", "data")))
    else:
        dsn = os.environ.get("POSTGRES_BOOKS_DSN")
        if not dsn:
            raise SystemExit("POSTGRES_BOOKS_DSN is required for PostgreSQL profiling.")
        database = PostgresRegistry({args.db_id: dsn})
    profile = profile_database(database, args.db_id, args.dialect)
    if args.bird_description_dir:
        profile = apply_bird_descriptions(profile, args.bird_description_dir)
    path = ProfileStore(args.output).save(profile)
    print(f"Wrote {len(profile.columns)} column profiles to {path}")


if __name__ == "__main__":
    main()
