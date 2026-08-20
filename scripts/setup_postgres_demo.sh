#!/usr/bin/env bash
set -euo pipefail

POSTGRES_BIN="/opt/homebrew/opt/postgresql@15/bin"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
READER_PASSWORD="${TEXT2SQL_POSTGRES_READER_PASSWORD:-text2sql_demo_reader_2026}"

if ! "$POSTGRES_BIN/pg_isready" -h 127.0.0.1 -p 5432 >/dev/null; then
  echo "PostgreSQL is not ready on 127.0.0.1:5432" >&2
  exit 1
fi

if ! "$POSTGRES_BIN/psql" -d postgres -Atqc \
  "SELECT 1 FROM pg_database WHERE datname='text2sql_books'" | grep -qx 1; then
  "$POSTGRES_BIN/createdb" text2sql_books
fi

"$POSTGRES_BIN/psql" -v ON_ERROR_STOP=1 -d text2sql_books \
  -f "$PROJECT_ROOT/sql/postgres_schema.sql"
"$POSTGRES_BIN/psql" -v ON_ERROR_STOP=1 -d text2sql_books \
  -f "$PROJECT_ROOT/sql/postgres_seed.sql"
"$POSTGRES_BIN/psql" -v ON_ERROR_STOP=1 -v reader_password="$READER_PASSWORD" \
  -d text2sql_books \
  -f "$PROJECT_ROOT/sql/postgres_security.sql"

echo "PostgreSQL demo database is ready."
