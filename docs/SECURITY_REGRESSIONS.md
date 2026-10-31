# SQL scope authorization and PostgreSQL timeout regression checks

## Physical-table authorization

`validator.physical_table_references` resolves lexical CTE visibility using
SQLGlot scopes on an identifier-normalized copy, then returns original AST nodes.
Schema/catalog-qualified references are never treated as CTE references. It
covers CTE bodies, nested scopes, non-recursive self references, forward
references, quoted alias case and SEMI/ANTI sources without rewriting user SQL.
Qualified generic allowlist entries require a matching qualified identity;
engine adapters still enforce their configured schema/catalog for bare entries.

`RemoteDatabaseRegistry._prepare` independently checks every resolved physical
table against its configured table allowlist before qualification/execution.
Schema/catalog restrictions and function restrictions remain in place. Tests
also force the generic validator to return valid, proving this second check is
not just an assertion about its output. These are defense-in-depth controls;
database credentials should remain least-privileged.

MySQL, SQL Server, Snowflake and BigQuery adversarial adapter tests use scripted
drivers. They verify no submitted query reaches those drivers, not real cloud
database access. Valid CTE chains and allowed qualified tables remain supported.

## PostgreSQL timeout

The dynamic timeout uses parameterized
`SELECT set_config('statement_timeout', %s, true)` with an explicit millisecond
string. The final `true` keeps it transaction-local. Existing read-only
transaction setup, rollback and 1–5000 ms bounds remain unchanged.

Opt-in live check (use a dedicated test database, not production credentials):

```bash
TEXT2SQL_TEST_POSTGRES_DSN='your dedicated test DSN' \
  .venv/bin/pytest -o addopts='' -q tests/test_postgres.py \
  -k live_postgres_bound_timeout_and_transaction_cleanup
```

Verified locally on PostgreSQL 15.19 / Psycopg 3 using a temporary UTF-8 database,
an unprivileged login and a private Unix socket with TCP disabled. The test checks
bound query parameters, the actual 125 ms timeout setting, transaction read-only
status, cancellation of `pg_sleep`, subsequent query recovery and restoration of
the original connection timeout. The temporary server was stopped afterward.

Full regression run with the dedicated live DSN: **556 passed, 1 skipped**.
The skip is the existing books-dataset integration test, not the new timeout
test. Final focused PostgreSQL run: **8 passed, 1 skipped**. Frontend: **170
passed**. Ruff, mypy and diff checks passed. No production database was modified.
