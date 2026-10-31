"""Allowlisted remote adapters sharing the existing SQL safety boundary.

SQLAlchemy drivers are optional; BigQuery uses its native client for dry-run cost
checks and cancellation. Credentials never appear in public schema metadata.
"""

from __future__ import annotations

import importlib
import math
from threading import Lock
from time import monotonic
from typing import Any, cast

from sqlglot import exp, parse_one

from semantic_text2sql.database import DatabaseError
from semantic_text2sql.dialects import DatabaseDialect
from semantic_text2sql.external_connections import ExternalConnection
from semantic_text2sql.models import (
    ColumnInfo,
    ForeignKeyInfo,
    SchemaInfo,
    TableInfo,
    ValidationResult,
)
from semantic_text2sql.postgres import PostgresRegistry
from semantic_text2sql.runtime import RequestBudgetExceeded, active_request_budget
from semantic_text2sql.validator import physical_table_references, validate_sql


class RemoteDatabaseRegistry(PostgresRegistry):
    """Compatibility extension of the existing non-SQLite registry boundary."""

    def __init__(
        self, postgres: dict[str, str], connections: dict[str, ExternalConnection]
    ) -> None:
        if set(postgres) & set(connections):
            raise ValueError("Database IDs must be unique across connection types.")
        super().__init__(postgres)
        self.connections = dict(connections)
        self._engines: dict[str, Any] = {}
        self._schemas: dict[str, tuple[float, SchemaInfo]] = {}
        self._lock = Lock()

    def configured_ids(self) -> list[str]:
        return sorted(set(self.databases) | set(self.connections))

    def dialect_for(self, db_id: str) -> DatabaseDialect:
        config = self.connections.get(db_id)
        return cast(DatabaseDialect, config.dialect) if config else super().dialect_for(db_id)

    def _config(self, db_id: str) -> ExternalConnection:
        config = self.connections[db_id]
        if not config.read_only_role_confirmed:
            raise DatabaseError(
                "READ_ONLY_ROLE_REQUIRED", "Configure a verified read-only DB role."
            )
        return config

    def _engine(self, config: ExternalConnection) -> Any:
        with self._lock:
            if config.db_id not in self._engines:
                try:
                    sa = importlib.import_module("sqlalchemy")
                    args: dict[str, Any] = {}
                    if config.engine == "mysql":
                        args = {
                            "connect_timeout": 5,
                            "read_timeout": config.timeout_seconds,
                            "write_timeout": config.timeout_seconds,
                        }
                    elif config.engine == "sqlserver":
                        args = {"timeout": 5}
                    elif config.engine == "snowflake":
                        args = {
                            "login_timeout": 5,
                            "network_timeout": config.timeout_seconds,
                            "session_parameters": {
                                "STATEMENT_TIMEOUT_IN_SECONDS": config.timeout_seconds,
                                "STATEMENT_QUEUED_TIMEOUT_IN_SECONDS": config.timeout_seconds,
                            },
                        }
                    self._engines[config.db_id] = sa.create_engine(
                        config.connection_url(),
                        connect_args=args,
                        pool_pre_ping=True,
                        pool_size=2,
                        max_overflow=0,
                        pool_timeout=5,
                    )
                except (ImportError, ModuleNotFoundError) as exc:
                    raise DatabaseError(
                        "DATABASE_DRIVER_MISSING", "Install the database extra."
                    ) from exc
                except Exception as exc:
                    raise DatabaseError(
                        "DATABASE_CONNECTION_FAILED", "Cannot initialize adapter."
                    ) from exc
            return self._engines[config.db_id]

    def _bigquery(self, config: ExternalConnection) -> Any:
        try:
            module = importlib.import_module("google.cloud.bigquery")
            project = config.schema_name.split(".")[0]
            return module.Client(project=project)
        except ImportError as exc:
            raise DatabaseError("DATABASE_DRIVER_MISSING", "Install the bigquery extra.") from exc
        except Exception as exc:
            raise DatabaseError(
                "DATABASE_CONNECTION_FAILED", "BigQuery credentials unavailable."
            ) from exc

    def inspect(self, db_id: str) -> SchemaInfo:
        if db_id not in self.connections:
            return super().inspect(db_id)
        config = self._config(db_id)
        cached = self._schemas.get(db_id)
        if cached and monotonic() - cached[0] < 60:
            return cached[1].model_copy(deep=True)
        tables: list[TableInfo] = []
        relationships: list[ForeignKeyInfo] = []
        deadline = monotonic() + config.timeout_seconds
        try:
            if config.engine == "bigquery":
                with self._bigquery(config) as client:
                    for name in config.allowed_tables:
                        if monotonic() >= deadline:
                            raise DatabaseError(
                                "SCHEMA_TIMEOUT", "Schema inspection deadline exceeded."
                            )
                        table = client.get_table(
                            f"{config.schema_name}.{name}",
                            timeout=min(5, config.timeout_seconds),
                        )
                        columns = [
                            ColumnInfo(
                                name=f.name,
                                data_type=f.field_type,
                                nullable=f.mode != "REQUIRED",
                                description=f.description,
                            )
                            for f in table.schema
                        ]
                        tables.append(TableInfo(name=name, columns=columns, create_sql=""))
            else:
                engine = self._engine(config)
                sa = importlib.import_module("sqlalchemy")
                with engine.connect() as connection:
                    if config.engine == "sqlserver":
                        connection.connection.driver_connection.timeout = config.timeout_seconds
                    inspector = sa.inspect(connection)
                    for name in config.allowed_tables:
                        if monotonic() >= deadline:
                            raise DatabaseError(
                                "SCHEMA_TIMEOUT", "Schema inspection deadline exceeded."
                            )
                        keys = inspector.get_pk_constraint(name, schema=config.schema_name)
                        pk = set(keys.get("constrained_columns") or [])
                        columns = [
                            ColumnInfo(
                                name=c["name"],
                                data_type=str(c["type"]),
                                nullable=c.get("nullable"),
                                primary_key=c["name"] in pk,
                            )
                            for c in inspector.get_columns(name, schema=config.schema_name)
                        ]
                        tables.append(TableInfo(name=name, columns=columns, create_sql=""))
                        for fk in inspector.get_foreign_keys(name, schema=config.schema_name):
                            if fk["referred_table"] not in config.allowed_tables or (
                                fk.get("referred_schema") not in {None, config.schema_name}
                            ):
                                continue
                            for ordinal, (left, right) in enumerate(
                                zip(
                                    fk["constrained_columns"],
                                    fk["referred_columns"],
                                    strict=True,
                                )
                            ):
                                relationships.append(
                                    ForeignKeyInfo(
                                        from_table=name,
                                        from_column=left,
                                        to_table=fk["referred_table"],
                                        to_column=right,
                                        constraint_id=fk.get("name"),
                                        ordinal=ordinal,
                                    )
                                )
            schema = SchemaInfo(
                db_id=db_id,
                dialect=self.dialect_for(db_id),
                tables=tables,
                relationships=relationships,
            )
            self._schemas[db_id] = (monotonic(), schema)
            return schema.model_copy(deep=True)
        except (DatabaseError, RequestBudgetExceeded):
            raise
        except Exception as exc:
            raise DatabaseError("DATABASE_INSPECTION_FAILED", "Schema inspection failed.") from exc

    def _prepare(self, db_id: str, sql: str) -> str:
        config = self._config(db_id)
        schema = self.inspect(db_id)
        validation = validate_sql(
            sql,
            schema,
            dialect=schema.dialect,
            allowed_tables=set(config.allowed_tables),
            allowed_columns={
                table.name: {column.name for column in table.columns} for table in schema.tables
            },
        )
        if not validation.valid:
            raise DatabaseError(validation.code, validation.message)
        root = parse_one(sql, read=schema.dialect)
        # Do not expose remote I/O, UDFs, stored procedures or table functions.
        if any(isinstance(node, exp.Anonymous) for node in root.walk()):
            raise DatabaseError("UNAPPROVED_FUNCTION", "Unapproved database function.")
        parts = config.schema_name.split(".")
        allowed = {name.casefold() for name in config.allowed_tables}
        for table in physical_table_references(root, dialect=schema.dialect):
            if not isinstance(table.this, exp.Identifier):
                raise DatabaseError(
                    "REMOTE_SOURCE_BLOCKED", "Only allowlisted tables are supported."
                )
            if table.name.casefold() not in allowed:
                raise DatabaseError(
                    "SQL_TABLE_NOT_AUTHORIZED", "Query references a non-allowlisted physical table."
                )
            expected = parts[-1]
            if (
                table.db
                and table.db != expected
                or (table.catalog and (len(parts) != 2 or table.catalog != parts[0]))
            ):
                raise DatabaseError("SCHEMA_NOT_AUTHORIZED", "Query references another schema.")
            table.set("db", exp.to_identifier(expected, quoted=True))
            if len(parts) == 2:
                table.set("catalog", exp.to_identifier(parts[0], quoted=True))
        return root.sql(dialect=schema.dialect)

    def execute(
        self,
        db_id: str,
        sql: str,
        *,
        max_rows: int,
        timeout_seconds: float = 5.0,
        parameters: tuple[Any, ...] = (),
    ) -> tuple[list[str], list[list[Any]], bool]:
        if db_id not in self.connections:
            return super().execute(
                db_id,
                sql,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
                parameters=parameters,
            )
        config = self._config(db_id)
        if not 1 <= max_rows <= 10000:
            raise DatabaseError("ROW_LIMIT_INVALID", "Row bound must be between 1 and 10000.")
        timeout = max(1, math.ceil(min(timeout_seconds, config.timeout_seconds)))
        prepared = self._prepare(db_id, sql)
        try:
            if config.engine == "bigquery":
                module = importlib.import_module("google.cloud.bigquery")
                with self._bigquery(config) as client:
                    policy = module.QueryJobConfig(
                        dry_run=True,
                        use_query_cache=False,
                        use_legacy_sql=False,
                        maximum_bytes_billed=config.maximum_bytes_billed,
                        query_parameters=[
                            module.ScalarQueryParameter(f"p{i}", "STRING", str(value))
                            for i, value in enumerate(parameters)
                        ],
                    )
                    estimate = client.query(prepared, job_config=policy, timeout=timeout)
                    if estimate.total_bytes_processed > (config.maximum_bytes_billed or 0):
                        raise DatabaseError(
                            "QUERY_COST_LIMIT", "Query exceeds the configured byte limit."
                        )
                    policy.dry_run = False
                    policy.job_timeout_ms = timeout * 1000
                    budget = active_request_budget()
                    if budget is not None:
                        budget.claim(
                            "database"
                        )  # Actual job plus the separately estimated dry run.
                    job = client.query(prepared, job_config=policy, timeout=timeout)
                    try:
                        result = job.result(timeout=timeout, max_results=max_rows + 1)
                        fetched = list(result)
                    except Exception:
                        job.cancel()
                        raise
                    return (
                        [f.name for f in result.schema],
                        [_row(row) for row in fetched[:max_rows]],
                        len(fetched) > max_rows,
                    )
            with self._engine(config).connect() as connection:
                if config.engine == "mysql":
                    connection.exec_driver_sql(f"SET SESSION MAX_EXECUTION_TIME={timeout * 1000}")
                    connection.exec_driver_sql("START TRANSACTION READ ONLY")
                elif config.engine == "sqlserver":
                    connection.connection.driver_connection.timeout = timeout
                elif config.engine == "snowflake":
                    connection.exec_driver_sql(
                        f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS={timeout}"
                    )
                try:
                    if parameters:
                        sa = importlib.import_module("sqlalchemy")
                        result = connection.execute(
                            sa.text(prepared), {f"p{i}": v for i, v in enumerate(parameters)}
                        )
                    else:
                        result = connection.exec_driver_sql(prepared)
                    fetched = result.fetchmany(max_rows + 1)
                    columns = list(result.keys())
                finally:
                    connection.rollback()
            return columns, [_row(row) for row in fetched[:max_rows]], len(fetched) > max_rows
        except (DatabaseError, RequestBudgetExceeded):
            raise
        except Exception as exc:
            raise DatabaseError("SQL_EXECUTION_FAILED", "Remote read-only query failed.") from exc

    def explain(self, db_id: str, sql: str, validation: ValidationResult) -> ValidationResult:
        if db_id not in self.connections:
            return super().explain(db_id, sql, validation)
        # Do not mislabel an execution as a dry run. SQL Server SHOWPLAN and BQ
        # cost estimates require different APIs, so only expose actual validation.
        self._prepare(db_id, sql)
        return validation.model_copy(update={"explain_plan": []})

    def lookup_values(
        self,
        db_id: str,
        table: str,
        column: str,
        search: str,
        *,
        exact: bool = False,
        limit: int = 101,
    ) -> tuple[list[str], list[list[Any]], bool]:
        """Parameterized, dialect-correct lookup over an approved field."""
        schema = self.inspect(db_id)
        selected = next((item for item in schema.tables if item.name == table), None)
        if selected is None or column not in {c.name for c in selected.columns}:
            raise DatabaseError("COLUMN_NOT_AUTHORIZED", "Unknown lookup field.")
        # Build identifiers through the AST; only data uses bind parameters.
        field = exp.column(column, quoted=True).sql(dialect="postgres")
        source = exp.to_identifier(table, quoted=True).sql(dialect="postgres")
        predicate = (
            f"LOWER(CAST({field} AS TEXT)) = LOWER(:p0)"
            if exact
            else (f"STRPOS(LOWER(CAST({field} AS TEXT)), LOWER(:p0)) > 0")
        )
        raw = f"SELECT DISTINCT CAST({field} AS TEXT) FROM {source} WHERE {field} IS NOT NULL"
        if search:
            raw += " AND " + predicate
        raw += f" ORDER BY 1 LIMIT {max(1, min(limit, 101))}"
        sql = parse_one(raw, read="postgres").sql(dialect=schema.dialect)
        return self.execute(
            db_id,
            sql,
            max_rows=max(1, min(limit, 101)),
            timeout_seconds=2,
            parameters=(search,) if search else (),
        )

    def close(self) -> None:
        for engine in self._engines.values():
            engine.dispose()


def _row(row: Any) -> list[Any]:
    return [
        value if value is None or isinstance(value, str | int | float | bool) else str(value)
        for value in row
    ]
