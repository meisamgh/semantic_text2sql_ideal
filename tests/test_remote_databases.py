"""Adapter contract tests. Drivers are scripted; these are not live DB tests."""

from __future__ import annotations

import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.api import create_app
from semantic_text2sql.database import DatabaseError, DatabaseRegistry
from semantic_text2sql.external_connections import ExternalConnection
from semantic_text2sql.models import TokenUsage, ValidationResult
from semantic_text2sql.remote_databases import RemoteDatabaseRegistry


def configuration(engine: str) -> ExternalConnection:
    return ExternalConnection.model_validate(
        {
            "db_id": "warehouse",
            "engine": engine,
            "url_env": "WAREHOUSE_URL",
            "schema_name": "project.analytics" if engine == "bigquery" else "analytics",
            "allowed_tables": ["orders"],
            "read_only_role_confirmed": True,
            "maximum_bytes_billed": 1000000,
        }
    )


class Result:
    schema = [SimpleNamespace(name="name"), SimpleNamespace(name="amount")]

    def keys(self) -> list[str]:
        return ["name", "amount"]

    def fetchmany(self, limit: int) -> list[tuple[Any, ...]]:
        return [("Ada", 12.5), ("Lin", 20.0)][:limit]

    def __iter__(self) -> Any:
        return iter(self.fetchmany(100))


class Connection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.connection = SimpleNamespace(driver_connection=SimpleNamespace(timeout=0))
        self.rollbacks = 0

    def __enter__(self) -> Connection:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def exec_driver_sql(self, sql: str) -> Result:
        self.calls.append((sql, None))
        return Result()

    def execute(self, sql: str, parameters: Any) -> Result:
        self.calls.append((sql, parameters))
        return Result()

    def rollback(self) -> None:
        self.rollbacks += 1


@pytest.fixture
def drivers(monkeypatch: pytest.MonkeyPatch) -> Any:
    connection = Connection()
    engine = SimpleNamespace(connect=lambda: connection, dispose=lambda: None)
    inspector = SimpleNamespace(
        get_pk_constraint=lambda *a, **k: {"constrained_columns": ["name"]},
        get_columns=lambda *a, **k: [
            {"name": "name", "type": "VARCHAR", "nullable": False},
            {"name": "amount", "type": "FLOAT", "nullable": True},
        ],
        get_foreign_keys=lambda *a, **k: [],
    )
    sa = SimpleNamespace(
        create_engine=lambda *a, **k: engine, inspect=lambda conn: inspector, text=lambda sql: sql
    )
    jobs: list[Any] = []

    class Job:
        total_bytes_processed = 100
        cancelled = False

        def result(self, **kwargs: Any) -> Result:
            return Result()

        def cancel(self) -> None:
            self.cancelled = True

    class Client:
        def __enter__(self) -> Client:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get_table(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(
                schema=[
                    SimpleNamespace(
                        name="name", field_type="STRING", mode="REQUIRED", description=None
                    ),
                    SimpleNamespace(
                        name="amount", field_type="FLOAT", mode="NULLABLE", description=None
                    ),
                ]
            )

        def query(self, sql: str, *, job_config: Any, timeout: Any) -> Job:
            job = Job()
            jobs.append((sql, job_config.dry_run, job_config.maximum_bytes_billed, job))
            return job

    bq = SimpleNamespace(
        Client=lambda **k: Client(),
        QueryJobConfig=lambda **k: SimpleNamespace(**k),
        ScalarQueryParameter=lambda *a: a,
    )
    original = importlib.import_module

    def load(name: str, *args: Any, **kwargs: Any) -> Any:
        return {"sqlalchemy": sa, "google.cloud.bigquery": bq}.get(name) or original(
            name,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(importlib, "import_module", load)
    return SimpleNamespace(connection=connection, jobs=jobs, bq=bq)


@pytest.mark.parametrize(
    "engine,scheme",
    [
        ("mysql", "mysql+pymysql"),
        ("sqlserver", "mssql+pyodbc"),
        ("snowflake", "snowflake"),
        ("bigquery", "bigquery"),
    ],
)
def test_remote_adapter_contract(
    engine: str,
    scheme: str,
    drivers: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WAREHOUSE_URL", f"{scheme}://example")
    remote = RemoteDatabaseRegistry({}, {"warehouse": configuration(engine)})
    schema = remote.inspect("warehouse")
    assert schema.dialect == configuration(engine).dialect
    assert [c.name for c in schema.tables[0].columns] == ["name", "amount"]
    columns, rows, truncated = remote.execute(
        "warehouse",
        "SELECT name, amount FROM orders",
        max_rows=1,
    )
    assert columns == ["name", "amount"]
    assert rows == [["Ada", 12.5]] and truncated
    for sql in [
        "DELETE FROM orders",
        "SELECT * FROM private",
        "SELECT * FROM other.orders",
        "SELECT secret_function() FROM orders",
    ]:
        with pytest.raises(DatabaseError):
            remote.execute("warehouse", sql, max_rows=1)
    before = len(drivers.jobs) if engine == "bigquery" else len(drivers.connection.calls)
    assert before > 0
    remote.lookup_values("warehouse", "orders", "name", "O'Reilly%", exact=True)
    if engine == "bigquery":
        assert drivers.jobs[0][1] is True and drivers.jobs[1][1] is False
    else:
        sql, parameters = drivers.connection.calls[-1]
        assert "O'Reilly" not in sql and parameters == {"p0": "O'Reilly%"}
        assert drivers.connection.rollbacks >= 2


@pytest.mark.parametrize("engine", ["mysql", "sqlserver", "snowflake", "bigquery"])
@pytest.mark.parametrize("permissive_validator", [False, True])
def test_cte_shadow_cannot_bypass_remote_table_authorization(
    engine: str, permissive_validator: bool, drivers: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_text2sql import remote_databases

    scheme = {"mysql": "mysql+pymysql", "sqlserver": "mssql+pyodbc",
              "snowflake": "snowflake", "bigquery": "bigquery"}[engine]
    monkeypatch.setenv("WAREHOUSE_URL", f"{scheme}://example")
    config = configuration(engine)
    remote = RemoteDatabaseRegistry({}, {"warehouse": config})
    remote.inspect("warehouse")
    if permissive_validator:
        # Defense in depth: even if the generic validator returns valid, the
        # adapter must independently authorize each resolved physical source.
        monkeypatch.setattr(remote_databases, "validate_sql", lambda *a, **k:
                            ValidationResult(valid=True, code="MOCK_VALID", message="Mock"))
    prefix = config.schema_name
    attacks = [
        f"WITH secret AS (SELECT name FROM orders) SELECT name FROM {prefix}.secret",
        f"WITH orders AS (SELECT name FROM {prefix}.secret) SELECT name FROM orders",
        "WITH secret AS (SELECT name FROM secret) SELECT name FROM secret",
        "SELECT name FROM secret WHERE EXISTS "
        "(WITH secret AS (SELECT name FROM orders) SELECT name FROM secret)",
    ]
    for sql in attacks:
        with pytest.raises(DatabaseError) as error:
            remote.execute("warehouse", sql, max_rows=5)
        assert error.value.code == "SQL_TABLE_NOT_AUTHORIZED"
    assert drivers.jobs == []
    assert drivers.connection.calls == []


@pytest.mark.parametrize("engine", ["mysql", "sqlserver", "snowflake", "bigquery"])
def test_remote_valid_ctes_remain_unqualified_and_physical_sources_are_scoped(
    engine: str, drivers: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlglot import parse_one

    from semantic_text2sql.validator import physical_table_references

    scheme = {"mysql": "mysql+pymysql", "sqlserver": "mssql+pyodbc",
              "snowflake": "snowflake", "bigquery": "bigquery"}[engine]
    monkeypatch.setenv("WAREHOUSE_URL", f"{scheme}://example")
    config = configuration(engine)
    remote = RemoteDatabaseRegistry({}, {"warehouse": config})
    for sql in [
        "WITH orders AS (SELECT name FROM orders) SELECT name FROM orders",
        "WITH a AS (SELECT name FROM orders), b AS (SELECT name FROM a) SELECT name FROM b",
        f"WITH orders AS (SELECT name FROM orders) SELECT name FROM {config.schema_name}.orders",
    ]:
        prepared = remote._prepare("warehouse", sql)
        physical = physical_table_references(parse_one(prepared, read=config.dialect),
                                             dialect=config.dialect)
        assert physical
        assert all(t.name.casefold() == "orders" and t.db == "analytics" for t in physical)
    assert drivers.jobs == [] and drivers.connection.calls == []


@pytest.mark.parametrize("engine", ["mysql", "sqlserver", "snowflake", "bigquery"])
def test_remote_connections_use_existing_chat_pipeline(
    engine: str,
    drivers: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = configuration(engine)
    scheme = {
        "mysql": "mysql+pymysql",
        "sqlserver": "mssql+pyodbc",
        "snowflake": "snowflake",
        "bigquery": "bigquery",
    }[engine]
    monkeypatch.setenv("WAREHOUSE_URL", f"{scheme}://example")
    monkeypatch.setenv("TEXT2SQL_EXTERNAL_CONNECTIONS", json.dumps([config.model_dump()]))
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(tmp_path))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    monkeypatch.setenv("TEXT2SQL_CONTEXT_MODE", "retrieval")
    monkeypatch.setenv("TEXT2SQL_CONVERSATIONAL_AGENT_ENABLED", "false")

    class Model:
        async def generate(self, **kwargs: Any) -> tuple[str, int, TokenUsage]:
            assert kwargs["dialect"] == config.dialect
            return (
                "SELECT name, amount FROM orders",
                1,
                TokenUsage(input_tokens=10, output_tokens=5),
            )

    remote = RemoteDatabaseRegistry({}, {"warehouse": config})
    app = create_app(TextToSQLAgent(DatabaseRegistry(tmp_path), Model(), remote))
    client = TestClient(app)
    options = client.get("/api/databases").json()
    assert any(o["db_id"] == "warehouse" and o["dialect"] == config.dialect for o in options)
    response = client.post(
        "/api/chat",
        json={
            "session_id": "remote",
            "db_id": "warehouse",
            "model": "test",
            "message": "Show name and amount from orders",
            "execute": True,
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()["generation"]
    assert result["accepted"], response.text
    assert result["dialect"] == config.dialect
    assert result["rows"] == [["Ada", 12.5], ["Lin", 20.0]]


def test_bigquery_rejects_cost_before_execution_and_cancels_timed_out_job(
    drivers: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = RemoteDatabaseRegistry({}, {"warehouse": configuration("bigquery")})
    remote.inspect("warehouse")
    calls: list[bool] = []
    job = SimpleNamespace(total_bytes_processed=2000000)

    def query(sql: str, *, job_config: Any, timeout: int) -> Any:
        calls.append(job_config.dry_run)
        return job

    monkeypatch.setattr(remote, "_bigquery", lambda cfg: nullcontext(SimpleNamespace(query=query)))
    with pytest.raises(DatabaseError, match="byte limit"):
        remote.execute("warehouse", "SELECT amount FROM orders", max_rows=5)
    assert calls == [True]
    calls.clear()
    job.total_bytes_processed = 100
    cancelled: list[bool] = []

    def timeout_result(**kwargs: Any) -> Any:
        raise TimeoutError("test timeout")

    job.result = timeout_result
    job.cancel = lambda: cancelled.append(True)
    with pytest.raises(DatabaseError, match="Remote read-only"):
        remote.execute("warehouse", "SELECT amount FROM orders", max_rows=5)
    assert calls == [True, False] and cancelled == [True]


def test_unconfirmed_roles_never_connect_and_configured_ids_do_not_imply_health(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = configuration("mysql").model_copy(update={"read_only_role_confirmed": False})
    remote = RemoteDatabaseRegistry({}, {"warehouse": config})
    with pytest.raises(DatabaseError, match="read-only"):
        remote.inspect("warehouse")
    monkeypatch.setenv("TEXT2SQL_EXTERNAL_CONNECTIONS", json.dumps([config.model_dump()]))
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(tmp_path))
    client = TestClient(create_app())
    assert client.get("/api/databases").status_code == 200
    response = client.get("/api/databases/warehouse/schema")
    assert response.status_code == 503
    assert response.json()["code"] == "READ_ONLY_ROLE_REQUIRED"
