import asyncio
from pathlib import Path

from semantic_text2sql.agent import TextToSQLAgent, _sqlglot_optimization_candidate
from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.models import GenerateRequest, SchemaInfo, StrategyHints
from semantic_text2sql.profiling import ProfileStore
from semantic_text2sql.validator import normalize_sql


def test_sqlglot_candidate_removes_redundant_true_predicate() -> None:
    original = "SELECT CustomerID FROM customers WHERE 1 = 1 AND CustomerID > 0"

    candidate = _sqlglot_optimization_candidate(original, dialect="sqlite")

    assert candidate is not None
    assert "1 = 1" not in candidate
    assert "CustomerID > 0" in candidate
    assert normalize_sql(candidate, dialect="sqlite") != normalize_sql(original, dialect="sqlite")


def test_sqlglot_candidate_preserves_identifier_case_and_query_shape() -> None:
    original = (
        "SELECT c.CustomerID, SUM(t.Price) AS TotalSpend "
        "FROM customers c JOIN transactions_1k t ON c.CustomerID = t.CustomerID "
        "GROUP BY c.CustomerID"
    )

    candidate = _sqlglot_optimization_candidate(original, dialect="sqlite")

    assert candidate is not None
    assert "CustomerID" in candidate
    assert "SUM(t.Price)" in candidate
    assert "JOIN transactions_1k" in candidate
    assert "GROUP BY" in candidate


class RedundantPredicateModel:
    async def generate(
        self,
        *,
        model: str,
        provider: str,
        question: str,
        evidence: str | None,
        schema: SchemaInfo,
        strategy: StrategyHints,
        dialect: str,
        profile_context: str,
        previous_sql: str | None,
        feedback: str | None,
        rejected_shapes: list[str],
        generation_style: str,
    ) -> tuple[str, int]:
        return "SELECT COUNT(*) AS n FROM orders WHERE 1 = 1", 1


def test_normal_generation_runs_automatic_sqlglot_stage(tmp_path: Path) -> None:
    directory = tmp_path / "orders"
    directory.mkdir()
    (directory / "orders.sqlite").write_bytes(_sqlite_database_bytes(tmp_path))
    registry = DatabaseRegistry(tmp_path)
    db_id = "orders"
    stages: list[str] = []
    agent = TextToSQLAgent(
        registry,
        RedundantPredicateModel(),  # type: ignore[arg-type]
        ProfileStore(tmp_path / "profiles"),
    )

    response = asyncio.run(
        agent.generate(
            GenerateRequest(
                db_id=db_id,
                question="How many orders?",
                provider="agentrouter",
                model="test",
                execute=True,
            ),
            progress=stages.append,
        )
    )

    assert response.accepted
    assert response.optimization is not None
    assert response.optimization.optimizer == "sqlglot"
    assert response.optimization.status in {"optimized", "equivalent_not_faster"}
    assert "optimization" in stages
    assert stages[-1] == "execution"


def _sqlite_database_bytes(tmp_path: Path) -> bytes:
    import sqlite3

    path = tmp_path / "source.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY)")
    connection.executemany("INSERT INTO orders VALUES (?)", [(1,), (2,)])
    connection.commit()
    connection.close()
    return path.read_bytes()
