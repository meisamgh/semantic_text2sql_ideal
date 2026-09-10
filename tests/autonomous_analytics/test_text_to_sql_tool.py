from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from autonomous_analytics.investigator.state import record_text_to_sql_result
from autonomous_analytics.models.investigation import InvestigationState
from autonomous_analytics.tools.text_to_sql import TextToSQLTool
from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.glossary import GlossaryStore
from semantic_text2sql.historical import HistoricalQueryStore
from semantic_text2sql.hybrid_retrieval import HybridSchemaRetriever
from semantic_text2sql.models import (
    ModelProvider,
    SchemaInfo,
    StrategyHints,
    ValidationResult,
)
from semantic_text2sql.postgres import PostgresRegistry
from semantic_text2sql.profiling import ProfileStore
from semantic_text2sql.service import TextToSQLService


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                float("customer" in text.casefold()),
                float("country" in text.casefold()),
                float("order" in text.casefold()),
            ]
            for text in texts
        ]


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    async def generate(
        self,
        *,
        provider: ModelProvider,
        model: str,
        question: str,
        evidence: str | None,
        schema: SchemaInfo,
        strategy: StrategyHints,
        dialect: Literal["sqlite", "postgres"],
        profile_context: str,
        previous_sql: str | None,
        feedback: str | None,
        rejected_shapes: list[str],
        generation_style: Literal["reasoning", "icl", "alternative"],
    ) -> tuple[str, int]:
        return self.responses.pop(0), 7


class MockPostgresRegistry(PostgresRegistry):
    """A PostgreSQL contract double; it never opens a network connection."""

    def __init__(self) -> None:
        super().__init__({"business_postgres": "postgresql://not-used"})
        self.executed_sql: list[str] = []

    def inspect(self, db_id: str) -> SchemaInfo:
        assert db_id == "business_postgres"
        return SchemaInfo.model_validate(
            {
                "db_id": db_id,
                "dialect": "postgres",
                "tables": [
                    {
                        "name": "signups",
                        "create_sql": (
                            "CREATE TABLE signups (signup_id bigint PRIMARY KEY, "
                            "channel text, signed_up_at timestamp)"
                        ),
                        "columns": [
                            {
                                "name": "signup_id",
                                "data_type": "bigint",
                                "primary_key": True,
                                "nullable": False,
                            },
                            {
                                "name": "channel",
                                "data_type": "text",
                                "primary_key": False,
                                "nullable": False,
                            },
                            {
                                "name": "signed_up_at",
                                "data_type": "timestamp without time zone",
                                "primary_key": False,
                                "nullable": False,
                            },
                        ],
                    }
                ],
            }
        )

    def explain(self, db_id: str, sql: str, validation: ValidationResult) -> ValidationResult:
        assert db_id == "business_postgres"
        return validation.model_copy(update={"explain_plan": ["Mock PostgreSQL aggregate"]})

    def execute(
        self,
        db_id: str,
        sql: str,
        *,
        max_rows: int,
        timeout_seconds: float = 5.0,
    ) -> tuple[list[str], list[list[object]], bool]:
        del timeout_seconds
        assert db_id == "business_postgres"
        self.executed_sql.append(sql)
        rows: list[list[object]] = [["Paid Search", 42], ["Organic", 31]]
        return ["channel", "signup_count"], rows[:max_rows], len(rows) > max_rows


def _tool(registry, tmp_path: Path, responses: list[str], **kwargs) -> TextToSQLTool:  # type: ignore[no-untyped-def]
    service = TextToSQLService(
        database=registry,
        profiles=ProfileStore(tmp_path / "profiles"),
        glossaries=GlossaryStore(),
        history=HistoricalQueryStore(),
        retriever=HybridSchemaRetriever(FakeEncoder()),
        agent=TextToSQLAgent(registry, FakeModel(responses)),
        context_completers={},
    )
    return TextToSQLTool(service, **kwargs)


def test_successful_question_becomes_bounded_sql_evidence(registry, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    tool = _tool(
        registry,
        tmp_path,
        [
            "SELECT country, COUNT(*) AS customer_count "
            "FROM customers GROUP BY country ORDER BY country"
        ],
        evidence_row_limit=1,
        max_attempts=1,
    )

    result = asyncio.run(tool.ask(db_id="shop", question="How many customers are in each country?"))

    assert result.success is True
    assert result.error is None
    assert result.evidence is not None
    assert result.evidence.sql is not None
    assert result.evidence.result_summary == {
        "columns": ["country", "customer_count"],
        "row_count": 1,
        "truncated": True,
        "rows": [{"country": "Germany", "customer_count": 1}],
    }
    assert result.evidence.metadata["attempt_count"] == 1
    assert result.evidence.metadata["attempts"][0]["validation"]["code"] == "SQL_SAFETY_VALID"


def test_unsafe_sql_is_rejected_and_returned_structurally(registry, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    tool = _tool(registry, tmp_path, ["DELETE FROM customers"], max_attempts=1)

    result = asyncio.run(tool.ask(db_id="shop", question="Delete every customer"))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "SQL_NOT_READ_ONLY"
    assert result.error.category == "safety"
    assert result.evidence is not None
    assert result.evidence.metadata["attempts"][0]["sql"] == "DELETE FROM customers"
    assert len(registry.execute("shop", "SELECT * FROM customers", max_rows=10)[1]) == 2


def test_database_failure_is_structured(registry, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    tool = _tool(registry, tmp_path, [], max_attempts=1)

    result = asyncio.run(tool.ask(db_id="missing", question="Count records"))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "DATABASE_NOT_FOUND"
    assert result.evidence is not None
    assert result.evidence.metadata["error"]["category"] == "database"


def test_investigation_state_consumes_result_and_charges_budget(registry, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    tool = _tool(registry, tmp_path, ["SELECT COUNT(*) AS count FROM customers"], max_attempts=1)
    result = asyncio.run(tool.ask(db_id="shop", question="How many customers are there?"))
    state = InvestigationState(
        case_id="case-1",
        observation="Customer count requires investigation.",
        primary_metrics=["customers"],
    )

    updated = record_text_to_sql_result(state, result)

    assert state.sql_calls == 0
    assert updated.sql_calls == 1
    assert updated.evidence == [result.evidence]


def test_postgres_question_becomes_investigation_evidence(registry, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    postgres = MockPostgresRegistry()
    model = FakeModel(
        [
            "SELECT channel, COUNT(*) AS signup_count "
            "FROM signups "
            "WHERE signed_up_at >= CURRENT_DATE - INTERVAL '7 days' "
            "GROUP BY channel ORDER BY signup_count DESC"
        ]
    )
    service = TextToSQLService(
        database=registry,
        postgres=postgres,
        profiles=ProfileStore(tmp_path / "profiles"),
        glossaries=GlossaryStore(),
        history=HistoricalQueryStore(),
        retriever=HybridSchemaRetriever(FakeEncoder()),
        agent=TextToSQLAgent(registry, model, postgres=postgres),
        context_completers={},
    )
    tool = TextToSQLTool(service, evidence_row_limit=10, max_attempts=1)
    state = InvestigationState(
        case_id="case-postgres",
        observation="Signup growth needs a channel breakdown.",
        primary_metrics=["signups"],
    )

    result = asyncio.run(
        tool.ask(
            db_id="business_postgres",
            question="Which channels contributed most to signups in the last 7 days?",
        )
    )
    updated = record_text_to_sql_result(state, result)

    assert result.success is True
    assert result.evidence is not None
    assert result.evidence.metadata["dialect"] == "postgres"
    assert result.evidence.result_summary["rows"][0] == {
        "channel": "Paid Search",
        "signup_count": 42,
    }
    assert "INTERVAL '7 days'" in (result.evidence.sql or "")
    assert postgres.executed_sql
    assert updated.sql_calls == 1
    assert updated.evidence == [result.evidence]
