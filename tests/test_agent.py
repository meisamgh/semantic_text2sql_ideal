from __future__ import annotations

import asyncio
from typing import Literal

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.models import (
    CheckRequest,
    GenerateRequest,
    ModelProvider,
    SchemaInfo,
    StrategyHints,
)


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.feedback: list[str | None] = []

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
        self.feedback.append(feedback)
        return self.responses.pop(0), 1


def test_sql_only_generation_executes_read_only_query(registry) -> None:  # type: ignore[no-untyped-def]
    result = asyncio.run(
        TextToSQLAgent(registry, FakeModel(["SELECT name FROM customers ORDER BY name"])).generate(
            GenerateRequest(db_id="shop", question="List customer names", execute=True)
        )
    )

    assert result.accepted is True
    assert result.rows == [["Anna"], ["Luca"]]
    assert result.attempts[0].validation.code == "SQL_SAFETY_VALID"


def test_database_error_is_returned_to_model_for_focused_repair(registry) -> None:  # type: ignore[no-untyped-def]
    model = FakeModel(["SELECT missing FROM orders", "SELECT amount FROM orders ORDER BY order_id"])
    result = asyncio.run(
        TextToSQLAgent(registry, model).generate(
            GenerateRequest(db_id="shop", question="List order amounts", execute=True)
        )
    )

    assert result.accepted is True
    assert result.attempts[0].validation.code == "DATABASE_ERROR"
    assert "DATABASE_ERROR" in (model.feedback[1] or "")
    assert result.rows == [[100.0], [50.0], [20.0]]


def test_write_statement_is_rejected_before_execution(registry) -> None:  # type: ignore[no-untyped-def]
    result = asyncio.run(
        TextToSQLAgent(registry, FakeModel(["DELETE FROM orders"])).generate(
            GenerateRequest(db_id="shop", question="Delete orders", max_attempts=1, execute=True)
        )
    )

    assert result.accepted is False
    assert result.attempts[0].validation.code == "SQL_NOT_READ_ONLY"


def test_check_endpoint_executes_only_safe_sql(registry) -> None:  # type: ignore[no-untyped-def]
    agent = TextToSQLAgent(registry, FakeModel([]))

    valid = agent.check(
        CheckRequest(db_id="shop", sql="SELECT name FROM customers ORDER BY name", execute=True)
    )
    invalid = agent.check(CheckRequest(db_id="shop", sql="DROP TABLE customers", execute=True))

    assert valid.rows == [["Anna"], ["Luca"]]
    assert invalid.validation.valid is False
    assert invalid.rows == []
