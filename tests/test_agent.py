from __future__ import annotations

import asyncio
import json
from typing import Literal

from semantic_text2sql.agent import (
    TextToSQLAgent,
    _has_meaningful_order,
    _normalized_rows,
    _validate_profile_aware_sql,
)
from semantic_text2sql.models import (
    CheckRequest,
    ContextRequest,
    DatabaseProfile,
    GenerateRequest,
    SchemaInfo,
    StrategyHints,
    ValidationResult,
)


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.feedback: list[str | None] = []

    async def generate(
        self,
        *,
        provider: Literal["ollama", "agentrouter"],
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


def test_model2_returns_semantic_plan_and_sql_in_one_response(registry) -> None:  # type: ignore[no-untyped-def]
    response = json.dumps(
        {
            "semantic_plan": {
                "operations": ["COUNT", "GROUP"],
                "outputs": ["orders.customer_id", "order_count"],
                "aggregations": [
                    {
                        "function": "count",
                        "input": "orders.order_id",
                        "output": "order_count",
                        "group_by": ["orders.customer_id"],
                    }
                ],
                "group_by": ["orders.customer_id"],
                "logical_steps": ["Count orders at customer grain."],
            },
            "sql": (
                "SELECT customer_id, COUNT(order_id) AS order_count "
                "FROM orders GROUP BY customer_id"
            ),
        }
    )
    result = asyncio.run(
        TextToSQLAgent(registry, FakeModel([response])).generate(
            GenerateRequest(
                db_id="shop",
                question="How many orders does each customer have?",
                execute=True,
                context_request=ContextRequest(
                    tables=["orders"],
                    columns={"orders": ["order_id", "customer_id"]},
                ),
            )
        )
    )

    assert result.accepted is True
    assert result.semantic_plan is not None
    assert result.semantic_plan.operations == ["COUNT", "GROUP"]
    assert result.semantic_contract.outputs == ["orders.customer_id", "order_count"]
    assert result.sql is not None and "COUNT(order_id)" in result.sql


def test_invalid_sql_gets_schema_rich_llm_repair(registry) -> None:  # type: ignore[no-untyped-def]
    model = FakeModel(
        [
            "SELECT c.name, SUM(c.amount) FROM customers c GROUP BY c.name",
            "SELECT c.name, SUM(o.amount) AS total_amount FROM customers c "
            "JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.name",
        ]
    )
    agent = TextToSQLAgent(registry, model)

    result = asyncio.run(
        agent.generate(
            GenerateRequest(
                db_id="shop",
                question="Total order amount per customer",
                execute=True,
            )
        )
    )

    assert result.accepted is True
    assert len(result.attempts) == 2
    assert result.attempts[0].validation.valid is False
    assert result.attempts[1].validation.valid is True
    assert model.feedback[0] is None
    assert "Available columns by table" in (model.feedback[1] or "")
    assert result.rows == [["Anna", 150.0], ["Luca", 20.0]]
    assert result.context_plan is not None
    assert result.telemetry.generation_attempts == 2
    assert result.telemetry.context_expansions <= 2


def test_attempt_limit_is_fail_closed(registry) -> None:  # type: ignore[no-untyped-def]
    model = FakeModel(["SELECT invented FROM orders"] * 3)
    result = asyncio.run(
        TextToSQLAgent(registry, model).generate(
            GenerateRequest(db_id="shop", question="Show orders", max_attempts=3)
        )
    )

    assert result.accepted is False
    assert result.sql is None
    assert result.termination_reason == "attempt_limit"
    assert len(result.attempts) == 3


def test_malformed_optimization_retains_accepted_baseline(registry) -> None:  # type: ignore[no-untyped-def]
    baseline = "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id"
    malformed = (
        "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id, consider git"
    )

    result = asyncio.run(
        TextToSQLAgent(registry, FakeModel([malformed])).generate(
            GenerateRequest(
                db_id="shop",
                question="Optimize it",
                previous_sql=baseline,
                optimization_required=True,
                max_attempts=1,
                execute=True,
            )
        )
    )

    assert result.accepted is True
    assert result.sql == baseline
    assert result.rows == [[1, 150.0], [2, 20.0]]
    assert result.attempts[0].validation.code == "SQL_PARSE_FAILED"
    assert result.optimization is not None
    assert result.optimization.status == "rejected"
    assert result.optimization.selected_sql == "baseline"


def test_semantic_contract_drives_targeted_repair(registry) -> None:  # type: ignore[no-untyped-def]
    model = FakeModel(
        [
            "SELECT COUNT(*) FROM orders",
            "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
        ]
    )

    result = asyncio.run(
        TextToSQLAgent(registry, model).generate(
            GenerateRequest(db_id="shop", question="How many orders for each customer?")
        )
    )

    assert result.accepted is True
    assert result.semantic_contract is not None
    assert result.semantic_contract.requires_grouping is True
    assert result.attempts[0].validation.code == "SEMANTIC_GRAIN_MISSING"
    assert "SEMANTIC_GRAIN_MISSING" in (model.feedback[1] or "")
    assert result.attempts[1].validation.semantic_checks == [
        "aggregation:count",
        "grain:grouped",
    ]


def test_check_executes_only_valid_read_only_sql(registry) -> None:  # type: ignore[no-untyped-def]
    agent = TextToSQLAgent(registry, FakeModel([]))

    valid = agent.check(
        CheckRequest(db_id="shop", sql="SELECT name FROM customers ORDER BY name", execute=True)
    )
    invalid = agent.check(CheckRequest(db_id="shop", sql="DROP TABLE customers", execute=True))

    assert valid.validation.valid is True
    assert valid.rows == [["Anna"], ["Luca"]]
    assert invalid.validation.valid is False
    assert invalid.rows == []


def test_profile_validator_rejects_strftime_for_yyyymm_text() -> None:
    profile = DatabaseProfile.model_validate(
        {
            "db_id": "cards",
            "dialect": "sqlite",
            "profiled_at": "2026-01-01T00:00:00+00:00",
            "columns": [
                {
                    "table": "yearmonth",
                    "column": "Date",
                    "database_type": "TEXT",
                    "semantic_type": "date",
                    "row_count": 1,
                    "null_count": 0,
                    "null_ratio": 0,
                    "observed_format": "YYYYMM",
                }
            ],
        }
    )
    valid = ValidationResult(valid=True, code="VALID", message="valid")

    rejected = _validate_profile_aware_sql(
        "SELECT strftime('%Y', y.Date) FROM yearmonth y", profile, valid, dialect="sqlite"
    )
    accepted = _validate_profile_aware_sql(
        "SELECT SUBSTR(y.Date, 1, 4) FROM yearmonth y", profile, valid, dialect="sqlite"
    )

    assert rejected.code == "PROFILE_DATE_FORMAT_MISMATCH"
    assert rejected.valid is False
    assert accepted.valid is True


def test_optimization_equivalence_helpers_preserve_bags_order_and_float_noise() -> None:
    assert _has_meaningful_order("SELECT value FROM items ORDER BY value", dialect="sqlite")
    assert not _has_meaningful_order("SELECT value FROM items", dialect="sqlite")
    assert _normalized_rows([[1.00000000001], [2], [2]]) == [
        (1.0,),
        (2,),
        (2,),
    ]
