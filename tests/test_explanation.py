from __future__ import annotations

import asyncio

import pytest

from semantic_text2sql.explanation import (
    deterministic_explanation,
    explain_turn_detailed,
    extract_sql_facts,
)
from semantic_text2sql.models import ConversationState


class ContradictingExplainer:
    async def complete(self, model: str, prompt: str) -> str:
        assert "LEFT JOIN annual_2013 AS a" in prompt
        assert '"type":"LEFT"' in prompt
        return "There is no LEFT JOIN in the accepted query. It only uses an INNER JOIN."


def test_extract_sql_facts_describes_left_join_and_null_rejecting_filter() -> None:
    facts = extract_sql_facts(
        """
        SELECT o.id, c.name
        FROM orders AS o
        LEFT JOIN customers AS c ON c.id = o.customer_id
        WHERE c.country = 'DE'
        ORDER BY o.id
        LIMIT 5
        """,
        "sqlite",
    )

    assert facts["tables"] == ["orders", "customers"]
    assert facts["outputs"] == ["o.id", "c.name"]
    assert facts["joins"] == [
        {
            "type": "LEFT",
            "right_relation": "customers AS c",
            "right_alias": "c",
            "condition": "c.id = o.customer_id",
        }
    ]
    assert facts["filters"] == ["c.country = 'DE'"]
    assert facts["order_by"] == ["o.id"]
    assert facts["limits"] == ["5"]


def test_deterministic_join_explanation_separates_behavior_from_intention() -> None:
    state = ConversationState(
        session_id="explain",
        db_id="shop",
        root_question="List every order with its customer name.",
        resolved_question="List every order with its customer name.",
        last_sql=(
            "SELECT o.id, c.name FROM orders o LEFT JOIN customers c ON c.id = o.customer_id"
        ),
    )

    explanation = deterministic_explanation("EXPLAIN_SQL", state, "sqlite")

    assert "LEFT JOIN customers AS c" in explanation
    assert "preserves every row from the left side" in explanation
    assert "do not prove why" in explanation


def test_model_explanation_cannot_deny_an_explicit_left_join() -> None:
    state = ConversationState(
        session_id="explain",
        db_id="debit_card_specializing",
        root_question="Find the highest-spending customer outside the top ten.",
        resolved_question="Find the highest-spending customer outside the top ten.",
        last_sql=(
            "WITH annual_2013 AS (SELECT CustomerID, SUM(Consumption) AS total "
            "FROM yearmonth GROUP BY CustomerID) "
            "SELECT s.CustomerID, a.total FROM spending AS s "
            "LEFT JOIN annual_2013 AS a ON a.CustomerID = s.CustomerID"
        ),
    )

    with pytest.raises(ValueError, match="contradicted"):
        asyncio.run(
            explain_turn_detailed(
                ContradictingExplainer(),
                model="test",
                operation="EXPLAIN_SQL",
                client_message="Why is LEFT JOIN used?",
                state=state,
                dialect="sqlite",
            )
        )
