import asyncio
import json

import pytest

from semantic_text2sql.models import ResultPresentation, TokenUsage
from semantic_text2sql.presentation import present_result, validate_chart
from semantic_text2sql.runtime import RequestBudget


class Presenter:
    async def complete_detailed(self, **kwargs):
        assert "returned" not in kwargs or kwargs["returned"]
        return json.dumps(
            {
                "summary": "EUR has 2 customers; CZK has 30 customers.",
                "chart_type": "bar",
                "x": "Currency",
                "y": "Count",
                "title": "Customers",
            }
        ), TokenUsage(input_tokens=30, output_tokens=20)


def test_presentation_uses_shared_budget_and_reports_usage():
    budget = RequestBudget(timeout_seconds=5)
    result, usage, entry = asyncio.run(
        present_result(
            Presenter(),
            question="Counts by currency?",
            sql="SELECT ...",
            columns=["Currency", "Count"],
            rows=[["EUR", 2], ["CZK", 30]],
            truncated=False,
            model="test",
            provider="test",
            budget=budget,
        )
    )
    assert result.chart_type == "bar"
    assert usage.total_tokens == 50
    assert budget.model_calls == 1
    assert entry.status == "SUCCEEDED"


def test_presentation_failure_preserves_scalar_and_partial_warning():
    result, _, entry = asyncio.run(
        present_result(
            None,
            question="Total?",
            sql="SELECT ...",
            columns=["Total"],
            rows=[[42]],
            truncated=True,
            model="test",
            provider="test",
            budget=RequestBudget(timeout_seconds=5),
        )
    )
    assert result.summary == "Total: 42"
    assert result.chart_type == "none"
    assert "Partial" in result.note
    assert entry.status == "FAILED"


@pytest.mark.parametrize(
    "chart,x,y,rows",
    [
        ("bar", "missing", "Count", [["A", 2], ["B", 3]]),
        ("bar", "Currency", "Count", [["A", 2], ["A", 3]]),
        ("line", "Currency", "Count", [["A", 2], ["B", 3]]),
        ("bar", "Currency", "Count", [["A", None], ["B", None]]),
    ],
)
def test_reject_invalid_chart(chart, x, y, rows):
    with pytest.raises(ValueError):
        validate_chart(
            ResultPresentation(summary="test", chart_type=chart, x=x, y=y),
            ["Currency", "Count"],
            rows,
        )
