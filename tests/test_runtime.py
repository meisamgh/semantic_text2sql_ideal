from __future__ import annotations

import asyncio

import pytest

from semantic_text2sql.runtime import RequestBudget, RequestBudgetExceeded


def test_request_budget_counts_model_and_database_calls() -> None:
    async def exercise() -> None:
        budget = RequestBudget(10, max_model_calls=1, max_database_calls=1)
        assert await budget.wait(asyncio.sleep(0, result="model"), kind="model") == "model"
        assert await budget.wait(asyncio.sleep(0, result="db"), kind="database") == "db"
        with pytest.raises(RequestBudgetExceeded, match="model-call"):
            await budget.wait(asyncio.sleep(0), kind="model")

    asyncio.run(exercise())


def test_request_budget_enforces_shared_deadline() -> None:
    async def exercise() -> None:
        budget = RequestBudget(0.001)
        with pytest.raises(RequestBudgetExceeded, match="deadline"):
            await budget.wait(asyncio.sleep(0.02), kind="work")

    asyncio.run(exercise())
