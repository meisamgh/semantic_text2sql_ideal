from __future__ import annotations

import asyncio

from semantic_text2sql.interpreter import interpret_question, reconcile_contract
from semantic_text2sql.models import PreliminarySemanticIR
from semantic_text2sql.semantic import plan_semantics


class FakeCompleter:
    async def complete(self, model: str, prompt: str) -> str:
        assert model == "claude-test"
        assert "customers" in prompt
        return """{
          "outputs":["customer","order_count"],
          "entities":["customer","order"],
          "tables":["customers","orders","invented"],
          "required_columns":["customers.customer_id","orders.customer_id","bad.x"],
          "joins":["customers.customer_id = orders.customer_id"],
          "filters":[],"aggregation":"count","grain":["customer"],
          "order_by":[],"limit":null,"ambiguities":[],"confidence":0.9
        }"""


def test_interpreter_and_reconciler_ground_scope_in_live_schema(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    interpreted = asyncio.run(
        interpret_question(
            FakeCompleter(),
            "claude-test",
            "How many orders for each customer?",
            None,
            schema,
        )
    )
    contract = reconcile_contract(
        plan_semantics("How many orders for each customer?"), interpreted, schema
    )

    assert contract.proposed_tables == ["customers", "orders"]
    assert contract.required_columns == ["customers.customer_id", "orders.customer_id"]
    assert contract.aggregation == "count"
    assert contract.requires_grouping is True


def test_low_confidence_interpretation_remains_advisory(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    interpreted = PreliminarySemanticIR(
        aggregation="sum",
        grain=["country"],
        order_by=["amount DESC"],
        confidence=0.4,
    )

    contract = reconcile_contract(plan_semantics("Show orders"), interpreted, schema)

    assert contract.aggregation is None
    assert contract.requires_grouping is False
    assert contract.requires_ordering is False
