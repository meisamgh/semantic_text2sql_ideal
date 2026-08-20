from __future__ import annotations

import asyncio

from semantic_text2sql.historical import (
    SemanticSignature,
    history_is_useful,
)
from semantic_text2sql.models import (
    ColumnInfo,
    ContextRequest,
    SchemaInfo,
    SemanticPlan,
    TableInfo,
    TokenUsage,
)
from semantic_text2sql.semantic import plan_semantics
from semantic_text2sql.semantic_planner import plan_semantics_detailed, verify_semantic_plan


class _RepairingCompleter:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete_detailed(self, model: str, prompt: str):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return '{"operations":["INVALID"]}', TokenUsage(input_tokens=4, output_tokens=2)
        return '{"operations":["LOOKUP"]}', TokenUsage(input_tokens=6, output_tokens=3)


def _schema() -> SchemaInfo:
    return SchemaInfo(
        db_id="shop",
        tables=[
            TableInfo(
                name="customers",
                create_sql="",
                columns=[
                    ColumnInfo(name="CustomerID", data_type="INTEGER"),
                    ColumnInfo(name="Currency", data_type="TEXT"),
                ],
            )
        ],
    )


def test_semantic_planner_repairs_invalid_structured_output_once() -> None:
    completer = _RepairingCompleter()

    plan, usage = asyncio.run(
        plan_semantics_detailed(
            completer,
            "qwen3:8b",
            "Show customers",
            {"tables": {"customers": {"columns": ["CustomerID"]}}},
        )
    )

    assert plan.operations == ["LOOKUP"]
    assert len(completer.prompts) == 2
    assert "YOUR PREVIOUS JSON WAS REJECTED" in completer.prompts[1]
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5


def test_controller_does_not_replace_model_operation_from_question_keywords() -> None:
    question = "What is the ratio of EUR customers to CZK customers?"
    selection = ContextRequest(
        tables=["customers"],
        columns={"customers": ["CustomerID", "Currency"]},
    )
    proposed = SemanticPlan(
        operations=["PERCENT_OF_TOTAL"],
        outputs=["ratio"],
        measures=["customers.CustomerID"],
    )

    verified, _ = verify_semantic_plan(
        proposed,
        selection,
        _schema(),
        plan_semantics(question),
        None,
        question,
    )

    assert verified.operations == ["PERCENT_OF_TOTAL"]


def test_ratio_plan_qualifies_columns_and_compiles_final_division() -> None:
    question = "What is the ratio of EUR customers to CZK customers?"
    selection = ContextRequest(
        tables=["customers"],
        columns={"customers": ["CustomerID", "Currency"]},
    )
    proposed = SemanticPlan(
        operations=["RATIO", "COUNT"],
        outputs=["eur_to_czk_ratio"],
        filters=[
            {"operand": "Currency", "operator": "=", "value": "EUR"},
            {"operand": "Currency", "operator": "=", "value": "CZK"},
        ],
        aggregations=[
            {"function": "count", "input": "CustomerID", "output": "eur_count"},
            {"function": "count", "input": "CustomerID", "output": "czk_count"},
        ],
        final_operations=[
            {
                "name": "eur_to_czk_ratio",
                "operator": "divide",
                "left": "eur_count",
                "right": "czk_count",
            }
        ],
    )

    verified, contract = verify_semantic_plan(
        proposed,
        selection,
        _schema(),
        plan_semantics(question),
        None,
        question,
    )

    assert [item.operand for item in verified.filters] == [
        "customers.Currency",
        "customers.Currency",
    ]
    assert all(
        item.input == "customers.CustomerID" for item in verified.aggregations
    )
    assert contract.outputs == ["eur_to_czk_ratio"]
    assert contract.named_outputs == ["eur_to_czk_ratio"]
    assert contract.output_operations[0].model_dump() == {
        "name": "eur_to_czk_ratio",
        "operator": "divide",
        "left": "eur_count",
        "right": "czk_count",
    }


def test_model_outputs_remain_authoritative_without_final_arithmetic() -> None:
    selection = ContextRequest(
        tables=["customers"],
        columns={"customers": ["CustomerID", "Currency"]},
    )
    proposed = SemanticPlan(
        operations=["COUNT", "GROUP"],
        outputs=["customers.Currency", "customer_count"],
        aggregations=[
            {
                "function": "count",
                "input": "customers.CustomerID",
                "output": "customer_count",
                "group_by": ["customers.Currency"],
            }
        ],
        group_by=["customers.Currency"],
    )

    _, contract = verify_semantic_plan(
        proposed,
        selection,
        _schema(),
        plan_semantics("Count customers per currency"),
        None,
        "Count customers per currency",
    )

    assert contract.outputs == ["customers.Currency", "customer_count"]
    assert contract.named_outputs == ["customers.Currency", "customer_count"]


def test_multiple_tables_alone_do_not_enable_history() -> None:
    signature = SemanticSignature(
        operations=frozenset({"LOOKUP"}),
        metrics=frozenset({"customer"}),
        tables=frozenset({"customers", "orders", "products"}),
        grain=frozenset(),
        temporal=frozenset(),
    )
    request = ContextRequest(tables=["customers", "orders", "products"])
    plan = SemanticPlan(operations=["LOOKUP"])

    assert history_is_useful(signature, request, plan) is False


def test_one_table_ratio_enables_history() -> None:
    signature = SemanticSignature(
        operations=frozenset({"RATIO"}),
        metrics=frozenset({"currency", "customer"}),
        tables=frozenset({"customers"}),
        grain=frozenset(),
        temporal=frozenset(),
    )
    request = ContextRequest(tables=["customers"])
    plan = SemanticPlan(operations=["RATIO"])

    assert history_is_useful(signature, request, plan) is True
