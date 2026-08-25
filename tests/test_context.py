from __future__ import annotations

from semantic_text2sql.context import (
    MAX_CONTEXT_EXPANSIONS,
    build_context_plan,
    failure_context_category,
    model_context_payload,
)
from semantic_text2sql.formulas import apply_structural_formulas
from semantic_text2sql.models import SemanticContract, StructuralFormula


def test_context_plan_records_selected_dependencies(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    contract = SemanticContract()
    plan = build_context_plan(schema, None, contract, "Total order amount per customer", None)

    assert set(plan.columns) == {"customers", "orders"}
    assert plan.metadata.relationships is True
    assert "quantiles" in plan.excluded
    assert plan.token_budget == 1_500
    assert any(item.kind == "COLUMN_SCHEMA" for item in plan.requirements)
    assert plan.coverage_complete is False


def test_formula_node_emits_formula_and_physical_type_requirements(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    contract = apply_structural_formulas(
        SemanticContract(),
        [
            StructuralFormula(
                id="amount_ratio",
                operator="DIVIDE",
                arguments=["orders.amount", "orders.customer_id"],
                zero_safe=True,
                source="APPROVED_GLOSSARY",
                usage="FILTER_OPERAND",
            )
        ],
    )
    plan = build_context_plan(schema, None, contract, "amount per customer", None)

    kinds = {(item.kind, item.target) for item in plan.requirements}
    assert ("FORMULA_DEFINITION", "amount_ratio") in kinds
    assert ("PHYSICAL_TYPE", "orders.amount") in kinds


def test_reactive_retrieval_categories_are_specific_and_bounded() -> None:
    assert MAX_CONTEXT_EXPANSIONS == 2
    assert failure_context_category("PROFILE_DATE_FORMAT_MISMATCH") == "DATE_FORMAT_UNKNOWN"
    assert failure_context_category("JOIN_CARDINALITY_RISK") == "CARDINALITY_UNKNOWN"
    assert failure_context_category("SQL_PARSE_FAILED") is None


def test_single_table_context_includes_primary_key_and_grain_only(registry) -> None:  # type: ignore[no-untyped-def]
    full_schema = registry.inspect("shop")
    schema = full_schema.model_copy(
        update={
            "tables": [table for table in full_schema.tables if table.name == "customers"],
            "relationships": [],
        }
    )
    question = "How many customers are there?"
    contract = SemanticContract()
    plan = build_context_plan(schema, None, contract, question, None)

    payload = model_context_payload(plan, None, contract, None, question, "sqlite")
    customers = payload["tables"]["customers"]

    assert customers["primary_key"] == ["customer_id"]
    assert customers["unique_keys"] == [["customer_id"]]
    assert customers["grain"] == "one row per customer_id"
    assert payload.get("relationships", []) == []


def test_multi_table_context_includes_key_uniqueness_and_fanout(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    question = "Show order amounts with their customer names."
    contract = SemanticContract()
    plan = build_context_plan(schema, None, contract, question, None)

    payload = model_context_payload(plan, None, contract, None, question, "sqlite")
    relationship = payload["relationships"][0]

    assert relationship == {
        "left": "orders.customer_id",
        "right": "customers.customer_id",
        "state": "VERIFIED_FK",
        "cardinality": "MANY_TO_ONE",
        "left_key_unique": False,
        "right_key_unique": True,
        "fanout_risk": True,
    }
