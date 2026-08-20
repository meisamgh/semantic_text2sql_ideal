from __future__ import annotations

from semantic_text2sql.context import (
    MAX_CONTEXT_EXPANSIONS,
    build_context_plan,
    failure_context_category,
    model_context_payload,
)
from semantic_text2sql.models import StructuralFormula
from semantic_text2sql.semantic import apply_structural_formulas, plan_semantics, resolution_report


def test_confidence_gate_skips_simple_resolved_question() -> None:
    question = "How many books are there?"
    report = resolution_report(question, plan_semantics(question))

    assert report.status == "RESOLVED"
    assert report.semantic_call_required is False


def test_confidence_gate_escalates_unresolved_complex_metric() -> None:
    question = "What is the average price per item for the top spending customer?"
    report = resolution_report(question, plan_semantics(question))

    assert report.status == "UNRESOLVED"
    assert report.semantic_call_required is True
    assert any(item.field == "formula" and item.critical for item in report.fields)


def test_context_plan_records_selected_dependencies(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    contract = plan_semantics("Total order amount per customer")
    plan = build_context_plan(schema, None, contract, "Total order amount per customer", None)

    assert set(plan.columns) == {"customers", "orders"}
    assert plan.metadata.relationships is True
    assert "quantiles" in plan.excluded
    assert plan.token_budget == 1_500
    assert any(item.kind == "COLUMN_SCHEMA" for item in plan.requirements)
    assert any(item.kind == "OUTPUT_GRAIN" for item in plan.requirements)
    assert plan.coverage_complete is False


def test_formula_node_emits_formula_and_physical_type_requirements(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    contract = apply_structural_formulas(
        plan_semantics("orders where amount per customer is over 10"),
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
    contract = plan_semantics(question)
    plan = build_context_plan(schema, None, contract, question, None)

    payload = model_context_payload(plan, None, contract, None, question, "sqlite")
    customers = payload["execution_context"]["tables"]["customers"]

    assert customers["primary_key"] == ["customer_id"]
    assert customers["unique_keys"] == [["customer_id"]]
    assert customers["grain"] == "one row per customer_id"
    assert "relationships" not in payload["execution_context"]


def test_multi_table_context_includes_key_uniqueness_and_fanout(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    question = "Show order amounts with their customer names."
    contract = plan_semantics(question)
    plan = build_context_plan(schema, None, contract, question, None)

    payload = model_context_payload(plan, None, contract, None, question, "sqlite")
    relationship = payload["execution_context"]["relationships"][0]

    assert relationship == {
        "left": "orders.customer_id",
        "right": "customers.customer_id",
        "state": "VERIFIED_FK",
        "cardinality": "ONE_TO_MANY",
        "left_key_unique": False,
        "right_key_unique": True,
        "fanout_risk": True,
    }
