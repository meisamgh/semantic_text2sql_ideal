from semantic_text2sql.models import StructuralFormula
from semantic_text2sql.semantic import apply_structural_formulas, plan_semantics, resolution_report


def test_planner_extracts_only_generic_surface_hints() -> None:
    contract = plan_semantics("What are the top 3 categories by number of books in 2024?")

    assert contract.aggregation == "count"
    assert contract.requires_ordering is True
    assert contract.limit == 3
    assert contract.required_literals == ["2024"]
    assert contract.proposed_tables == []
    assert contract.required_columns == []


def test_planner_never_injects_dataset_specific_schema() -> None:
    contract = plan_semantics(
        "For people paying more than 29 per unit for product 5, show August 2012 consumption."
    )

    assert contract.proposed_tables == []
    assert contract.proposed_joins == []
    assert contract.outputs == []
    assert contract.filters == []


def test_glossary_formula_adds_only_its_declared_dependencies() -> None:
    formula = StructuralFormula(
        id="unit_price",
        operator="DIVIDE",
        arguments=["sales.price", "sales.quantity"],
        zero_safe=True,
        source="APPROVED_GLOSSARY",
        usage="FILTER_OPERAND",
    )

    contract = apply_structural_formulas(plan_semantics("price per unit"), [formula])

    assert contract.required_columns == ["sales.price", "sales.quantity"]
    assert contract.structural_formulas == [formula]
    assert resolution_report("price per unit", contract).status == "RESOLVED"


def test_unapproved_formula_remains_unresolved() -> None:
    report = resolution_report(
        "What is the ratio of active to inactive accounts?", plan_semantics("ratio")
    )

    assert report.status == "UNRESOLVED"
    assert report.semantic_call_required is True
