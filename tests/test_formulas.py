from semantic_text2sql.formulas import apply_structural_formulas
from semantic_text2sql.models import SemanticContract, StructuralFormula


def test_glossary_formula_adds_only_its_declared_dependencies() -> None:
    formula = StructuralFormula(
        id="unit_price",
        operator="DIVIDE",
        arguments=["sales.price", "sales.quantity"],
        zero_safe=True,
        source="APPROVED_GLOSSARY",
        usage="FILTER_OPERAND",
    )

    contract = apply_structural_formulas(SemanticContract(), [formula])

    assert contract.required_columns == ["sales.price", "sales.quantity"]
    assert contract.structural_formulas == [formula]
