"""Deterministic attachment of approved business-glossary formulas."""

from __future__ import annotations

from semantic_text2sql.models import SemanticContract, StructuralFormula


def apply_structural_formulas(
    contract: SemanticContract, formulas: list[StructuralFormula]
) -> SemanticContract:
    """Attach database-specific approved formulas supplied by the glossary."""
    if not formulas:
        return contract
    columns = [
        argument for formula in formulas for argument in formula.arguments if "." in argument
    ]
    return contract.model_copy(
        update={
            "structural_formulas": list({formula.id: formula for formula in formulas}.values()),
            "required_columns": list(dict.fromkeys([*contract.required_columns, *columns])),
            "rationale": [
                *contract.rationale,
                "Approved glossary formulas were attached as database-specific context.",
            ],
        }
    )
