"""Dataset-agnostic semantic hints and deterministic context reconciliation."""

from __future__ import annotations

import re
from typing import Literal

from semantic_text2sql.models import (
    FieldResolution,
    ResolutionReport,
    SemanticContract,
    StructuralFormula,
)

_TOP_N = re.compile(r"\b(?:top|first)\s+(\d+)\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])")
_QUOTED = re.compile(r"['\"]([^'\"]{1,100})['\"]")


def plan_semantics(question: str, evidence: str | None = None) -> SemanticContract:
    """Extract generic surface hints without selecting schema objects or solving SQL."""
    text = " ".join(value for value in (question, evidence or "") if value)
    lowered = text.casefold()
    aggregation: Literal["count", "sum", "average", "min", "max"] | None = None
    rationale: list[str] = []
    for kind, phrases in (
        ("average", ("average", "mean ")),
        ("sum", ("sum of", "total ")),
        ("count", ("how many", "number of", "count of")),
        ("max", ("maximum", "highest")),
        ("min", ("minimum", "lowest")),
    ):
        if any(phrase in lowered for phrase in phrases):
            aggregation = kind  # type: ignore[assignment]
            rationale.append(f"Explicit {kind} language was detected.")
            break
    top_match = _TOP_N.search(text)
    limit = int(top_match.group(1)) if top_match else None
    numbers = _NUMBER.findall(text)
    if limit is not None:
        numbers = [value for value in numbers if value != str(limit)]
    literals = list(dict.fromkeys([*_YEAR.findall(text), *numbers, *_QUOTED.findall(text)]))
    grouped = aggregation is not None and bool(
        re.search(r"\b(?:each|per|grouped by|broken down by)\b", lowered)
    )
    ranking = bool(re.search(r"\b(?:top|bottom|highest|lowest)\b", lowered))
    return SemanticContract(
        aggregation=aggregation,
        requires_grouping=grouped,
        requires_distinct=bool(re.search(r"\b(?:distinct|unique)\b", lowered)),
        requires_ordering=ranking,
        limit=limit,
        required_literals=literals,
        rationale=rationale,
    )


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


def resolution_report(question: str, contract: SemanticContract) -> ResolutionReport:
    """Describe unresolved formulas without adding dataset-specific answers."""
    formula_requested = bool(
        re.search(
            r"\b(per unit|average price per|percentage|ratio|share|difference|growth)\b",
            question.casefold(),
        )
    )
    formula_resolved = bool(contract.structural_formulas or contract.formula_metrics)
    resolved = formula_resolved or not formula_requested
    field = FieldResolution(
        field="formula",
        status="RESOLVED" if resolved else "UNRESOLVED",
        confidence=0.95 if resolved else 0.3,
        provenance=["approved database glossary" if formula_resolved else "question surface hints"],
        critical=formula_requested,
    )
    unresolved = field.critical and field.status != "RESOLVED"
    return ResolutionReport(
        status="UNRESOLVED" if unresolved else "RESOLVED",
        fields=[field],
        semantic_call_required=unresolved,
    )
