from __future__ import annotations

from pathlib import Path

from semantic_text2sql.glossary import GlossaryStore


def test_debit_card_glossary_retrieves_metric_and_disambiguation() -> None:
    root = Path(__file__).parents[1] / "data" / "business_glossaries"
    context = GlossaryStore(root).retrieve(
        "debit_card_specializing",
        "What percentage of premium stations are in Slovakia?",
    )

    assert "station segment" in context
    assert "gasstations.Segment" in context
    assert "percentage or ratio" in context
    assert "question > trusted question evidence > live schema" in context


def test_missing_glossary_fails_to_neutral_context(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert "No curated" in GlossaryStore(tmp_path).retrieve("unknown", "question")


def test_glossary_returns_matched_structural_formula() -> None:
    root = Path(__file__).parents[1] / "data" / "business_glossaries"
    formulas = GlossaryStore(root).structural_formulas(
        "debit_card_specializing", "customers paying more than 29 per unit"
    )

    assert len(formulas) == 1
    assert formulas[0].id == "unit_price"
    assert formulas[0].arguments == [
        "transactions_1k.Price",
        "transactions_1k.Amount",
    ]
