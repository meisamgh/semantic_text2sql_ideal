"""Versioned, deterministic business-glossary retrieval."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from semantic_text2sql.models import StructuralFormula

_TOKEN = re.compile(r"[A-Za-z0-9]+")


class GlossaryTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str
    definition: str
    synonyms: list[str] = Field(default_factory=list)
    formula: str | None = None
    structural_formula: StructuralFormula | None = None
    columns: list[str] = Field(default_factory=list)
    grain: str | None = None
    caveats: list[str] = Field(default_factory=list)
    core: bool = False


class BusinessGlossary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_id: str
    version: str
    source: str
    precedence: list[str]
    terms: list[GlossaryTerm]


class GlossaryStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root.resolve() if root else None
        self._cache: dict[str, tuple[int, BusinessGlossary]] = {}

    def load(self, db_id: str) -> BusinessGlossary | None:
        if self.root is None:
            return None
        path = self.root / f"{db_id}.json"
        if not path.is_file():
            return None
        modified_ns = path.stat().st_mtime_ns
        cached = self._cache.get(db_id)
        if cached is not None and cached[0] == modified_ns:
            return cached[1]
        glossary = BusinessGlossary.model_validate_json(path.read_text(encoding="utf-8"))
        self._cache[db_id] = (modified_ns, glossary)
        return glossary

    def retrieve(
        self, db_id: str, question: str, *, top_k: int = 8, include_formulas: bool = True
    ) -> str:
        glossary = self.load(db_id)
        if glossary is None:
            return "No curated business glossary is available."
        query = _tokens(question)
        ranked = sorted(
            glossary.terms,
            key=lambda item: (
                -len(query & _tokens(f"{item.term} {' '.join(item.synonyms)}")),
                -_score(item, query),
                item.term,
            ),
        )
        selected = [
            item for item in ranked if query & _tokens(f"{item.term} {' '.join(item.synonyms)}")
        ][:top_k]
        lines = [
            f"Business glossary {glossary.db_id} version={glossary.version}",
            "Precedence: " + " > ".join(glossary.precedence),
        ]
        for item in selected:
            facts = [item.definition]
            if include_formulas and item.formula:
                facts.append(f"formula={item.formula}")
            if include_formulas and item.structural_formula:
                facts.append("structural_formula=" + item.structural_formula.model_dump_json())
            if item.columns:
                facts.append(f"columns={item.columns}")
            if item.grain:
                facts.append(f"grain={item.grain}")
            if item.caveats:
                facts.append(f"caveats={item.caveats}")
            lines.append(f"- {item.term}: " + "; ".join(facts))
        return "\n".join(lines)

    def structural_formulas(self, db_id: str, question: str) -> list[StructuralFormula]:
        """Return only explicitly matched, approved structural formulas."""
        glossary = self.load(db_id)
        if glossary is None:
            return []
        query = _tokens(question)
        return [
            item.structural_formula
            for item in glossary.terms
            if item.structural_formula is not None
            and query & _tokens(f"{item.term} {' '.join(item.synonyms)}")
        ]

    def relevant_concept_ids(self, db_id: str, question: str) -> set[str]:
        """Return only glossary concepts with direct term/synonym overlap."""
        glossary = self.load(db_id)
        if glossary is None:
            return set()
        query = _tokens(question)
        return {
            item.term.casefold().replace(" ", "_")
            for item in glossary.terms
            if query & _tokens(f"{item.term} {' '.join(item.synonyms)}")
        }


def _score(item: GlossaryTerm, query: set[str]) -> int:
    return len(query & _tokens(f"{item.term} {' '.join(item.synonyms)} {item.definition}"))


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(value)}
