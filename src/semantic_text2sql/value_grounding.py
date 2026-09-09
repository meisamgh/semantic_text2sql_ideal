"""Conservative deterministic grounding of explicit categorical literals."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from semantic_text2sql.glossary import BusinessGlossary, GlossaryTerm
from semantic_text2sql.models import (
    ContextRequest,
    DatabaseProfile,
    ValueFrequency,
    ValueGroundingIssue,
)


@dataclass(frozen=True)
class ValueGroundingResult:
    evidence: list[str] = field(default_factory=list)
    issue: ValueGroundingIssue | None = None


def ground_question_values(
    question: str,
    context: ContextRequest,
    profile: DatabaseProfile | None,
    glossary: BusinessGlossary | None,
) -> ValueGroundingResult:
    """Resolve approved aliases and reject explicit values absent from bounded domains."""
    if profile is None or glossary is None:
        return ValueGroundingResult()
    selected = {
        f"{table}.{column}" for table, columns in context.columns.items() for column in columns
    }
    profiles = {f"{item.table}.{item.column}": item for item in profile.columns}
    normalized_question = _normalize(question)

    for term in glossary.terms:
        phrases = [term.term, *term.synonyms]
        matched_phrases = [phrase for phrase in phrases if _contains(normalized_question, phrase)]
        if not matched_phrases:
            continue
        for qualified in term.columns:
            column_profile = profiles.get(qualified)
            if column_profile is None or (selected and qualified not in selected):
                continue
            values = _domain_values(column_profile.allowed_values, column_profile.top_values)
            if not values or len(values) > 50:
                continue
            comparison_candidate = _comparison_candidate(question, values)
            if comparison_candidate is not None:
                return ValueGroundingResult(
                    issue=ValueGroundingIssue(
                        user_value=comparison_candidate,
                        column=qualified,
                        available_values=values[:20],
                        reason=(
                            f"The explicit value {comparison_candidate!r} is not present in the "
                            f"profiled categorical domain for {qualified}."
                        ),
                    )
                )
            if any(_contains(normalized_question, value) for value in values):
                continue
            alias = _matched_alias(normalized_question, term, values)
            if alias is not None:
                source, canonical = alias
                return ValueGroundingResult(
                    evidence=[
                        f"Approved categorical alias: {source!r} means {canonical!r} "
                        f"for {qualified}."
                    ]
                )
            candidate = _explicit_candidate(question, qualified, matched_phrases)
            if candidate is None:
                continue
            return ValueGroundingResult(
                issue=ValueGroundingIssue(
                    user_value=candidate,
                    column=qualified,
                    available_values=values[:20],
                    reason=(
                        f"The explicit value {candidate!r} is not present in the profiled "
                        f"categorical domain for {qualified}."
                    ),
                )
            )
    return ValueGroundingResult()


def _domain_values(allowed: list[str], top_values: Sequence[ValueFrequency]) -> list[str]:
    values = list(allowed)
    if not values:
        values = [str(item.value) for item in top_values]
    return list(dict.fromkeys(value for value in values if value))


def _matched_alias(
    normalized_question: str, term: GlossaryTerm, values: list[str]
) -> tuple[str, str] | None:
    canonical = {value.casefold(): value for value in values}
    for alias, target in term.value_aliases.items():
        if _contains(normalized_question, alias) and target.casefold() in canonical:
            return alias, canonical[target.casefold()]
    return None


def _explicit_candidate(
    question: str, qualified_column: str, matched_phrases: list[str]
) -> str | None:
    leaf = qualified_column.rsplit(".", 1)[-1]
    triggers = [leaf, *matched_phrases]
    for trigger in sorted(set(triggers), key=len, reverse=True):
        if not (trigger.casefold().endswith(" in") or trigger.casefold() == leaf.casefold()):
            continue
        match = re.search(
            rf"\b{re.escape(trigger)}\b\s*(?:is|=|of)?\s*[\"']?"
            r"(?P<value>[A-Za-z][A-Za-z0-9_-]{1,30})",
            question,
            re.IGNORECASE,
        )
        if match:
            value = match.group("value").strip("\"'")
            if value.casefold() not in {"the", "a", "an", "all", "each"}:
                return value
    return None


def _comparison_candidate(question: str, values: list[str]) -> str | None:
    """Find an unknown code in an explicit between/among categorical list.

    Requiring at least one known domain value in the same list avoids treating
    unrelated uppercase literals elsewhere in the question as column values.
    """
    match = re.search(r"\b(?:between|among)\b(?P<body>[^?;.]+)", question, re.IGNORECASE)
    if match is None:
        return None
    body = match.group("body")
    canonical = {value.casefold() for value in values}
    tokens = re.findall(r"\b[A-Z][A-Z0-9_-]{1,19}\b", body)
    if not any(token.casefold() in canonical for token in tokens):
        return None
    return next((token for token in tokens if token.casefold() not in canonical), None)


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _contains(normalized_question: str, phrase: str) -> bool:
    pattern = rf"(?<!\w){re.escape(_normalize(phrase))}(?!\w)"
    return re.search(pattern, normalized_question) is not None
