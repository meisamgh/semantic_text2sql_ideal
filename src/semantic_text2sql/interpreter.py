"""Schema-grounded Claude interpretation and deterministic contract reconciliation."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, cast

from pydantic import ValidationError

from semantic_text2sql.models import (
    PreliminarySemanticIR,
    SchemaInfo,
    SemanticContract,
    TokenUsage,
)

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.I | re.S)


class Completer(Protocol):
    async def complete(self, model: str, prompt: str) -> str: ...


def compact_schema_memory(schema: SchemaInfo) -> str:
    """Level-1 memory: identities and relationships, without full value profiles."""
    tables = []
    for table in schema.tables:
        columns = ", ".join(
            column.name
            + (" PK" if column.primary_key else "")
            + (f" [{column.semantic_type}]" if column.semantic_type else "")
            for column in table.columns
        )
        tables.append(f"{table.name}: {columns}")
    relationships = [
        f"{item.from_table}.{item.from_column} -> {item.to_table}.{item.to_column}"
        for item in schema.relationships
    ]
    return (
        "Tables:\n"
        + "\n".join(tables)
        + "\nRelationships:\n"
        + ("\n".join(relationships) or "None")
    )


async def interpret_question(
    completer: Completer,
    model: str,
    question: str,
    evidence: str | None,
    schema: SchemaInfo,
    business_context: str = "No curated business glossary is available.",
) -> PreliminarySemanticIR:
    interpreted, _ = await interpret_question_detailed(
        completer,
        model,
        question,
        evidence,
        schema,
        business_context,
    )
    return interpreted


async def interpret_question_detailed(
    completer: Completer,
    model: str,
    question: str,
    evidence: str | None,
    schema: SchemaInfo,
    business_context: str = "No curated business glossary is available.",
) -> tuple[PreliminarySemanticIR, TokenUsage]:
    prompt = f"""Interpret a text-to-SQL question using the compact live schema memory below.
Return exactly one JSON object matching this shape:
{{"outputs":[],"entities":[],"tables":[],"required_columns":[],"joins":[],
"filters":[],"aggregation":null,"grain":[],"order_by":[],"limit":null,
"ambiguities":[],"confidence":0.0}}

Rules:
- Use only exact table and table.column identifiers present in schema memory.
- Describe filters and joins compactly; do not generate SQL.
- Put unresolved business meaning in ambiguities instead of guessing.
- aggregation is one of count, sum, average, min, max, or null.
- confidence is between 0 and 1.

Question: {question}
Trusted evidence: {evidence or "None"}

Retrieved business glossary (advisory; question and trusted evidence take precedence):
{business_context}

Compact live schema memory:
{compact_schema_memory(schema)}
"""
    detailed = getattr(completer, "complete_detailed", None)
    if callable(detailed):
        raw, usage = await cast(Any, detailed)(model, prompt)
    else:
        raw = await completer.complete(model, prompt)
        usage = TokenUsage()
    raw = raw.strip()
    match = _FENCE.fullmatch(raw)
    value = match.group(1) if match else raw
    try:
        return PreliminarySemanticIR.model_validate(json.loads(value)), usage
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("Claude interpreter returned invalid Semantic IR.") from exc


def reconcile_contract(
    deterministic: SemanticContract,
    interpreted: PreliminarySemanticIR,
    schema: SchemaInfo,
) -> SemanticContract:
    """Keep grounded Claude proposals; deterministic high-confidence rules win conflicts."""
    tables = {table.name: table for table in schema.tables}
    proposed_tables = list(dict.fromkeys(name for name in interpreted.tables if name in tables))
    available_columns = {
        f"{table.name}.{column.name}" for table in schema.tables for column in table.columns
    }
    required_columns = list(
        dict.fromkeys(name for name in interpreted.required_columns if name in available_columns)
    )
    trusted_interpretation = interpreted.confidence >= 0.8
    aggregation = deterministic.aggregation or (
        interpreted.aggregation if trusted_interpretation else None
    )
    limit = deterministic.limit or (interpreted.limit if trusted_interpretation else None)
    rationale = [*deterministic.rationale]
    if interpreted.confidence >= 0.5:
        rationale.append("Schema-grounded Claude interpretation contributed advisory scope.")
    if deterministic.aggregation and interpreted.aggregation not in {
        None,
        deterministic.aggregation,
    }:
        rationale.append("Aggregation conflict resolved in favor of explicit deterministic text.")
    return deterministic.model_copy(
        update={
            "aggregation": aggregation,
            "requires_grouping": deterministic.requires_grouping
            or (trusted_interpretation and aggregation is not None and bool(interpreted.grain)),
            "requires_ordering": deterministic.requires_ordering
            or (trusted_interpretation and bool(interpreted.order_by)),
            "limit": limit,
            "outputs": deterministic.outputs
            or (interpreted.outputs if interpreted.confidence >= 0.5 else []),
            "advisory_outputs": [],
            "entities": interpreted.entities,
            "proposed_tables": proposed_tables,
            "required_columns": required_columns,
            "proposed_joins": interpreted.joins,
            "proposed_filters": interpreted.filters,
            "grain": interpreted.grain,
            "ambiguities": interpreted.ambiguities,
            "interpreter_confidence": interpreted.confidence,
            "rationale": rationale,
        }
    )
