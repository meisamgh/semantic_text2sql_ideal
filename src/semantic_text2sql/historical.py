"""Leakage-safe BM25 and semantic-feature retrieval over validated query history."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from semantic_text2sql.models import (
    ContextRequest,
    HistoricalExample,
    SemanticContract,
    SemanticPlan,
)

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NORMALIZATION = {
    "spent": "spending",
    "spend": "spending",
    "cost": "spending",
    "paid": "payment",
    "paying": "payment",
    "purchased": "purchase",
    "bought": "purchase",
    "highest": "maximum",
    "biggest": "maximum",
    "top": "maximum",
    "lowest": "minimum",
    "least": "minimum",
    "avg": "average",
    "mean": "average",
    "yearly": "annual",
    "monthly": "month",
}
_BUSINESS_CONCEPTS = {
    "spending": {"spending", "payment", "price", "cost"},
    "quantity": {"amount", "quantity", "units", "items"},
    "consumption": {"consumption", "usage"},
    "unit_price": {"unit", "per", "price", "item"},
    "customer": {"customer", "client", "people", "person"},
    "currency": {"currency"},
    "segment": {"segment"},
    "station": {"station"},
}
_OPERATIONS = {
    "count": {"count", "many", "number"},
    "sum": {"sum", "total", "spending"},
    "average": {"average", "mean"},
    "minimum": {"minimum"},
    "maximum": {"maximum"},
    "difference": {"difference", "subtract", "between"},
    "ratio": {"ratio"},
    "group": {"each", "per", "group"},
    "rank": {"rank", "maximum", "minimum"},
}
_COMPLEX_OPERATIONS = {
    "RATIO",
    "PERCENT_OF_TOTAL",
    "PERCENT_CHANGE",
    "DIFFERENCE",
    "ARGMAX",
    "ARGMIN",
}


@dataclass(frozen=True)
class SemanticSignature:
    operations: frozenset[str]
    metrics: frozenset[str]
    tables: frozenset[str]
    grain: frozenset[str]
    temporal: frozenset[str]


def build_semantic_signature(
    question: str,
    request: ContextRequest,
    contract: SemanticContract,
    plan: SemanticPlan | None = None,
) -> SemanticSignature:
    """Build the retrieval key from the verified post-grounding semantic plan."""
    operations = set(plan.operations if plan is not None else _operation_types(question))
    aggregations = plan.aggregations if plan is not None else request.aggregations
    ranking = plan.ranking if plan is not None else request.ranking
    concepts = plan.business_concepts if plan is not None else request.business_concepts
    measures = plan.measures if plan is not None else request.measures
    group_by = plan.group_by if plan is not None else request.group_by
    temporal = plan.temporal_operations if plan is not None else list(_temporal_features(question))
    operations.update(item.function.upper() for item in aggregations)
    operations.update("ARGMAX" if item.direction == "DESC" else "ARGMIN" for item in ranking)
    metrics = {
        *concepts,
        *measures,
        *contract.measures,
        *(item.input for item in aggregations),
        *(item.output for item in aggregations if item.output),
    }
    return SemanticSignature(
        operations=frozenset(operations),
        metrics=frozenset(item.casefold() for item in metrics),
        tables=frozenset(item.casefold() for item in request.tables),
        grain=frozenset(item.casefold() for item in (group_by or contract.grain)),
        temporal=frozenset(item.casefold() for item in temporal),
    )


def history_is_useful(
    signature: SemanticSignature,
    request: ContextRequest,
    plan: SemanticPlan | None = None,
) -> bool:
    """Use examples for semantic complexity, never merely because several tables are used."""
    return bool(
        signature.operations & _COMPLEX_OPERATIONS
        or len(plan.aggregations if plan is not None else request.aggregations) > 1
        or (plan.ranking if plan is not None else request.ranking)
    )


class CrossEncoderReranker(Protocol):
    """Optional local experiment boundary; normal retrieval needs no extra dependency."""

    def rerank(
        self, question: str, candidates: list[HistoricalExample]
    ) -> list[HistoricalExample]: ...


class HistoricalQueryStore:
    def __init__(
        self, path: Path | None = None, reranker: CrossEncoderReranker | None = None
    ) -> None:
        self.records: list[dict[str, str]] = []
        self.reranker = reranker
        if path and path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            records = raw.get("records", raw) if isinstance(raw, dict) else raw
            if not isinstance(records, list):
                raise ValueError("Historical query corpus must be a JSON list.")
            for item in records:
                if not isinstance(item, dict) or item.get("success") is not True:
                    continue
                question = item.get("question")
                sql = item.get("sql") or item.get("SQL")
                db_id = item.get("db_id")
                if all(isinstance(value, str) and value for value in (question, sql, db_id)):
                    self.records.append(
                        {"question": str(question), "sql": str(sql), "db_id": str(db_id)}
                    )

    def search(
        self,
        question: str,
        db_id: str,
        *,
        top_k: int = 3,
        min_score: float = 0.65,
        bm25_pool: int = 20,
        semantic_pool: int = 5,
        candidate_tables: set[str] | None = None,
        signature: SemanticSignature | None = None,
    ) -> list[HistoricalExample]:
        """BM25 and semantic reranking with fail-closed zero-to-two example selection."""
        if top_k == 0:
            return []
        records = [item for item in self.records if item["db_id"] == db_id]
        if not records:
            return []
        query_tokens = _normalized_tokens(question)
        documents = [_normalized_tokens(item["question"]) for item in records]
        bm25_scores = _bm25_scores(query_tokens, documents)
        bm25_ranked = sorted(
            zip(records, bm25_scores, strict=True),
            key=lambda item: (-item[1], item[0]["question"]),
        )[: max(top_k, bm25_pool)]
        max_bm25 = max((item[1] for item in bm25_ranked), default=0.0)
        query_signature = signature or _signature_from_text(question, candidate_tables or set())
        scored: list[HistoricalExample] = []
        for record, bm25_score in bm25_ranked:
            candidate_signature = _signature_from_text(
                record["question"], _sql_tables(record["sql"])
            )
            if not _operation_compatible(query_signature, candidate_signature):
                continue
            if (
                query_signature.metrics
                and _overlap(query_signature.metrics, candidate_signature.metrics) < 0.5
            ):
                continue
            semantic_score = _signature_score(query_signature, candidate_signature)
            normalized_bm25 = bm25_score / max_bm25 if max_bm25 else 0.0
            score = 0.45 * normalized_bm25 + 0.55 * semantic_score
            if score >= min_score:
                scored.append(HistoricalExample(**record, score=score))
        semantic_ranked = sorted(scored, key=lambda item: (-item.score, item.question))[
            : max(top_k, semantic_pool)
        ]
        if self.reranker is not None and semantic_ranked:
            semantic_ranked = self.reranker.rerank(question, semantic_ranked)
        if not semantic_ranked:
            return []
        selected = [semantic_ranked[0]]
        if min(2, top_k) > 1:
            first_signature = _signature_from_text(
                semantic_ranked[0].question, _sql_tables(semantic_ranked[0].sql)
            )
            for candidate in semantic_ranked[1:]:
                candidate_signature = _signature_from_text(
                    candidate.question, _sql_tables(candidate.sql)
                )
                if _complements(query_signature, first_signature, candidate_signature):
                    selected.append(candidate)
                    break
        return selected


def format_examples(examples: list[HistoricalExample]) -> str:
    if not examples:
        return ""
    blocks = [
        f"Historical pattern {index} (retrieval_score={item.score:.3f}):\n"
        f"Question: {item.question}\nSQL pattern: {item.sql}"
        for index, item in enumerate(examples, 1)
    ]
    return (
        "\n\nHistorical examples are non-authoritative patterns. Never copy literals, filters, "
        "or identifiers unless required by the current question and live schema.\n\n"
        + "\n\n".join(blocks)
    )


def _bm25_scores(query: list[str], documents: list[list[str]]) -> list[float]:
    if not documents:
        return []
    document_frequencies = Counter(token for document in documents for token in set(document))
    average_length = sum(len(document) for document in documents) / len(documents)
    scores: list[float] = []
    k1 = 1.5
    b = 0.75
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for token in set(query):
            frequency = frequencies[token]
            if not frequency:
                continue
            document_frequency = document_frequencies[token]
            inverse_frequency = math.log(
                1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * len(document) / max(average_length, 1))
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def _signature_from_text(value: str, tables: set[str]) -> SemanticSignature:
    tokens = set(_normalized_tokens(value))
    return SemanticSignature(
        operations=frozenset(_operation_types(value)),
        metrics=frozenset(
            name for name, vocabulary in _BUSINESS_CONCEPTS.items() if tokens & vocabulary
        ),
        tables=frozenset(item.casefold() for item in tables),
        grain=frozenset(),
        temporal=frozenset(_temporal_features(value)),
    )


def _signature_score(query: SemanticSignature, candidate: SemanticSignature) -> float:
    return (
        0.35 * _overlap(query.metrics, candidate.metrics)
        + 0.30 * _overlap(query.operations, candidate.operations)
        + 0.20 * _overlap(query.tables, candidate.tables)
        + 0.15 * _overlap(query.temporal, candidate.temporal)
    )


def _operation_compatible(query: SemanticSignature, candidate: SemanticSignature) -> bool:
    required = query.operations & _COMPLEX_OPERATIONS
    return not required or required <= candidate.operations


def _complements(
    query: SemanticSignature,
    first: SemanticSignature,
    candidate: SemanticSignature,
) -> bool:
    """Allow a second example only when it adds a relevant table or temporal pattern."""
    if not _operation_compatible(query, candidate):
        return False
    additions = (candidate.tables - first.tables) | (candidate.temporal - first.temporal)
    relevant = query.tables | query.temporal
    return bool(additions & relevant)


def _operation_types(value: str) -> set[str]:
    tokens = set(_normalized_tokens(value))
    operations = {name.upper() for name, vocabulary in _OPERATIONS.items() if tokens & vocabulary}
    lowered = value.casefold()
    if "ratio" in tokens:
        operations.add("RATIO")
        operations.discard("PERCENT_OF_TOTAL")
    if {"percentage", "percent"} & tokens:
        if {"increase", "decrease", "change", "growth"} & tokens:
            operations.add("PERCENT_CHANGE")
        else:
            operations.add("PERCENT_OF_TOTAL")
    if "difference" in tokens:
        operations.add("DIFFERENCE")
    if {"top", "highest", "biggest", "maximum"} & set(_raw_tokens(lowered)):
        operations.add("ARGMAX")
    if {"least", "lowest", "minimum"} & set(_raw_tokens(lowered)):
        operations.add("ARGMIN")
    return operations


def _temporal_features(value: str) -> set[str]:
    tokens = set(_normalized_tokens(value))
    temporal = set(_YEAR.findall(value))
    months = {
        "month",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
    if months & tokens:
        temporal.add("month")
    if "annual" in tokens or "year" in tokens:
        temporal.add("year")
    return temporal


def _overlap(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _normalized_tokens(value: str) -> list[str]:
    return [_NORMALIZATION.get(token, token) for token in _raw_tokens(value)]


def _raw_tokens(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(value)]


def _sql_tables(sql: str) -> set[str]:
    try:
        return {table.name for table in parse_one(sql).find_all(exp.Table)}
    except ParseError:
        return set()
