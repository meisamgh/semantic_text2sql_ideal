"""Dataset-agnostic BM25, dense, and value-aware schema retrieval with RRF."""

from __future__ import annotations

import math
import re
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from semantic_text2sql.glossary import BusinessGlossary, GlossaryTerm
from semantic_text2sql.models import (
    ColumnInfo,
    ColumnProfile,
    DatabaseProfile,
    RetrievalTrace,
    SchemaInfo,
    SchemaSelection,
    TableInfo,
)

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_NUMERIC_INTENT = re.compile(
    r"\b(?:above|below|between|greater|less|minimum|maximum|min|max|range|over|under)\b|[<>]=?",
    re.IGNORECASE,
)


class DenseEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SchemaCandidateReranker(Protocol):
    """Production boundary for an optional candidate-level learning-to-rank model."""

    def score(self, features: list[list[float]]) -> list[float]: ...


RERANKER_FEATURES = (
    "bm25_reciprocal_rank",
    "embedding_reciprocal_rank",
    "value_reciprocal_rank",
    "rrf_score",
    "is_table",
    "is_primary_key",
    "is_relationship_key",
    "historical_max_similarity",
    "historical_weighted_usage",
    "historical_support_count",
    "historical_operation_match",
)


class LightGBMSchemaReranker:
    """Load a versioned LightGBM text model only when production enables it."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Schema reranker model not found: {model_path}")
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - depends on optional production extra
            raise RuntimeError("Install the 'ml' extra to enable the schema reranker.") from exc
        self.model_path = model_path
        self._model = lgb.Booster(model_file=str(model_path))
        if self._model.num_feature() != len(RERANKER_FEATURES):
            raise ValueError(
                f"Reranker expects {self._model.num_feature()} features, "
                f"but runtime supplies {len(RERANKER_FEATURES)}."
            )

    def score(self, features: list[list[float]]) -> list[float]:
        values = self._model.predict(features)
        return [float(value) for value in values]


class FastEmbedEncoder:
    """Lazy local BGE encoder; no vector database or remote embedding API is used."""

    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", *, cache_size: int = 20_000
    ) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self.cache_size = cache_size
        self._cache: dict[str, list[float]] = {}

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            vectors = [list(map(float, vector)) for vector in self._model.embed(missing)]
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[text] = vector
            while len(self._cache) > self.cache_size:
                self._cache.pop(next(iter(self._cache)))
        return [self._cache[text] for text in texts]


class HybridSchemaRetriever:
    def __init__(
        self,
        encoder: DenseEncoder,
        *,
        rrf_k: int = 60,
        max_tables: int = 5,
        max_columns_per_table: int = 5,
        reranker: SchemaCandidateReranker | None = None,
        reranker_pool: int = 30,
    ) -> None:
        self.encoder = encoder
        self.rrf_k = rrf_k
        self.max_tables = max_tables
        self.max_columns_per_table = max_columns_per_table
        self.reranker = reranker
        self.reranker_pool = reranker_pool

    def retrieve(
        self,
        question: str,
        evidence: str | None,
        schema: SchemaInfo,
        profile: DatabaseProfile | None,
        glossary: BusinessGlossary | None,
        historical_evidence: dict[str, dict[str, float]] | None = None,
    ) -> tuple[SchemaInfo, SchemaSelection, RetrievalTrace]:
        query = " ".join(value for value in (question, evidence or "") if value)
        profiles = {(item.table, item.column): item for item in profile.columns} if profile else {}
        table_profiles = {item.table: item for item in profile.tables} if profile else {}
        relevant_terms = _relevant_glossary_terms(query, glossary)
        column_docs: dict[str, str] = {}
        table_docs: dict[str, str] = {}
        for table in schema.tables:
            column_texts: list[str] = []
            for column in table.columns:
                identifier = f"column:{table.name}.{column.name}"
                document = _column_document(
                    table.name,
                    column,
                    profiles.get((table.name, column.name)),
                    relevant_terms,
                )
                column_docs[identifier] = document
                column_texts.append(document)
            profile_table = table_profiles.get(table.name)
            table_docs[f"table:{table.name}"] = " ".join(
                value
                for value in (
                    table.name,
                    profile_table.summary if profile_table else "",
                    profile_table.grain if profile_table else "",
                    " ".join(column_texts),
                )
                if value
            )

        documents = {**table_docs, **column_docs}
        bm25_scores = _bm25(query, documents)
        dense_scores = _dense_scores(query, documents, self.encoder)
        value_scores = _value_scores(query, schema, profiles)
        bm25_ranks = _ranks(bm25_scores, positive_only=True)
        dense_ranks = _ranks(dense_scores, positive_only=False)
        value_ranks = _ranks(value_scores, positive_only=True)
        fused = _rrf((bm25_ranks, dense_ranks, value_ranks), self.rrf_k)

        ml_scores: dict[str, float] = {}
        if self.reranker is not None:
            candidate_pool = sorted(fused, key=lambda key: (-fused[key], key))[: self.reranker_pool]
            features = [
                _reranker_features(
                    identifier,
                    schema,
                    bm25_ranks,
                    dense_ranks,
                    value_ranks,
                    fused,
                    historical_evidence or {},
                )
                for identifier in candidate_pool
            ]
            try:
                predicted = self.reranker.score(features)
                if len(predicted) != len(candidate_pool):
                    raise ValueError("Schema reranker returned the wrong number of scores.")
                ml_scores = dict(zip(candidate_pool, predicted, strict=True))
            except Exception:
                # Retrieval must remain available when an optional ML artifact fails at runtime.
                ml_scores = {}

        table_ids = set(table_docs)
        column_ids = set(column_docs)
        # RRF table recall is the safety baseline; learned ranking is limited to columns.
        ranked_tables = _rank_ids(table_ids, fused)
        ranked_columns = _rank_ids(column_ids, fused, ml_scores)
        leading_table_score = fused.get(ranked_tables[0], 0.0) if ranked_tables else 0.0
        selected_tables = [
            item.removeprefix("table:")
            for item in ranked_tables[: self.max_tables]
            if fused.get(item, 0.0) >= leading_table_score * 0.55
        ]
        for identifier in ranked_columns[: self.max_tables]:
            table_name = identifier.removeprefix("column:").split(".", 1)[0]
            if table_name not in selected_tables and len(selected_tables) < self.max_tables:
                selected_tables.append(table_name)
        if not selected_tables and schema.tables:
            selected_tables = [schema.tables[0].name]

        required_columns = _glossary_columns(relevant_terms, schema)
        for qualified in sorted(required_columns):
            table_name = qualified.split(".", 1)[0]
            if table_name not in selected_tables:
                selected_tables.append(table_name)
        before_bridges = set(selected_tables)
        selected_tables = _expand_bridges(selected_tables, schema)
        bridge_tables = sorted(set(selected_tables) - before_bridges)
        selected_table_set = set(selected_tables)
        selected_columns: dict[str, list[str]] = {}
        reduced_tables: list[TableInfo] = []
        table_map = {table.name: table for table in schema.tables}
        for table_name in selected_tables:
            table = table_map[table_name]
            table_column_ids = [
                item for item in ranked_columns if item.startswith(f"column:{table.name}.")
            ]
            relevant = {
                item.rsplit(".", 1)[-1] for item in table_column_ids[: self.max_columns_per_table]
            }
            required = {
                column.name for column in table.columns if column.primary_key
            } | _relationship_columns(table.name, schema)
            required.update(
                qualified.rsplit(".", 1)[-1]
                for qualified in required_columns
                if qualified.startswith(f"{table.name}.")
            )
            keep = required if table.name in bridge_tables else required | relevant
            columns = [column for column in table.columns if column.name in keep]
            selected_columns[table.name] = [column.name for column in columns]
            reduced_tables.append(table.model_copy(update={"columns": columns, "create_sql": ""}))

        relationships = [
            item
            for item in schema.relationships
            if item.from_table in selected_table_set and item.to_table in selected_table_set
        ]
        reduced = SchemaInfo(
            db_id=schema.db_id,
            dialect=schema.dialect,
            tables=reduced_tables,
            relationships=relationships,
        )
        table_scores = {
            table.name: fused.get(f"table:{table.name}", 0.0) for table in schema.tables
        }
        column_scores = {
            table.name: {
                column.name: fused.get(f"column:{table.name}.{column.name}", 0.0)
                for column in table.columns
            }
            for table in schema.tables
        }
        trace = RetrievalTrace(
            bm25_ranks=bm25_ranks,
            embedding_ranks=dense_ranks,
            value_match_ranks=value_ranks,
            rrf_scores={key: round(value, 8) for key, value in fused.items()},
            ml_reranker_applied=bool(ml_scores),
            ml_reranker_scores={key: round(value, 8) for key, value in ml_scores.items()},
            selected_tables=[table.name for table in reduced_tables],
            selected_columns=selected_columns,
            bridge_tables_added=bridge_tables,
            metadata_supplied=_metadata_inventory(query, reduced, profile),
        )
        return (
            reduced,
            SchemaSelection(
                tables=trace.selected_tables,
                columns=selected_columns,
                table_scores=table_scores,
                column_scores=column_scores,
            ),
            trace,
        )


def metadata_requests(
    question: str, schema: SchemaInfo, profile: DatabaseProfile | None
) -> list[tuple[str, str]]:
    """Choose metadata types deterministically after retrieval, never during ranking."""
    profiles = {(item.table, item.column): item for item in profile.columns} if profile else {}
    numeric_requested = bool(_NUMERIC_INTENT.search(question))
    result: list[tuple[str, str]] = []
    for table in schema.tables:
        for column in table.columns:
            qualified = f"{table.name}.{column.name}"
            item = profiles.get((table.name, column.name))
            semantic_type = item.semantic_type if item else (column.semantic_type or "")
            if semantic_type in {"date", "datetime"}:
                result.append(("DATE_FORMAT", qualified))
            elif semantic_type in {"categorical", "text"}:
                result.append(("CATEGORY_VALUES", qualified))
            elif semantic_type in {"integer", "numeric", "number", "float"} and numeric_requested:
                result.append(("NUMERIC_PROFILE", qualified))
    return result


def _column_document(
    table: str,
    column: ColumnInfo,
    profile: ColumnProfile | None,
    terms: list[GlossaryTerm],
) -> str:
    glossary = [
        f"{term.term} {' '.join(term.synonyms)} {term.definition}"
        for term in terms
        if f"{table}.{column.name}" in term.columns
    ]
    values: list[str] = []
    if profile:
        values.extend(item.value for item in profile.top_values[:5])
        values.extend(profile.allowed_values[:5])
    return " ".join(
        value
        for value in (
            column.name,
            column.description or "",
            " ".join(column.aliases),
            profile.description if profile and profile.description else "",
            " ".join(profile.aliases) if profile else "",
            profile.semantic_type if profile else (column.semantic_type or ""),
            " ".join(values),
            " ".join(glossary),
        )
        if value
    )


def _relevant_glossary_terms(query: str, glossary: BusinessGlossary | None) -> list[GlossaryTerm]:
    if glossary is None:
        return []
    tokens = set(_tokens(query))
    return [
        term
        for term in glossary.terms
        if tokens & set(_tokens(f"{term.term} {' '.join(term.synonyms)}"))
    ]


def _glossary_columns(terms: Iterable[GlossaryTerm], schema: SchemaInfo) -> set[str]:
    live = {f"{table.name}.{column.name}" for table in schema.tables for column in table.columns}
    return {column for term in terms for column in term.columns if column in live}


def _bm25(
    query: str, documents: dict[str, str], *, k1: float = 1.5, b: float = 0.75
) -> dict[str, float]:
    query_tokens = _tokens(query)
    tokenized = {key: _tokens(value) for key, value in documents.items()}
    count = len(tokenized)
    average_length = sum(map(len, tokenized.values())) / count if count else 1.0
    document_frequency = Counter(token for tokens in tokenized.values() for token in set(tokens))
    scores: dict[str, float] = {}
    for key, tokens in tokenized.items():
        frequencies = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse = math.log(
                1 + (count - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * len(tokens) / max(average_length, 1))
            score += inverse * frequency * (k1 + 1) / denominator
        scores[key] = score
    return scores


def _dense_scores(query: str, documents: dict[str, str], encoder: DenseEncoder) -> dict[str, float]:
    keys = list(documents)
    vectors = encoder.encode([f"query: {query}", *[f"passage: {documents[key]}" for key in keys]])
    query_vector = vectors[0]
    return {
        key: _cosine(query_vector, vector) for key, vector in zip(keys, vectors[1:], strict=True)
    }


def _value_scores(
    query: str,
    schema: SchemaInfo,
    profiles: dict[tuple[str, str], ColumnProfile],
) -> dict[str, float]:
    normalized = query.casefold()
    result: dict[str, float] = {}
    for table in schema.tables:
        table_score = 0.0
        for column in table.columns:
            profile = profiles.get((table.name, column.name))
            if profile is None:
                continue
            values = [item.value for item in profile.top_values]
            values.extend(profile.allowed_values)
            values.extend(profile.examples)
            matches = {
                value.casefold()
                for value in values
                if value and re.search(rf"(?<!\w){re.escape(value.casefold())}(?!\w)", normalized)
            }
            if matches:
                score = float(len(matches))
                result[f"column:{table.name}.{column.name}"] = score
                table_score += score
        if table_score:
            result[f"table:{table.name}"] = table_score
    return result


def _ranks(scores: dict[str, float], *, positive_only: bool) -> dict[str, int]:
    ordered = [(key, score) for key, score in scores.items() if not positive_only or score > 0]
    ordered.sort(key=lambda item: (-item[1], item[0]))
    return {key: rank for rank, (key, _) in enumerate(ordered, 1)}


def _rrf(rankings: Iterable[dict[str, int]], k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for key, rank in ranking.items():
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def _rank_ids(
    identifiers: set[str], scores: dict[str, float], ml_scores: dict[str, float] | None = None
) -> list[str]:
    learned = ml_scores or {}
    return sorted(
        identifiers,
        key=lambda key: (
            0 if key in learned else 1,
            -learned.get(key, 0.0),
            -scores.get(key, 0.0),
            key,
        ),
    )


def _reranker_features(
    identifier: str,
    schema: SchemaInfo,
    bm25_ranks: dict[str, int],
    embedding_ranks: dict[str, int],
    value_ranks: dict[str, int],
    rrf_scores: dict[str, float],
    historical_evidence: dict[str, dict[str, float]] | None = None,
) -> list[float]:
    primary_key = False
    relationship_key = False
    if identifier.startswith("column:"):
        qualified = identifier.removeprefix("column:")
        table_name, column_name = qualified.split(".", 1)
        table = next((item for item in schema.tables if item.name == table_name), None)
        column = (
            next((item for item in table.columns if item.name == column_name), None)
            if table
            else None
        )
        primary_key = bool(column and column.primary_key)
        relationship_key = column_name in _relationship_columns(table_name, schema)

    def reciprocal(ranking: dict[str, int]) -> float:
        rank = ranking.get(identifier)
        return 1.0 / rank if rank else 0.0

    history = (historical_evidence or {}).get(identifier, {})
    return [
        reciprocal(bm25_ranks),
        reciprocal(embedding_ranks),
        reciprocal(value_ranks),
        rrf_scores.get(identifier, 0.0),
        float(identifier.startswith("table:")),
        float(primary_key),
        float(relationship_key),
        history.get("max_similarity", 0.0),
        history.get("weighted_usage", 0.0),
        min(history.get("support_count", 0.0) / 3.0, 1.0),
        history.get("operation_match", 0.0),
    ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return numerator / denominator if denominator else 0.0


def _tokens(value: str) -> list[str]:
    tokens = [token.casefold() for token in _TOKEN.findall(value)]
    expanded = list(tokens)
    for token in tokens:
        if token.endswith("ies"):
            expanded.append(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3:
            expanded.append(token[:-1])
    return expanded


def _relationship_columns(table: str, schema: SchemaInfo) -> set[str]:
    result: set[str] = set()
    for relationship in schema.relationships:
        if relationship.from_table == table:
            result.add(relationship.from_column)
        if relationship.to_table == table:
            result.add(relationship.to_column)
    return result


def _expand_bridges(selected: list[str], schema: SchemaInfo) -> list[str]:
    graph: dict[str, set[str]] = {table.name: set() for table in schema.tables}
    for relationship in schema.relationships:
        graph[relationship.from_table].add(relationship.to_table)
        graph[relationship.to_table].add(relationship.from_table)
    expanded = list(dict.fromkeys(selected))
    for index, start in enumerate(selected):
        for target in selected[index + 1 :]:
            queue = deque([(start, [start])])
            visited = {start}
            while queue:
                current, path = queue.popleft()
                if current == target:
                    for table in path:
                        if table not in expanded:
                            expanded.append(table)
                    break
                for neighbor in sorted(graph.get(current, set())):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, [*path, neighbor]))
    return expanded


def _metadata_inventory(
    question: str, schema: SchemaInfo, profile: DatabaseProfile | None
) -> list[str]:
    always = [
        f"{table.name}.{column.name}:type,nulls,key_role"
        for table in schema.tables
        for column in table.columns
    ]
    conditional = [
        f"{kind}:{target}" for kind, target in metadata_requests(question, schema, profile)
    ]
    return [*always, *conditional]
