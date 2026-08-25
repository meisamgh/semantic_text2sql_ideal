"""Deterministic table-first and column-second schema linking."""

from __future__ import annotations

import re
from collections import deque

from semantic_text2sql.models import (
    ColumnProfile,
    DatabaseProfile,
    SchemaInfo,
    SchemaSelection,
    TableInfo,
    TableProfile,
)

_WORDS = re.compile(r"[A-Za-z0-9]+")


def select_schema(
    question: str,
    evidence: str | None,
    schema: SchemaInfo,
    profile: DatabaseProfile | None = None,
    *,
    max_tables: int = 5,
    max_columns_per_table: int = 5,
    required_columns: list[str] | None = None,
    required_tables: list[str] | None = None,
) -> tuple[SchemaInfo, SchemaSelection]:
    query_tokens = _tokens(f"{question} {evidence or ''}")
    profiles = {(item.table, item.column): item for item in profile.columns} if profile else {}
    table_profiles = {item.table: item for item in profile.tables} if profile else {}
    table_scores = {
        table.name: _table_score(
            table,
            question,
            query_tokens,
            profiles,
            table_profiles.get(table.name),
        )
        for table in schema.tables
    }
    ranked = sorted(schema.tables, key=lambda item: (-table_scores[item.name], item.name))
    top_score = table_scores[ranked[0].name] if ranked else 0.0
    threshold = max(0.5, top_score * 0.2)
    relevant = [item for item in ranked if table_scores[item.name] >= threshold]
    live_names = {item.name for item in schema.tables}
    explicit_tables = {name for name in (required_tables or []) if name in live_names}
    if explicit_tables:
        selected_names = explicit_tables
    else:
        selected_names = {item.name for item in (relevant or ranked[:1])[:max_tables]}
        selected_names.update(_bridge_tables(selected_names, schema))

    selected_tables: list[TableInfo] = []
    column_scores: dict[str, dict[str, float]] = {}
    selected_columns: dict[str, list[str]] = {}
    for table in schema.tables:
        if table.name not in selected_names:
            continue
        semantic_required = {
            name.rsplit(".", 1)[-1]
            for name in (required_columns or [])
            if name.casefold().startswith(table.name.casefold() + ".")
        }
        required = semantic_required or (
            {column.name for column in table.columns if column.primary_key}
            | _foreign_key_columns(table.name, schema)
        )
        scores = {
            column.name: _column_score(
                table.name,
                column.name,
                column.description,
                column.aliases,
                query_tokens,
                profiles.get((table.name, column.name)),
            )
            for column in table.columns
        }
        ranked_columns = sorted(table.columns, key=lambda item: (-scores[item.name], item.name))
        keep = (
            required
            if semantic_required
            else required | {item.name for item in ranked_columns[:max_columns_per_table]}
        )
        columns = [item for item in table.columns if item.name in keep]
        selected_tables.append(table.model_copy(update={"columns": columns, "create_sql": ""}))
        selected_columns[table.name] = [item.name for item in columns]
        column_scores[table.name] = scores

    relationships = [
        item
        for item in schema.relationships
        if item.from_table in selected_names and item.to_table in selected_names
    ]
    reduced = SchemaInfo(
        db_id=schema.db_id,
        dialect=schema.dialect,
        tables=selected_tables,
        relationships=relationships,
    )
    selection = SchemaSelection(
        tables=[item.name for item in selected_tables],
        columns=selected_columns,
        table_scores=table_scores,
        column_scores=column_scores,
    )
    return reduced, selection


def _tokens(value: str) -> set[str]:
    tokens = {token.casefold() for token in _WORDS.findall(value)}
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("ies"):
            expanded.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
    return expanded


def _table_score(
    table: TableInfo,
    question: str,
    query_tokens: set[str],
    profiles: dict[tuple[str, str], ColumnProfile],
    table_profile: TableProfile | None,
) -> float:
    score = 3.0 * len(_tokens(table.name) & query_tokens)
    if table_profile is not None:
        metadata = _tokens(
            f"{table_profile.summary} {' '.join(table_profile.supported_terms)} "
            f"{' '.join(table_profile.metrics)} {' '.join(table_profile.dimensions)}"
        )
        score += 2.5 * len(metadata & query_tokens)
        metric_tokens = _tokens(" ".join(table_profile.metrics))
        score += 5.0 * len(metric_tokens & query_tokens)
        requested_years = {int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", question)}
        exact_coverages = [item for item in table_profile.date_coverage if item.exact]
        if requested_years and exact_coverages:
            covered = any(
                _year(coverage.minimum) is not None
                and _year(coverage.maximum) is not None
                and any(
                    _year(coverage.minimum) <= year <= _year(coverage.maximum)  # type: ignore[operator]
                    for year in requested_years
                )
                for coverage in exact_coverages
            )
            score += 4.0 if covered else -20.0
    for column in table.columns:
        score += _column_score(
            table.name,
            column.name,
            column.description,
            column.aliases,
            query_tokens,
            profiles.get((table.name, column.name)),
        )
    return score


def _year(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.match(r"^((?:19|20)\d{2})", value)
    return int(match.group(1)) if match else None


def _column_score(
    table: str,
    column: str,
    description: str | None,
    aliases: list[str],
    query_tokens: set[str],
    profile: ColumnProfile | None,
) -> float:
    identity = _tokens(f"{table} {column}")
    profile_description = profile.description if profile else ""
    profile_aliases = profile.aliases if profile else []
    semantic = _tokens(
        f"{description or ''} {' '.join(aliases)} "
        f"{profile_description or ''} {' '.join(profile_aliases)}"
    )
    values = _tokens(" ".join(value.value for value in profile.top_values)) if profile else set()
    return (
        2.0 * len(identity & query_tokens)
        + 1.5 * len(semantic & query_tokens)
        + len(values & query_tokens)
    )


def _foreign_key_columns(table: str, schema: SchemaInfo) -> set[str]:
    result: set[str] = set()
    for item in schema.relationships:
        if item.from_table == table:
            result.add(item.from_column)
        if item.to_table == table:
            result.add(item.to_column)
    return result


def _bridge_tables(selected: set[str], schema: SchemaInfo) -> set[str]:
    graph: dict[str, set[str]] = {table.name: set() for table in schema.tables}
    for item in schema.relationships:
        graph.setdefault(item.from_table, set()).add(item.to_table)
        graph.setdefault(item.to_table, set()).add(item.from_table)
    additions: set[str] = set()
    names = sorted(selected)
    for index, start in enumerate(names):
        for target in names[index + 1 :]:
            queue = deque([(start, [start])])
            visited = {start}
            while queue:
                current, path = queue.popleft()
                if current == target:
                    additions.update(path)
                    break
                for neighbor in sorted(graph.get(current, set())):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, [*path, neighbor]))
    return additions
