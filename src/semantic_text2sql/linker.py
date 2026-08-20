"""Deterministic table-first and column-second schema linking."""

from __future__ import annotations

import json
import re
from collections import deque

from semantic_text2sql.models import (
    ColumnProfile,
    DatabaseProfile,
    SchemaInfo,
    SchemaSelection,
    SemanticContract,
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


def compact_profile_context(
    schema: SchemaInfo,
    profile: DatabaseProfile | None,
    question: str = "",
    evidence: str | None = None,
) -> str:
    if profile is None:
        return "No cached value profiles available."
    query = _tokens(f"{question} {evidence or ''}")
    needs = _context_needs(question, evidence) if question or evidence else None
    selected = {(table.name, column.name) for table in schema.tables for column in table.columns}
    lines = _compact_table_lines(
        schema,
        profile,
        include_dates=needs is None or needs["date"],
    )
    for item in profile.columns:
        if (item.table, item.column) not in selected:
            continue
        lines.append(_profile_line(item, detailed=True, needs=needs, query=query))
    return "\n".join(lines) or "No profiles matched the retrieved columns."


def cardinality_plan_context(
    schema: SchemaInfo,
    profile: DatabaseProfile | None,
    contract: SemanticContract,
) -> str:
    """Build a deterministic, compact join plan; this performs no request-time scans."""
    selected = {table.name for table in schema.tables}
    if profile is None:
        relationships = [
            {
                "parent_table": item.to_table,
                "parent_columns": [item.to_column],
                "child_table": item.from_table,
                "child_columns": [item.from_column],
                "type": "UNKNOWN",
            }
            for item in schema.relationships
        ]
    else:
        relationships = [
            item.model_dump()
            for item in profile.relationships
            if {item.parent_table, item.child_table} <= selected
        ]
    return "Join cardinality plan (deterministic, authoritative):\n" + json.dumps(
        {
            "target_grain": contract.grain,
            "measures": contract.measures,
            "relationships": relationships,
        },
        separators=(",", ":"),
    )


def _compact_table_lines(
    schema: SchemaInfo,
    profile: DatabaseProfile,
    *,
    include_dates: bool = True,
) -> list[str]:
    selected = {table.name for table in schema.tables}
    lines: list[str] = []
    for item in profile.tables:
        if item.table not in selected:
            continue
        lines.append(f"TABLE {item.table}: grain={item.grain}")
    return lines


def compact_interpreter_profile_context(
    question: str,
    evidence: str | None,
    profile: DatabaseProfile | None,
    *,
    max_columns: int = 24,
) -> str:
    """Question-relevant level-1 metadata; never dumps every deep distribution."""
    if profile is None:
        return "No cached column metadata is available."
    query = _tokens(f"{question} {evidence or ''}")
    ranked = sorted(
        profile.columns,
        key=lambda item: (
            -len(
                query
                & _tokens(
                    f"{item.table} {item.column} {item.description or ''} "
                    f"{' '.join(item.aliases)} "
                    f"{' '.join(value.value for value in item.top_values[:5])}"
                )
            ),
            -int(bool(item.suspected_sentinels or item.observed_format)),
            item.table,
            item.column,
        ),
    )
    relevant = [
        item
        for item in ranked
        if query
        & _tokens(
            f"{item.table} {item.column} {item.description or ''} "
            f"{' '.join(item.aliases)} {' '.join(value.value for value in item.top_values[:5])}"
        )
    ][:max_columns]
    return "\n".join(_profile_line(item, detailed=False) for item in relevant) or (
        "No question-relevant column metadata was retrieved."
    )


def _profile_line(
    item: ColumnProfile,
    *,
    detailed: bool,
    needs: dict[str, bool] | None = None,
    query: set[str] | None = None,
) -> str:
    include = needs or {key: True for key in ("date", "missing", "stats", "text")}
    query_tokens = query or set()
    facts = [f"storage_type={item.database_type}"]
    if not detailed:
        facts.append(f"semantic_type={item.semantic_type}")
        if item.description:
            facts.append(f"description={item.description[:300]}")
        if item.aliases:
            facts.append(f"aliases={item.aliases[:5]}")
    if include["missing"]:
        facts.extend(
            [
                "sql_null=" + ("observed" if item.null_count else "not_observed"),
                f"empty_strings={item.empty_string_count}",
            ]
        )
    if include["missing"] and item.suspected_sentinels:
        sentinels = [
            {
                "value": value.value,
                "confidence": value.confidence,
                "reason": value.reason,
            }
            for value in item.suspected_sentinels
        ]
        facts.append(f"suspected_missing_representations={sentinels}")
    if include["date"] and item.observed_format:
        facts.append(f"observed_format={item.observed_format}")
        if item.observed_format == "YYYYMM":
            facts.append(
                "safe_operations={year: SUBSTR(column, 1, 4), "
                "month: SUBSTR(column, 5, 2), chronological_sort: column}"
            )
    value_tokens = _tokens(" ".join(value.value for value in item.top_values))
    column_referenced = bool(_tokens(item.column) & query_tokens)
    value_referenced = bool(value_tokens & query_tokens)
    if item.allowed_values:
        facts.append(f"allowed_values={item.allowed_values}")
    if item.top_values and (needs is None or column_referenced or value_referenced):
        values = ", ".join(f"{value.value} ({value.count})" for value in item.top_values[:5])
        facts.append(f"top_values=[{values}]")
    if detailed:
        if include["stats"] and (item.minimum is not None or item.maximum is not None):
            facts.append(f"range={item.minimum}..{item.maximum}")
        if include["text"] and item.examples and not item.top_values:
            facts.append(f"examples={item.examples[:3]}")
        if include["text"] and item.detected_pattern:
            facts.append(f"detected_pattern={item.detected_pattern}")
    return f"{item.table}.{item.column}: " + "; ".join(facts)


def _context_needs(question: str, evidence: str | None) -> dict[str, bool]:
    value = f"{question} {evidence or ''}".casefold()
    tokens = _tokens(value)
    return {
        "date": bool(
            tokens & {"date", "day", "week", "month", "quarter", "year", "annual", "time"}
            or re.search(r"\b(?:19|20)\d{2}\b", value)
        ),
        "missing": bool(tokens & {"missing", "null", "empty", "blank", "unknown", "sentinel"}),
        "stats": bool(
            tokens
            & {
                "average",
                "avg",
                "median",
                "quantile",
                "percentile",
                "distribution",
                "range",
                "minimum",
                "maximum",
                "lowest",
                "highest",
                "least",
                "most",
            }
        ),
        "text": bool(
            tokens
            & {
                "text",
                "pattern",
                "contains",
                "starts",
                "ends",
                "like",
                "similar",
                "fuzzy",
                "match",
            }
        ),
    }


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
