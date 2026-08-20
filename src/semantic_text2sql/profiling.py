"""Offline, bounded column profiling and JSON profile storage."""

from __future__ import annotations

import csv
import io
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.models import (
    ColumnInfo,
    ColumnProfile,
    DatabaseProfile,
    DateCoverage,
    ForeignKeyInfo,
    RelationshipProfile,
    SuspectedSentinel,
    TableProfile,
    ValueFrequency,
)
from semantic_text2sql.postgres import PostgresRegistry

Database = DatabaseRegistry | PostgresRegistry
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f-]{27,}$", re.I)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ][^ ]+)?$")
_COMPACT_MONTH = re.compile(r"^\d{6}$")
_SLASH_DATE = re.compile(r"^\d{4}/\d{2}/\d{2}$")
_CLOCK_TIME = re.compile(r"^\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$")
_TIMEZONE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


class ProfileStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self._cache: dict[tuple[str, str], tuple[int, DatabaseProfile]] = {}

    def path(self, dialect: str, db_id: str) -> Path:
        return self.root / f"{dialect}__{db_id}.json"

    def load(self, dialect: str, db_id: str) -> DatabaseProfile | None:
        path = self.path(dialect, db_id)
        if not path.is_file():
            return None
        key = (dialect, db_id)
        modified_ns = path.stat().st_mtime_ns
        cached = self._cache.get(key)
        if cached is not None and cached[0] == modified_ns:
            return cached[1]
        profile = DatabaseProfile.model_validate_json(path.read_text())
        self._cache[key] = (modified_ns, profile)
        return profile

    def save(self, profile: DatabaseProfile) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path(profile.dialect, profile.db_id)
        path.write_text(profile.model_dump_json(indent=2) + "\n")
        self._cache[(profile.dialect, profile.db_id)] = (path.stat().st_mtime_ns, profile)
        return path


def profile_database(
    database: Database,
    db_id: str,
    dialect: Literal["sqlite", "postgres"],
    *,
    sample_limit: int = 10_000,
    top_limit: int = 10,
) -> DatabaseProfile:
    schema = database.inspect(db_id)
    relationships = {
        (item.from_table, item.from_column): f"{item.to_table}.{item.to_column}"
        for item in schema.relationships
    }
    profiles: list[ColumnProfile] = []
    for table in schema.tables:
        table_name = _quote(table.name, dialect)
        _, count_rows, _ = database.execute(
            db_id, f"SELECT COUNT(*) AS row_count FROM {table_name}", max_rows=1
        )
        row_count = int(count_rows[0][0])
        for column in table.columns:
            column_name = _quote(column.name, dialect)
            _, rows, truncated = database.execute(
                db_id,
                f"SELECT {column_name} FROM {table_name} LIMIT {sample_limit + 1}",
                max_rows=sample_limit,
            )
            values = [row[0] for row in rows]
            item = _profile_column(
                table.name,
                column,
                values,
                row_count,
                relationships.get((table.name, column.name)),
                top_limit,
                exact=not truncated and row_count <= sample_limit,
            )
            if item.semantic_type not in {
                "identifier",
                "email",
                "personal_name",
                "phone",
                "postal_address",
                "secret",
                "numeric",
                "date",
                "datetime",
            }:
                _, distinct_rows, distinct_truncated = database.execute(
                    db_id,
                    f"SELECT DISTINCT {column_name} FROM {table_name} "
                    f"WHERE {column_name} IS NOT NULL LIMIT 51",
                    max_rows=51,
                )
                distinct_values = sorted(
                    _safe_value(str(row[0])) for row in distinct_rows if not _sensitive(str(row[0]))
                )
                if not distinct_truncated and len(distinct_rows) <= 50:
                    item = item.model_copy(update={"allowed_values": distinct_values})
            if item.semantic_type in {"date", "datetime"}:
                _, bounds, _ = database.execute(
                    db_id,
                    f"SELECT MIN({column_name}), MAX({column_name}) FROM {table_name}",
                    max_rows=1,
                )
                if bounds:
                    item = item.model_copy(
                        update={
                            "minimum": str(bounds[0][0]) if bounds[0][0] is not None else None,
                            "maximum": str(bounds[0][1]) if bounds[0][1] is not None else None,
                            "range_exact": True,
                        }
                    )
            profiles.append(item)
    return DatabaseProfile(
        db_id=db_id,
        dialect=dialect,
        profiled_at=datetime.now(UTC).isoformat(),
        columns=profiles,
        tables=_table_profiles(profiles),
        relationships=_profile_relationships(database, db_id, dialect, schema.relationships),
    )


def _profile_relationships(
    database: Database,
    db_id: str,
    dialect: Literal["sqlite", "postgres"],
    relationships: list[ForeignKeyInfo],
) -> list[RelationshipProfile]:
    grouped: dict[tuple[str, str, str], list[ForeignKeyInfo]] = {}
    for relationship in relationships:
        identity = relationship.constraint_id or (
            f"{relationship.from_table}.{relationship.from_column}->"
            f"{relationship.to_table}.{relationship.to_column}"
        )
        grouped.setdefault((relationship.from_table, relationship.to_table, identity), []).append(
            relationship
        )
    result: list[RelationshipProfile] = []
    seen: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: item.ordinal)
        child_columns = [item.from_column for item in ordered]
        parent_columns = [item.to_column for item in ordered]
        key = (
            ordered[0].to_table,
            tuple(parent_columns),
            ordered[0].from_table,
            tuple(child_columns),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(
            _profile_relationship(
                database,
                db_id,
                dialect,
                ordered[0].to_table,
                parent_columns,
                ordered[0].from_table,
                child_columns,
            )
        )
    return result


def _profile_relationship(
    database: Database,
    db_id: str,
    dialect: Literal["sqlite", "postgres"],
    parent_table: str,
    parent_columns: list[str],
    child_table: str,
    child_columns: list[str],
) -> RelationshipProfile:
    parent_unique = _key_is_unique(
        database,
        db_id,
        dialect,
        parent_table,
        parent_columns,
    )
    child_unique = _key_is_unique(
        database,
        db_id,
        dialect,
        child_table,
        child_columns,
    )
    relationship_type: Literal["ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY"]
    if parent_unique and child_unique:
        relationship_type = "ONE_TO_ONE"
    elif parent_unique:
        relationship_type = "ONE_TO_MANY"
    elif child_unique:
        relationship_type = "MANY_TO_ONE"
    else:
        relationship_type = "MANY_TO_MANY"
    return RelationshipProfile(
        parent_table=parent_table,
        parent_column=parent_columns[0],
        child_table=child_table,
        child_column=child_columns[0],
        parent_columns=parent_columns,
        child_columns=child_columns,
        type=relationship_type,
        parent_key_unique=parent_unique,
        child_key_unique=child_unique,
    )


def _key_is_unique(
    database: Database,
    db_id: str,
    dialect: Literal["sqlite", "postgres"],
    table: str,
    columns: list[str],
) -> bool:
    table_name = _quote(table, dialect)
    column_names = [_quote(column, dialect) for column in columns]
    non_null = " AND ".join(f"{column} IS NOT NULL" for column in column_names)
    group_by = ", ".join(column_names)
    _, rows, _ = database.execute(
        db_id,
        f"SELECT COUNT(*) FROM (SELECT 1 FROM {table_name} "
        f"WHERE {non_null} GROUP BY {group_by}) AS unique_keys",
        max_rows=1,
    )
    distinct_count = int(rows[0][0])
    _, total_rows, _ = database.execute(
        db_id,
        f"SELECT COUNT(*) FROM {table_name} WHERE {non_null}",
        max_rows=1,
    )
    return int(total_rows[0][0]) == distinct_count


def apply_bird_descriptions(
    profile: DatabaseProfile, description_directory: Path
) -> DatabaseProfile:
    """Attach BIRD per-table CSV descriptions without reading labelled SQL."""
    metadata: dict[tuple[str, str], tuple[str | None, list[str], str | None]] = {}
    for path in sorted(description_directory.glob("*.csv")):
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
        for row in csv.DictReader(io.StringIO(text, newline="")):
            column = (row.get("original_column_name") or "").strip()
            if not column:
                continue
            alias = (row.get("column_name") or "").strip()
            description = (row.get("column_description") or "").strip()
            value_description = (row.get("value_description") or "").strip()
            combined = "; ".join(item for item in (description, value_description) if item)
            metadata[(path.stem.casefold(), column.casefold())] = (
                combined or None,
                [alias] if alias and alias.casefold() != column.casefold() else [],
                _bird_semantic_type(row.get("data_format") or ""),
            )
    columns = []
    for item in profile.columns:
        details = metadata.get((item.table.casefold(), item.column.casefold()))
        if details is None:
            columns.append(item)
            continue
        description, aliases, semantic_type = details
        if item.primary_key or item.foreign_key_target or item.semantic_type == "identifier":
            semantic_type = "identifier"
        if item.observed_format in {"YYYYMM", "YYYY/MM/DD", "YYYY-MM-DD"}:
            semantic_type = "date"
        elif item.observed_format == "ISO-8601 datetime":
            semantic_type = "datetime"
        columns.append(
            item.model_copy(
                update={
                    "description": description or item.description,
                    "aliases": aliases or item.aliases,
                    "semantic_type": semantic_type or item.semantic_type,
                }
            )
        )
    return profile.model_copy(update={"columns": columns, "tables": _table_profiles(columns)})


def _table_profiles(columns: list[ColumnProfile]) -> list[TableProfile]:
    grouped: dict[str, list[ColumnProfile]] = {}
    for item in columns:
        grouped.setdefault(item.table, []).append(item)
    result: list[TableProfile] = []
    for table, items in grouped.items():
        keys = [item.column for item in items if item.primary_key]
        metrics = {
            item.column: item.description or item.column
            for item in items
            if item.semantic_type == "numeric" and not item.primary_key
        }
        dimensions = {
            item.column: item.description or item.semantic_type
            for item in items
            if item.column not in metrics
        }
        dates = [
            DateCoverage(
                column=item.column,
                minimum=item.minimum,
                maximum=item.maximum,
                observed_format=item.observed_format,
                exact=item.range_exact,
            )
            for item in items
            if item.semantic_type in {"date", "datetime"}
        ]
        terms = sorted(
            {
                token.casefold()
                for item in items
                for token in re.findall(
                    r"[A-Za-z][A-Za-z0-9_-]+",
                    f"{item.column} {item.description or ''} {' '.join(item.aliases)}",
                )
                if len(token) > 2
            }
        )
        metric_text = ", ".join(metrics) or "no inferred numeric measure"
        summary = (
            f"{table} records at {('key ' + ', '.join(keys)) if keys else 'row'} grain; "
            f"inferred measures: {metric_text}; dimensions: {', '.join(dimensions)}."
        )
        warnings = [
            f"Observed {date.column} coverage is {date.minimum} through {date.maximum} "
            f"({'exact' if date.exact else 'sampled'})."
            for date in dates
            if date.minimum is not None and date.maximum is not None
        ]
        result.append(
            TableProfile(
                table=table,
                summary=summary,
                grain=("one row per " + " + ".join(keys)) if keys else "one database row",
                metrics=metrics,
                dimensions=dimensions,
                date_coverage=dates,
                supported_terms=terms,
                warnings=warnings,
                provenance={
                    "summary": "deterministically inferred",
                    "columns": "live schema and BIRD column descriptions",
                    "date_coverage": "offline observed profile",
                    "reviewed": "false",
                },
            )
        )
    return result


def _profile_column(
    table: str,
    column: ColumnInfo,
    values: list[Any],
    row_count: int,
    foreign_key_target: str | None,
    top_limit: int,
    *,
    exact: bool,
) -> ColumnProfile:
    non_null = [value for value in values if value is not None]
    strings = [str(value) for value in non_null]
    semantic_type = _semantic_type(table, column, non_null)
    observed_format = _observed_format(strings)
    if observed_format in {"YYYYMM", "YYYY/MM/DD", "YYYY-MM-DD"}:
        semantic_type = "date"
    elif observed_format == "ISO-8601 datetime":
        semantic_type = "datetime"
    numeric = [float(value) for value in non_null if isinstance(value, int | float)]
    counts: dict[str, int] = {}
    for value in strings:
        counts[value] = counts.get(value, 0) + 1
    sensitive_column = semantic_type in {
        "email",
        "personal_name",
        "phone",
        "postal_address",
        "secret",
    }
    safe_counts = [
        ValueFrequency(value=_safe_value(value), count=count)
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_limit]
        if not sensitive_column and not _sensitive(value)
    ]
    allowed_values = (
        sorted(_safe_value(value) for value in counts if not _sensitive(value))
        if exact and semantic_type == "categorical" and len(counts) <= 50 and not sensitive_column
        else []
    )
    sentinels = _sentinels(non_null, semantic_type)
    lengths = sorted(len(value) for value in strings)
    null_count = len(values) - len(non_null)
    denominator = len(values) or 1
    return ColumnProfile(
        table=table,
        column=column.name,
        database_type=column.data_type,
        semantic_type=semantic_type,
        nullable=column.nullable,
        primary_key=column.primary_key,
        foreign_key_target=foreign_key_target,
        description=column.description,
        aliases=column.aliases,
        row_count=row_count,
        null_count=null_count,
        null_ratio=null_count / denominator,
        empty_string_count=sum(value == "" for value in strings),
        suspected_sentinels=sentinels,
        distinct_count=len(counts),
        minimum=str(min(non_null)) if non_null else None,
        maximum=str(max(non_null)) if non_null else None,
        range_exact=exact,
        median=median(numeric) if numeric else None,
        quantiles=_quantiles(numeric),
        observed_format=observed_format,
        timezone=_observed_timezone(strings) if semantic_type in {"date", "datetime"} else None,
        top_values=safe_counts if len(counts) <= 100 else [],
        allowed_values=allowed_values,
        text_length_min=lengths[0] if lengths and not numeric else None,
        text_length_median=median(lengths) if lengths and not numeric else None,
        text_length_max=lengths[-1] if lengths and not numeric else None,
        examples=[item.value for item in safe_counts[:3]],
        detected_pattern=_detected_pattern(strings),
        exact=exact,
    )


def _semantic_type(table: str, column: ColumnInfo, values: list[Any]) -> str:
    name = column.name.casefold()
    table_name = table.casefold()
    data_type = column.data_type.casefold()
    if "email" in name:
        return "email"
    person_tables = {"author", "authors", "customer", "customers", "user", "users", "people"}
    if name in {"first_name", "last_name", "full_name"} or (
        name == "name" and table_name in person_tables
    ):
        return "personal_name"
    if "phone" in name or "mobile" in name:
        return "phone"
    if "address" in name or "postcode" in name or "postal" in name:
        return "postal_address"
    if any(term in name for term in ("password", "secret", "token", "api_key")):
        return "secret"
    if column.primary_key or name.endswith("_id") or name.endswith("id"):
        return "identifier"
    if "date" in data_type or "date" in name:
        return "datetime" if "time" in data_type or "time" in name else "date"
    if any(word in data_type for word in ("int", "real", "numeric", "decimal", "float")):
        return "numeric"
    distinct = len({str(value) for value in values})
    if values and distinct <= min(100, max(20, len(values) // 5)):
        return "categorical"
    return "text"


def _bird_semantic_type(data_format: str) -> str | None:
    value = data_format.casefold().strip()
    if any(item in value for item in ("date", "year", "time")):
        return "date"
    if any(item in value for item in ("integer", "number", "float", "decimal")):
        return "numeric"
    if "boolean" in value:
        return "categorical"
    if "text" in value or "string" in value:
        return "text"
    return None


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    points = (("p25", 0.25), ("p50", 0.5), ("p75", 0.75))
    return {name: ordered[math.floor((len(ordered) - 1) * point)] for name, point in points}


def _sentinels(values: list[Any], semantic_type: str) -> list[SuspectedSentinel]:
    text_candidates = {
        "-",
        "missing",
        "unknown",
        "n/a",
        "n.a.",
        "na",
        "none",
        "null",
        "not available",
    }
    candidates = {"-1"} if semantic_type == "numeric" else text_candidates
    found = sorted({str(value) for value in values if str(value).casefold() in candidates})
    return [
        SuspectedSentinel(
            value=value,
            confidence=0.7,
            reason="common missing-value sentinel",
        )
        for value in found
    ]


def _observed_format(values: list[str]) -> str | None:
    sample = values[:20]
    if sample and all(_COMPACT_MONTH.match(value) for value in sample):
        return "YYYYMM"
    if sample and all(_SLASH_DATE.match(value) for value in sample):
        return "YYYY/MM/DD"
    if sample and all(_CLOCK_TIME.match(value) for value in sample):
        return "HH:MM[:SS[.fraction]]"
    if sample and all(_DATE.match(value) for value in sample):
        if any("T" in value or " " in value for value in sample):
            return "ISO-8601 datetime"
        return "YYYY-MM-DD"
    return None


def _observed_timezone(values: list[str]) -> str | None:
    sample = values[:20]
    if not sample:
        return None
    matches = [_TIMEZONE.search(value) for value in sample]
    zones = {match.group(0) for match in matches if match is not None}
    if len(zones) == 1 and all(match is not None for match in matches):
        return next(iter(zones))
    return "not encoded" if any(_DATE.match(value) for value in sample) else None


def _detected_pattern(values: list[str]) -> str | None:
    if values and all(_UUID.match(value) for value in values[:20]):
        return "UUID"
    if values and all(_EMAIL.match(value) for value in values[:20]):
        return "email"
    return None


def _sensitive(value: str) -> bool:
    return bool(_EMAIL.match(value) or _UUID.match(value))


def _safe_value(value: str) -> str:
    return value[:100]


def _quote(identifier: str, dialect: str) -> str:
    del dialect
    return '"' + identifier.replace('"', '""') + '"'
