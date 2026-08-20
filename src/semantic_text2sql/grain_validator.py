"""Deterministic grain and join-cardinality validation for generated SQL."""

from __future__ import annotations

from typing import Literal

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from semantic_text2sql.models import (
    ColumnProfile,
    DatabaseProfile,
    SemanticContract,
    ValidationResult,
)

_AGGREGATES: dict[str, type[exp.AggFunc]] = {
    "count": exp.Count,
    "sum": exp.Sum,
    "average": exp.Avg,
    "min": exp.Min,
    "max": exp.Max,
}


def validate_grain_cardinality(
    sql: str,
    contract: SemanticContract,
    profile: DatabaseProfile,
    validation: ValidationResult,
    *,
    dialect: Literal["sqlite", "postgres"],
) -> ValidationResult:
    if not validation.valid:
        return validation
    try:
        root = parse_one(sql, read=dialect)
    except ParseError:
        return validation

    incomplete = _incomplete_composite_join(root, profile)
    if incomplete:
        return _failure(
            validation,
            "COMPOSITE_JOIN_INCOMPLETE",
            f"The join between {incomplete} does not use every column of its composite key.",
        )

    if _unsafe_distinct_aggregation(root, profile):
        return _failure(
            validation,
            "DISTINCT_DUPLICATION_REPAIR",
            "SELECT DISTINCT cannot repair duplication around an aggregate over a "
            "ONE_TO_MANY join; fix the input grain before aggregation.",
        )

    fanout = _unsafe_measure_fanout(root, contract, profile)
    if fanout:
        return _failure(
            validation,
            "JOIN_FANOUT_RISK",
            f"Direct join between {fanout} can duplicate the requested measure grain; "
            "filter or preaggregate the many-side relation before combining results.",
        )

    stage_failure = _validate_aggregation_stages(root, contract, profile)
    if stage_failure:
        return _failure(validation, stage_failure[0], stage_failure[1])

    output_failure = _validate_final_grain(root, contract, profile)
    if output_failure:
        return _failure(validation, "OUTPUT_GRAIN_MISMATCH", output_failure)

    ranking_failure = _validate_ranking_grain(root, contract, profile)
    if ranking_failure:
        return _failure(validation, "RANKING_GRAIN_MISMATCH", ranking_failure)

    checks = [*validation.semantic_checks]
    if contract.aggregation_stages:
        checks.append("grain:aggregation_stages")
    if contract.grain:
        checks.append("grain:final_output")
    if profile.relationships:
        checks.append("cardinality:join_keys")
    return validation.model_copy(update={"semantic_checks": list(dict.fromkeys(checks))})


def _incomplete_composite_join(root: exp.Expression, profile: DatabaseProfile) -> str | None:
    for select in root.find_all(exp.Select):
        aliases = _physical_aliases(select)
        present = set(aliases.values())
        equality_pairs = _join_equality_pairs(select, aliases)
        for relationship in profile.relationships:
            parent_columns = relationship.parent_columns or [relationship.parent_column]
            child_columns = relationship.child_columns or [relationship.child_column]
            if len(parent_columns) < 2:
                continue
            parent = relationship.parent_table.casefold()
            child = relationship.child_table.casefold()
            if {parent, child} - present:
                continue
            required = {
                frozenset(((parent, left.casefold()), (child, right.casefold())))
                for left, right in zip(parent_columns, child_columns, strict=True)
            }
            if not required <= equality_pairs:
                return f"{relationship.parent_table} and {relationship.child_table}"
    return None


def _validate_aggregation_stages(
    root: exp.Expression,
    contract: SemanticContract,
    profile: DatabaseProfile,
) -> tuple[str, str] | None:
    known = {item.column.casefold() for item in profile.columns}
    profile_columns = {
        (item.table.casefold(), item.column.casefold()): item for item in profile.columns
    }
    selects = list(root.find_all(exp.Select))
    for stage in contract.aggregation_stages:
        aggregate_type = _AGGREGATES[stage.function]
        input_name = stage.input.rsplit(".", 1)[-1].casefold()
        candidates: list[exp.Select] = []
        for select in selects:
            aggregates = [
                node
                for node in select.find_all(aggregate_type)
                if node.find_ancestor(exp.Select) is select
            ]
            if input_name in known:
                aggregates = [
                    node
                    for node in aggregates
                    if _aggregate_matches_stage_input(
                        node,
                        select,
                        stage.input,
                        profile_columns,
                    )
                ]
            if aggregates:
                candidates.append(select)
        if not candidates:
            return (
                "AGGREGATION_STAGE_MISSING",
                f"Missing {stage.function.upper()} stage {stage.name} over {stage.input}.",
            )
        required_grain = {
            value.rsplit(".", 1)[-1].casefold()
            for value in (stage.output_grain or stage.group_by)
            if value.rsplit(".", 1)[-1].casefold() in known
        }
        if required_grain and not any(
            required_grain <= _group_columns(select) for select in candidates
        ):
            return (
                "AGGREGATION_GRAIN_MISMATCH",
                f"Stage {stage.name} must output grain {sorted(required_grain)}.",
            )
    return None


def _aggregate_matches_stage_input(
    aggregate: exp.AggFunc,
    select: exp.Select,
    stage_input: str,
    profile_columns: dict[tuple[str, str], ColumnProfile],
) -> bool:
    table_name, separator, column_name = stage_input.casefold().rpartition(".")
    if not separator:
        table_name = ""
    if any(column.name.casefold() == column_name for column in aggregate.find_all(exp.Column)):
        return True
    if not isinstance(aggregate, exp.Count) or aggregate.find(exp.Star) is None:
        return False
    present_tables = set(_physical_aliases(select).values())
    candidates = [
        item
        for (table, column), item in profile_columns.items()
        if column == column_name
        and (table == table_name if table_name else table in present_tables)
    ]
    if len(candidates) != 1:
        return False
    candidate = candidates[0]
    candidate_table = candidate.table.casefold()
    non_null_entity_key = candidate.primary_key or candidate.nullable is False
    return non_null_entity_key and candidate_table in present_tables


def _unsafe_distinct_aggregation(root: exp.Expression, profile: DatabaseProfile) -> bool:
    for select in root.find_all(exp.Select):
        if not select.args.get("distinct"):
            continue
        present = set(_physical_aliases(select).values())
        risky_join = any(
            relationship.type == "ONE_TO_MANY"
            and {
                relationship.parent_table.casefold(),
                relationship.child_table.casefold(),
            }
            <= present
            for relationship in profile.relationships
        )
        has_aggregate = any(
            aggregate.find_ancestor(exp.Select) is select
            for aggregate in select.find_all(exp.AggFunc)
        )
        if risky_join and has_aggregate:
            return True
    return False


def _unsafe_measure_fanout(
    root: exp.Expression,
    contract: SemanticContract,
    profile: DatabaseProfile,
) -> str | None:
    measure_tables = {item.split(".", 1)[0].casefold() for item in contract.measures if "." in item}
    filter_tables = {
        item.operand.split(".", 1)[0].casefold() for item in contract.filters if "." in item.operand
    } - measure_tables
    if not measure_tables or not filter_tables:
        return None
    for select in root.find_all(exp.Select):
        present = set(_physical_aliases(select).values())
        projected_tables = {
            _physical_aliases(select).get(column.table.casefold(), column.table.casefold())
            for projection in select.expressions
            for column in projection.find_all(exp.Column)
        }
        for relationship in profile.relationships:
            parent = relationship.parent_table.casefold()
            child = relationship.child_table.casefold()
            if {parent, child} - present:
                continue
            measure_parent = parent in measure_tables and parent in projected_tables
            measure_child = child in measure_tables and child in projected_tables
            risky = relationship.type == "MANY_TO_MANY" or (
                relationship.type == "ONE_TO_MANY" and measure_parent and child in filter_tables
            )
            if risky and (measure_parent or measure_child):
                return f"{relationship.parent_table} and {relationship.child_table}"
    return None


def _validate_final_grain(
    root: exp.Expression,
    contract: SemanticContract,
    profile: DatabaseProfile,
) -> str | None:
    if not contract.requires_grouping or not contract.grain:
        return None
    known = {item.column.casefold() for item in profile.columns}
    required = {
        value.rsplit(".", 1)[-1].casefold()
        for value in contract.grain
        if value.rsplit(".", 1)[-1].casefold() in known
    }
    if not required:
        return None
    outer = root if isinstance(root, exp.Select) else root.find(exp.Select)
    if not isinstance(outer, exp.Select):
        return f"Final query must group by {sorted(required)}."
    if required <= _group_columns(outer):
        return None
    outer_aggregates = any(
        projection.find(exp.AggFunc) is not None for projection in outer.expressions
    )
    nested_grouped = any(
        select is not outer and required <= _group_columns(select)
        for select in root.find_all(exp.Select)
    )
    if outer_aggregates or not nested_grouped:
        return f"Final query must group by {sorted(required)}."
    return None


def _validate_ranking_grain(
    root: exp.Expression,
    contract: SemanticContract,
    profile: DatabaseProfile,
) -> str | None:
    if not contract.selectors:
        return None
    known = {item.column.casefold() for item in profile.columns}
    for selector in contract.selectors:
        required = {
            value.rsplit(".", 1)[-1].casefold()
            for value in selector.partition_by
            if value.rsplit(".", 1)[-1].casefold() in known
        }
        if not required:
            continue
        windows = list(root.find_all(exp.Window))
        if not windows:
            continue
        if not any(required <= _partition_columns(window) for window in windows):
            return f"Selector {selector.name} must rank within {sorted(required)}."
    return None


def _physical_aliases(select: exp.Select) -> dict[str, str]:
    tables = [
        table for table in select.find_all(exp.Table) if table.find_ancestor(exp.Select) is select
    ]
    return {table.alias_or_name.casefold(): table.name.casefold() for table in tables}


def _join_equality_pairs(
    select: exp.Select, aliases: dict[str, str]
) -> set[frozenset[tuple[str, str]]]:
    pairs: set[frozenset[tuple[str, str]]] = set()
    for join in select.args.get("joins") or []:
        condition = join.args.get("on")
        if not isinstance(condition, exp.Expression):
            continue
        for equality in condition.find_all(exp.EQ):
            if not isinstance(equality.this, exp.Column) or not isinstance(
                equality.expression, exp.Column
            ):
                continue
            left = equality.this
            right = equality.expression
            pairs.add(
                frozenset(
                    (
                        (
                            aliases.get(left.table.casefold(), left.table.casefold()),
                            left.name.casefold(),
                        ),
                        (
                            aliases.get(right.table.casefold(), right.table.casefold()),
                            right.name.casefold(),
                        ),
                    )
                )
            )
    return pairs


def _group_columns(select: exp.Select) -> set[str]:
    group = select.args.get("group")
    if not isinstance(group, exp.Group):
        return set()
    return {column.name.casefold() for column in group.find_all(exp.Column)}


def _partition_columns(window: exp.Window) -> set[str]:
    partitions = window.args.get("partition_by") or []
    return {
        column.name.casefold()
        for partition in partitions
        for column in partition.find_all(exp.Column)
    }


def _failure(validation: ValidationResult, code: str, message: str) -> ValidationResult:
    return validation.model_copy(update={"valid": False, "code": code, "message": message})
