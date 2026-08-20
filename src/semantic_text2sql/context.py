"""Deterministic context planning and bounded failure-triggered expansion."""

from __future__ import annotations

import json
from typing import Any, Literal, cast

from semantic_text2sql.models import (
    ContextMetadata,
    ContextPlan,
    ContextRelationship,
    ContextRequest,
    ContextRequirement,
    ContextTable,
    DatabaseProfile,
    HistoricalExample,
    SchemaInfo,
    SemanticContract,
    SemanticPlan,
    VerifiedColumn,
    VerifiedContext,
    VerifiedRelationship,
    VerifiedTable,
)

MAX_CONTEXT_EXPANSIONS = 2

_FAILURE_CATEGORIES = {
    "PROFILE_DATE_FORMAT_MISMATCH": "DATE_FORMAT_UNKNOWN",
    "JOIN_CARDINALITY_RISK": "CARDINALITY_UNKNOWN",
    "GRAIN_CARDINALITY_RISK": "CARDINALITY_UNKNOWN",
    "UNKNOWN_COLUMN": "COLUMN_SEMANTICS_UNKNOWN",
    "UNKNOWN_TABLE": "JOIN_PATH_UNKNOWN",
    "SEMANTIC_FILTER_MISSING": "UNKNOWN_CATEGORY",
    "SEMANTIC_METRIC_LINEAGE_MISMATCH": "FORMULA_UNKNOWN",
}


def build_context_plan(
    schema: SchemaInfo,
    profile: DatabaseProfile | None,
    contract: SemanticContract,
    question: str,
    evidence: str | None,
    context_request: ContextRequest | None = None,
) -> ContextPlan:
    """Resolve only dependencies already approved by schema linking and the contract."""
    measures = {item.casefold() for item in contract.measures}
    required = {item.casefold() for item in contract.required_columns}
    join_tables = {item.from_table for item in schema.relationships} | {
        item.to_table for item in schema.relationships
    }
    tables: list[ContextTable] = []
    table_profiles = {item.table: item for item in profile.tables} if profile else {}
    profiled_relationships = profile.relationships if profile else []
    reasons: dict[str, str] = {}
    columns: dict[str, list[str]] = {}
    for table in schema.tables:
        primary_key = [column.name for column in table.columns if column.primary_key]
        profiled_grain = table_profiles[table.name].grain if table.name in table_profiles else None
        derived_grain = "one row per " + " + ".join(primary_key) if primary_key else None
        qualified = {f"{table.name}.{column.name}".casefold() for column in table.columns}
        role: Literal["MEASURE", "FILTER", "DIMENSION", "BRIDGE", "UNKNOWN"]
        if qualified & measures:
            role = "MEASURE"
            reason = "Provides a requested measure."
        elif qualified & required:
            role = "FILTER"
            reason = "Provides a required output, filter, grouping, or ranking dependency."
        elif table.name in join_tables:
            role = "BRIDGE"
            reason = "Retained for the verified join path."
        else:
            role = "UNKNOWN"
            reason = "Selected by deterministic schema relevance."
        unique_keys = [primary_key] if primary_key else []
        for relationship in profiled_relationships:
            if relationship.parent_table == table.name and relationship.parent_key_unique:
                key = relationship.parent_columns or [relationship.parent_column]
                if key not in unique_keys:
                    unique_keys.append(key)
            if relationship.child_table == table.name and relationship.child_key_unique:
                key = relationship.child_columns or [relationship.child_column]
                if key not in unique_keys:
                    unique_keys.append(key)
        tables.append(
            ContextTable(
                name=table.name,
                role=role,
                grain=profiled_grain or derived_grain,
                primary_key=primary_key,
                unique_keys=unique_keys,
                reason=reason,
            )
        )
        columns[table.name] = [column.name for column in table.columns]
        for column in table.columns:
            reasons[f"{table.name}.{column.name}"] = "Selected schema dependency."

    date_profiles: list[str] = []
    if profile:
        selected = {(table, column) for table, names in columns.items() for column in names}
        date_profiles = [
            f"{item.table}.{item.column}"
            for item in profile.columns
            if (item.table, item.column) in selected and item.observed_format
        ]
    formulas = [item.id for item in contract.structural_formulas]
    formulas.extend(item.name for item in contract.formula_metrics)
    formulas.extend(item.name for item in contract.derived_metrics)
    relationships = _relationships(schema, profile, tables)
    requirements = _requirements(schema, profile, contract, relationships)
    if context_request is not None:
        requirements.extend(_planner_requirements(context_request, profile, relationships))
    missing = [
        f"{item.kind}:{item.target}"
        for item in requirements
        if item.priority == "HARD" and not item.resolved
    ]
    return ContextPlan(
        tables=tables,
        columns=columns,
        relationships=relationships,
        metadata=ContextMetadata(
            formulas=list(dict.fromkeys(formulas)),
            date_profiles=date_profiles,
            relationships=bool(relationships),
            cardinality=any(item.state != "UNRESOLVED" for item in relationships),
        ),
        excluded=["quantiles", "timezone", "text_length_statistics", "full_distributions"],
        inclusion_reasons=reasons,
        requirements=requirements,
        coverage_complete=not missing,
        missing_requirements=missing,
        token_budget=_token_budget(question, contract),
    )


def render_context(
    plan: ContextPlan,
    schema: SchemaInfo,
    profile: DatabaseProfile | None,
    contract: SemanticContract,
    question: str,
    evidence: str | None,
    business_context: str | None,
    historical_examples: list[HistoricalExample] | None = None,
    semantic_plan: SemanticPlan | None = None,
) -> str:
    return json.dumps(
        model_context_payload(
            plan,
            profile,
            contract,
            business_context,
            question,
            schema.dialect,
            historical_examples,
            semantic_plan,
        ),
        separators=(",", ":"),
    )


def model_context_payload(
    plan: ContextPlan,
    profile: DatabaseProfile | None,
    contract: SemanticContract,
    business_context: str | None,
    question: str,
    dialect: str,
    historical_examples: list[HistoricalExample] | None = None,
    semantic_plan: SemanticPlan | None = None,
) -> dict[str, Any]:
    """Return the single authoritative JSON object supplied to SQL generation."""
    verified = _verified_context(
        plan,
        profile,
        question,
        dialect,
        business_context,
    )
    payload: dict[str, Any] = verified.model_dump(mode="json", exclude_none=True)
    formulas = [
        item.model_dump(mode="json", exclude={"source", "usage"})
        for item in contract.structural_formulas
    ]
    if formulas:
        payload["approved_formulas"] = formulas
    if semantic_plan is not None:
        payload["verified_semantic_plan"] = semantic_plan.model_dump(mode="json", exclude_none=True)
    if historical_examples:
        payload["historical_examples"] = [
            {
                "question": item.question,
                "sql": item.sql,
                "similarity": round(item.score, 4),
                "authority": "advisory_pattern_only",
            }
            for item in historical_examples[:2]
        ]
    return cast(dict[str, Any], _sparse(payload))


def _verified_context(
    plan: ContextPlan,
    profile: DatabaseProfile | None,
    question: str,
    dialect: str,
    business_context: str | None,
) -> VerifiedContext:
    column_profiles = (
        {(item.table, item.column): item for item in profile.columns} if profile else {}
    )
    relationship_keys: dict[str, list[str]] = {}
    date_format_targets = {
        item.target
        for item in plan.requirements
        if item.kind == "PHYSICAL_DATE_FORMAT" and item.resolved
    }
    for relationship in plan.relationships:
        relationship_keys.setdefault(relationship.left_table, []).append(relationship.left_column)
        relationship_keys.setdefault(relationship.right_table, []).append(relationship.right_column)
    return VerifiedContext(
        question=question,
        dialect=cast(Any, dialect),
        tables={
            item.name: VerifiedTable(
                grain=item.grain,
                primary_key=item.primary_key,
                relationship_keys=list(dict.fromkeys(relationship_keys.get(item.name, []))),
                unique_keys=item.unique_keys,
                columns={
                    column: _execution_column(
                        column_profiles.get((item.name, column)),
                        include_format=f"{item.name}.{column}" in date_format_targets,
                    )
                    for column in plan.columns.get(item.name, [])
                },
            )
            for item in plan.tables
        },
        relationships=[
            VerifiedRelationship(
                left=f"{item.left_table}.{item.left_column}",
                right=f"{item.right_table}.{item.right_column}",
                state=item.state,
                cardinality=item.join_cardinality,
                left_key_unique=item.left_key_unique,
                right_key_unique=item.right_key_unique,
                fanout_risk=item.fanout_risk,
            )
            for item in plan.relationships
        ],
        business_context=business_context,
        optional_metadata=_dependency_profile_context(plan, profile),
    )


def _execution_column(profile: Any | None, *, include_format: bool) -> VerifiedColumn:
    if profile is None:
        return VerifiedColumn(type="UNKNOWN")
    example_values: list[str] = []
    if profile.semantic_type in {"categorical", "text"}:
        observed = [item.value for item in profile.top_values]
        observed.extend(profile.examples)
        example_values = [str(value) for value in dict.fromkeys(observed)][:5]
    return VerifiedColumn(
        type=profile.database_type,
        observed_nulls=bool(profile.null_count),
        example_values=example_values,
        format=profile.observed_format if include_format else None,
    )


def failure_context_category(code: str) -> str | None:
    return _FAILURE_CATEGORIES.get(code)


def reactive_context(category: str, schema: SchemaInfo, profile: DatabaseProfile | None) -> str:
    """Return one focused expansion; callers prevent duplicate category retrieval."""
    if category in {"CARDINALITY_UNKNOWN", "JOIN_PATH_UNKNOWN"}:
        relationships = [item.model_dump_json() for item in schema.relationships]
        if profile:
            relationships.extend(item.model_dump_json() for item in profile.relationships)
        return "Reactive relationship facts:\n" + ("\n".join(relationships) or "None")
    if profile is None:
        return f"Reactive context {category}: no cached profile is available."
    selected = {(table.name, column.name) for table in schema.tables for column in table.columns}
    facts: list[str] = []
    for item in profile.columns:
        if (item.table, item.column) not in selected:
            continue
        prefix = f"{item.table}.{item.column}"
        if category == "DATE_FORMAT_UNKNOWN" and item.observed_format:
            facts.append(
                f"{prefix}: format={item.observed_format}; min={item.minimum}; max={item.maximum}"
            )
        elif category == "UNKNOWN_CATEGORY" and item.top_values:
            facts.append(f"{prefix}: top_values={[value.value for value in item.top_values[:5]]}")
        elif category == "FORMULA_UNKNOWN" and item.description:
            facts.append(f"{prefix}: {item.description}")
        elif category == "COLUMN_SEMANTICS_UNKNOWN":
            facts.append(f"{prefix}: storage_type={item.database_type}")
    return f"Reactive context {category}:\n" + ("\n".join(facts) or "No matching facts.")


def record_context_expansion(plan: ContextPlan, category: str) -> ContextPlan:
    """Reflect a reactive retrieval in the typed plan returned to clients."""
    updates: dict[str, object] = {}
    if category in {"CARDINALITY_UNKNOWN", "JOIN_PATH_UNKNOWN"}:
        updates.update(relationships=True, cardinality=True)
    elif category == "UNKNOWN_CATEGORY":
        updates["categorical_values"] = True
    elif category == "FORMULA_UNKNOWN":
        updates["numeric_profile"] = True
    elif category == "COLUMN_SEMANTICS_UNKNOWN":
        updates["text_profile"] = True
    reasons = {
        **plan.inclusion_reasons,
        f"reactive:{category}": "Retrieved after a deterministic validation failure.",
    }
    return plan.model_copy(
        update={
            "metadata": plan.metadata.model_copy(update=updates),
            "inclusion_reasons": reasons,
        }
    )


def estimate_tokens(value: str) -> int:
    return max(1, (len(value) + 3) // 4)


def _requirements(
    schema: SchemaInfo,
    profile: DatabaseProfile | None,
    contract: SemanticContract,
    relationships: list[ContextRelationship],
) -> list[ContextRequirement]:
    requirements: list[ContextRequirement] = []
    selected_columns = {
        f"{table.name}.{column.name}": column for table in schema.tables for column in table.columns
    }
    profiles = {f"{item.table}.{item.column}": item for item in profile.columns} if profile else {}
    for name in selected_columns:
        requirements.append(
            ContextRequirement(
                kind="COLUMN_SCHEMA",
                target=name,
                required_by="selected schema dependency",
                resolved=True,
                source="live schema",
            )
        )
    for formula in contract.structural_formulas:
        requirements.append(
            ContextRequirement(
                kind="FORMULA_DEFINITION",
                target=formula.id,
                required_by=formula.usage.casefold(),
                resolved=True,
                source=formula.source,
            )
        )
        for argument in formula.arguments:
            requirements.append(
                ContextRequirement(
                    kind="PHYSICAL_TYPE",
                    target=argument,
                    required_by=formula.id,
                    resolved=argument in selected_columns,
                    source="live schema" if argument in selected_columns else None,
                )
            )
    for name, item in profiles.items():
        if name in selected_columns and item.semantic_type in {"date", "datetime"}:
            requirements.append(
                ContextRequirement(
                    kind="PHYSICAL_DATE_FORMAT",
                    target=name,
                    required_by="selected date or datetime column",
                    resolved=bool(item.observed_format),
                    source="offline profile" if item.observed_format else None,
                )
            )
    if len(schema.tables) > 1:
        has_join_path = any(item.state != "UNRESOLVED" for item in relationships)
        has_cardinality = has_join_path
        requirements.extend(
            [
                ContextRequirement(
                    kind="JOIN_PATH",
                    target=",".join(table.name for table in schema.tables),
                    required_by="multi-table query",
                    resolved=has_join_path,
                    source="live FK graph" if has_join_path else None,
                ),
                ContextRequirement(
                    kind="JOIN_CARDINALITY",
                    target=",".join(table.name for table in schema.tables),
                    required_by="multi-table query",
                    resolved=has_cardinality,
                    source="offline relationship profile" if has_cardinality else None,
                ),
            ]
        )
        table_profiles = {item.table: item for item in profile.tables} if profile else {}
        for table in schema.tables:
            requirements.append(
                ContextRequirement(
                    kind="TABLE_GRAIN",
                    target=table.name,
                    required_by="multi-table grain/cardinality plan",
                    resolved=table.name in table_profiles
                    and bool(table_profiles[table.name].grain),
                    source="offline table profile" if table.name in table_profiles else None,
                )
            )
    unique = {(item.kind, item.target, item.required_by): item for item in requirements}
    return list(unique.values())


def _planner_requirements(
    request: ContextRequest,
    profile: DatabaseProfile | None,
    relationships: list[ContextRelationship],
) -> list[ContextRequirement]:
    profile_columns = (
        {f"{item.table}.{item.column}": item for item in profile.columns} if profile else {}
    )
    table_profiles = {item.table for item in profile.tables} if profile else set()
    mapped = {
        "RELATIONSHIP": "JOIN_PATH",
        "JOIN_CARDINALITY": "JOIN_CARDINALITY",
        "TABLE_GRAIN": "TABLE_GRAIN",
        "DATE_FORMAT": "PHYSICAL_DATE_FORMAT",
        "CATEGORY_VALUES": "ALLOWED_VALUES",
        "FORMULA_DEFINITION": "FORMULA_DEFINITION",
        "PHYSICAL_TYPE": "PHYSICAL_TYPE",
        "NULL_SEMANTICS": "NULL_REPRESENTATION",
        "TEXT_EXAMPLES": "TEXT_EXAMPLES",
        "TEXT_PATTERN": "TEXT_PATTERN",
        "NUMERIC_PROFILE": "NUMERIC_PROFILE",
        "COLUMN_DESCRIPTION": "COLUMN_DESCRIPTION",
    }
    result: list[ContextRequirement] = []
    for item in request.metadata_requirements:
        for target in item.targets:
            profile_item = profile_columns.get(target)
            if item.kind == "RELATIONSHIP":
                resolved = bool(relationships)
            elif item.kind == "JOIN_CARDINALITY":
                resolved = any(rel.join_cardinality != "UNKNOWN" for rel in relationships)
            elif item.kind == "TABLE_GRAIN":
                resolved = target in table_profiles
            elif item.kind == "DATE_FORMAT":
                resolved = bool(profile_item and profile_item.observed_format)
            elif item.kind == "CATEGORY_VALUES":
                resolved = bool(profile_item and profile_item.top_values)
            elif item.kind == "NULL_SEMANTICS":
                resolved = profile_item is not None
            elif item.kind == "TEXT_PATTERN":
                resolved = bool(profile_item and profile_item.detected_pattern)
            elif item.kind in {"TEXT_EXAMPLES", "NUMERIC_PROFILE", "COLUMN_DESCRIPTION"}:
                resolved = profile_item is not None
            else:
                resolved = True
            result.append(
                ContextRequirement(
                    kind=cast(Any, mapped[item.kind]),
                    target=target,
                    required_by="verified Context Planner request",
                    resolved=resolved,
                    source="verified schema/glossary/profile" if resolved else None,
                )
            )
    return result


def _relationships(
    schema: SchemaInfo,
    profile: DatabaseProfile | None,
    tables: list[ContextTable],
) -> list[ContextRelationship]:
    if len(schema.tables) < 2:
        return []
    selected = {item.name for item in schema.tables}
    roles = {item.name: item.role for item in tables}
    relationships: list[ContextRelationship] = []
    if profile:
        for profiled_relationship in profile.relationships:
            if (
                not {
                    profiled_relationship.parent_table,
                    profiled_relationship.child_table,
                }
                <= selected
            ):
                continue
            filter_side = next((name for name, role in roles.items() if role == "FILTER"), None)
            fanout = profiled_relationship.type == "MANY_TO_MANY" or not (
                profiled_relationship.parent_key_unique and profiled_relationship.child_key_unique
            )
            relationships.append(
                ContextRelationship(
                    left_table=profiled_relationship.parent_table,
                    right_table=profiled_relationship.child_table,
                    left_column=profiled_relationship.parent_column,
                    right_column=profiled_relationship.child_column,
                    state=(
                        "INFERRED_KEY_RELATIONSHIP"
                        if profiled_relationship.inferred
                        else "VERIFIED_FK"
                    ),
                    join_cardinality=profiled_relationship.type,
                    left_key_unique=profiled_relationship.parent_key_unique,
                    right_key_unique=profiled_relationship.child_key_unique,
                    fanout_risk=fanout,
                    filter_side=filter_side,
                    recommended_strategy="EXISTS" if fanout and filter_side else "JOIN",
                )
            )
    if relationships:
        return relationships
    for foreign_key in schema.relationships:
        if {foreign_key.from_table, foreign_key.to_table} <= selected:
            left_unique = _single_column_primary_key(
                schema, foreign_key.from_table, foreign_key.from_column
            )
            right_unique = _single_column_primary_key(
                schema, foreign_key.to_table, foreign_key.to_column
            )
            relationships.append(
                ContextRelationship(
                    left_table=foreign_key.from_table,
                    right_table=foreign_key.to_table,
                    left_column=foreign_key.from_column,
                    right_column=foreign_key.to_column,
                    state="VERIFIED_FK",
                    join_cardinality=_cardinality(left_unique, right_unique),
                    left_key_unique=left_unique,
                    right_key_unique=right_unique,
                    fanout_risk=not (left_unique and right_unique),
                    filter_side=next(
                        (name for name, role in roles.items() if role == "FILTER"), None
                    ),
                    recommended_strategy="EXISTS",
                )
            )
    if not relationships:
        names = [item.name for item in schema.tables]
        relationships.append(
            ContextRelationship(
                left_table=names[0],
                right_table=names[1],
                left_column="",
                right_column="",
                state="UNRESOLVED",
                join_cardinality="UNKNOWN",
                left_key_unique=None,
                right_key_unique=None,
                fanout_risk=True,
                recommended_strategy="UNRESOLVED",
            )
        )
    return relationships


def _single_column_primary_key(schema: SchemaInfo, table_name: str, column_name: str) -> bool:
    table = next((item for item in schema.tables if item.name == table_name), None)
    if table is None:
        return False
    primary_key = [column.name for column in table.columns if column.primary_key]
    return primary_key == [column_name]


def _cardinality(
    left_unique: bool, right_unique: bool
) -> Literal["ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY"]:
    if left_unique and right_unique:
        return "ONE_TO_ONE"
    if left_unique:
        return "ONE_TO_MANY"
    if right_unique:
        return "MANY_TO_ONE"
    return "MANY_TO_MANY"


def _dependency_profile_context(
    plan: ContextPlan, profile: DatabaseProfile | None
) -> dict[str, dict[str, object]]:
    if profile is None:
        return {}
    required = {(item.kind, item.target) for item in plan.requirements if item.resolved}
    profiles = {f"{item.table}.{item.column}": item for item in profile.columns}
    facts: dict[str, dict[str, object]] = {}
    for kind, target in sorted(required):
        item = profiles.get(target)
        if item is None:
            continue
        if kind == "TEXT_PATTERN" and item.detected_pattern:
            facts[target] = {"pattern": item.detected_pattern}
        elif kind == "TEXT_EXAMPLES" and item.examples:
            facts[target] = {"examples": item.examples[:5]}
        elif kind == "NUMERIC_PROFILE":
            facts[target] = {"minimum": item.minimum, "maximum": item.maximum}
        elif kind == "COLUMN_DESCRIPTION" and item.description:
            facts[target] = {"description": item.description}
    return facts


def _token_budget(question: str, contract: SemanticContract) -> int:
    complexity = (
        len(contract.aggregation_stages)
        + len(contract.formula_metrics)
        + len(contract.derived_metrics)
        + len(contract.selectors)
        + len(contract.output_operations)
    )
    if complexity >= 4 or len(question) > 500:
        return 4_000
    if complexity or len(question) > 180:
        return 2_500
    return 1_500


def _sparse(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _sparse(item)) not in (None, [], {})
        }
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _sparse(item)) not in (None, [], {})]
    return value
