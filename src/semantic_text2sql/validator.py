"""Minimal fail-closed validation for SQL syntax and read-only safety."""

from __future__ import annotations

import re
from typing import Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError, TokenError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import Scope, traverse_scope

from semantic_text2sql.dialects import DatabaseDialect
from semantic_text2sql.models import (
    SchemaInfo,
    SemanticContract,
    SemanticFinding,
    ValidationResult,
)

_FENCE = re.compile(r"^```(?:sql)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)
_EMBEDDED_SQL_FENCE = re.compile(r"```sql\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_FORBIDDEN = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
    exp.Merge,
    exp.Lock,
)


def clean_model_sql(value: str) -> str:
    stripped = value.strip()
    match = _FENCE.fullmatch(stripped)
    if match:
        stripped = match.group(1)
    else:
        embedded = _EMBEDDED_SQL_FENCE.search(stripped)
        if embedded:
            stripped = embedded.group(1)
    return stripped.strip().removesuffix(";").strip()


def normalize_sql(value: str, *, dialect: DatabaseDialect = "sqlite") -> str:
    try:
        return sqlglot.parse_one(value, read=dialect).sql(dialect=dialect, pretty=False)
    except (ParseError, TokenError):
        return " ".join(value.split())


def validate_sql(
    sql: str,
    schema: SchemaInfo,
    *,
    dialect: DatabaseDialect = "sqlite",
    allowed_tables: set[str] | None = None,
    allowed_columns: dict[str, set[str]] | None = None,
) -> ValidationResult:
    available = {table.name: [column.name for column in table.columns] for table in schema.tables}
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except (ParseError, TokenError) as exc:
        return _failure("SQL_PARSE_FAILED", f"Generated SQL could not be parsed: {exc}", available)
    if len(statements) != 1:
        return _failure("SQL_MULTIPLE_STATEMENTS", "Exactly one statement is allowed.", available)
    root = statements[0]
    if root is None or not isinstance(root, exp.Query):
        return _failure("SQL_NOT_READ_ONLY", "Only SELECT or WITH...SELECT is allowed.", available)
    if any(isinstance(node, _FORBIDDEN) for node in root.walk()):
        return _failure(
            "SQL_NOT_READ_ONLY",
            "Write and administrative operations are blocked.",
            available,
        )
    if any(isinstance(node, exp.Into) for node in root.walk()):
        return _failure("SQL_SELECT_INTO_BLOCKED", "SELECT INTO is blocked.", available)

    try:
        physical_refs = physical_table_references(root, dialect=dialect)
    except (OptimizeError, ValueError) as exc:
        return _failure(
            "SQL_SCOPE_UNRESOLVED", f"SQL scopes could not be resolved: {exc}", available
        )
    physical_tables = [table.name for table in physical_refs if table.name]
    if allowed_tables is not None:
        authorized = {name.casefold() for name in allowed_tables}
        unauthorized = sorted(
            {table.sql(dialect=dialect) for table in physical_refs
             if not table_is_allowlisted(table, authorized)}
        )
        if unauthorized:
            return _failure(
                "SQL_TABLE_NOT_AUTHORIZED",
                "SQL references tables outside the authorized context: " + ", ".join(unauthorized),
                available,
                tables=physical_tables,
            )
    columns = list(
        dict.fromkeys(column.sql(dialect=dialect) for column in root.find_all(exp.Column))
    )
    if allowed_columns is not None:
        normalized_columns = {
            table.casefold(): {column.casefold() for column in names}
            for table, names in allowed_columns.items()
        }
        aliases = {
            (table.alias_or_name or table.name).casefold(): table.name.casefold()
            for table in physical_refs if table.name
        }
        derived_columns = {
            id(column) for scope in traverse_scope(root) for column in scope.columns
            if column.table and not column.db and not column.catalog
            and isinstance(scope.sources.get(column.table), Scope)
        }
        select_aliases = {
            alias.alias.casefold() for alias in root.find_all(exp.Alias) if alias.alias
        }
        unauthorized_columns: list[str] = []
        for column in root.find_all(exp.Column):
            if not column.name or id(column) in derived_columns:
                continue
            if column.table:
                owner = aliases.get(column.table.casefold(), column.table.casefold())
                allowed = normalized_columns.get(owner)
                if allowed is not None and column.name.casefold() not in allowed:
                    unauthorized_columns.append(column.sql(dialect=dialect))
            elif column.name.casefold() not in select_aliases and not any(
                column.name.casefold() in names for names in normalized_columns.values()
            ):
                unauthorized_columns.append(column.sql(dialect=dialect))
        if unauthorized_columns:
            return _failure(
                "SQL_COLUMN_NOT_AUTHORIZED",
                "SQL references columns outside the authorized context: "
                + ", ".join(sorted(set(unauthorized_columns))),
                available,
                tables=physical_tables,
            )
    return ValidationResult(
        valid=True,
        code="SQL_SAFETY_VALID",
        message="SQL is one syntactically valid read-only statement.",
        tables=list(dict.fromkeys(physical_tables)),
        columns=columns,
        available_columns=available,
    )


def physical_table_references(
    root: exp.Expression, *, dialect: DatabaseDialect = "sqlite",
) -> list[exp.Table]:
    """Actual AST table nodes, excluding only lexically visible unqualified CTEs.

    Resolve a copy using engine case rules (including quoted names), returning
    original nodes without rewriting SQL. Qualified identifiers can never bind
    a CTE. All table nodes include SEMI/ANTI joins, which selected_sources omits.
    """
    resolved = normalize_identifiers(root.copy(), dialect=dialect)
    original_tables = list(root.find_all(exp.Table))
    resolved_tables = list(resolved.find_all(exp.Table))
    if len(original_tables) != len(resolved_tables):
        raise ValueError("Identifier normalization changed the table-reference structure")
    cte_references = {
        id(table) for scope in traverse_scope(resolved) for table in scope.tables
        if not table.db and not table.catalog
        and isinstance(table.this, exp.Identifier)
        and isinstance(scope.cte_sources.get(table.name), Scope)
    }
    return [original for original, table in zip(original_tables, resolved_tables, strict=True)
            if id(table) not in cte_references]


def table_is_allowlisted(table: exp.Table, allowed: set[str]) -> bool:
    """Bare allowlist entries authorize names; qualified entries require identity.

    Engine adapters must independently restrict schemas/catalogs when their
    configuration uses bare table names.
    """
    full_name = ".".join(part.name.casefold() for part in table.parts)
    return table.name.casefold() in allowed or full_name in allowed


def validate_sql_semantics(
    sql: str,
    contract: SemanticContract,
    verified_context: dict[str, object] | None = None,
    *,
    dialect: DatabaseDialect = "sqlite",
) -> list[SemanticFinding]:
    """Reject only contract mismatches provable from the SQL AST.

    Unverified business meaning, equivalence of alternate formulas, and join
    fanout require data-backed review and are explicitly marked UNRESOLVED.
    The caller must run ``validate_sql`` first; this function never executes SQL.
    """
    root = sqlglot.parse_one(sql, read=dialect)
    findings: list[SemanticFinding] = []
    aliases = {
        alias.alias.casefold() for alias in root.find_all(exp.Alias) if alias.alias
    }
    for required in contract.required_output_aliases:
        if required.casefold() not in aliases:
            findings.append(SemanticFinding(
                status="FAIL", code="REQUIRED_OUTPUT_MISSING",
                evidence=f"Required output alias {required!r} is absent from the SELECT list.",
            ))
        else:
            findings.append(SemanticFinding(
                status="PASS", code="REQUIRED_OUTPUT_PRESENT",
                evidence=f"Required output alias {required!r} is present.",
            ))
    aggregates = {type(node) for node in root.find_all(exp.AggFunc)}
    expected: dict[str, type[exp.AggFunc]] = {
        "count": exp.Count, "sum": exp.Sum, "average": exp.Avg,
        "min": exp.Min, "max": exp.Max,
    }
    if contract.aggregation and expected[contract.aggregation] not in aggregates:
        # SUM(x)/COUNT(x) and similar rewrites can implement AVG without AVG().
        # They need stronger lineage proof rather than a structural rejection.
        alternate = contract.aggregation == "average" and bool(
            list(root.find_all(exp.Div))
        )
        findings.append(SemanticFinding(
            status="UNRESOLVED" if alternate else "FAIL",
            code="AGGREGATION_NOT_FOUND",
            evidence=f"The SQL has no {contract.aggregation.upper()} aggregation node.",
        ))
    elif contract.aggregation:
        findings.append(SemanticFinding(
            status="PASS", code="AGGREGATION_PRESENT",
            evidence=f"Found a {contract.aggregation.upper()} aggregation node.",
        ))
    required_grain = contract.output_grain or contract.grain
    groups = list(root.find_all(exp.Group))
    if contract.requires_grouping and required_grain and not groups:
        status: Literal["FAIL", "UNRESOLVED"] = (
            "UNRESOLVED" if any(root.find_all(exp.Window)) else "FAIL"
        )
        findings.append(SemanticFinding(
            status=status, code="GROUPING_NOT_FOUND",
            evidence="The contract requires grouped output, but SQL has no GROUP BY clause.",
        ))
    elif contract.requires_grouping and required_grain:
        findings.extend(_grouping_findings(root, required_grain, groups))
    findings.extend(_measure_findings(root, contract, verified_context))
    findings.extend(_approved_metric_findings(root, contract, dialect))
    findings.extend(_filter_findings(root, contract, verified_context))
    findings.extend(_fanout_findings(root, verified_context, dialect=dialect))
    findings.extend(_ranking_findings(root, contract))
    if contract.requires_distinct and not any(root.find_all(exp.Distinct)):
        findings.append(SemanticFinding(
            status="UNRESOLVED", code="DISTINCT_NOT_OBVIOUS",
            evidence="No DISTINCT node was found; grouping or keys might still provide uniqueness.",
        ))
    return findings


def _approved_metric_findings(
    root: exp.Expression, contract: SemanticContract,
    dialect: DatabaseDialect,
) -> list[SemanticFinding]:
    findings: list[SemanticFinding] = []
    aliases = {
        table.alias_or_name.casefold(): table.name.casefold()
        for table in root.find_all(exp.Table)
    }
    complex_lineage = bool(list(root.find_all(exp.CTE)) or list(root.find_all(exp.Subquery)))
    for metric in contract.approved_metrics:
        try:
            formula = sqlglot.parse_one(metric.formula, read=dialect)
        except (ParseError, TokenError):
            formula = None
        if not isinstance(formula, exp.AggFunc) or not isinstance(formula.this, exp.Column):
            findings.append(SemanticFinding(
                status="UNRESOLVED", code="METRIC_FORMULA_EQUIVALENCE_UNPROVEN",
                evidence=f"The approved {metric.id} formula needs derived-expression lineage.",
            ))
            continue
        required_column = formula.this.name.casefold()
        candidates = [
            node for node in root.find_all(type(formula))
            if isinstance(node.this, exp.Column)
            and node.this.name.casefold() == required_column
        ]
        if candidates:
            source = metric.source_relation.casefold()
            proven = any(
                (node.this.table and aliases.get(node.this.table.casefold(),
                                                  node.this.table.casefold()) == source)
                or (not node.this.table and set(aliases.values()) == {source})
                for node in candidates
            )
            proven = proven and not complex_lineage and len(list(root.find_all(type(formula)))) == 1
            findings.append(SemanticFinding(
                status="PASS" if proven else "UNRESOLVED",
                code="APPROVED_METRIC_INPUT_MATCH" if proven
                else "METRIC_FORMULA_EQUIVALENCE_UNPROVEN",
                evidence=(f"The {metric.id} aggregate uses its approved input."
                          if proven else f"The {metric.id} input matches by name, but "
                          "its source relation is not proven."),
            ))
        else:
            direct_wrong_input = bool(list(root.find_all(type(formula)))) and all(
                isinstance(node.this, exp.Column)
                for node in root.find_all(type(formula))
            )
            findings.append(SemanticFinding(
                status="FAIL" if direct_wrong_input and not complex_lineage else "UNRESOLVED",
                code="APPROVED_METRIC_INPUT_MISMATCH",
                evidence=f"No direct {type(formula).__name__.upper()} on "
                         f"{metric.source_relation}.{required_column} implements {metric.id}.",
            ))
    return findings


def _grouping_findings(
    root: exp.Expression, required_grain: list[str], groups: list[exp.Group]
) -> list[SemanticFinding]:
    aliases = {
        alias.alias.casefold(): alias.this.name.casefold()
        for alias in root.find_all(exp.Alias)
        if isinstance(alias.this, exp.Column)
    }
    actual = {
        column.name.casefold()
        for group in groups for expression in group.expressions
        for column in expression.find_all(exp.Column)
    }
    actual.update(
        expression.name.casefold()
        for group in groups for expression in group.expressions
        if isinstance(expression, exp.Column)
    )
    actual.update(aliases[name] for name in list(actual) if name in aliases)
    for group in groups:
        parent = group.parent
        if not isinstance(parent, exp.Select):
            continue
        for expression in group.expressions:
            if not isinstance(expression, exp.Literal) or not expression.is_int:
                continue
            position = int(expression.this) - 1
            if 0 <= position < len(parent.expressions):
                actual.update(
                    item.name.casefold()
                    for item in parent.expressions[position].find_all(exp.Column)
                )
    missing = [name for name in required_grain if name.split(".")[-1].casefold() not in actual]
    if not missing:
        return [SemanticFinding(
            status="PASS", code="GROUPING_DIMENSIONS_PRESENT",
            evidence="Required grouping columns occur in GROUP BY expressions.",
        )]
    # A CTE/window can transform a dimension before the final grouping. Without
    # lineage proof, do not turn that into a false rejection.
    uncertain = bool(list(root.find_all(exp.CTE)) or list(root.find_all(exp.Window)))
    return [SemanticFinding(
        status="UNRESOLVED" if uncertain else "FAIL",
        code="GROUPING_DIMENSION_MISMATCH",
        evidence=f"Required grouping dimension(s) absent from GROUP BY: {', '.join(missing)}.",
    )]


def _measure_findings(
    root: exp.Expression,
    contract: SemanticContract,
    verified_context: dict[str, object] | None,
) -> list[SemanticFinding]:
    if not contract.aggregation or not contract.measures or not verified_context:
        return []
    tables = verified_context.get("tables")
    if not isinstance(tables, dict):
        return []
    physical = {
        column.casefold()
        for table in tables.values() if isinstance(table, dict)
        for column in table.get("columns", {})
    }
    measures = {name.split(".")[-1].casefold() for name in contract.measures}
    known_measures = measures & physical
    if not known_measures:
        return [SemanticFinding(
            status="UNRESOLVED", code="MEASURE_LINEAGE_UNKNOWN",
            evidence=("Requested measure is symbolic; no verified physical "
                      "column mapping was supplied."),
        )]
    expected: dict[str, type[exp.AggFunc]] = {
        "count": exp.Count, "sum": exp.Sum, "average": exp.Avg,
        "min": exp.Min, "max": exp.Max,
    }
    aggregate_nodes = list(root.find_all(expected[contract.aggregation]))
    if not aggregate_nodes:
        return []  # The aggregation-presence finding already covers this case.
    matched = any(
        column.name.casefold() in known_measures
        for aggregate in aggregate_nodes for column in aggregate.find_all(exp.Column)
    )
    if matched:
        return [SemanticFinding(
            status="PASS", code="AGGREGATED_MEASURE_PRESENT",
            evidence="The requested physical measure occurs inside the required aggregate.",
        )]
    if contract.aggregation == "count" and any(
        isinstance(node.this, exp.Star) for node in aggregate_nodes
    ):
        return [SemanticFinding(
            status="UNRESOLVED", code="COUNT_MEASURE_EQUIVALENCE_UNKNOWN",
            evidence="COUNT(*) may count the intended entity only if the input grain is verified.",
        )]
    uncertain = bool(list(root.find_all(exp.CTE)) or list(root.find_all(exp.Subquery)))
    return [SemanticFinding(
        status="UNRESOLVED" if uncertain else "FAIL", code="AGGREGATED_MEASURE_MISMATCH",
        evidence="Required aggregate does not use the requested physical measure.",
    )]


def _filter_findings(
    root: exp.Expression,
    contract: SemanticContract,
    verified_context: dict[str, object] | None,
) -> list[SemanticFinding]:
    findings: list[SemanticFinding] = []
    aliases = {
        table.alias_or_name.casefold(): table.name.casefold()
        for table in root.find_all(exp.Table)
    }
    context_tables = verified_context.get("tables", {}) if verified_context else {}
    predicates = [
        node for where in root.find_all(exp.Where) for node in where.walk()
        if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In,
                             exp.Between))
    ]
    for required, temporal in [
        *((item, False) for item in contract.filters),
        *((item, False) for item in contract.mandatory_metric_filters),
        *((item, True) for item in contract.temporal_boundaries),
    ]:
        column_name = required.operand.split(".")[-1].casefold()
        required_table = (
            required.operand.split(".", 1)[0].casefold()
            if "." in required.operand else None
        )
        relevant = [
            node for node in predicates
            if any(
                col.name.casefold() == column_name
                and (
                    required_table is None
                    or (col.table and aliases.get(col.table.casefold(), col.table.casefold())
                        == required_table)
                    or (not col.table and set(aliases.values()) == {required_table})
                )
                for col in node.find_all(exp.Column)
            )
        ]
        requested = (
            {str(value).casefold() for value in required.value}
            if isinstance(required.value, list)
            else {str(required.value).casefold()}
        )
        matches = [
            node for node in relevant
            if _operator_name(node) == required.operator
            and requested.issubset({
                str(item.this).casefold() for item in node.find_all(exp.Literal)
            })
        ]
        if matches:
            ambiguous = (
                required_table is None and isinstance(context_tables, dict)
                and sum(
                    column_name in {str(name).casefold() for name in table.get("columns", {})}
                    for table in context_tables.values() if isinstance(table, dict)
                ) > 1
                and any(not col.table for node in matches for col in node.find_all(exp.Column))
            )
            status: Literal["PASS", "UNRESOLVED"] = (
                "UNRESOLVED"
                if ambiguous or any(node.find_ancestor(exp.Or) for node in matches)
                else "PASS"
            )
            findings.append(SemanticFinding(
                status=status, code="FILTER_PREDICATE_MATCH",
                evidence=(f"Found the required {required.operand} predicate "
                          "with its operator and value."),
            ))
        elif relevant:
            uncertain = temporal and bool(
                list(root.find_all(exp.CTE)) or list(root.find_all(exp.Subquery))
            )
            findings.append(SemanticFinding(
                status="UNRESOLVED" if uncertain else "FAIL",
                code="REQUIRED_FILTER_PREDICATE_MISMATCH",
                evidence=(f"SQL filters {required.operand}, but not with "
                          "the required operator/value."),
            ))
        elif not (len(requested) == 1 and re.fullmatch(r"\d{4}", next(iter(requested)))):
            uncertain = temporal and bool(
                list(root.find_all(exp.CTE)) or list(root.find_all(exp.Subquery))
            )
            findings.append(SemanticFinding(
                status="UNRESOLVED" if uncertain else "FAIL",
                code="REQUIRED_FILTER_VALUE_MISSING",
                evidence=(f"No required predicate for {required.operand} "
                          "appears in SQL WHERE clauses."),
            ))
        else:
            findings.append(SemanticFinding(
                status="UNRESOLVED", code="TEMPORAL_PREDICATE_EQUIVALENCE_UNKNOWN",
                evidence="A year may be represented by date bounds; equivalence is not proven.",
            ))
    return findings


def _operator_name(node: exp.Expression) -> str:
    operators: dict[type[exp.Expression], str] = {
        exp.EQ: "=", exp.NEQ: "!=", exp.GT: ">", exp.GTE: ">=",
        exp.LT: "<", exp.LTE: "<=", exp.In: "IN", exp.Between: "BETWEEN",
    }
    return operators.get(type(node), "")


def _fanout_findings(
    root: exp.Expression, verified_context: dict[str, object] | None,
    *, dialect: DatabaseDialect = "sqlite",
) -> list[SemanticFinding]:
    if not verified_context or not any(root.find_all(exp.Sum, exp.Avg)):
        return []
    relationships = verified_context.get("relationships")
    if not isinstance(relationships, list):
        return []
    physical = {
        table.name.casefold() for table in physical_table_references(root, dialect=dialect)
    }
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        left = str(relationship.get("left") or "").split(".")[0].casefold()
        right = str(relationship.get("right") or "").split(".")[0].casefold()
        if (left in physical and right in physical and left != right
                and (relationship.get("cardinality") == "MANY_TO_MANY"
                     or relationship.get("fanout_risk") is True)):
            return [SemanticFinding(
                status="UNRESOLVED",
                code="FACT_JOIN_FANOUT_RISK",
                evidence=(f"Aggregating across {left} and {right} may multiply measures; "
                          "the measure grain and join keys need a data-backed check."),
            )]
    return []


def _ranking_findings(root: exp.Expression, contract: SemanticContract) -> list[SemanticFinding]:
    if (not contract.requires_ordering and not contract.ranking_direction
            and contract.limit is None and not contract.ranked_metric):
        return []
    findings: list[SemanticFinding] = []
    outer_order = root.args.get("order")
    complex_ranking = bool(list(root.find_all(exp.Window)) or list(root.find_all(exp.CTE)))
    if contract.requires_ordering and not isinstance(outer_order, exp.Order):
        findings.append(SemanticFinding(
            status="UNRESOLVED" if complex_ranking else "FAIL",
            code="RANKING_ORDER_MISSING",
            evidence="No final ORDER BY is present; an intermediate ranking may still select rows.",
        ))
    if contract.ranking_direction and isinstance(outer_order, exp.Order):
        first = outer_order.expressions[0] if outer_order.expressions else None
        if isinstance(first, exp.Ordered):
            actual_direction = "DESC" if first.args.get("desc") else "ASC"
            findings.append(SemanticFinding(
                status="PASS" if actual_direction == contract.ranking_direction else "FAIL",
                code="RANKING_DIRECTION_MATCH" if actual_direction == contract.ranking_direction
                else "RANKING_DIRECTION_MISMATCH",
                evidence=f"The first final ORDER BY key is {actual_direction}; expected "
                         f"{contract.ranking_direction}.",
            ))
    if contract.ranked_metric:
        first = (
            outer_order.expressions[0]
            if isinstance(outer_order, exp.Order) and outer_order.expressions else None
        )
        key = first.this if isinstance(first, exp.Ordered) else None
        expected_name = re.sub(
            r"[\s_-]+", "", contract.ranked_metric.split(".")[-1].casefold()
        )
        actual_name = key.name.casefold() if isinstance(key, exp.Column) else None
        if actual_name is not None:
            actual_name = re.sub(r"[\s_-]+", "", actual_name)
        if actual_name == expected_name:
            findings.append(SemanticFinding(
                status="PASS", code="RANKED_METRIC_MATCH",
                evidence=f"The final ORDER BY uses {contract.ranked_metric}.",
            ))
        else:
            projected_names = {
                re.sub(r"[\s_-]+", "", expression.alias_or_name.casefold())
                for expression in root.expressions if expression.alias_or_name
            }
            uncertain = (
                complex_ranking or not isinstance(key, exp.Column)
                or expected_name not in projected_names
            )
            findings.append(SemanticFinding(
                status="UNRESOLVED" if uncertain else "FAIL",
                code="RANKED_METRIC_UNPROVEN" if uncertain else "RANKED_METRIC_MISMATCH",
                evidence=f"The final ORDER BY does not directly use {contract.ranked_metric}.",
            ))
    outer_limit = root.args.get("limit")
    if contract.limit is not None:
        actual_limit = outer_limit.expression if isinstance(outer_limit, exp.Limit) else None
        if isinstance(actual_limit, exp.Literal) and actual_limit.is_int:
            actual_count = int(actual_limit.this)
            findings.append(SemanticFinding(
                status="PASS" if actual_count == contract.limit else "FAIL",
                code="RANKING_LIMIT_MATCH" if actual_count == contract.limit
                else "RANKING_LIMIT_MISMATCH",
                evidence=f"The final LIMIT is {actual_count}; expected {contract.limit}.",
            ))
        else:
            findings.append(SemanticFinding(
                status="UNRESOLVED" if complex_ranking else "FAIL",
                code="RANKING_LIMIT_MISSING",
                evidence="No literal final LIMIT implements the requested rank count.",
            ))
    return findings


def repair_context(result: ValidationResult) -> str:
    guidance = ""
    if (
        result.code == "DATABASE_ERROR"
        and "ORDER BY clause should come after" in result.message
        and "not before" in result.message
    ):
        guidance = (
            "\nRequired structural repair: SQLite does not allow branch-level ORDER BY/LIMIT "
            "directly before a compound operator. Wrap each ranked branch in its own subquery "
            "or CTE, or replace the compound query with window-function ranking. Do not repeat "
            "the rejected compound-query structure."
        )
    return (f"Validation code: {result.code}\nValidation message: {result.message}{guidance}")[
        :12_000
    ]


def _failure(
    code: str,
    message: str,
    available: dict[str, list[str]],
    *,
    tables: list[str] | None = None,
) -> ValidationResult:
    return ValidationResult(
        valid=False,
        code=code,
        message=message,
        tables=tables or [],
        available_columns=available,
    )
