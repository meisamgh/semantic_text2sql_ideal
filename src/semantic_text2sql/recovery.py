"""Bounded LangGraph recovery and database-evidence tools."""

from __future__ import annotations

from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlglot import exp, parse_one

from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.models import (
    DatabaseProfile,
    RecoveryToolCall,
    RecoveryTrace,
    RecoveryUsage,
    SchemaInfo,
)
from semantic_text2sql.postgres import PostgresRegistry


class RecoveryState(TypedDict):
    mode: str
    question: str
    failed_sql: str
    failure_code: str
    failure_message: str
    allowed_tables: list[str]
    category: str
    diagnosis_code: str | None
    diagnosis_summary: str | None
    filter_checks: list[dict[str, Any]]
    evidence: list[str]
    tool_calls: list[dict[str, str]]
    requires_human_review: bool


class RecoveryTools:
    """Capability-scoped tools; the model never receives a database connection."""

    def __init__(
        self,
        database: DatabaseRegistry | PostgresRegistry,
        db_id: str,
        schema: SchemaInfo,
        profile: DatabaseProfile | None,
        *,
        max_rows: int = 20,
    ) -> None:
        self.database = database
        self.db_id = db_id
        self.schema = schema
        self.profile = profile
        self.max_rows = min(max_rows, 20)

    def inspect_schema(self, tables: list[str]) -> dict[str, Any]:
        allowed = set(tables)
        selected = [table for table in self.schema.tables if table.name in allowed]
        return {
            "tables": {
                table.name: {
                    "columns": {column.name: column.data_type for column in table.columns},
                    "primary_key": [column.name for column in table.columns if column.primary_key],
                }
                for table in selected
            },
            "relationships": [
                item.model_dump()
                for item in self.schema.relationships
                if item.from_table in allowed and item.to_table in allowed
            ],
        }

    def inspect_profiles(self, tables: list[str]) -> list[dict[str, Any]]:
        if self.profile is None:
            return []
        allowed = set(tables)
        return [
            {
                "column": f"{item.table}.{item.column}",
                "semantic_type": item.semantic_type,
                "observed_format": item.observed_format,
                "observed_nulls": item.null_count > 0,
                "examples": item.examples[:3],
                "top_values": [value.value for value in item.top_values[:5]],
            }
            for item in self.profile.columns
            if item.table in allowed
            and (item.observed_format or item.examples or item.top_values or item.null_count)
        ][:20]

    def sample_values(
        self, table: str, column: str, *, allowed_tables: list[str]
    ) -> dict[str, Any]:
        live_table = next((item for item in self.schema.tables if item.name == table), None)
        if live_table is None or table not in set(allowed_tables):
            raise ValueError("Table is outside the recovery allowlist.")
        if column not in {item.name for item in live_table.columns}:
            raise ValueError("Column is outside the recovery allowlist.")
        dialect = self.schema.dialect
        query = (
            exp.select(exp.Distinct(expressions=[exp.column(column)]))
            .from_(exp.to_table(table))
            .where(exp.column(column).is_(exp.null()).not_())
            .limit(self.max_rows)
            .sql(dialect=dialect)
        )
        columns, rows, truncated = self.database.execute(
            self.db_id, query, max_rows=self.max_rows
        )
        return {"columns": columns, "rows": rows, "truncated": truncated}

    def inspect_filters(self, sql: str, *, allowed_tables: list[str]) -> list[dict[str, Any]]:
        """Enumerate every explicit SQL filter with bounded profile evidence."""
        try:
            tree = parse_one(sql, dialect=self.schema.dialect)
        except Exception:
            return []
        allowed = set(allowed_tables)
        aliases = {
            table.alias_or_name: table.name
            for table in tree.find_all(exp.Table)
            if table.name in allowed
        }
        profiles = (
            {(item.table, item.column): item for item in self.profile.columns}
            if self.profile
            else {}
        )
        predicates: list[dict[str, Any]] = []
        filter_types = (
            exp.EQ,
            exp.NEQ,
            exp.GT,
            exp.GTE,
            exp.LT,
            exp.LTE,
            exp.In,
            exp.Between,
            exp.Like,
            exp.ILike,
            exp.Is,
        )
        for where in tree.find_all(exp.Where):
            for predicate in where.walk():
                if not isinstance(predicate, filter_types):
                    continue
                columns = list(predicate.find_all(exp.Column))
                if not columns:
                    continue
                column_evidence: list[dict[str, Any]] = []
                for column in columns:
                    table = aliases.get(column.table, column.table)
                    if table and table not in allowed:
                        # Aliases are still useful in the trace, but never authorize a new table.
                        profile = None
                    else:
                        candidates = [
                            item
                            for (profile_table, profile_column), item in profiles.items()
                            if profile_column == column.name
                            and (not table or profile_table == table)
                            and profile_table in allowed
                        ]
                        profile = candidates[0] if len(candidates) == 1 else None
                    column_evidence.append(
                        {
                            "column": column.sql(dialect=self.schema.dialect),
                            "semantic_type": profile.semantic_type if profile else None,
                            "observed_format": profile.observed_format if profile else None,
                            "observed_nulls": profile.null_count > 0 if profile else None,
                            "domain_complete": bool(profile and profile.allowed_values),
                            "known_values": (
                                profile.allowed_values[:20]
                                if profile and profile.allowed_values
                                else [value.value for value in profile.top_values[:5]]
                                if profile
                                else []
                            ),
                        }
                    )
                predicates.append(
                    {
                        "expression": predicate.sql(dialect=self.schema.dialect),
                        "plain_language": _plain_filter(predicate),
                        "columns": column_evidence,
                        "literals": [literal.this for literal in predicate.find_all(exp.Literal)],
                    }
                )
        # Nested predicates can repeat; preserve order while deduplicating exact expressions.
        unique: dict[str, dict[str, Any]] = {}
        for item in predicates:
            unique.setdefault(item["expression"], item)
        return list(unique.values())[:20]

    def probe_filter_counts(
        self, sql: str, *, allowed_tables: list[str]
    ) -> list[dict[str, Any]]:
        """Run one bounded count probe per top-level AND filter."""
        try:
            tree = parse_one(sql, dialect=self.schema.dialect)
        except Exception:
            return []
        select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
        if select is None or select.args.get("where") is None:
            return []
        cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
        physical_tables = {
            table.name for table in tree.find_all(exp.Table) if table.name not in cte_names
        }
        if not physical_tables.issubset(set(allowed_tables)):
            return []
        predicates = _split_and(select.args["where"].this)[:10]
        checks: list[dict[str, Any]] = []
        for predicate in predicates:
            probe_tree = tree.copy()
            probe_select = (
                probe_tree if isinstance(probe_tree, exp.Select) else probe_tree.find(exp.Select)
            )
            if probe_select is None:
                continue
            probe_select.set(
                "expressions",
                [exp.alias_(exp.Count(this=exp.Star()), "match_count")],
            )
            probe_select.set("where", exp.Where(this=predicate.copy()))
            for clause in ("group", "having", "qualify", "order", "limit", "offset"):
                probe_select.set(clause, None)
            probe_select.set("distinct", None)
            probe_sql = probe_tree.sql(dialect=self.schema.dialect)
            try:
                _, rows, _ = self.database.execute(
                    self.db_id, probe_sql, max_rows=1, timeout_seconds=2.0
                )
                count = int(rows[0][0]) if rows else 0
                checks.append(
                    {
                        "filter": predicate.sql(dialect=self.schema.dialect),
                        "plain_language": _plain_filter(predicate),
                        "match_count": count,
                        "status": "MATCH" if count > 0 else "NO_MATCH",
                    }
                )
            except Exception as exc:
                checks.append(
                    {
                        "filter": predicate.sql(dialect=self.schema.dialect),
                        "plain_language": _plain_filter(predicate),
                        "match_count": None,
                        "status": "PROBE_FAILED",
                        "error": str(exc)[:200],
                    }
                )
        return checks

class RecoveryCoordinator:
    """A deterministic controller expressed as a bounded LangGraph."""

    def __init__(self, tools: RecoveryTools) -> None:
        self.tools = tools
        graph = StateGraph(RecoveryState)
        graph.add_node("classify", self._classify)
        graph.add_node("gather", self._gather)
        graph.add_edge(START, "classify")
        graph.add_edge("classify", "gather")
        graph.add_edge("gather", END)
        self.graph = graph.compile()

    def investigate(
        self,
        *,
        question: str,
        failed_sql: str,
        failure_code: str,
        failure_message: str,
        allowed_tables: list[str],
        mode: str = "FAILURE",
    ) -> RecoveryTrace:
        started = perf_counter()
        result = self.graph.invoke(
            RecoveryState(
                mode=mode,
                question=question,
                failed_sql=failed_sql,
                failure_code=failure_code,
                failure_message=failure_message,
                allowed_tables=allowed_tables,
                category="sql",
                diagnosis_code=None,
                diagnosis_summary=None,
                filter_checks=[],
                evidence=[],
                tool_calls=[],
                requires_human_review=False,
            )
        )
        return RecoveryTrace(
            mode=mode,  # type: ignore[arg-type]
            failure_code=failure_code,
            failure_category=result["category"],
            diagnosis_code=result["diagnosis_code"],
            diagnosis_summary=result["diagnosis_summary"],
            filter_checks=result["filter_checks"],
            usage=RecoveryUsage(
                llm_calls=0,
                database_probe_count=len(result["filter_checks"]),
                latency_ms=round((perf_counter() - started) * 1_000),
                estimated_llm_cost_usd=0.0,
            ),
            evidence=result["evidence"],
            tool_calls=[RecoveryToolCall.model_validate(item) for item in result["tool_calls"]],
            requires_human_review=result["requires_human_review"],
        )

    def _classify(self, state: RecoveryState) -> dict[str, Any]:
        if state["mode"] in {"ZERO_RESULT", "NULL_RESULT", "FILTER"}:
            return {"category": "data_grounding"}
        if state["mode"] == "CORRECTNESS":
            return {"category": "correctness"}
        text = f"{state['failure_code']} {state['failure_message']}".casefold()
        if any(token in text for token in ("column", "table", "alias", "schema")):
            category = "schema"
        elif any(token in text for token in ("date", "type", "value", "null", "empty")):
            category = "data_grounding"
        elif any(token in text for token in ("timeout", "rate limit", "unavailable")):
            category = "provider"
        elif "safety" in text or "write" in text:
            category = "safety"
        else:
            category = "sql"
        return {"category": category}

    def _gather(self, state: RecoveryState) -> dict[str, Any]:
        evidence = [f"Failure {state['failure_code']}: {state['failure_message']}"]
        calls: list[dict[str, str]] = []
        filters: list[dict[str, Any]] = []
        if state["category"] in {"schema", "data_grounding", "correctness"}:
            schema = self.tools.inspect_schema(state["allowed_tables"])
            evidence.append(f"Verified schema: {schema}")
            calls.append(
                {
                    "tool": "inspect_schema",
                    "purpose": "Verify referenced schema after the failed attempt.",
                    "result_summary": f"Inspected {len(schema['tables'])} allowed tables.",
                }
            )
        if state["category"] in {"data_grounding", "correctness"} and len(calls) < 3:
            profiles = self.tools.inspect_profiles(state["allowed_tables"])
            evidence.append(f"Relevant column profiles: {profiles}")
            calls.append(
                {
                    "tool": "inspect_column",
                    "purpose": "Verify date, value, and NULL representations.",
                    "result_summary": f"Returned {len(profiles)} bounded profile entries.",
                }
            )
        if state["mode"] in {"ZERO_RESULT", "NULL_RESULT", "FILTER", "CORRECTNESS"}:
            filters = self.tools.inspect_filters(
                state["failed_sql"], allowed_tables=state["allowed_tables"]
            )
            evidence.append(f"All explicit SQL filters: {filters}")
            calls.append(
                {
                    "tool": "inspect_filters",
                    "purpose": "Check every explicit filter and its grounded column evidence.",
                    "result_summary": f"Inspected {len(filters)} distinct filter predicates.",
                }
            )
        filter_checks: list[dict[str, Any]] = []
        if state["mode"] in {"ZERO_RESULT", "NULL_RESULT", "FILTER"}:
            filter_checks = self.tools.probe_filter_counts(
                state["failed_sql"], allowed_tables=state["allowed_tables"]
            )
            evidence.append(f"Independent filter counts: {filter_checks}")
            calls.append(
                {
                    "tool": "probe_filter_counts",
                    "purpose": "Test each top-level filter independently with a bounded SELECT.",
                    "result_summary": f"Probed {len(filter_checks)} filters with a 2 second limit.",
                }
            )
        diagnosis_code, diagnosis_summary = _diagnose(
            mode=state["mode"],
            sql=state["failed_sql"],
            filters=filters,
            checks=filter_checks,
        )
        if diagnosis_code:
            evidence.append(f"Diagnosis {diagnosis_code}: {diagnosis_summary}")
        return {
            "evidence": evidence,
            "tool_calls": calls,
            "requires_human_review": state["category"] == "provider"
            or state["mode"] in {"ZERO_RESULT", "NULL_RESULT", "FILTER"},
            "diagnosis_code": diagnosis_code,
            "diagnosis_summary": diagnosis_summary,
            "filter_checks": filter_checks,
        }


def recovery_feedback(trace: RecoveryTrace) -> str:
    evidence = "\n".join(f"- {item}" for item in trace.evidence)
    return (
        "Bounded recovery was activated after the first failure. Fix only the diagnosed "
        "violation while preserving the original question and grounded context.\n" + evidence
    )


def _split_and(expression: exp.Expression) -> list[exp.Expression]:
    if isinstance(expression, exp.And):
        return [*_split_and(expression.left), *_split_and(expression.right)]
    return [expression]


def _diagnose(
    *,
    mode: str,
    sql: str,
    filters: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    for item in filters:
        literals = {str(value).casefold() for value in item.get("literals", [])}
        for column in item.get("columns", []):
            known = {str(value).casefold() for value in column.get("known_values", [])}
            if column.get("domain_complete") and known and literals and literals.isdisjoint(known):
                return (
                    "FILTER_VALUE_NOT_FOUND",
                    f"The requested value in '{item['plain_language']}' is not stored in the "
                    "selected database column.",
                )
    no_match = [
        item.get("plain_language", item["filter"])
        for item in checks
        if item.get("status") == "NO_MATCH"
    ]
    matched = [item for item in checks if item.get("status") == "MATCH"]
    if no_match:
        return (
            "FILTER_NO_MATCH",
            "No data matched the following condition(s): " + "; ".join(no_match) + ".",
        )
    if mode == "ZERO_RESULT" and checks and len(matched) == len(checks):
        return (
            "FILTER_COMBINATION_EMPTY",
            "Every checked filter matched independently, but their complete combination "
            "returned no rows.",
        )
    if mode == "ZERO_RESULT" and not filters:
        return "VALID_EMPTY_RESULT", "The query has no explicit filter to relax or remap."
    if mode == "NULL_RESULT":
        try:
            tree = parse_one(sql)
        except Exception:
            tree = None
        if tree is not None and any(
            join.args.get("side") == "LEFT" for join in tree.find_all(exp.Join)
        ):
            return "QUERY_INDUCED_NULL", "A LEFT JOIN can introduce NULL for unmatched rows."
        if tree is not None and tree.find(exp.Nullif) is not None:
            return "DIVISION_BY_ZERO", "NULLIF can intentionally turn a zero denominator into NULL."
        aggregate_types = (exp.Sum, exp.Avg, exp.Min, exp.Max)
        if tree is not None and any(tree.find(kind) is not None for kind in aggregate_types):
            return (
                "AGGREGATION_OVER_EMPTY_SET",
                "An aggregate may return NULL when its qualifying input contains no non-NULL "
                "values.",
            )
        return (
            "SOURCE_DATA_NULL_OR_EXPRESSION",
            "The NULL may come from stored data or a SQL expression; human confirmation is "
            "required.",
        )
    return None, None


def _plain_filter(predicate: exp.Expression) -> str:
    """Render a common SQL predicate as concise user-facing language."""
    column = next(predicate.find_all(exp.Column), None)
    subject = column.name if column is not None else "calculated value"
    literals = [str(item.this) for item in predicate.find_all(exp.Literal)]
    value = literals[0] if literals else "the requested value"
    if isinstance(predicate, exp.EQ):
        return f"{subject} must equal {value}"
    if isinstance(predicate, exp.NEQ):
        return f"{subject} must not equal {value}"
    if isinstance(predicate, exp.GT):
        return f"{subject} must be greater than {value}"
    if isinstance(predicate, exp.GTE):
        return f"{subject} must be at least {value}"
    if isinstance(predicate, exp.LT):
        return f"{subject} must be less than or earlier than {value}"
    if isinstance(predicate, exp.LTE):
        return f"{subject} must be at most or no later than {value}"
    if isinstance(predicate, exp.In):
        return f"{subject} must be one of {', '.join(literals) or 'the requested values'}"
    if isinstance(predicate, exp.Between) and len(literals) >= 2:
        return f"{subject} must be between {literals[0]} and {literals[1]}"
    if isinstance(predicate, (exp.Like, exp.ILike)):
        return f"{subject} must match {value}"
    if isinstance(predicate, exp.Is):
        return f"{subject} must have the requested missing-value state"
    return f"The condition on {subject} must be satisfied"
