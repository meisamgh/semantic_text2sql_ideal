"""Bounded LangGraph recovery and database-evidence tools."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlglot import exp, parse_one

from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.models import (
    DatabaseProfile,
    RecoveryToolCall,
    RecoveryTrace,
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
                        "columns": column_evidence,
                        "literals": [literal.this for literal in predicate.find_all(exp.Literal)],
                    }
                )
        # Nested predicates can repeat; preserve order while deduplicating exact expressions.
        unique: dict[str, dict[str, Any]] = {}
        for item in predicates:
            unique.setdefault(item["expression"], item)
        return list(unique.values())[:20]

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
        result = self.graph.invoke(
            RecoveryState(
                mode=mode,
                question=question,
                failed_sql=failed_sql,
                failure_code=failure_code,
                failure_message=failure_message,
                allowed_tables=allowed_tables,
                category="sql",
                evidence=[],
                tool_calls=[],
                requires_human_review=False,
            )
        )
        return RecoveryTrace(
            mode=mode,  # type: ignore[arg-type]
            failure_code=failure_code,
            failure_category=result["category"],
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
        return {
            "evidence": evidence,
            "tool_calls": calls,
            "requires_human_review": state["category"] == "provider",
        }


def recovery_feedback(trace: RecoveryTrace) -> str:
    evidence = "\n".join(f"- {item}" for item in trace.evidence)
    return (
        "Bounded recovery was activated after the first failure. Fix only the diagnosed "
        "violation while preserving the original question and grounded context.\n" + evidence
    )
