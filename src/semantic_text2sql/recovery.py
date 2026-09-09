"""Bounded LangGraph recovery and database-evidence tools."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlglot import exp

from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.models import (
    DatabaseProfile,
    RecoveryToolCall,
    RecoveryTrace,
    SchemaInfo,
)
from semantic_text2sql.postgres import PostgresRegistry


class RecoveryState(TypedDict):
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
    ) -> RecoveryTrace:
        result = self.graph.invoke(
            RecoveryState(
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
            failure_code=failure_code,
            failure_category=result["category"],
            evidence=result["evidence"],
            tool_calls=[RecoveryToolCall.model_validate(item) for item in result["tool_calls"]],
            requires_human_review=result["requires_human_review"],
        )

    def _classify(self, state: RecoveryState) -> dict[str, Any]:
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
        if state["category"] in {"schema", "data_grounding"}:
            schema = self.tools.inspect_schema(state["allowed_tables"])
            evidence.append(f"Verified schema: {schema}")
            calls.append(
                {
                    "tool": "inspect_schema",
                    "purpose": "Verify referenced schema after the failed attempt.",
                    "result_summary": f"Inspected {len(schema['tables'])} allowed tables.",
                }
            )
        if state["category"] == "data_grounding" and len(calls) < 3:
            profiles = self.tools.inspect_profiles(state["allowed_tables"])
            evidence.append(f"Relevant column profiles: {profiles}")
            calls.append(
                {
                    "tool": "inspect_column",
                    "purpose": "Verify date, value, and NULL representations.",
                    "result_summary": f"Returned {len(profiles)} bounded profile entries.",
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
