"""Bounded LangGraph recovery and database-evidence tools."""

from __future__ import annotations

import asyncio
import calendar
import json
import re
from time import monotonic, perf_counter
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
    TokenUsage,
)
from semantic_text2sql.postgres import PostgresRegistry
from semantic_text2sql.validator import validate_sql

MAX_RECOVERY_TOOL_CALLS = 3
MAX_RECOVERY_MODEL_CALLS = 4


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
        max_database_probes: int = 8,
        max_recovery_seconds: float = 8.0,
    ) -> None:
        self.database = database
        self.db_id = db_id
        self.schema = schema
        self.profile = profile
        self.max_rows = min(max_rows, 20)
        self.max_database_probes = min(max(1, max_database_probes), 8)
        self.max_recovery_seconds = min(max(0.1, max_recovery_seconds), 8.0)
        self._deadline: float | None = None
        self.budget_exhausted = False

    def begin_recovery(self) -> None:
        self._deadline = monotonic() + self.max_recovery_seconds
        self.budget_exhausted = False

    def remaining_seconds(self) -> float:
        if self._deadline is None:
            return self.max_recovery_seconds
        return max(0.0, self._deadline - monotonic())

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

    def inspect_schema_rich(
        self, tables: list[str], columns: dict[str, list[str]] | None = None
    ) -> dict[str, Any]:
        """One bounded metadata tool for schema, profiles, grain and relationships."""
        allowed = set(tables)
        requested = columns or {}
        profile_columns = (
            {(item.table, item.column): item for item in self.profile.columns}
            if self.profile
            else {}
        )
        table_profiles = {item.table: item for item in self.profile.tables} if self.profile else {}
        result: dict[str, Any] = {"tables": {}, "relationships": []}
        for table in self.schema.tables:
            if table.name not in allowed:
                continue
            selected = set(requested.get(table.name, []))
            live_columns = (
                table.columns
                if not selected
                else [item for item in table.columns if item.name in selected]
            )
            table_profile = table_profiles.get(table.name)
            result["tables"][table.name] = {
                "grain": table_profile.grain if table_profile else None,
                "primary_key": [item.name for item in table.columns if item.primary_key],
                "columns": {
                    item.name: _rich_column_metadata(
                        item.data_type,
                        item.primary_key,
                        profile_columns.get((table.name, item.name)),
                    )
                    for item in live_columns
                },
            }
        result["relationships"] = [
            item.model_dump()
            for item in self.schema.relationships
            if item.from_table in allowed and item.to_table in allowed
        ]
        return result

    def query_database(self, sql: str, *, allowed_tables: list[str]) -> dict[str, Any]:
        """Execute one SQLGlot-checked, allowlisted, bounded read-only SELECT probe."""
        validation = validate_sql(
            sql,
            self.schema,
            dialect=self.schema.dialect,
            allowed_tables={item.casefold() for item in allowed_tables},
        )
        if not validation.valid:
            raise ValueError(f"{validation.code}: {validation.message}")
        tree = parse_one(sql, dialect=self.schema.dialect)
        remaining = self.remaining_seconds()
        if remaining <= 0:
            self.budget_exhausted = True
            raise TimeoutError("The recovery database budget is exhausted.")
        columns, rows, truncated = self.database.execute(
            self.db_id,
            tree.sql(dialect=self.schema.dialect),
            max_rows=self.max_rows,
            timeout_seconds=min(2.0, remaining),
        )
        return {"columns": columns, "rows": rows, "truncated": truncated}

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

    def inspect_temporal_coverage(self, tables: list[str]) -> list[dict[str, Any]]:
        """Return verified storage format and bounds for date columns in allowed tables."""
        if self.profile is None:
            return []
        allowed = set(tables)
        return [
            {
                "column": f"{item.table}.{item.column}",
                "observed_format": item.observed_format,
                "minimum": item.minimum,
                "maximum": item.maximum,
                "range_exact": item.range_exact,
            }
            for item in self.profile.columns
            if item.table in allowed
            and item.semantic_type in {"date", "datetime"}
            and (item.observed_format or item.minimum or item.maximum)
        ][:10]

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
        columns, rows, truncated = self.database.execute(self.db_id, query, max_rows=self.max_rows)
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

    def probe_filter_counts(self, sql: str, *, allowed_tables: list[str]) -> list[dict[str, Any]]:
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
        all_predicates = _split_and(select.args["where"].this)
        predicates = all_predicates[: self.max_database_probes]
        if len(all_predicates) > self.max_database_probes:
            self.budget_exhausted = True
        checks: list[dict[str, Any]] = []
        for predicate in predicates:
            remaining = self.remaining_seconds()
            if remaining <= 0:
                self.budget_exhausted = True
                break
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
                    self.db_id,
                    probe_sql,
                    max_rows=1,
                    timeout_seconds=min(2.0, remaining),
                )
                count = int(rows[0][0]) if rows else 0
                subject_kind = _filter_subject_kind(predicate)
                checks.append(
                    {
                        "filter": predicate.sql(dialect=self.schema.dialect),
                        "plain_language": _plain_filter(predicate),
                        "subject_kind": subject_kind,
                        "no_match_explanation": (
                            _no_match_explanation(predicate, subject_kind) if count == 0 else None
                        ),
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
        reset_budget: bool = True,
    ) -> RecoveryTrace:
        started = perf_counter()
        if reset_budget:
            self.tools.begin_recovery()
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
                max_tool_calls=MAX_RECOVERY_TOOL_CALLS,
                max_database_probes=self.tools.max_database_probes,
                max_recovery_ms=round(self.tools.max_recovery_seconds * 1_000),
                budget_exhausted=self.tools.budget_exhausted,
                estimated_llm_cost_usd=0.0,
            ),
            evidence=result["evidence"],
            tool_calls=[RecoveryToolCall.model_validate(item) for item in result["tool_calls"]],
            requires_human_review=result["requires_human_review"],
        )

    async def ainvestigate(
        self,
        *,
        question: str,
        failed_sql: str,
        failure_code: str,
        failure_message: str,
        allowed_tables: list[str],
        mode: str = "FAILURE",
        completer: Any | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> RecoveryTrace:
        """Run one bounded agent with two tools; fall back to deterministic recovery."""

        async def fallback() -> RecoveryTrace:
            return await asyncio.to_thread(
                self.investigate,
                question=question,
                failed_sql=failed_sql,
                failure_code=failure_code,
                failure_message=failure_message,
                allowed_tables=allowed_tables,
                mode=mode,
                reset_budget=False,
            )

        self.tools.begin_recovery()
        category = _failure_category(mode, failure_code, failure_message)
        if completer is None or model is None or category in {"provider", "safety"}:
            return await fallback()
        observations: list[dict[str, Any]] = []
        calls: list[RecoveryToolCall] = []
        total_usage = TokenUsage(input_tokens=0, output_tokens=0)
        started = perf_counter()
        try:
            detailed = getattr(completer, "complete_detailed", None)
            if not callable(detailed):
                return await fallback()
            for model_call in range(MAX_RECOVERY_MODEL_CALLS):
                prompt = _agent_recovery_prompt(
                    question,
                    failed_sql,
                    failure_code,
                    failure_message,
                    allowed_tables,
                    observations,
                )
                try:
                    raw, usage = await asyncio.wait_for(
                        detailed(provider, model, prompt),
                        timeout=max(0.01, self.tools.remaining_seconds()),
                    )
                except TypeError:
                    raw, usage = await asyncio.wait_for(
                        detailed(model, prompt),
                        timeout=max(0.01, self.tools.remaining_seconds()),
                    )
                if isinstance(usage, TokenUsage):
                    total_usage = TokenUsage(
                        input_tokens=(total_usage.input_tokens or 0) + (usage.input_tokens or 0),
                        output_tokens=(total_usage.output_tokens or 0) + (usage.output_tokens or 0),
                    )
                payload = json.loads(_strip_json_fence(raw))
                action = str(payload.get("action") or "").upper()
                if action in {"REPAIR", "INFORM", "ESCALATE"}:
                    diagnosis = str(payload.get("diagnosis") or "").strip()
                    if not diagnosis:
                        return await fallback()
                    if (
                        category in {"data_grounding", "correctness"}
                        and action in {"INFORM", "ESCALATE"}
                        and not observations
                    ):
                        return await fallback()
                    base = RecoveryTrace(
                        mode=mode,  # type: ignore[arg-type]
                        failure_code=failure_code,
                        failure_category=category,
                    )
                    return base.model_copy(
                        update={
                            "agent_diagnosis": diagnosis[:1_000],
                            "agent_confidence": max(
                                0.0, min(float(payload.get("confidence", 0.0)), 1.0)
                            ),
                            "agent_action": action,
                            "repair_instruction": (
                                str(payload.get("repair_instruction") or "")[:1_000] or None
                            ),
                            "diagnosis_summary": diagnosis[:1_000],
                            "evidence": [
                                f"{item['tool']}: {item['result']}" for item in observations
                            ],
                            "tool_calls": calls,
                            "requires_human_review": action == "ESCALATE",
                            "usage": RecoveryUsage(
                                llm_calls=model_call + 1,
                                token_usage=total_usage,
                                database_probe_count=sum(
                                    item["tool"] == "query_database" for item in observations
                                ),
                                latency_ms=round((perf_counter() - started) * 1_000),
                                max_tool_calls=MAX_RECOVERY_TOOL_CALLS,
                                max_database_probes=self.tools.max_database_probes,
                                max_recovery_ms=round(self.tools.max_recovery_seconds * 1_000),
                                budget_exhausted=self.tools.budget_exhausted,
                                estimated_llm_cost_usd=None,
                            ),
                        }
                    )
                if action != "CALL_TOOL" or len(calls) >= MAX_RECOVERY_TOOL_CALLS:
                    return await fallback()
                tool = str(payload.get("tool") or "")
                arguments = payload.get("arguments") or {}
                if tool == "inspect_schema":
                    requested_tables = [
                        item
                        for item in arguments.get("tables", allowed_tables)
                        if item in allowed_tables
                    ]
                    result = self.tools.inspect_schema_rich(
                        requested_tables, arguments.get("columns")
                    )
                elif tool == "query_database":
                    result = await asyncio.to_thread(
                        self.tools.query_database,
                        str(arguments.get("sql") or ""),
                        allowed_tables=allowed_tables,
                    )
                else:
                    return await fallback()
                purpose = str(payload.get("purpose") or "Gather verified recovery evidence.")
                observations.append({"tool": tool, "result": result})
                calls.append(
                    RecoveryToolCall(
                        tool=tool,
                        purpose=purpose[:500],
                        result_summary=f"Returned bounded {tool} evidence.",
                    )
                )
        except Exception:
            return await fallback()
        return await fallback()

    def _classify(self, state: RecoveryState) -> dict[str, Any]:
        return {
            "category": _failure_category(
                state["mode"], state["failure_code"], state["failure_message"]
            )
        }

    def _gather(self, state: RecoveryState) -> dict[str, Any]:
        evidence = [f"Failure {state['failure_code']}: {state['failure_message']}"]
        calls: list[dict[str, str]] = []
        filters: list[dict[str, Any]] = []
        if state["category"] in {"schema", "data_grounding", "correctness"} and len(calls) < 6:
            schema = self.tools.inspect_schema(state["allowed_tables"])
            evidence.append(f"Verified schema: {schema}")
            calls.append(
                {
                    "tool": "inspect_schema",
                    "purpose": "Verify referenced schema after the failed attempt.",
                    "result_summary": f"Inspected {len(schema['tables'])} allowed tables.",
                }
            )
        if state["category"] in {"data_grounding", "correctness"} and len(calls) < 6:
            profiles = self.tools.inspect_profiles(state["allowed_tables"])
            evidence.append(f"Relevant column profiles: {profiles}")
            calls.append(
                {
                    "tool": "inspect_column",
                    "purpose": "Verify date, value, and NULL representations.",
                    "result_summary": f"Returned {len(profiles)} bounded profile entries.",
                }
            )
        if state["category"] in {"data_grounding", "correctness"} and len(calls) < 6:
            temporal = self.tools.inspect_temporal_coverage(state["allowed_tables"])
            if temporal:
                evidence.append(f"Verified temporal coverage: {temporal}")
                calls.append(
                    {
                        "tool": "inspect_temporal_coverage",
                        "purpose": (
                            "Compare requested periods with observed date formats and bounds."
                        ),
                        "result_summary": f"Returned coverage for {len(temporal)} date columns.",
                    }
                )
        if (
            state["mode"] in {"ZERO_RESULT", "NULL_RESULT", "FILTER", "CORRECTNESS"}
            and len(calls) < 6
        ):
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
        if state["mode"] in {"ZERO_RESULT", "NULL_RESULT", "FILTER"} and len(calls) < 6:
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
    instruction = (
        f"\nAgent repair instruction: {trace.repair_instruction}"
        if trace.agent_action == "REPAIR" and trace.repair_instruction
        else ""
    )
    return (
        "Bounded recovery was activated after the first failure. Fix only the diagnosed "
        "violation while preserving the original question and grounded context.\n"
        + evidence
        + instruction
    )


def _failure_category(mode: str, failure_code: str, failure_message: str) -> str:
    if mode in {"ZERO_RESULT", "NULL_RESULT", "FILTER"}:
        return "data_grounding"
    if mode == "CORRECTNESS":
        return "correctness"
    text = f"{failure_code} {failure_message}".casefold()
    if any(token in text for token in ("column", "table", "alias", "schema")):
        return "schema"
    if any(token in text for token in ("date", "type", "value", "null", "empty")):
        return "data_grounding"
    if any(token in text for token in ("timeout", "rate limit", "unavailable")):
        return "provider"
    if "safety" in text or "write" in text:
        return "safety"
    return "sql"


def _agent_recovery_prompt(
    question: str,
    failed_sql: str,
    failure_code: str,
    failure_message: str,
    allowed_tables: list[str],
    observations: list[dict[str, Any]],
) -> str:
    return f"""You are one bounded Text-to-SQL recovery agent with exactly two tools.
Return one JSON object only, using one of these shapes:
{{"action":"CALL_TOOL","tool":"inspect_schema","arguments":{{"tables":["name"],
"columns":{{"name":["column"]}}}},"purpose":"short reason"}}
{{"action":"CALL_TOOL","tool":"query_database","arguments":{{"sql":"SELECT ..."}},
"purpose":"short reason"}}
{{"action":"INFORM","diagnosis":"plain-language fact","confidence":0.0}}
{{"action":"REPAIR","diagnosis":"plain-language cause","confidence":0.0,
"repair_instruction":"focused instruction for the SQL generator"}}
{{"action":"ESCALATE","diagnosis":"plain-language uncertainty","confidence":0.0}}

Rules:
- Use inspect_schema for types, grain, keys, relationships, NULLs and temporal coverage.
- Schema/profile examples are advisory and cannot prove that a filter value exists or is absent.
- Use query_database whenever the diagnosis depends on actual categorical or identifier values.
- Prefer one tool at a time and stop as soon as evidence is sufficient.
- Before a final action, check every independent explicit filter that could explain an empty or
  NULL result. Report all confirmed issues, not only the first one discovered.
- For temporal filters, inspect the date format and coverage. For identifier and categorical
  filters, verify whether the requested value exists. Distinguish an individually invalid filter
  from a valid set of filters whose combination has no rows.
- Never silently change an explicit value or date and never expose chain-of-thought.
- Final diagnosis must be at most three short sentences for a non-technical stakeholder.
- Separate verified facts from uncertainty. Do not call a value a typo unless you say you cannot
  determine the intended replacement.

Question: {question}
Failure: {failure_code}: {failure_message}
SQL: {failed_sql or "No SQL was produced."}
Allowed tables: {json.dumps(allowed_tables)}
Tool observations: {json.dumps(observations, default=str)[:12_000]}
"""


def _rich_column_metadata(data_type: str, primary_key: bool, profile: Any | None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": data_type, "primary_key": primary_key}
    if profile is None:
        return result
    result.update(
        {
            "semantic_type": profile.semantic_type,
            "observed_nulls": profile.null_count > 0,
            "format": profile.observed_format,
            "minimum": profile.minimum if profile.semantic_type in {"date", "datetime"} else None,
            "maximum": profile.maximum if profile.semantic_type in {"date", "datetime"} else None,
            "range_exact": profile.range_exact,
        }
    )
    return {key: value for key, value in result.items() if value is not None}


def _strip_json_fence(value: str) -> str:
    cleaned = value.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.I | re.S)
    return match.group(1) if match else cleaned


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
        item.get("no_match_explanation") or item.get("plain_language", item["filter"])
        for item in checks
        if item.get("status") == "NO_MATCH"
    ]
    matched = [item for item in checks if item.get("status") == "MATCH"]
    if no_match:
        return "FILTER_NO_MATCH", " ".join(no_match)
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
    values = [_human_literal(item) for item in literals]
    value = values[0] if values else "the requested value"
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
        return f"{subject} must be one of {', '.join(values) or 'the requested values'}"
    if isinstance(predicate, exp.Between) and len(values) >= 2:
        return f"{subject} must be from {values[0]} through {values[1]}"
    if isinstance(predicate, (exp.Like, exp.ILike)):
        return f"{subject} must match {value}"
    if isinstance(predicate, exp.Is):
        return f"{subject} must have the requested missing-value state"
    return f"The condition on {subject} must be satisfied"


def _filter_subject_kind(predicate: exp.Expression) -> str:
    column = next(predicate.find_all(exp.Column), None)
    name = column.name.casefold() if column is not None else ""
    if name.endswith("id") or name.endswith("_id"):
        return "identifier"
    if any(token in name for token in ("date", "year", "month", "time")):
        return "date"
    if isinstance(predicate, (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between)):
        return "numeric_or_date"
    return "categorical_or_text"


def _no_match_explanation(predicate: exp.Expression, subject_kind: str) -> str:
    column = next(predicate.find_all(exp.Column), None)
    subject = column.name if column is not None else "selected field"
    literal = next(predicate.find_all(exp.Literal), None)
    value = _human_literal(str(literal.this)) if literal is not None else "the requested value"
    if subject_kind == "identifier":
        entity = subject[:-2].replace("_", " ").strip() or "record"
        return (
            f"There is no {entity.lower()} with ID {value} in this database. "
            f"Because the selected {entity.lower()} is unavailable, the requested result "
            "cannot be calculated."
        )
    if subject_kind == "date":
        literals = [_human_literal(str(item.this)) for item in predicate.find_all(exp.Literal)]
        if isinstance(predicate, exp.Between) and len(literals) >= 2:
            return f"No records were found from {literals[0]} through {literals[1]}."
        return f"No records were found for the requested date or period: {value}."
    if subject_kind == "numeric_or_date":
        return f"No records satisfy the requested condition on {subject} using {value}."
    return f"The value {value} is not used by any matching record in {subject}."


def _human_literal(value: str) -> str:
    compact_month = re.fullmatch(r"(?P<year>\d{4})(?P<month>\d{2})", value)
    if compact_month:
        month = int(compact_month.group("month"))
        if 1 <= month <= 12:
            return f"{calendar.month_name[month]} {compact_month.group('year')}"
    iso_date = re.fullmatch(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})", value)
    if iso_date:
        month = int(iso_date.group("month"))
        day = int(iso_date.group("day"))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{calendar.month_name[month]} {day}, {iso_date.group('year')}"
    return value
