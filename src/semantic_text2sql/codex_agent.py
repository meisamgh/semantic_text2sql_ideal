"""Bounded Claude investigation agent with auditable actions and hidden-gold isolation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.historical import HistoricalQueryStore, format_examples
from semantic_text2sql.linker import compact_profile_context, select_schema
from semantic_text2sql.llm import AgentRouterClaudeModel, ModelError
from semantic_text2sql.profiling import ProfileStore
from semantic_text2sql.validator import clean_model_sql, normalize_sql, repair_context, validate_sql


class AgentBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_ir: dict[str, Any]
    logical_plan: dict[str, Any]
    sql: str = Field(min_length=1)


class CodexClaudeAgent:
    def __init__(
        self,
        databases: DatabaseRegistry,
        profiles: ProfileStore,
        history: HistoricalQueryStore,
        claude: AgentRouterClaudeModel,
    ) -> None:
        self.databases = databases
        self.profiles = profiles
        self.history = history
        self.claude = claude

    async def solve(
        self,
        *,
        run_id: str,
        dataset_index: int,
        db_id: str,
        question: str,
        evidence: str | None,
        model: str,
        max_repairs: int = 2,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        started = perf_counter()
        trace: list[dict[str, Any]] = []

        def record(action: str, reason: str, arguments: Any, result: Any) -> None:
            trace.append(
                {
                    "run_id": run_id,
                    "dataset_index": dataset_index,
                    "step": len(trace) + 1,
                    "action": action,
                    "reason": reason,
                    "arguments": arguments,
                    "result_summary": result,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        record(
            "interpret_question",
            "Freeze the target question and trusted evidence before retrieval.",
            {"question": question},
            {"evidence_present": bool(evidence)},
        )
        examples = self.history.search(question, db_id, top_k=3)
        record(
            "search_similar_queries",
            "Retrieve same-database structural patterns from approved history.",
            {"db_id": db_id, "top_k": 3},
            [{"question": item.question, "score": item.score} for item in examples],
        )
        full_schema = self.databases.inspect(db_id)
        profile = self.profiles.load("sqlite", db_id)
        reduced, selection = select_schema(question, evidence, full_schema, profile)
        record(
            "search_tables",
            "Rank tables, then verify them against the live schema.",
            {"query": question},
            {"tables": selection.tables, "scores": selection.table_scores},
        )
        record(
            "search_columns",
            "Rank columns only inside verified tables and preserve join keys.",
            {"tables": selection.tables},
            selection.columns,
        )
        relationships = [item.model_dump() for item in reduced.relationships]
        record(
            "get_relationships",
            "Inspect live relationships among selected tables.",
            {"tables": selection.tables},
            relationships,
        )
        profile_context = compact_profile_context(reduced, profile, question, evidence)
        record(
            "profile_columns",
            "Attach offline profiles only for retrieved columns.",
            {"profile_available": profile is not None},
            {"context_characters": len(profile_context)},
        )

        schema_text = "\n".join(
            f"{table.name}("
            + ", ".join(
                f"{column.name} {column.data_type}" + (" PRIMARY KEY" if column.primary_key else "")
                for column in table.columns
            )
            + ")"
            for table in reduced.tables
        )
        prompt = _prompt(
            question,
            evidence,
            schema_text,
            relationships,
            profile_context,
            format_examples(examples),
        )
        rejected: list[dict[str, str]] = []
        bundle: AgentBundle | None = None
        final_validation = None
        for attempt in range(max_repairs + 1):
            try:
                raw = await self.claude.complete(model, prompt)
                bundle = _parse_bundle(raw)
            except (ModelError, ValueError) as exc:
                record(
                    "claude_response",
                    "Claude must return the structured Semantic IR, plan, and SQL contract.",
                    {"attempt": attempt + 1},
                    {"accepted": False, "error": str(exc)},
                )
                prompt += (
                    f"\nPrevious response violated the JSON contract: {exc}. "
                    "Return valid JSON only."
                )
                continue
            record(
                "submit_semantic_ir",
                "Freeze Claude's interpretation before accepting SQL.",
                {"attempt": attempt + 1},
                bundle.semantic_ir,
            )
            record(
                "submit_plan",
                "Record the logical query plan for audit and repair.",
                {"attempt": attempt + 1},
                bundle.logical_plan,
            )
            sql = clean_model_sql(bundle.sql)
            validation = validate_sql(sql, full_schema, dialect="sqlite")
            if validation.valid:
                validation = self.databases.explain(db_id, sql, validation)
            final_validation = validation
            record(
                "validate_sql",
                "Apply full-schema SQLGlot and SQLite EXPLAIN gates.",
                {"attempt": attempt + 1, "sql": sql},
                {"valid": validation.valid, "code": validation.code},
            )
            if validation.valid:
                bundle = bundle.model_copy(update={"sql": sql})
                break
            rejected.append({"sql": normalize_sql(sql), "code": validation.code})
            prompt += (
                "\nREPAIR REQUIRED. Keep the question and frozen interpretation authoritative.\n"
                + repair_context(validation)
                + f"\nRejected SQL: {sql}\nReturn the complete JSON contract again."
            )
            bundle = None

        accepted = bool(bundle and final_validation and final_validation.valid)
        prediction = {
            "dataset_index": dataset_index,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "model": model,
            "semantic_ir": bundle.semantic_ir if bundle else None,
            "logical_plan": bundle.logical_plan if bundle else None,
            "sql": bundle.sql if accepted and bundle else None,
            "accepted": accepted,
            "validation_code": final_validation.code if final_validation else "NO_VALID_SQL",
            "selected_tables": selection.tables,
            "selected_columns": selection.columns,
            "historical_examples": [
                {"question": item.question, "score": item.score} for item in examples
            ],
            "rejected_sql": rejected,
            "action_count": len(trace),
            "latency_ms": round((perf_counter() - started) * 1_000),
        }
        return prediction, trace


def _parse_bundle(raw: str) -> AgentBundle:
    value = raw.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1])
    try:
        return AgentBundle.model_validate_json(value)
    except Exception as exc:
        raise ValueError("Claude returned malformed agent JSON.") from exc


def _prompt(
    question: str,
    evidence: str | None,
    schema: str,
    relationships: list[dict[str, Any]],
    profiles: str,
    history: str,
) -> str:
    return f"""You are a bounded SQLite Text-to-SQL investigation agent.
Return JSON only with exactly three keys: semantic_ir, logical_plan, sql.
semantic_ir and logical_plan must be JSON objects. sql must be one SELECT or WITH...SELECT.
Do not include markdown or explanation outside JSON. Do not invent requirements, identifiers, or
values. Historical SQL is a non-authoritative structural pattern. Preserve requested output count,
order, filters, time logic, aggregation, ranking, and grain. Use only the retrieved live schema.

Question:
{question}

Trusted evidence:
{evidence or "None"}

Retrieved schema:
{schema}

Verified relationships:
{json.dumps(relationships, indent=2)}

Column profiles:
{profiles}

Approved historical patterns:
{history or "None"}
"""
