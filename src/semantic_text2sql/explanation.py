"""Grounded explanations for previously accepted SQL and its result."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, cast

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from semantic_text2sql.models import ConversationState, TokenUsage

_FENCE = re.compile(r"^```(?:text|markdown)?\s*(.*?)\s*```$", re.I | re.S)


class ExplanationCompleter(Protocol):
    async def complete(self, model: str, prompt: str) -> str: ...


def extract_sql_facts(sql: str, dialect: str) -> dict[str, Any]:
    """Extract inspectable SQL facts without inferring the author's business intention."""
    try:
        expression = parse_one(sql, read=dialect)
    except (ParseError, ValueError):
        return {"parse_error": "The accepted SQL could not be parsed for explanation."}

    root_select = expression if isinstance(expression, exp.Select) else expression.find(exp.Select)
    joins: list[dict[str, str | None]] = []
    for join in expression.find_all(exp.Join):
        side = str(join.args.get("side") or "").upper()
        kind = str(join.args.get("kind") or "").upper()
        join_type = " ".join(item for item in (side, kind) if item) or "INNER"
        target = join.this
        joins.append(
            {
                "type": join_type,
                "right_relation": target.sql(dialect=dialect),
                "right_alias": getattr(target, "alias_or_name", None),
                "condition": (
                    cast(exp.Expression, join.args["on"]).sql(dialect=dialect)
                    if join.args.get("on") is not None
                    else None
                ),
            }
        )

    def clauses(node_type: type[exp.Expression]) -> list[str]:
        return [
            item.this.sql(dialect=dialect)
            for item in expression.find_all(node_type)
            if item.this is not None
        ]

    groups = [
        item.sql(dialect=dialect)
        for group in expression.find_all(exp.Group)
        for item in group.expressions
    ]
    orders = [
        item.sql(dialect=dialect)
        for order in expression.find_all(exp.Order)
        for item in order.expressions
    ]
    limits = [item.expression.sql(dialect=dialect) for item in expression.find_all(exp.Limit)]
    return {
        "tables": list(dict.fromkeys(item.name for item in expression.find_all(exp.Table))),
        "ctes": list(
            dict.fromkeys(item.alias_or_name for item in expression.find_all(exp.CTE))
        ),
        "outputs": (
            [item.sql(dialect=dialect) for item in root_select.expressions]
            if root_select is not None
            else []
        ),
        "joins": joins,
        "filters": clauses(exp.Where),
        "group_by": groups,
        "having": clauses(exp.Having),
        "order_by": orders,
        "limits": limits,
        "distinct": bool(root_select and root_select.args.get("distinct")),
    }


async def explain_turn_detailed(
    completer: ExplanationCompleter,
    *,
    model: str,
    operation: str,
    client_message: str,
    state: ConversationState,
    dialect: str,
) -> tuple[str, TokenUsage]:
    """Ask the selected model to verbalize deterministic facts without changing SQL."""
    facts = extract_sql_facts(state.last_sql or state.last_failed_sql or "", dialect)
    context = {
        key: state.last_model_context[key]
        for key in ("tables", "relationships", "approved_formulas")
        if state.last_model_context.get(key)
    }
    prompt = _explanation_prompt(
        operation=operation,
        client_message=client_message,
        state=state,
        dialect=dialect,
        sql_facts=facts,
        relationship_context=context,
    )
    detailed = getattr(completer, "complete_detailed", None)
    if callable(detailed):
        raw, usage = await cast(Any, detailed)(model, prompt)
    else:
        raw = await completer.complete(model, prompt)
        usage = TokenUsage()
    value = raw.strip()
    match = _FENCE.fullmatch(value)
    explanation = (match.group(1) if match else value).strip()
    if not explanation:
        raise ValueError("Explanation model returned no text.")
    if _contradicts_sql_facts(explanation, client_message, facts):
        raise ValueError("Explanation contradicted the parsed SQL structure.")
    return explanation, usage


def _contradicts_sql_facts(
    explanation: str,
    client_message: str,
    facts: dict[str, Any],
) -> bool:
    """Reject high-confidence contradictions about explicit SQL structure."""
    requested = client_message.upper()
    rendered = explanation.upper()
    joins = cast(list[dict[str, str | None]], facts.get("joins", []))
    actual_types = {str(item.get("type") or "INNER").upper() for item in joins}

    for join_type in ("LEFT", "RIGHT", "FULL", "CROSS", "INNER"):
        if f"{join_type} JOIN" not in requested:
            continue
        exists = any(join_type in actual for actual in actual_types)
        denies = bool(
            re.search(
                rf"(?:NO|NOT|DOESN'T|DOES NOT|ISN'T|IS NOT|WITHOUT)\s+"
                rf"(?:AN?\s+)?{join_type}\s+JOIN",
                rendered,
            )
            or re.search(rf"THERE\s+IS\s+NO\s+{join_type}\s+JOIN", rendered)
        )
        claims = f"{join_type} JOIN" in rendered and not denies
        if (exists and denies) or (not exists and claims):
            return True
    return False


def deterministic_explanation(operation: str, state: ConversationState, dialect: str) -> str:
    """Provide an evidence-only fallback when the selected model cannot explain."""
    facts = extract_sql_facts(state.last_sql or state.last_failed_sql or "", dialect)
    if operation == "EXPLAIN_FAILURE":
        return state.last_failure or "No recorded failure is available in this conversation."
    if operation == "EXPLAIN_RESULT":
        columns = ", ".join(state.last_columns) or "unknown"
        truncation = " The displayed result was truncated." if state.last_truncated else ""
        return (
            f"The accepted query returned {state.last_row_count or 0} displayed rows with "
            f"columns: {columns}.{truncation} Execution alone does not prove business correctness."
        )
    if operation == "EXPLAIN_INTERPRETATION":
        return (
            f"The original request was: {state.root_question} The current resolved request is: "
            f"{state.resolved_question}"
        )
    if operation == "EXPLAIN_CONTEXT":
        tables = ", ".join(facts.get("tables", [])) or ", ".join(state.approved_tables) or "none"
        return f"The accepted SQL references these tables: {tables}."
    joins = cast(list[dict[str, str | None]], facts.get("joins", []))
    if joins:
        details: list[str] = []
        for item in joins:
            rendered_join = (
                f"{item['type']} JOIN {item['right_relation']} "
                f"ON {item['condition'] or 'no condition'}"
            )
            if "LEFT" in str(item["type"]):
                rendered_join += (
                    ". It preserves every row from the left side and returns NULL for "
                    "right-side columns when no match exists"
                )
            details.append(rendered_join)
        return (
            f"The accepted SQL uses {'; '.join(details)}. "
            "This states the join's mechanical behavior; "
            "the available facts do not prove why that join type was intended."
        )
    return "The accepted SQL contains no explicit JOIN operation."


def _explanation_prompt(
    *,
    operation: str,
    client_message: str,
    state: ConversationState,
    dialect: str,
    sql_facts: dict[str, Any],
    relationship_context: dict[str, Any],
) -> str:
    return f"""You explain an already accepted SQL query to a client.
Do not generate, modify, optimize, or execute SQL. Use only the supplied information.
Never invent a business definition or claim that successful execution proves correctness.
Distinguish verified SQL behavior from an inferred rationale. If the reason for a design choice
cannot be proven, explain its mechanical effect and say that its necessity is uncertain.
Answer only what the client asked, in concise plain language.

EXPLANATION_TYPE: {operation}
CLIENT_REQUEST: {client_message}
ORIGINAL_QUESTION: {state.root_question}
RESOLVED_REQUEST: {state.resolved_question}
DIALECT: {dialect}

ACCEPTED_SQL:
{state.last_sql or "None"}

VERIFIED_SQL_FACTS:
{json.dumps(sql_facts, separators=(",", ":"))}

VERIFIED_SCHEMA_CONTEXT:
{json.dumps(relationship_context, separators=(",", ":"))}

RESULT_SUMMARY:
{json.dumps(
    {
        "columns": state.last_columns,
        "row_count": state.last_row_count,
        "truncated": state.last_truncated,
    },
    separators=(",", ":"),
)}

RECORDED_FAILURE:
{state.last_failure or "None"}
"""
