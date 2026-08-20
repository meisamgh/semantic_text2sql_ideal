"""Legacy compatibility helpers; the default runtime has no semantic-planner LLM call."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, cast

from pydantic import ValidationError

from semantic_text2sql.context_planner import (
    reconcile_context_contract,
    verify_context_request,
)
from semantic_text2sql.models import (
    ContextRequest,
    DatabaseProfile,
    PlannerAggregation,
    PlannerRanking,
    SchemaInfo,
    SemanticContract,
    SemanticFilter,
    SemanticPlan,
    TokenUsage,
)


class SemanticPlannerCompleter(Protocol):
    async def complete(self, model: str, prompt: str) -> str: ...


async def plan_semantics_detailed(
    completer: SemanticPlannerCompleter,
    model: str,
    question: str,
    grounded_context: dict[str, Any],
) -> tuple[SemanticPlan, TokenUsage]:
    """Ask the selected model for semantics only; SQL and historical examples are forbidden."""
    shape = SemanticPlan.model_json_schema()
    prompt = f"""You are the semantic-planning stage of a Text-to-SQL system.
Interpret the question using only the verified grounded context. Return JSON only and match the
required schema exactly. Describe the requested operation, outputs, filters, measures,
aggregation stages, explicit final operations, output grain, ranking, temporal logic, business
concepts, and concise logical steps. If the requested output is derived from intermediate metrics,
represent that derivation in final_operations rather than leaving it only in prose. Do not generate
SQL. Do not invent tables, columns, formulas, literals, or relationships.
Use final_operations only for arithmetic combinations with operator add, subtract, divide, or
multiply. Express top/lowest selection only through ranking; never encode argmax or argmin as a
final_operation. Do not restate a row-level approved glossary formula in final_operations; reference
its formula ID as an aggregation input instead. Use approved glossary formulas exactly when their
business concepts are requested.
Explicit user literals and approved formulas in the grounded context are authoritative.

QUESTION:
{question}

VERIFIED GROUNDED CONTEXT:
{json.dumps(grounded_context, separators=(",", ":"))}

REQUIRED JSON SCHEMA:
{json.dumps(shape, separators=(",", ":"))}
"""
    total_usage = TokenUsage()
    validation_feedback = ""
    for attempt in range(2):
        attempt_prompt = prompt + validation_feedback
        detailed = getattr(completer, "complete_detailed", None)
        if callable(detailed):
            raw, usage = await cast(Any, detailed)(model, attempt_prompt)
        else:
            raw = await completer.complete(model, attempt_prompt)
            usage = TokenUsage()
        total_usage = _merge_usage(total_usage, usage)
        try:
            return SemanticPlan.model_validate_json(_json_object(raw)), total_usage
        except (ValidationError, ValueError) as exc:
            if attempt == 1:
                raise ValueError("Semantic planner returned an invalid SemanticPlan.") from exc
            validation_feedback = f"""

YOUR PREVIOUS JSON WAS REJECTED.
Validation error: {exc}
Return one corrected JSON object only. Preserve the intended semantics, use only enum values from
the schema, and do not add SQL or commentary.
"""
    raise AssertionError("unreachable")


def _merge_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=_merge_known(left.input_tokens, right.input_tokens),
        output_tokens=_merge_known(left.output_tokens, right.output_tokens),
        cache_read_tokens=_merge_known(left.cache_read_tokens, right.cache_read_tokens),
        cache_creation_tokens=_merge_known(left.cache_creation_tokens, right.cache_creation_tokens),
    )


def _merge_known(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def verify_semantic_plan(
    plan: SemanticPlan,
    selection: ContextRequest,
    schema: SchemaInfo,
    contract: SemanticContract,
    profile: DatabaseProfile | None,
    question: str,
) -> tuple[SemanticPlan, SemanticContract]:
    """Ground plan identifiers and preserve deterministic facts and explicit user literals."""
    plan = _ground_identifiers(plan, schema)
    verified_plan = _verify_final_operations(plan)
    semantic_request = ContextRequest(
        outputs=verified_plan.outputs,
        tables=selection.tables,
        columns=selection.columns,
        filters=verified_plan.filters,
        measures=verified_plan.measures,
        aggregations=verified_plan.aggregations,
        group_by=verified_plan.group_by,
        ranking=verified_plan.ranking,
        temporal_operations=verified_plan.temporal_operations,
        business_concepts=list(
            dict.fromkeys([*selection.business_concepts, *verified_plan.business_concepts])
        ),
        metadata_requirements=selection.metadata_requirements,
    )
    grounded = verify_context_request(semantic_request, schema, contract, profile)
    reconciled = reconcile_context_contract(contract, grounded)
    final_names = [item.name for item in verified_plan.final_operations]
    outputs = verified_plan.outputs or final_names
    reconciled = reconciled.model_copy(
        update={
            "outputs": outputs,
            "named_outputs": outputs,
            "output_operations": verified_plan.final_operations,
            "grain": verified_plan.group_by,
        }
    )
    return verified_plan, reconciled


def _ground_identifiers(plan: SemanticPlan, schema: SchemaInfo) -> SemanticPlan:
    by_name: dict[str, list[str]] = {}
    live = {f"{table.name}.{column.name}" for table in schema.tables for column in table.columns}
    for qualified in live:
        by_name.setdefault(qualified.rsplit(".", 1)[-1].casefold(), []).append(qualified)

    def qualify(value: str) -> str:
        if value in live or "." in value:
            return value
        matches = by_name.get(value.casefold(), [])
        return matches[0] if len(matches) == 1 else value

    return plan.model_copy(
        update={
            "filters": [
                SemanticFilter(
                    operand=qualify(item.operand),
                    operator=item.operator,
                    value=item.value,
                )
                for item in plan.filters
            ],
            "measures": [qualify(item) for item in plan.measures],
            "aggregations": [
                PlannerAggregation(
                    function=item.function,
                    input=qualify(item.input),
                    output=item.output,
                    group_by=[qualify(value) for value in item.group_by],
                )
                for item in plan.aggregations
            ],
            "group_by": [qualify(item) for item in plan.group_by],
            "ranking": [
                PlannerRanking(
                    metric=qualify(item.metric),
                    direction=item.direction,
                    limit=item.limit,
                )
                for item in plan.ranking
            ],
        }
    )


def _verify_final_operations(plan: SemanticPlan) -> SemanticPlan:
    available = {item.output for item in plan.aggregations if item.output}
    for operation in plan.final_operations:
        missing = {operation.left, operation.right} - available
        if missing:
            raise ValueError(
                "Semantic plan final operation references undefined metrics: "
                + ", ".join(sorted(missing))
            )
    return plan


def _json_object(raw: str) -> str:
    value = raw.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Semantic planner returned no JSON object.")
    return value[start : end + 1]
