"""Bounded, opt-in conversational agent around the governed Text-to-SQL path.

The agent chooses capabilities; it never receives a database connection or
permission to execute model-authored SQL. Analytical work delegates to chat.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from statistics import mean, median
from time import perf_counter
from typing import Any, Literal, cast, get_args
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError
from sqlglot import exp, parse_one

from semantic_text2sql.conversational_bridge import (
    ConversationalAnalyticsBridge,
    MetadataToolInput,
)
from semantic_text2sql.database import DatabaseError
from semantic_text2sql.llm import (
    ModelError,
    ProviderRequestError,
    StructuredOutputFallbackRequired,
    StructuredOutputUnsupported,
)
from semantic_text2sql.models import (
    CallLedgerEntry,
    ChatRequest,
    ChatResponse,
    ConversationState,
    PresentationState,
    RequestedChange,
    ResultPresentation,
    StrictModel,
    TokenUsage,
    TurnContext,
)
from semantic_text2sql.plotting_service import design_chart
from semantic_text2sql.result_cache import RecentResult
from semantic_text2sql.runtime import RequestBudget, RequestBudgetExceeded, use_request_budget
from semantic_text2sql.visualization import parse_intent
from semantic_text2sql.visualization_recommender import LLMChartProposal, profile_chart_result

ToolName = Literal[
    "list_tables",
    "search_schema",
    "describe_table",
    "get_relationships",
    "get_column_profile",
    "get_available_metrics",
    "get_business_context",
    "ask_data",
    "analyze_result",
    "check_sql_correctness",
    "change_visualization",
    "present_result",
    "explain_existing_result",
    "reset_context",
    "inspect_result",
    "search_conversation_memory",
    "refresh_analysis",
    "ai_dashboard_designer",
    "undo_dashboard",
]


class ToolInput(StrictModel):
    """The only model-controlled tool arguments; never executable SQL."""

    objective: str = Field(min_length=1, max_length=4_000)
    operation: Literal["NEW_QUERY", "REFINE", "CORRECTION", "COMPARE", "OPTIMIZE"] | None = None
    changes: list[RequestedChange] = Field(default_factory=list, max_length=10)
    source_run_id: str | None = Field(default=None, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


class AgentDecision(StrictModel):
    action: Literal["ANSWER", "CALL_TOOL"]
    intent: Literal[
        "DIRECT_ANSWER",
        "NEW_QUERY",
        "MODIFY_QUERY",
        "VISUALIZE",
        "EXPLORE_SCHEMA",
        "BUSINESS_CONTEXT",
        "EXPLAIN",
        "INVESTIGATE",
        "CORRECT",
        "REVIEW_SQL",
        "OPTIMIZE",
        "RESET_CONTEXT",
    ]
    tool: ToolName | None = None
    tool_arguments: ToolInput | None = None
    instruction: str | None = Field(default=None, max_length=4_000)
    answer: str | None = Field(default=None, max_length=2_000)
    finish_after_tool: bool = False
    chart_proposal: LLMChartProposal | None = None


_ACTION_VALUES: tuple[str, ...] = tuple(get_args(AgentDecision.model_fields["action"].annotation))
_INTENT_VALUES: tuple[str, ...] = tuple(get_args(AgentDecision.model_fields["intent"].annotation))
_TOOL_VALUES: tuple[str, ...] = tuple(get_args(ToolName))


_DECISION_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "action": "CALL_TOOL",
        "intent": "NEW_QUERY",
        "tool": "ask_data",
        "tool_arguments": {
            "objective": "Show total consumption by month and plot the trend",
            "operation": "NEW_QUERY",
            "changes": [],
            "source_run_id": None,
        },
        "instruction": None,
        "answer": None,
        "finish_after_tool": True,
        "chart_proposal": {
            "chart_type": "line", "x": "month", "x_components": [],
            "y": "total_consumption", "y_measures": [], "series": None,
            "reason": "Show consumption over time.",
        },
    },
    {
        "action": "CALL_TOOL",
        "intent": "VISUALIZE",
        "tool": "change_visualization",
        "tool_arguments": {
            "objective": "Plot the accepted monthly result",
            "operation": None,
            "changes": [],
            "source_run_id": None,
        },
        "instruction": None,
        "answer": None,
        "finish_after_tool": True,
        "chart_proposal": None,
    },
    {
        "action": "ANSWER",
        "intent": "DIRECT_ANSWER",
        "tool": None,
        "tool_arguments": None,
        "instruction": None,
        "answer": "Hello. I can help analyze your database.",
        "finish_after_tool": False,
        "chart_proposal": None,
    },
)


_TOOL_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["objective", "operation", "changes", "source_run_id"],
    "properties": {
        "objective": {"type": "string", "minLength": 1, "maxLength": 4_000},
        "source_run_id": {"anyOf": [
            {"type":"string", "minLength":1, "maxLength":100, "pattern":"^[A-Za-z0-9_-]+$"},
            {"type":"null"},
        ]},
        "operation": {
            "anyOf": [
                {
                    "type": "string",
                    "enum": ["NEW_QUERY", "REFINE", "CORRECTION", "COMPARE", "OPTIMIZE"],
                },
                {"type": "null"},
            ]
        },
        "changes": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["operation", "target", "previous_value", "new_value", "source"],
                "properties": {
                    "operation": {"type": "string", "enum": [
                        "ADD_DIMENSION", "REMOVE_DIMENSION", "ADD_FILTER", "REMOVE_FILTER",
                        "REPLACE_FILTER", "CHANGE_METRIC", "CHANGE_AGGREGATION", "CHANGE_GRAIN",
                        "CHANGE_PERIOD", "CHANGE_RANKING", "COMPARE", "REQUEST_VISUALIZATION",
                        "CHANGE_VISUALIZATION", "OTHER",
                    ]},
                    "target": {"type": "string", "maxLength": 200},
                    "previous_value": {"type": ["string", "integer", "number", "null"]},
                    "new_value": {"type": ["string", "integer", "number", "null"]},
                    "source": {
                        "type": "string",
                        "enum": ["model_interpretation", "user_correction"],
                    },
                },
            },
        },
    },
}


_CHART_PROPOSAL_SCHEMA = LLMChartProposal.model_json_schema()
# Native strict providers require every key; local parsing remains compatible
# with existing decisions that omit this optional proposal.
_CHART_PROPOSAL_SCHEMA["required"] = list(_CHART_PROPOSAL_SCHEMA["properties"])
for _chart_field in _CHART_PROPOSAL_SCHEMA["properties"].values():
    _chart_field.pop("default", None)

AGENT_DECISION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action", "intent", "tool", "tool_arguments", "instruction", "answer",
        "finish_after_tool", "chart_proposal",
    ],
    "properties": {
        "action": {"type": "string", "enum": list(_ACTION_VALUES)},
        "intent": {"type": "string", "enum": list(_INTENT_VALUES)},
        "tool": {
            "anyOf": [
                {"type": "string", "enum": list(_TOOL_VALUES)},
                {"type": "null"},
            ]
        },
        "tool_arguments": {"anyOf": [_TOOL_INPUT_SCHEMA, {"type": "null"}]},
        "instruction": {"type": ["string", "null"], "maxLength": 4_000},
        "answer": {"type": ["string", "null"], "maxLength": 2_000},
        "finish_after_tool": {"type": "boolean"},
        "chart_proposal": {"anyOf": [_CHART_PROPOSAL_SCHEMA, {"type": "null"}]},
    },
}


def _decision_json_line(example: dict[str, Any]) -> str:
    return json.dumps(example, separators=(",", ":"))


_DECISION_INSTRUCTIONS = (
    "Decision schema rules: Return ONLY one structured AgentDecision JSON object with keys "
    "action, intent, tool, tool_arguments, answer, finish_after_tool, chart_proposal; "
    "no prose, markdown, or code fences. "
    "action must be exactly one of: "
    + ", ".join(_ACTION_VALUES)
    + ". intent must be exactly one of: "
    + ", ".join(_INTENT_VALUES)
    + "; it is a machine identifier, never a natural-language sentence. "
    "tool must be exactly one of: "
    + ", ".join(_TOOL_VALUES)
    + " or null. "
    'For CALL_TOOL, tool_arguments contains objective, operation and typed changes; '
    'use operation NEW_QUERY only for independent questions, REFINE for follow-ups, '
    'and null for non-SQL tools. Each change names an operation and target. '
    'Preserve the previous accepted metric and filters unless a change explicitly '
    'replaces them. Otherwise use null. The legacy instruction key is accepted '
    'Set finish_after_tool=true when one governed SQL result completes the '
    'analysis, or when SQL plus the requested chart will complete it. The '
    'controller will finish only after a verified chart exists. Set false for '
    'refinement, investigation, or another analytical tool step. '
    "for compatibility, but prefer tool_arguments. Complete JSON schema:\n"
    + json.dumps(AGENT_DECISION_SCHEMA, separators=(",", ":"))
    + "\nExamples:\n"
    + "\n".join(_decision_json_line(example) for example in _DECISION_EXAMPLES)
)


def _extract_decision_object(value: str) -> str:
    """Require the complete response to be JSON; never salvage embedded prose."""
    return str(value).strip()


def _validation_issue_lines(exc: ValidationError) -> list[str]:
    """Render bounded, non-exception validation issues for model repair prompts."""
    lines: list[str] = []
    for error in exc.errors()[:4]:
        location = ".".join(str(part) for part in error["loc"]) or "decision"
        if error["type"] == "literal_error":
            ctx = error.get("ctx") or {}
            expected = error.get("expected") or ctx.get("expected")
            lines.append(
                f"{location} must be one of {expected}"
                if isinstance(expected, str)
                else f"{location} used an invalid literal value"
            )
        elif error["type"] == "missing":
            lines.append(f"{location} is required")
        elif error["type"] in {"json_invalid", "invalid_json"}:
            lines.append("response is not a valid JSON object")
        else:
            lines.append(f"{location} {error['msg']}")
    if not lines:
        lines.append("response could not be parsed as a valid AgentDecision")
    return lines


def _consistency_issue_lines(decision: AgentDecision) -> list[str]:
    issues: list[str] = []
    if decision.action == "CALL_TOOL" and decision.tool is None:
        issues.append(
            "tool is required when action is CALL_TOOL; choose one of "
            + ", ".join(_TOOL_VALUES)
        )
    if decision.action == "ANSWER" and decision.tool is not None:
        issues.append(
            f"tool must be null when action is {decision.action}; observed {decision.tool}"
        )
    if decision.action == "CALL_TOOL" and not (
        decision.tool_arguments or (decision.instruction and decision.instruction.strip())
    ):
        issues.append("CALL_TOOL requires tool_arguments.objective or instruction")
    if decision.action != "CALL_TOOL" and decision.tool_arguments is not None:
        issues.append("tool_arguments must be null unless action is CALL_TOOL")
    if decision.action == "ANSWER" and not (decision.answer or "").strip():
        issues.append(f"{decision.action} requires a nonempty answer")
    if decision.chart_proposal is not None and (
        decision.action != "CALL_TOOL"
        or decision.tool not in {"ask_data", "change_visualization", "refresh_analysis"}
    ):
        issues.append("chart_proposal requires ask_data, refresh_analysis or change_visualization")
    intent_tools = {
        "NEW_QUERY": {"ask_data"},
        "MODIFY_QUERY": {"ask_data"},
        "VISUALIZE": {
            "change_visualization", "present_result", "ai_dashboard_designer", "undo_dashboard"
        },
        "EXPLORE_SCHEMA": {"list_tables", "search_schema", "get_column_profile"},
        "BUSINESS_CONTEXT": {"get_available_metrics", "get_business_context"},
        "EXPLAIN": {"explain_existing_result", "present_result", "inspect_result",
                    "search_conversation_memory"},
        "INVESTIGATE": {"analyze_result", "ask_data", "inspect_result",
                        "search_conversation_memory", "refresh_analysis"},
        "CORRECT": {"check_sql_correctness", "ask_data"},
        "REVIEW_SQL": {"check_sql_correctness"},
        "OPTIMIZE": {"ask_data"},
        "RESET_CONTEXT": {"reset_context"},
    }
    if (
        decision.action == "CALL_TOOL"
        and decision.intent in intent_tools
        and decision.tool not in intent_tools[decision.intent]
    ):
        issues.append(f"tool {decision.tool} is incompatible with intent {decision.intent}")
    tool_input = decision.tool_arguments
    if tool_input is not None and tool_input.source_run_id and decision.tool != "ask_data":
        issues.append("source_run_id is only valid for modifying historical SQL through ask_data")
    if tool_input is not None and tool_input.source_run_id and (
        decision.intent not in {"MODIFY_QUERY", "CORRECT", "OPTIMIZE"}
        or tool_input.operation == "NEW_QUERY"
    ):
        issues.append("Historical modifications must use a modification intent, not NEW_QUERY")
    if tool_input is not None and tool_input.operation is not None:
        if decision.tool != "ask_data":
            issues.append("tool_arguments.operation is only valid for ask_data")
        elif decision.intent == "NEW_QUERY" and tool_input.operation != "NEW_QUERY":
            issues.append("NEW_QUERY intent requires NEW_QUERY operation")
        elif decision.intent == "MODIFY_QUERY" and tool_input.operation == "NEW_QUERY":
            issues.append("MODIFY_QUERY must preserve the previous accepted analysis")
    return issues


class AgentState(StrictModel):
    pending_dashboard_objective: str | None = None
    turn_context: TurnContext | None = None
    observations: list[dict[str, Any]] = Field(default_factory=list, max_length=4)
    decision: AgentDecision | None = None
    response: ChatResponse | None = None
    base_response: ChatResponse | None = None
    steps: int = 0
    finish_after_tool: bool = False
    chart_proposal: LLMChartProposal | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    ledger: list[CallLedgerEntry] = Field(default_factory=list)


class ToolObservation(StrictModel):
    tool: ToolName
    facts: dict[str, Any]
    evidence_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class _DecisionCompletion:
    """Outcome of one decision-fetch round; failures keep usage and ledger entries."""

    decision: AgentDecision | None
    usage: TokenUsage
    ledger: list[CallLedgerEntry]
    issues: list[str]
    message: str | None = None


CACHED_RESULT_TOOLS: frozenset[ToolName] = frozenset({
    "analyze_result", "present_result", "ai_dashboard_designer", "undo_dashboard"
})
PRIOR_SQL_TOOLS: frozenset[ToolName] = frozenset(
    {"change_visualization", "check_sql_correctness"}
)
ANALYTICAL_CHANGE_OPERATIONS = frozenset({
    "ADD_DIMENSION", "REMOVE_DIMENSION", "ADD_FILTER", "REMOVE_FILTER",
    "REPLACE_FILTER", "CHANGE_METRIC", "CHANGE_AGGREGATION", "CHANGE_GRAIN",
    "CHANGE_PERIOD", "CHANGE_RANKING", "COMPARE",
})
CHART_FOLLOWUPS = frozenset({"plot it", "chart it", "graph it"})


def _chart_followup(message: str) -> bool:
    return message.strip().casefold().rstrip(".!?") in CHART_FOLLOWUPS


def _result_presentation_request(message: str) -> bool:
    return bool(re.fullmatch(
        r"(?:show|summarize|describe|explain)\s+(?:(?:the|those|these|my)\s+)?"
        r"(?:results?|answer|data)(?:\s+again)?[.!?]?",
        message.strip(), re.I,
    ))


def _database_question(message: str) -> bool:
    """Conservative guard against unsupported model-only database answers."""
    return bool(re.search(
        r"\b(?:how many|how much|total|sum|average|count|top|highest|lowest|"
        r"most|least|revenue|sales|spending|consumption|customers?|orders?|"
        r"transactions?|profit|metric|kpi)\b",
        message, re.I,
    ))


def _route_content_terms(value: str) -> set[str]:
    """Reject model-proposed requery objectives with new business concepts."""
    ignored = {"a", "an", "and", "as", "by", "for", "in", "is", "of", "the",
               "to", "what", "which", "show", "calculate", "give", "me"}
    terms = set()
    for raw in re.findall(r"[a-z]+", value.casefold()):
        term = {"monthly": "month", "months": "month", "yearly": "year",
                "years": "year", "daily": "day", "days": "day"}.get(raw, raw)
        if term not in ignored:
            terms.add(term)
    return terms


class ConversationalAnalyticsAgent:
    """One LangGraph reason/act loop with explicit tool and model budgets."""

    def __init__(
        self,
        *,
        bridge: ConversationalAnalyticsBridge,
        completers: dict[str, Any],
        get_state: Callable[[str], ConversationState | None],
        get_last_response: Callable[[str, str], RecentResult | ChatResponse | None],
        sql_handler: Callable[[ChatRequest], Awaitable[ChatResponse]],
        investigation_handler: Callable[[ChatRequest], Awaitable[ChatResponse]] | None = None,
        reset_handler: Callable[[str], None] | None = None,
        history_inspector: Callable[[str, str, str], dict[str, Any] | None] | None = None,
        history_refresher: Callable[[ChatRequest, str], Awaitable[ChatResponse]] | None = None,
        dashboard_handler: Callable[
            [ChatRequest, str, RecentResult | ChatResponse, bool], Awaitable[ChatResponse]
        ] | None = None,
        dashboard_inspector: Callable[[str, str, str], dict[str, Any] | None] | None = None,
    ) -> None:
        self.bridge = bridge
        self.completers = completers
        self.get_state = get_state
        self.get_last_response = get_last_response
        self.sql_handler = sql_handler
        self.investigation_handler = investigation_handler
        self.reset_handler = reset_handler
        self.history_inspector = history_inspector
        self.history_refresher = history_refresher
        self.dashboard_handler = dashboard_handler
        self.dashboard_inspector = dashboard_inspector
        graph = StateGraph(AgentState)
        graph.add_node("reason", self._reason)
        graph.add_node("act", self._act)
        graph.add_edge(START, "reason")
        graph.add_conditional_edges("reason", self._next, {"act": "act", "end": END})
        graph.add_conditional_edges("act", self._after_act, {"act": "reason", "end": END})
        self.graph = graph.compile()
        self._request: ChatRequest | None = None
        self._budget: RequestBudget | None = None
        self._investigation_id: str | None = None

    async def run(self, request: ChatRequest) -> ChatResponse:
        # The compiled graph is immutable, but request-local data must not be
        # stored on a shared agent instance. Create a request-local runner.
        runner = self._for_request(request)
        assert runner._budget is not None
        with use_request_budget(runner._budget):
            return await runner._run_request(request)

    async def _run_request(self, request: ChatRequest) -> ChatResponse:
        started = perf_counter()
        try:
            result = await self.graph.ainvoke(AgentState(turn_context=TurnContext(
                request_id=uuid4().hex,
                session_id=request.session_id,
                db_id=request.db_id,
                pending_goal="CHART" if parse_intent(request.message).requested else "NONE",
            )))
            state = AgentState.model_validate(result)
            if state.response is not None:
                response = state.response
                timings: dict[str, int] = {
                    "total": round((perf_counter() - started) * 1000)
                }
                stages: dict[str, Callable[[CallLedgerEntry], bool]] = {
                    "top_decisions": lambda item: item.component == "conversational_agent"
                    and item.task != "final_answer",
                    "top_final_answer": lambda item: item.component == "conversational_agent"
                    and item.task == "final_answer",
                    "context_planning": lambda item: item.component == "context_planner",
                    "analytics_planning": lambda item: item.component == "analytics_planner",
                    "sql_generation": lambda item: item.component in {
                        "sql_generation", "sql_repair"
                    },
                    "sql_validation": lambda item: item.component == "sql_validation",
                    "sql_execution": lambda item: item.component == "sql_execution",
                    "result_analysis": lambda item: item.component == (
                        "assistant_tool:analyze_result"
                    ),
                    "result_explanation": lambda item: item.component in {
                        "assistant_tool:explain_existing_result", "assistant_tool:present_result",
                        "result_interpretation", "explanation",
                    },
                    "charting_model": lambda item: item.component == "plotting_model",
                }
                for name, selected in stages.items():
                    durations = [
                        item.latency_ms for item in response.call_ledger
                        if item.latency_ms is not None and selected(item)
                    ]
                    if durations:
                        timings[name] = sum(durations)
                return response.model_copy(update={"timings_ms": timings})
            return self._failure("Investigation limit reached without enough evidence.")
        except ModelError as exc:
            return self._failure(self._provider_failure_message(exc))
        except (DatabaseError, RequestBudgetExceeded, ValueError) as exc:
            return self._failure(f"The assistant could not complete this safely: {exc}")
        except ValidationError:
            return self._failure(
                "I couldn't interpret that request safely. Please try rephrasing it."
            )
        except Exception as exc:  # Provider adapters can raise transport-specific errors.
            return self._failure(f"Assistant response unavailable ({type(exc).__name__}).")

    @staticmethod
    def _provider_failure_message(error: Exception) -> str:
        """Give actionable, non-sensitive provider diagnostics to the client."""
        if isinstance(error, ProviderRequestError):
            if error.status in {401, 403}:
                cause = "The model provider rejected the configured credentials or access."
            elif error.status == 404:
                cause = f"The configured model {error.model} is unavailable or was not found."
            elif error.status == 429:
                cause = "The model provider is rate-limiting requests. Please try again shortly."
            elif error.status == 400:
                cause = (
                    f"The model provider rejected the {error.phase} request "
                    f"({error.code or 'invalid_request'}). Check the model ID and request format."
                )
            else:
                cause = f"The model provider returned HTTP {error.status}."
            return (
                f"{cause} Provider: {error.provider}; model: {error.model}. "
                "No query was run; please retry or select another configured model."
            )
        detail = str(error).lower()
        if "not configured" in detail or ("api key" in detail and "missing" in detail):
            cause = "The selected model's API configuration is missing."
        elif any(
            term in detail for term in ("rate limit", "rate_limit", "too many requests", "429")
        ):
            cause = "The model provider is rate-limiting requests. Please try again shortly."
        elif any(term in detail for term in ("insufficient credits", "quota exceeded", "billing")):
            cause = "The model provider's quota or credits are unavailable."
        elif any(
            term in detail
            for term in ("unauthorized", "forbidden", "invalid api key", "401", "403")
        ):
            cause = "The model provider rejected the configured credentials or access."
        elif isinstance(error, StructuredOutputUnsupported):
            cause = "The selected model does not support the required decision format."
        elif any(term in detail for term in ("timeout", "timed out")):
            cause = "The model provider did not respond before the timeout."
        elif any(term in detail for term in (
            "nodename nor servname", "name resolution", "dns", "connecterror",
            "connection refused", "network is unreachable",
        )):
            cause = "The application could not reach the model provider over the network."
        elif "no text output" in detail or "no structured decision" in detail:
            cause = "The model provider returned an empty decision."
        else:
            cause = "The model provider rejected or failed the decision request."
        return cause + " No query was run; please retry or select another configured model."

    @staticmethod
    def _capabilities() -> list[str]:
        return [
            "answer analytical questions through the governed Text-to-SQL pipeline",
            "explain accepted results and SQL",
            "inspect schema and approved business definitions",
            "make validated charts from complete accepted results",
        ]

    def _failure(self, message: str) -> ChatResponse:
        return self._plain(message, provenance=["assistant_service_failure"])

    def _for_request(self, request: ChatRequest) -> ConversationalAnalyticsAgent:
        runner = object.__new__(ConversationalAnalyticsAgent)
        runner.__dict__ = self.__dict__.copy()
        runner._request = request
        runner._investigation_id = uuid4().hex
        runner._budget = RequestBudget(
            timeout_seconds=float(os.environ.get("TEXT2SQL_ASSISTANT_TIMEOUT_SECONDS", "180")),
            max_model_calls=int(os.environ.get("TEXT2SQL_ASSISTANT_MAX_MODEL_CALLS", "8")),
            max_database_calls=int(os.environ.get("TEXT2SQL_ASSISTANT_MAX_DATABASE_CALLS", "24")),
        )
        # Bind graph nodes to this request-local instance.
        graph = StateGraph(AgentState)
        graph.add_node("reason", runner._reason)
        graph.add_node("act", runner._act)
        graph.add_edge(START, "reason")
        graph.add_conditional_edges("reason", runner._next, {"act": "act", "end": END})
        graph.add_conditional_edges("act", runner._after_act, {"act": "reason", "end": END})
        runner.graph = graph.compile()
        return runner

    def _plain(self, message: str, *, provenance: list[str] | None = None) -> ChatResponse:
        assert self._request is not None
        return ChatResponse(
            session_id=self._request.session_id,
            operation="EXPLAIN_CONTEXT",
            resolved_question=self._request.message,
            message=message,
            provenance=provenance or [],
        )

    @staticmethod
    def _sum_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=(left.input_tokens or 0) + (right.input_tokens or 0),
            output_tokens=(left.output_tokens or 0) + (right.output_tokens or 0),
            cache_read_tokens=(left.cache_read_tokens or 0) + (right.cache_read_tokens or 0),
            cache_creation_tokens=(left.cache_creation_tokens or 0)
            + (right.cache_creation_tokens or 0),
        )

    def _reconcile_route(
        self,
        decision: AgentDecision,
        state: AgentState,
        previous: ConversationState | None,
        last: RecentResult | ChatResponse | None,
    ) -> AgentDecision:
        """Check model routing against accepted evidence without interpreting SQL."""
        assert self._request is not None
        if decision.tool in {"ai_dashboard_designer", "undo_dashboard"}:
            return decision
        message = self._request.message
        has_result = bool(last and last.generation and last.generation.accepted)
        prior_chart = (
            last.presentation.chart_spec
            if last and last.presentation else None
        )
        contract = previous.semantic_contract if previous else None
        changes = decision.tool_arguments.changes if decision.tool_arguments else []
        analytical_change = any(
            item.operation in ANALYTICAL_CHANGE_OPERATIONS for item in changes
        )
        chart_requested = parse_intent(message, has_previous=has_result).requested
        chart_observations = [
            (index, item) for index, item in enumerate(state.observations)
            if item.get("tool") == "change_visualization"
        ]
        latest_chart = chart_observations[-1] if chart_observations else None
        # An explicit historical source must not be replaced with the latest
        # accepted analysis by the legacy chart/analytical reconciliation path.
        if decision.action == "CALL_TOOL" and (
            decision.tool in {"search_conversation_memory", "inspect_result", "refresh_analysis"}
            or decision.tool_arguments and decision.tool_arguments.source_run_id
        ):
            return decision

        failed_refresh = next((
            item for item in reversed(state.observations)
            if item.get("tool") == "refresh_analysis"
            and item.get("status") != "executed"
        ), None)
        if failed_refresh is not None and (
            decision.tool == "change_visualization" or latest_chart
        ):
            return AgentDecision(
                action="ANSWER", intent="VISUALIZE",
                answer=str(failed_refresh.get("failure")
                           or "The historical analysis could not be refreshed safely."),
            )

        if latest_chart and latest_chart[1].get("status") == "REQUERY_REQUIRED":
            chart_index, issue = latest_chart
            if not has_result or any(
                item.get("tool") == "ask_data"
                for item in state.observations[:chart_index]
            ):
                return AgentDecision(
                    action="ANSWER", intent="VISUALIZE",
                    answer=str(issue.get("reason") or issue.get("error")
                               or "The current result cannot support that chart."),
                )
            later_sql = next((
                item for item in state.observations[chart_index + 1:]
                if item.get("tool") == "ask_data"
            ), None)
            if later_sql is not None:
                if later_sql.get("status") == "executed" and (
                    state.base_response and state.base_response.generation
                    and state.base_response.generation.accepted
                ):
                    return AgentDecision(
                        action="CALL_TOOL", intent="VISUALIZE", tool="change_visualization",
                        tool_arguments=ToolInput(objective=message),
                    )
                return AgentDecision(
                    action="ANSWER", intent="VISUALIZE",
                    answer="I couldn't obtain the data needed for that chart. "
                    "Your previous accepted result was not changed.",
                )
            reason = str(issue.get("reason") or issue.get("error") or "")
            if "truncated" in reason.casefold() or "incomplete" in reason.casefold():
                return AgentDecision(action="ANSWER", intent="VISUALIZE", answer=reason)
            accepted_question = (
                (contract.resolved_question or contract.original_question) if contract else None
            ) or (previous.resolved_question if previous else None)
            if not accepted_question or state.steps >= 4:
                return AgentDecision(action="ANSWER", intent="VISUALIZE", answer=reason)
            # The accepted question carries unaffected metric/filter/period constraints;
            # the follow-up adds only the requested change. The SQL resolver still
            # validates the resulting contract before execution.
            proposed_objective = (
                (decision.tool_arguments.objective if decision.tool_arguments
                 else decision.instruction)
                if decision.action == "CALL_TOOL" and decision.tool == "ask_data"
                else None
            )
            if proposed_objective and not _route_content_terms(proposed_objective) <= (
                _route_content_terms(accepted_question) | _route_content_terms(message)
            ):
                proposed_objective = None
            chart_objective = proposed_objective or str(issue.get("objective") or message)
            objective = f"{accepted_question}. {chart_objective}"
            requested_changes = [
                RequestedChange.model_validate(item)
                for item in issue.get("requested_changes", [])
            ]
            return AgentDecision(
                action="CALL_TOOL", intent="MODIFY_QUERY", tool="ask_data",
                tool_arguments=ToolInput(
                    objective=objective, operation="REFINE",
                    changes=requested_changes,
                ),
            )

        if (
            state.observations and chart_requested and not chart_observations
            and state.base_response and state.base_response.generation
            and state.base_response.generation.accepted
            and any(item.get("tool") == "ask_data" and item.get("status") == "executed"
                    for item in state.observations)
        ):
            return AgentDecision(
                action="CALL_TOOL", intent="VISUALIZE", tool="change_visualization",
                tool_arguments=ToolInput(
                    objective=(decision.tool_arguments.objective
                               if decision.tool == "change_visualization"
                               and decision.tool_arguments
                               else decision.instruction
                               if decision.tool == "change_visualization"
                               and decision.instruction else message),
                ),
            )

        if not state.observations and analytical_change and has_result:
            return AgentDecision(
                action="CALL_TOOL", intent="MODIFY_QUERY", tool="ask_data",
                tool_arguments=ToolInput(
                    objective=(decision.tool_arguments.objective
                               if decision.tool_arguments else message),
                    operation="REFINE", changes=changes,
                ),
            )
        if not state.observations and chart_requested and not has_result:
            accepted_question = (
                (contract.resolved_question or contract.original_question) if contract else None
            ) or (previous.resolved_question if previous else None)
            if previous and previous.last_sql and accepted_question:
                # A restart or bounded-cache eviction may leave the accepted
                # contract without complete rows. Regenerate through governed
                # SQL before charting; never fabricate data from a preview.
                bare_plot = message.strip().casefold().rstrip(".") in {
                    "plot", "plot it", "chart", "chart it", "graph", "show the chart",
                }
                return AgentDecision(
                    action="CALL_TOOL",
                    intent="NEW_QUERY" if bare_plot else "MODIFY_QUERY",
                    tool="ask_data",
                    tool_arguments=ToolInput(
                        objective=(accepted_question if bare_plot
                                   else f"{accepted_question}. {message}"),
                        operation="NEW_QUERY" if bare_plot else "REFINE",
                        changes=[] if bare_plot else changes,
                    ),
                )
            if _chart_followup(message) or message.strip().casefold().rstrip(".") in {
                "plot", "plot it", "chart", "chart it", "graph", "show the chart",
            }:
                return AgentDecision(
                    action="ANSWER", intent="DIRECT_ANSWER",
                    answer="I don't have an accepted result to plot. "
                    "Please ask the original data question again.",
                )
        if not state.observations and has_result and chart_requested and (
            decision.tool == "change_visualization"
            or _chart_followup(message)
            or (parse_intent(message, has_previous=True).followup
                and (prior_chart is not None or decision.tool == "ask_data"))
        ):
            return AgentDecision(
                action="CALL_TOOL", intent="VISUALIZE", tool="change_visualization",
                tool_arguments=ToolInput(
                    objective=message, changes=changes,
                ),
            )
        if not state.observations and has_result and _result_presentation_request(message):
            return AgentDecision(
                action="CALL_TOOL", intent="EXPLAIN", tool="present_result",
                tool_arguments=ToolInput(objective=message),
            )
        if (
            not state.observations and decision.action == "ANSWER"
            and decision.intent == "DIRECT_ANSWER" and _database_question(message)
            and not chart_requested
        ):
            return AgentDecision(
                action="CALL_TOOL", intent="NEW_QUERY", tool="ask_data",
                tool_arguments=ToolInput(objective=message, operation="NEW_QUERY"),
            )
        if (
            not state.observations and not has_result and chart_requested
            and decision.tool == "change_visualization" and _database_question(message)
        ):
            return AgentDecision(
                action="CALL_TOOL", intent="NEW_QUERY", tool="ask_data",
                tool_arguments=ToolInput(objective=message, operation="NEW_QUERY"),
            )
        return decision

    @staticmethod
    def _model_entry(
        provider: str, model: str, usage: TokenUsage, *, succeeded: bool,
        latency_ms: int | None = None, task: str | None = None,
        retry_count: int | None = None,
    ) -> CallLedgerEntry:
        """Ledger entry for one top-layer model call, valid output or not."""
        return CallLedgerEntry(
            component="conversational_agent",
            kind="MODEL",
            status="SUCCEEDED" if succeeded else "FAILED",
            provider=provider,
            requested_model=model,
            effective_model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            task=task,
            retry_count=(retry_count or 0) + usage.provider_retries,
        )

    async def _complete_decision(
        self,
        prompt: str,
        state: AgentState,
        provider: str,
        model: str,
        completer: Any,
    ) -> _DecisionCompletion:
        """Fetch and validate one AgentDecision with at most one bounded repair.

        Provider-native JSON-schema output is used when the completer exposes it;
        local validation always applies. Invalid output never executes a tool and
        never converts prose into intent: the exact issues are fed back once, and
        a still-invalid response fails closed. Usage and ledger survive failures.
        """
        assert self._budget is not None
        structured = getattr(completer, "complete_structured", None)
        structured_task = getattr(completer, "complete_structured_task", None)
        usage_total = TokenUsage()
        ledger: list[CallLedgerEntry] = []
        issues: list[str] = []
        native_capability = getattr(completer, "native_schema_supported", None)
        native_supported = (callable(structured_task) or callable(structured)) and (
            not callable(native_capability)
            or bool(native_capability(model, "AgentDecision"))
        )
        task_completion = getattr(completer, "complete_task", None)
        for attempt in range(2):
            call_started = perf_counter()
            request_prompt = (
                prompt if attempt == 0 or not issues else prompt + self._repair_tail(issues)
            )
            task = (
                "decision_repair" if attempt else "final_answer" if state.observations
                else "decision"
            )
            if native_supported:
                call = (
                    cast(Any, structured_task)(
                        model, request_prompt, "AgentDecision", AGENT_DECISION_SCHEMA, task
                    )
                    if callable(structured_task)
                    else cast(Any, structured)(
                        model, request_prompt, "AgentDecision", AGENT_DECISION_SCHEMA
                    )
                )
            elif callable(task_completion):
                call = task_completion(model, request_prompt, task, json_mode=True)
            else:
                call = completer.complete_detailed(model, request_prompt)
            try:
                raw, usage = await self._budget.wait(call, kind="model")
            except (StructuredOutputFallbackRequired, StructuredOutputUnsupported):
                ledger.append(self._model_entry(
                    provider, model, TokenUsage(), succeeded=False,
                    latency_ms=round((perf_counter() - call_started) * 1000),
                    task=task, retry_count=attempt,
                ))
                native_supported = False
                call_started = perf_counter()
                try:
                    fallback_call = (
                        task_completion(model, request_prompt, task, json_mode=True)
                        if callable(task_completion)
                        else completer.complete_detailed(model, request_prompt)
                    )
                    raw, usage = await self._budget.wait(
                        fallback_call, kind="model"
                    )
                except Exception as exc:
                    failed_usage = getattr(exc, "usage", TokenUsage())
                    usage_total = self._sum_usage(usage_total, failed_usage)
                    ledger.append(self._model_entry(
                        provider, model, failed_usage, succeeded=False,
                        latency_ms=round((perf_counter() - call_started) * 1000),
                        task=task, retry_count=attempt + 1,
                    ))
                    return _DecisionCompletion(
                        None,
                        usage_total,
                        ledger,
                        ["structured output unavailable"],
                        message=self._provider_failure_message(exc),
                    )
            except RequestBudgetExceeded:
                ledger.append(self._model_entry(
                    provider, model, TokenUsage(), succeeded=False,
                    latency_ms=round((perf_counter() - call_started) * 1000),
                    task=task, retry_count=attempt,
                ))
                return _DecisionCompletion(
                    None,
                    usage_total,
                    ledger,
                    ["the assistant call budget was exhausted before a valid decision"],
                    message=(
                        "The assistant could not complete this safely: the assistant call "
                        "budget was exhausted before a valid decision was produced."
                    ),
                )
            except Exception as exc:
                failed_usage = getattr(exc, "usage", TokenUsage())
                usage_total = self._sum_usage(usage_total, failed_usage)
                ledger.append(self._model_entry(
                    provider, model, failed_usage, succeeded=False,
                    latency_ms=round((perf_counter() - call_started) * 1000),
                    task=task, retry_count=attempt,
                ))
                transport_message = self._provider_failure_message(exc)
                issues = [
                    *issues,
                    f"then the repair call failed ({type(exc).__name__})",
                ]
                return _DecisionCompletion(
                    None,
                    usage_total,
                    ledger,
                    issues,
                    message=transport_message,
                )
            usage_total = self._sum_usage(usage_total, usage)
            try:
                decision = AgentDecision.model_validate_json(
                    _extract_decision_object(str(raw))
                )
            except ValidationError as exc:
                issues = _validation_issue_lines(exc)
                ledger.append(self._model_entry(
                    provider, model, usage, succeeded=False,
                    latency_ms=round((perf_counter() - call_started) * 1000),
                    task=task, retry_count=attempt,
                ))
                continue
            issues = _consistency_issue_lines(decision)
            if issues:
                ledger.append(self._model_entry(
                    provider, model, usage, succeeded=False,
                    latency_ms=round((perf_counter() - call_started) * 1000),
                    task=task, retry_count=attempt,
                ))
                continue
            ledger.append(self._model_entry(
                provider, model, usage, succeeded=True,
                latency_ms=round((perf_counter() - call_started) * 1000),
                task=task, retry_count=attempt,
            ))
            return _DecisionCompletion(decision, usage_total, ledger, [])
        return _DecisionCompletion(None, usage_total, ledger, issues)

    @staticmethod
    def _repair_tail(issues: list[str]) -> str:
        return (
            "\n\nThe previous response was not a valid AgentDecision. Return ONLY one "
            "corrected JSON object that matches the AgentDecision schema: no prose, "
            "markdown, or code fences.\nValidation issues:\n- "
            + "\n- ".join(issues)
            + "\nAllowed values: action="
            + "|".join(_ACTION_VALUES)
            + "; intent="
            + "|".join(_INTENT_VALUES)
            + "; tool="
            + "|".join(_TOOL_VALUES)
            + "|null.\n"
            + "Consistency: action CALL_TOOL requires a tool; ANSWER requires "
            "tool null.\nExamples:\n"
            + "\n".join(_decision_json_line(example) for example in _DECISION_EXAMPLES)
        )

    @staticmethod
    def _decision_failure_message(issues: list[str]) -> str:
        return (
            "I couldn't interpret that request safely. Please try rephrasing it. "
            "No tool was run and your previous result was preserved."
        )

    @staticmethod
    def _result_facts(response: ChatResponse) -> dict[str, Any]:
        generation = response.generation
        return {
            "kind": "governed_sql_result",
            "status": "executed" if generation and generation.accepted else "not_accepted",
            "run_id": response.run_id,
            "question": response.resolved_question,
            "execution_timestamp":generation.execution_timestamp if generation else None,
            "source_provenance":response.provenance,
            "columns": generation.columns if generation else [],
            "rows_preview": generation.rows[:10] if generation else [],
            "row_count": generation.row_count if generation else 0,
            "truncated": generation.truncated if generation else False,
            "chart": (
                response.presentation.chart_spec.model_dump(exclude={"data"})
                if response.presentation and response.presentation.chart_spec
                else None
            ),
            "failure": response.message if not generation or not generation.accepted else None,
        }

    @staticmethod
    def _next(state: AgentState) -> Literal["act", "end"]:
        return (
            "act"
            if state.decision and state.decision.action == "CALL_TOOL" and state.response is None
            else "end"
        )

    @staticmethod
    def _after_act(state: AgentState) -> Literal["act", "end"]:
        return "end" if state.response is not None else "act"

    async def _reason(self, state: AgentState) -> AgentState:
        assert self._request is not None and self._budget is not None
        if state.observations and state.observations[-1].get("status") == "DASHBOARD_READY":
            assert state.base_response is not None
            return state.model_copy(update={"response": state.base_response.model_copy(update={
                "token_usage": state.usage, "call_ledger": state.ledger,
            })})
        if (state.pending_dashboard_objective and state.observations
                and state.observations[-1].get("tool") == "ask_data"
                and state.observations[-1].get("status") == "executed"):
            return state.model_copy(update={"decision": AgentDecision(
                action="CALL_TOOL", intent="VISUALIZE", tool="ai_dashboard_designer",
                tool_arguments=ToolInput(objective=state.pending_dashboard_objective),
            )})
        previous = self.get_state(self._request.session_id)
        if state.base_response and state.base_response.state:
            previous = state.base_response.state
        if previous and previous.db_id != self._request.db_id:
            previous = None
        last = (
            state.base_response
            if state.base_response and state.base_response.generation
            and state.base_response.generation.accepted
            else self.get_last_response(self._request.session_id, self._request.db_id)
        )
        chart_issues = [
            item for item in state.observations
            if item.get("tool") == "change_visualization"
            and item.get("status") == "REQUERY_REQUIRED"
        ]
        if len(chart_issues) >= 2 and state.base_response is not None:
            reason = str(chart_issues[-1].get("reason") or "The result cannot support that chart.")
            return state.model_copy(update={
                "response": state.base_response.model_copy(update={
                    "message": reason,
                    "plot_status": "REQUERY_REQUIRED",
                    "token_usage": self._sum_usage(state.base_response.token_usage, state.usage),
                    "call_ledger": [*state.base_response.call_ledger, *state.ledger],
                }),
            })
        if (
            chart_issues and state.observations
            and state.observations[-1].get("tool") == "ask_data"
            and state.observations[-1].get("status") == "executed"
            and state.base_response and state.base_response.generation
            and state.base_response.generation.accepted
        ):
            return state.model_copy(update={"decision": AgentDecision(
                action="CALL_TOOL", intent="VISUALIZE", tool="change_visualization",
                tool_arguments=ToolInput(objective=self._request.message),
            )})
        # A successful governed result already supplies the client-facing message.
        # Do not make another model call merely to paraphrase it. A combined
        # SQL-and-chart request still needs the plotting tool before completion.
        if state.finish_after_tool and state.observations and state.base_response is not None:
            latest = state.observations[-1]
            wants_chart = (
                parse_intent(self._request.message).requested or state.chart_proposal is not None
            )
            chart_ready = any(
                item.get("tool") == "change_visualization"
                and item.get("status") == "CHART_READY"
                for item in state.observations
            )
            if (
                latest.get("tool") in {"ask_data", "refresh_analysis"}
                and latest.get("status") == "executed"
                and wants_chart and not chart_ready and not chart_issues
            ):
                return state.model_copy(update={"decision": AgentDecision(
                    action="CALL_TOOL", intent="VISUALIZE",
                    tool="change_visualization",
                    tool_arguments=ToolInput(objective=self._request.message),
                )})
            if (
                state.base_response.generation is not None
                and state.base_response.generation.accepted
                and (
                    latest.get("tool") in {"ask_data", "refresh_analysis"} and not wants_chart
                    or latest.get("tool") == "change_visualization" and chart_ready
                )
            ):
                base = state.base_response
                message = (
                    "The chart is ready from the validated query result."
                    if chart_ready else base.message
                )
                presentation = base.presentation
                if presentation is not None:
                    presentation = presentation.model_copy(update={"summary": message})
                return state.model_copy(update={
                    "response": base.model_copy(update={
                        "message": message,
                        "presentation": presentation,
                        "token_usage": self._sum_usage(base.token_usage, state.usage),
                        "call_ledger": [*base.call_ledger, *state.ledger],
                    }),
                })
        compact = {
            "saved_dashboard_snapshot_only": self._request._dashboard_snapshot_only,
            "current_dashboard": (
                self.dashboard_inspector(
                    self._request.session_id, self._request.db_id, last.result_id
                ) if self.dashboard_inspector and last and last.result_id else None
            ),
            "question": self._request.message,
            "capabilities": self._capabilities(),
            "previous_question": previous.resolved_question if previous else None,
            "accepted_analysis_reference": ({"run_id":last.run_id, "result_id":last.result_id}
                if last else None),
            "accepted_sql_available": bool(previous and previous.last_sql),
            "accepted_result_available": bool(
                last and last.generation and last.generation.accepted
            ),
            "accepted_result_completeness": ({
                "result_id": last.result_id,
                "retained_rows": len(last.generation.rows),
                "row_count": last.generation.row_count,
                "truncated": last.generation.truncated,
                "complete": not last.generation.truncated
                and last.generation.row_count == len(last.generation.rows),
            } if last and last.generation and last.generation.accepted else None),
            "result_columns": previous.last_columns if previous else [],
            "chart_column_profiles_advisory": (
                profile_chart_result(
                    last.generation.columns, last.generation.rows, sample_rows=0
                )[0]
                if last and last.generation and last.generation.accepted else None
            ),
            "accepted_analytical_contract": (
                previous.semantic_contract.model_dump(
                    mode="json", exclude_none=True, exclude_defaults=True,
                    include={
                        "approved_metrics", "mandatory_metric_filters", "structural_formulas",
                        "aggregation_stages", "output_operations", "outputs", "filters",
                        "measures", "aggregation", "input_grain", "output_grain",
                        "temporal_boundaries", "active_period", "comparison_periods",
                        "grain", "requires_grouping", "requires_ordering", "ranking_direction",
                        "ranked_metric", "limit", "tie_policy", "required_columns",
                    },
                ) if previous and previous.semantic_contract else {}
            ),
            "corrections": previous.corrections[-3:] if previous else [],
            "chart": (
                last.presentation.chart_spec.model_dump(exclude={"data"})
                if last and last.presentation and last.presentation.chart_spec
                else None
            ),
            "observations": state.observations[-3:],
        }
        prompt = (
            "You are the single client-facing AI Business Analytics Assistant and the "
            "primary decision-maker for conversational analytics.\n\n"
            "For every user message, understand the user's objective using the current "
            "request, previous conversation, accepted SemanticContract, available SQL "
            "results, chart state, and corrections. Select the simplest valid execution "
            "path that fully satisfies the user's request.\n\n"
            "Routing rules:\n"
            "SAVED STUDIO: When saved_dashboard_snapshot_only is true, edit the retained "
            "dashboard with ai_dashboard_designer. No SQL refresh or metadata DB tools are "
            "available in this scope. If new analytical data is required, explain that the "
            "client must run a new governed analysis in Conversations, then Save as New.\n"
            "DASHBOARDS: For dashboard creation or presentation-only revisions use "
            "ai_dashboard_designer with the precise design objective; undo_dashboard "
            "restores the previous design. Both reuse the current complete accepted result. "
            "If REQUERY_REQUIRED is returned, preserve the dashboard goal, use ask_data "
            "for missing analytical requirements, then retry ai_dashboard_designer. "
            "Do not claim a dashboard is ready until DASHBOARD_READY is observed.\n"
            "1. DIRECT_ANSWER: Answer directly only when no new database evidence or tool "
            "action is required.\n"
            "2. REUSE EXISTING RESULT: When accepted results already contain sufficient "
            "verified evidence, reuse them instead of generating SQL.\n"
            "3. CHART CREATION OR MODIFICATION: Use change_visualization when the request "
            "concerns chart type, axes, series, or presentation and can be supported by "
            "existing data. Preserve all unaffected chart settings.\n"
            "Chart selection requiring reasoning uses the existing visualization planner, "
            "not ask_data. A type-only edit preserves accepted axes, series and result_id "
            "and needs no plotting-model call. Explicit UI color/style/zoom edits are local "
            "display actions, not new analyses.\n"
            "Integrated visualization planning: When a user requests SQL and a chart, "
            "you may attach chart_proposal to the ask_data decision if intended output "
            "columns are sufficiently known. Use the existing chart schema; this is a "
            "provisional design, not verified data. Set finish_after_tool=true when SQL "
            "and the chart complete the goal. The controller validates actual columns, "
            "types, units, grain and completeness after execution; a valid proposal "
            "needs no plotting-model call. Unknown aliases must not be invented as "
            "facts: use chart_proposal=null when uncertain, and the existing planner "
            "will inspect the executed result. For chart-only requests you may propose "
            "against accepted columns. Select the smallest useful X/Y/series set, "
            "exclude irrelevant identifiers, and preserve unaffected encodings. "
            "Never attach a chart merely to an SQL-only request.\n"
            "4. ANALYTICAL MODIFICATION: Use ask_data when metrics, formulas, filters, "
            "periods, dimensions, aggregations, ranking, or comparisons require changes "
            "to the accepted analysis. Preserve unaffected SemanticContract constraints.\n"
            "5. NEW ANALYSIS: Use ask_data for independent analytical questions.\n"
            "6. EXPLANATION: Use the appropriate existing-result or SQL explanation tool, "
            "obtaining additional evidence only when needed.\n"
            "7. INVESTIGATION: Break complex analytical questions into distinct, "
            "evidence-driven steps. Avoid unnecessary tool calls.\n"
            "8. INSUFFICIENT DATA: If a tool returns REQUERY_REQUIRED, determine the missing "
            "requirements. Check accepted_result_completeness: a truncated preview is not "
            "complete evidence. When the user has specified a metric and grouping, ask_data "
            "for that aggregated result while preserving filters and periods; do not add an "
            "arbitrary LIMIT or repeat the same incomplete detail query. Do not invent a "
            "new metric or grouping merely to fit a chart. Determine the missing "
            "requirements, call ask_data to obtain them when feasible, then retry the "
            "original visualization. Do not invent missing data.\n"
            "9. AMBIGUITY: Request clarification only when materially different "
            "interpretations cannot be resolved safely from the accepted context.\n\n"
            "Execution principles: Distinguish presentation-only changes from changes to "
            "analytical meaning. Do not regenerate SQL when verified existing data is "
            "sufficient. Do not answer database-dependent questions from unsupported "
            "assumptions. Keep the user's original goal across tool calls. Inspect every "
            "tool result before deciding whether another step is necessary. Never silently "
            "modify accepted metrics, formulas, filters, periods, or chart encodings. "
            "Never claim success without supporting tool evidence. Return only a valid "
            "AgentDecision JSON matching the existing application schema. Prioritize "
            "correctness, continuity, efficiency, and stakeholder-friendly answers.\n"
            + _DECISION_INSTRUCTIONS
            + "\nTools: list_tables/search_schema/describe_table/get_relationships/"
            "get_column_profile inspect allowlisted metadata; get_available_metrics/"
            "get_business_context return approved definitions; ask_data delegates new or "
            "modified questions; inspect_result/search_conversation_memory retrieve accepted "
            "SQL and semantic context only, never numerical evidence. Their objective is an "
            "exact run/result ID, first, latest, or analysis:N (1-based; negative counts from "
            "newest). Translate 'my first query' into first and 'third analysis' into analysis:3. "
            "Do not invent IDs. Inspecting history never restores active state. For current "
            "data from an unchanged prior analysis, use refresh_analysis with the retrieved "
            "run_id as objective and intent INVESTIGATE; it revalidates and executes SQL "
            "without a generator "
            "call. For modifications, use ask_data with source_run_id set to the retrieved "
            "run_id and operation REFINE/COMPARE/CORRECTION, preserving unaffected metrics, "
            "joins, filters, aggregation and grain. For chart-only changes, reuse a complete "
            "cached result instead of refreshing. Refreshed figures are CURRENT data, never "
            "original historical figures. Exact old figures require a complete timestamped "
            "snapshot or governed time-travel source; SQL definitions and summaries are not "
            "numerical proof. Only for explicitly requested old figures, inspect_result "
            "may use objective snapshot:RUN_ID to request a retained snapshot. An unavailable "
            "snapshot is a terminal limitation, not permission to relabel refreshed data. "
            "If unavailable, explain that limitation rather than refresh "
            "and mislabel it. ask_data delegates new or "
            "modified analytical questions to governed Text-to-SQL; analyze_result/"
            "present_result inspect bounded accepted result rows; check_sql_correctness "
            "reviews accepted SQL; change_visualization reuses or requeries as the "
            "existing gate decides; explain_existing_result inspects accepted SQL, "
            "its referenced tables/columns and result evidence; reset_context "
            "delegates an explicit reset request to the existing conversation service. "
            "For database facts, use a grounding tool before answering; for a plot of an "
            "existing result use change_visualization, and for a new plot requiring data "
            "use ask_data. Do not claim a chart exists before observing a chart result. "
            "Use ask_data for new, corrected or optimized analytical questions; "
            "If one request needs both new data and a chart, call ask_data first, then "
            "change_visualization on the accepted result, then answer. "
            "use check_sql_correctness only when the client requests a correctness review. "
            "Only ask_data may delegate governed SQL generation. Never ask tools to "
            "execute arbitrary SQL. "
            "After sufficient observations, answer with only supported facts; do not infer "
            "causality from descriptive data.\nContext: " + json.dumps(compact, default=str)
        )
        provider = os.environ.get("TEXT2SQL_ASSISTANT_PROVIDER") or self._request.provider
        model = os.environ.get("TEXT2SQL_ASSISTANT_MODEL") or self._request.model
        if provider not in self.completers:
            available = ", ".join(sorted(self.completers)) or "none"
            raise ValueError(
                f"Assistant provider {provider!r} is not configured. "
                f"Set TEXT2SQL_ASSISTANT_PROVIDER to one of: {available}."
            )
        completer = self.completers[provider]
        completion = await self._complete_decision(prompt, state, provider, model, completer)
        if completion.decision is None:
            failed_ledger = [*state.ledger, *completion.ledger]
            merged_failed_usage = self._sum_usage(state.usage, completion.usage)
            if state.base_response is not None:
                # SQL may already have executed and updated the conversation.
                # Preserve that governed result if final prose generation fails.
                return state.model_copy(
                    update={
                        "ledger": failed_ledger,
                        "response": state.base_response.model_copy(
                            update={
                                "token_usage": merged_failed_usage,
                                "call_ledger": failed_ledger,
                                "provenance": [
                                    *state.base_response.provenance,
                                    "assistant_response_unavailable",
                                ],
                            }
                        ),
                    }
                )
            failure = self._plain(
                (
                    completion.message
                    if completion.message is not None
                    else self._decision_failure_message(completion.issues)
                ),
                provenance=["assistant_service_failure"],
            )
            return state.model_copy(
                update={
                    "ledger": failed_ledger,
                    "response": failure.model_copy(
                        update={
                            "token_usage": merged_failed_usage,
                            "call_ledger": failed_ledger,
                        }
                    ),
                }
            )
        decision = self._reconcile_route(completion.decision, state, previous, last)
        if completion.decision.finish_after_tool and decision.action == "CALL_TOOL":
            decision = decision.model_copy(update={"finish_after_tool": True})
        if (
            completion.decision.chart_proposal is not None
            and decision.action == "CALL_TOOL"
            and decision.tool in {"ask_data", "refresh_analysis", "change_visualization"}
            and decision.chart_proposal is None
        ):
            decision = decision.model_copy(update={
                "chart_proposal": completion.decision.chart_proposal,
            })
        if (
            parse_intent(self._request.message).requested
            and any(item.get("tool") == "change_visualization"
                    and item.get("status") == "CHART_READY" for item in state.observations)
            and state.base_response is not None
            and state.base_response.presentation is not None
            and state.base_response.presentation.chart_spec is not None
        ):
            decision = AgentDecision(
                action="ANSWER",
                intent="VISUALIZE",
                answer=(
                    decision.answer
                    if decision.action == "ANSWER" and decision.answer
                    else "Here is the chart of the accepted result."
                ),
            )
        if not state.observations and decision.action == "ANSWER":
            required_tool: ToolName | None = {
                "NEW_QUERY": "ask_data",
                "MODIFY_QUERY": "ask_data",
                "INVESTIGATE": "ask_data",
                "CORRECT": "ask_data",
                "REVIEW_SQL": "check_sql_correctness",
                "OPTIMIZE": "ask_data",
                "RESET_CONTEXT": "reset_context",
                "VISUALIZE": "change_visualization" if last else "ask_data",
                "EXPLORE_SCHEMA": "search_schema",
                "BUSINESS_CONTEXT": "get_business_context",
                "EXPLAIN": "explain_existing_result",
            }.get(decision.intent)
            if required_tool is not None:
                # The top model selects intent, but cannot assert database facts,
                # accepted SQL details or a chart before observing the tool result.
                decision = decision.model_copy(
                    update={"action": "CALL_TOOL", "tool": required_tool, "answer": None}
                )
        if (
            decision.action == "ANSWER"
            and state.base_response is not None
            and state.base_response.generation is not None
            and state.base_response.generation.accepted
            and parse_intent(self._request.message).requested
            and not any(item.get("tool") == "change_visualization" for item in state.observations)
            and not (
                state.base_response.presentation
                and state.base_response.presentation.chart_spec
            )
        ):
            decision = decision.model_copy(update={
                "action": "CALL_TOOL",
                "intent": "VISUALIZE",
                "tool": "change_visualization",
                "tool_arguments": ToolInput(objective=self._request.message),
                "answer": None,
            })
        merged_usage = self._sum_usage(state.usage, completion.usage)
        ledger = [*state.ledger, *completion.ledger]
        if decision.action != "CALL_TOOL":
            answer = decision.answer or "I could not complete that request safely."
            base = state.base_response or self._plain(answer)
            failed_required = None
            completed_tools: set[str] = set()
            for item in reversed(state.observations):
                tool_name = str(item.get("tool") or "")
                if tool_name not in {"ask_data", "change_visualization", "refresh_analysis",
                                     "inspect_result", "search_conversation_memory",
                                     "ai_dashboard_designer", "undo_dashboard"}:
                    continue
                if tool_name in completed_tools:
                    continue
                completed_tools.add(tool_name)
                if (item.get("status") in {"not_accepted", "REQUERY_REQUIRED", "UNSUPPORTED",
                                           "UNAVAILABLE"}
                        or item.get("error")):
                    failed_required = item
            if failed_required is not None:
                answer = str(
                    failed_required.get("failure") or failed_required.get("reason")
                    or failed_required.get("error") or "The requested action did not complete."
                )
            recovery = base.generation.recovery if base.generation else None
            if recovery and recovery.diagnosis_code == "REQUESTED_KEY_ABSENT":
                # A live exact primary-key probe is conclusive. Do not let a
                # conversational paraphrase turn it back into speculation.
                answer = recovery.diagnosis_summary or answer
            plot_issue = next(
                (
                    item for item in reversed(state.observations)
                    if item.get("tool") == "change_visualization"
                    and item.get("status") != "CHART_READY"
                ),
                None,
            )
            if plot_issue is not None and not (
                base.presentation and base.presentation.chart_spec
            ):
                # The final conversational model cannot claim an unavailable
                # chart exists. Keep the governed SQL result, if any, intact.
                answer = str(plot_issue.get("reason") or plot_issue.get("error")
                             or "The requested chart is unavailable.")
            evidence_runs = [
                str(item.get("run_id") or item["evidence_id"])
                for item in state.observations
                if item.get("run_id") or item.get("evidence_id")
            ]
            presentation = base.presentation
            if presentation is not None:
                presentation = presentation.model_copy(
                    update={
                        "summary": answer[:1800], "source": "model",
                        "chart_reason": (
                            answer[:300] if plot_issue is not None else presentation.chart_reason
                        ),
                    }
                )
            return state.model_copy(
                update={
                    "decision": decision,
                    "usage": merged_usage,
                    "ledger": ledger,
                    "response": base.model_copy(
                        update={
                            "message": answer,
                            "plot_status": (
                                str(plot_issue.get("status") or "UNSUPPORTED")
                                if plot_issue is not None else base.plot_status
                            ),
                            "explanation": answer if base.explanation is not None else None,
                            "presentation": presentation,
                            "provenance": [*base.provenance, *evidence_runs],
                            "clarification_required": base.clarification_required,
                            "clarification_question": base.clarification_question,
                            "token_usage": self._sum_usage(base.token_usage, merged_usage),
                            "call_ledger": [*base.call_ledger, *ledger],
                        }
                    ),
                }
            )
        if not decision.tool or state.steps >= 5:
            return state.model_copy(
                update={
                    "response": self._failure(
                        "The investigation limit was reached without enough evidence."
                    ),
                    "usage": merged_usage,
                    "ledger": ledger,
                }
            )
        return state.model_copy(update={
            "decision": decision,
            "finish_after_tool": decision.finish_after_tool,
            "usage": merged_usage,
            "ledger": ledger,
        })

    async def _act(self, state: AgentState) -> AgentState:
        assert self._request is not None and self._budget is not None and state.decision is not None
        tool = state.decision.tool
        if tool is None:
            raise ValueError("The agent did not select a tool.")
        tool_input = state.decision.tool_arguments or ToolInput(
            objective=state.decision.instruction or self._request.message
        )
        question = tool_input.objective
        started = perf_counter()
        facts: dict[str, Any]
        if self._request._dashboard_snapshot_only and tool not in {
            "ai_dashboard_designer", "undo_dashboard", "present_result", "analyze_result",
            "explain_existing_result",
        }:
            return state.model_copy(update={
                "observations": [*state.observations, {"tool": tool, "status": "UNSUPPORTED",
                    "error": "Saved dashboards use retained data. Run a new governed analysis "
                    "in Conversations for different data, then save a new dashboard."}],
                "decision": None, "steps": state.steps + 1,
                "ledger": [*state.ledger, CallLedgerEntry(
                    component=f"assistant_tool:{tool}", kind="TOOL", status="FAILED"
                )],
            })
        last = (
            state.base_response
            if state.base_response and state.base_response.generation
            and state.base_response.generation.accepted
            else self.get_last_response(self._request.session_id, self._request.db_id)
        )
        previous = self.get_state(self._request.session_id)
        if state.base_response and state.base_response.state:
            previous = state.base_response.state
        has_prior_sql = bool(
            previous and previous.db_id == self._request.db_id and previous.last_sql
        )
        if (tool in CACHED_RESULT_TOOLS and last is None) or (
            tool in PRIOR_SQL_TOOLS and not has_prior_sql
            and not (last and last.generation and last.generation.accepted)
        ):
            facts = {"tool": tool, "error": "No accepted result is available."}
            return state.model_copy(
                update={
                    "observations": [*state.observations, facts],
                    "decision": None,
                    "steps": state.steps + 1,
                    "ledger": [
                        *state.ledger,
                        CallLedgerEntry(
                            component=f"assistant_tool:{tool}",
                            kind="TOOL",
                            status="FAILED",
                            latency_ms=round((perf_counter() - started) * 1000),
                        ),
                    ],
                }
            )
        base = state.base_response
        downstream_usage = TokenUsage()
        downstream_ledger: list[CallLedgerEntry] = []
        if tool == "reset_context" and self.reset_handler is not None:
            self.reset_handler(self._request.session_id)
            facts = {"tool": tool, "status": "reset"}
            base = ChatResponse(
                session_id=self._request.session_id,
                operation="RESET_CONTEXT",
                resolved_question="",
                message="Conversation context was reset.",
            )
        elif tool in {"ask_data", "check_sql_correctness", "reset_context"}:
            if tool == "check_sql_correctness":
                request_message = "Check correctness of the previous SQL. " + self._request.message
            elif state.decision.intent == "INVESTIGATE":
                # An investigation must be able to ask genuinely different
                # governed analytical questions. The original request remains
                # in evidence; the SQL engine still owns validation/execution.
                request_message = question
                if request_message.strip().casefold() == self._request.message.strip().casefold():
                    facts = {
                        "tool": tool,
                        "error": (
                            "Investigation objective repeats the original question; "
                            "request a distinct analytical breakdown."
                        ),
                    }
                    return state.model_copy(update={
                        "observations": [*state.observations, facts],
                        "decision": None,
                        "steps": state.steps + 1,
                    })
            else:
                # The specialist must receive the objective the orchestrator
                # actually resolved, not the short surface follow-up.
                request_message = question
            assistant_operation = None
            if tool == "ask_data" and state.decision.intent != "INVESTIGATE":
                assistant_operation = tool_input.operation or {
                    "NEW_QUERY": "NEW_QUERY",
                    "MODIFY_QUERY": "REFINE",
                    "CORRECT": "CORRECTION",
                    "OPTIMIZE": "OPTIMIZE",
                }.get(state.decision.intent)
            investigation_evidence = self._request.evidence
            if state.decision.intent == "INVESTIGATE":
                accepted = previous if previous and previous.db_id == self._request.db_id else None
                constraints = (
                    accepted.semantic_contract.model_dump(
                        mode="json", exclude_none=True, exclude_defaults=True
                    )
                    if accepted and accepted.semantic_contract else {}
                )
                investigation_evidence = json.dumps({
                    "original_client_request": self._request.message,
                    "previous_accepted_question": accepted.resolved_question if accepted else None,
                    "approved_analytical_constraints": constraints,
                    "trusted_evidence": self._request.evidence,
                }, separators=(",", ":"))
            request = self._request.model_copy(
                update={
                    "message": request_message,
                    "session_id": (
                        f"inv_{self._investigation_id}_{state.steps}"
                        if state.decision.intent == "INVESTIGATE"
                        else self._request.session_id
                    ),
                    "parent_run_id": (
                        None
                        if state.decision.intent == "INVESTIGATE"
                        else tool_input.source_run_id or self._request.parent_run_id
                    ),
                    "evidence": investigation_evidence,
                    "assistant_operation": assistant_operation,
                    "assistant_changes": tool_input.changes,
                    "assistant_original_question": self._request.message,
                    "visualization": (
                        False
                        if tool == "ask_data" and parse_intent(self._request.message).requested
                        else self._request.visualization
                    ),
                }
            )
            request._history_source_run_id = tool_input.source_run_id
            if state.base_response and state.base_response.state and not tool_input.source_run_id:
                request._turn_state = previous
                if state.base_response.state_commit_pending:
                    request._turn_expected_version = state.base_response.state_expected_version
            handler = (
                (self.investigation_handler or self.sql_handler)
                if state.decision.intent == "INVESTIGATE"
                else self.sql_handler
            )
            response = await handler(request)
            if response.generation and response.generation.accepted and not response.result_id:
                response = response.model_copy(update={"result_id": uuid4().hex})
            facts = {"tool": tool, **self._result_facts(response)}
            if state.decision.intent == "INVESTIGATE" and response.generation:
                facts["evidence_id"] = request.session_id
                facts["calculated_evidence"] = self._analyze_result(response)
            downstream_usage = response.token_usage
            downstream_ledger = response.call_ledger
            if state.decision.intent != "INVESTIGATE" and (
                (response.generation is not None and response.generation.accepted)
                or base is None
            ):
                # Account each delegated call once, including when the model
                # invokes more than one tool in the same conversation turn.
                base = response.model_copy(
                    update={"token_usage": TokenUsage(), "call_ledger": []}
                )
        elif tool in {"inspect_result", "search_conversation_memory"}:
            reference = question.strip()
            inspected = None
            if self.history_inspector:
                inspected = await asyncio.to_thread(
                    self.history_inspector, self._request.session_id,
                    self._request.db_id, reference,
                )
            if inspected and not reference.startswith("snapshot:"):
                inspected = {k:v for k,v in inspected.items()
                             if k not in {"rows", "rows_preview", "result_sample", "answer"}}
            unsupported = inspected and inspected.get("evidence_kind") == "SNAPSHOT_UNAVAILABLE"
            facts = {"tool":tool, "status":"UNSUPPORTED" if unsupported else
                     "OK" if inspected else "UNAVAILABLE", "historical_result":inspected,
                     "reason":(inspected or {}).get("limitation") if unsupported else
                     None if inspected else "No accepted SQL is available for that reference."}
        elif tool == "refresh_analysis":
            if self.history_refresher is None:
                facts = {"tool":tool, "error":"Historical refresh is not configured."}
            else:
                refresh_request = self._request.model_copy()
                if state.base_response and state.base_response.state:
                    refresh_request._turn_state = previous
                    refresh_request._turn_expected_version = (
                        state.base_response.state_expected_version
                        if state.base_response.state_commit_pending else previous.version
                        if previous else None
                    )
                response = await self.history_refresher(refresh_request, question.strip())
                facts = {"tool":tool, **self._result_facts(response),
                         "data_provenance":"REFRESHED_CURRENT_DATA_NOT_HISTORICAL_SNAPSHOT"}
                downstream_usage = response.token_usage
                downstream_ledger = response.call_ledger
                if response.generation and response.generation.accepted:
                    base = response.model_copy(update={
                        "token_usage":TokenUsage(), "call_ledger":[],
                    })
        elif tool in {"get_business_context", "get_available_metrics"}:
            output = self.bridge.get_business_context(
                MetadataToolInput(
                    db_id=self._request.db_id,
                    question=question
                    if tool == "get_business_context"
                    else "What metrics are available?",
                )
            )
            facts = ToolObservation(
                tool=tool, facts={"definitions": output.facts, "found": output.found},
                evidence_id=output.evidence_id,
            ).model_dump()
        elif tool in {
            "list_tables",
            "search_schema",
            "describe_table",
            "get_relationships",
            "get_column_profile",
        }:
            self._budget.claim("database")
            output = await self.bridge.inspect_metadata(
                self._request.db_id,
                cast(
                    Literal[
                        "list_tables",
                        "search_schema",
                        "describe_table",
                        "get_relationships",
                        "get_column_profile",
                    ],
                    tool,
                ),
                question,
            )
            facts = ToolObservation(
                tool=tool, facts={"metadata": output.facts, "found": output.found},
                evidence_id=output.evidence_id,
            ).model_dump()
        elif tool == "analyze_result":
            if last is None:
                facts = {"tool": tool, "error": "No accepted result is available."}
            else:
                facts = {"tool": tool, **self._analyze_result(last)}
        elif tool == "explain_existing_result":
            sql = last.generation.sql if last and last.generation else None
            if not sql and previous and previous.db_id == self._request.db_id:
                sql = previous.last_sql
            references = None
            if sql:
                tree = parse_one(sql)
                references = {
                    "tables": sorted({node.name for node in tree.find_all(exp.Table)}),
                    "columns": sorted({node.sql() for node in tree.find_all(exp.Column)}),
                }
            facts = (
                {
                    "tool": tool,
                    "result": self._cached_facts(last, include_sql=True) if last else None,
                    "accepted_sql_references": references,
                }
                if sql or last
                else {"tool": tool, "error": "No accepted SQL or result is available."}
            )
        elif tool in {"ai_dashboard_designer", "undo_dashboard"}:
            if last is None or self.dashboard_handler is None:
                facts = {"tool": tool, "error": "Dashboard design is unavailable."}
            else:
                base = await self.dashboard_handler(
                    self._request, question, last, tool == "undo_dashboard"
                )
                downstream_usage = base.token_usage
                downstream_ledger = base.call_ledger
                facts = {"tool": tool, "status": "DASHBOARD_READY" if base.dashboard
                         else "REQUERY_REQUIRED", "message": base.message,
                         "result_id": base.result_id}
                if not base.dashboard:
                    facts["reason"] = base.message
                    # A design limitation is not a new accepted analysis. Keep
                    # same-turn SQL/contract available for the governed requery.
                    base = state.base_response
        elif tool == "present_result":
            facts = {"tool": tool, "result": self._cached_facts(last) if last else None}
            if last and last.generation:
                base = ChatResponse(
                    session_id=self._request.session_id,
                    result_id=last.result_id,
                    chart_id=last.chart_id,
                    operation="EXPLAIN_RESULT",
                    resolved_question=last.generation.question,
                    generation=last.generation,
                    presentation=last.presentation,
                    message=last.message,
                    state=previous,
                    provenance=[*last.provenance, "accepted_result_reused"],
                )
        elif tool == "change_visualization":
            # The plotting specialist receives the exact accepted rows, including
            # a result produced earlier in this same user turn. It cannot run SQL.
            failed_current_sql = any(
                item.get("tool") in {"ask_data", "refresh_analysis"}
                and item.get("status") != "executed"
                for item in state.observations
            )
            source = (
                state.base_response
                if state.base_response and state.base_response.generation
                and state.base_response.generation.accepted
                else None if failed_current_sql else last
            )
            plot_provider = os.environ.get("TEXT2SQL_PLOTTING_PROVIDER") or self._request.provider
            plot_model = os.environ.get("TEXT2SQL_PLOTTING_MODEL") or self._request.model
            plot_completer = self.completers.get(plot_provider)
            local_type_edit = bool(
                source and source.presentation and source.presentation.chart_spec
                and parse_intent(question, has_previous=True).explicit_type
            )
            proposal = state.decision.chart_proposal or state.chart_proposal
            if plot_completer is None and not local_type_edit and proposal is None:
                facts = {"tool": tool, "error": "Plotting model provider is unavailable."}
            else:
                plot_objective = (
                    self._request.message
                    if self._request.message.strip().casefold().rstrip(".")
                    in {"plot", "plot it", "chart", "chart it", "graph", "show the chart"}
                    else question
                )
                outcome = await design_chart(
                    result=source,
                    objective=plot_objective,
                    completer=plot_completer,
                    provider=plot_provider,
                    model=plot_model,
                    budget=self._budget,
                    previous=previous,
                    db_id=self._request.db_id,
                    fresh_result=source is state.base_response and source is not None,
                    changes=tool_input.changes,
                    provisional_proposal=proposal,
                )
                facts = {
                    "tool": tool,
                    "status": outcome.status,
                    "reason": outcome.reason,
                    "objective": plot_objective,
                    "requested_changes": [item.model_dump() for item in tool_input.changes],
                }
                downstream_usage = outcome.usage
                downstream_ledger = list(outcome.ledger)
                if outcome.spec is None or source is None or source.generation is None:
                    facts["error"] = outcome.reason
                else:
                    presentation = source.presentation or ResultPresentation(summary=source.message)
                    presentation = presentation.model_copy(update={
                        "chart_spec": outcome.spec,
                        "chart_type": outcome.spec.chart_type,
                        "x": outcome.spec.x,
                        "y": outcome.spec.y,
                        "title": outcome.spec.title,
                        "x_label": outcome.spec.x_label,
                        "y_label": outcome.spec.y_label,
                        "number_format": outcome.spec.number_format,
                        "time_format": outcome.spec.time_format,
                    })
                    base = ChatResponse(
                        session_id=self._request.session_id,
                        result_id=source.result_id,
                        chart_id=uuid4().hex,
                        state=(source.state if isinstance(source, ChatResponse) else previous),
                        state_commit_pending=(
                            source.state_commit_pending
                            if isinstance(source, ChatResponse) else False
                        ),
                        state_expected_version=(
                            source.state_expected_version
                            if isinstance(source, ChatResponse) else None
                        ),
                        operation="VISUALIZE",
                        resolved_question=source.generation.question,
                        generation=source.generation,
                        message=source.message,
                        presentation=presentation,
                        presentation_state=PresentationState(
                            mode="chart", chart_type=outcome.spec.chart_type,
                            x=outcome.spec.x, y=outcome.spec.y,
                            title=outcome.spec.title, source_run_id=source.run_id,
                            source_result_id=source.result_id,
                        ),
                        chart_recommendations=outcome.recommendations,
                        plot_status="CHART_READY",
                        provenance=[*source.provenance, "validated_plotting_result"],
                    )
                    facts["chart"] = outcome.spec.model_dump(exclude={"data"})
                    facts["row_count"] = source.generation.row_count
                    downstream_ledger.append(CallLedgerEntry(
                        component="result_cache_reuse", kind="TOOL", status="SUCCEEDED"
                    ))
        else:
            facts = {"tool": tool, "error": "Unsupported tool."}
        context = state.turn_context
        if context is not None and base is not None:
            context = context.model_copy(update={
                "source_result_id": base.result_id or context.source_result_id,
                "source_run_id": (
                    base.presentation_state.source_run_id
                    if base.presentation_state and base.presentation_state.source_run_id
                    else base.run_id or context.source_run_id
                ),
                "chart_id": base.chart_id or context.chart_id,
                "pending_goal": (
                    "NONE" if base.chart_id
                    else "CHART" if context.pending_goal in {"CHART", "SQL_AND_CHART"}
                    else context.pending_goal
                ),
            })
        return state.model_copy(
            update={
                "turn_context": context,
                "observations": [*state.observations, facts],
                "base_response": base,
                "pending_dashboard_objective": (
                    question if tool == "ai_dashboard_designer"
                    and facts.get("status") == "REQUERY_REQUIRED"
                    else None if facts.get("status") == "DASHBOARD_READY"
                    else state.pending_dashboard_objective
                ),
                "chart_proposal": (
                    state.decision.chart_proposal
                    if tool in {
                        "ask_data", "refresh_analysis", "reset_context", "check_sql_correctness"
                    }
                    else state.chart_proposal
                ),
                "decision": None,
                "steps": state.steps + 1,
                "usage": self._sum_usage(state.usage, downstream_usage),
                "ledger": [
                    *state.ledger,
                    *downstream_ledger,
                    CallLedgerEntry(
                        component=f"assistant_tool:{tool}",
                        kind="TOOL",
                        status=(
                            "FAILED" if "error" in facts
                            or facts.get("status") in {
                                "not_accepted", "REQUERY_REQUIRED", "UNSUPPORTED"
                            }
                            else "SUCCEEDED"
                        ),
                        latency_ms=round((perf_counter() - started) * 1000),
                    ),
                ],
            }
        )

    @staticmethod
    def _cached_facts(
        last: RecentResult | ChatResponse, *, include_sql: bool = False
    ) -> dict[str, Any]:
        generation = last.generation
        facts = {
            "columns": generation.columns if generation else [],
            "rows_preview": generation.rows[:10] if generation else [],
            "row_count": generation.row_count if generation else 0,
            "truncated": generation.truncated if generation else False,
            "chart": (
                last.presentation.chart_spec.model_dump(exclude={"data"})
                if last.presentation and last.presentation.chart_spec
                else None
            ),
        }
        if include_sql:
            facts["accepted_sql"] = generation.sql if generation else None
        return facts

    @staticmethod
    def _analyze_result(last: RecentResult | ChatResponse) -> dict[str, Any]:
        generation = last.generation
        if (
            generation is None
            or generation.truncated
            or generation.row_count != len(generation.rows)
        ):
            return {"error": "A complete accepted result is required for numerical analysis."}
        summary: dict[str, Any] = {"row_count": generation.row_count, "columns": {}}
        numeric_columns: list[tuple[int, str]] = []
        for index, name in enumerate(generation.columns):
            values = [row[index] for row in generation.rows]
            numbers = [
                float(value) for value in values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if not numbers:
                continue
            numeric_columns.append((index, name))
            summary["columns"][name] = {
                "count": len(numbers),
                "missing": len(values) - len(numbers),
                "minimum": min(numbers),
                "maximum": max(numbers),
                "mean": mean(numbers),
                "median": median(numbers),
                "sum": sum(numbers),
            }
        dimension_candidates = [
            (index, name)
            for index, name in enumerate(generation.columns)
            if index not in {item[0] for item in numeric_columns}
        ]
        if len(dimension_candidates) == 1 and numeric_columns:
            dimension_index, dimension_name = dimension_candidates[0]
            rankings: dict[str, Any] = {}
            for numeric_index, numeric_name in numeric_columns[:4]:
                ranked = [
                    (str(row[dimension_index]), float(row[numeric_index]))
                    for row in generation.rows
                    if row[dimension_index] is not None
                    and isinstance(row[numeric_index], (int, float))
                    and not isinstance(row[numeric_index], bool)
                ]
                if not ranked:
                    continue
                highest = max(value for _, value in ranked)
                lowest = min(value for _, value in ranked)
                rankings[numeric_name] = {
                    "dimension": dimension_name,
                    "highest_rows": [item for item in ranked if item[1] == highest],
                    "lowest_rows": [item for item in ranked if item[1] == lowest],
                    "note": "Ranking among returned rows only; no causal interpretation.",
                }
            if rankings:
                summary["rankings"] = rankings
        time_columns = [
            (index, name)
            for index, name in enumerate(generation.columns)
            if any(word in name.casefold() for word in ("date", "month", "year", "period"))
        ]
        if len(time_columns) == 1 and numeric_columns and len(generation.rows) >= 2:
            time_index, time_name = time_columns[0]
            measure_candidates = [item for item in numeric_columns if item[0] != time_index]
            if not measure_candidates:
                return summary
            numeric_index, numeric_name = measure_candidates[0]
            points = [
                (str(row[time_index]), float(row[numeric_index]))
                for row in generation.rows
                if row[time_index] is not None
                and isinstance(row[numeric_index], (int, float))
                and not isinstance(row[numeric_index], bool)
            ]
            if (
                len(points) >= 2
                and len({item[0] for item in points}) == len(points)
                and all(
                    re.fullmatch(r"\d{4}(?:-?\d{2}(?:-?\d{2})?)?", item[0])
                    for item in points
                )
            ):
                points.sort(key=lambda item: item[0])
                first, last_point = points[0], points[-1]
                delta = last_point[1] - first[1]
                summary["observed_trend"] = {
                    "time_column": time_name,
                    "measure_column": numeric_name,
                    "from": first,
                    "to": last_point,
                    "absolute_change": delta,
                    "percent_change": (
                        delta / first[1] * 100 if first[1] != 0 else None
                    ),
                    "decline_observed": delta < 0,
                    "note": "Descriptive comparison only; no causal conclusion.",
                }
        return summary
