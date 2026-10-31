from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.api import create_app
from semantic_text2sql.conversational_agent import (
    AGENT_DECISION_SCHEMA,
    ConversationalAnalyticsAgent,
)
from semantic_text2sql.conversational_bridge import (
    ConversationalAnalyticsBridge,
    _route,
)
from semantic_text2sql.data_requirement_validator import _stored_grain_capability
from semantic_text2sql.glossary import GlossaryStore
from semantic_text2sql.hybrid_retrieval import HybridSchemaRetriever
from semantic_text2sql.llm import GroqSQLModel
from semantic_text2sql.models import (
    CallLedgerEntry,
    ChatRequest,
    ChatResponse,
    ConversationState,
    GenerateResponse,
    SemanticContract,
    SemanticFilter,
    StrategyHints,
    TokenUsage,
)
from semantic_text2sql.profiling import ProfileStore
from semantic_text2sql.result_cache import RecentResult
from semantic_text2sql.run_history import RunHistoryStore
from semantic_text2sql.runtime import RequestBudget, active_request_budget, use_request_budget


class Encoder:
    def encode(self, texts: Any) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class Completer:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = iter(decisions)
        self.calls = 0
        self.prompts: list[str] = []

    async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        self.calls += 1
        self.prompts.append(prompt)
        return json.dumps(next(self.decisions)), TokenUsage(input_tokens=10, output_tokens=5)


def _agent(
    registry: Any, tmp_path: Any, completer: Completer, calls: list[str]
) -> ConversationalAnalyticsAgent:
    async def sql_handler(request: ChatRequest) -> ChatResponse:
        calls.append(request.message)
        return ChatResponse(
            session_id=request.session_id,
            operation="NEW_QUERY",
            resolved_question=request.message,
            message="The query ran.",
            generation=GenerateResponse(
                db_id=request.db_id, question=request.message, provider=request.provider,
                model=request.model, strategy=StrategyHints(mode="exact"), sql="SELECT 1",
                accepted=True, execution_status="ACCEPTED", termination_reason="accepted",
                rows=[[1]], columns=["count"], row_count=1,
            ),
        )

    bridge = ConversationalAnalyticsBridge(
        database=registry,
        postgres=None,
        profiles=ProfileStore(tmp_path / "profiles"),
        glossaries=GlossaryStore(tmp_path / "glossaries"),
        retriever=HybridSchemaRetriever(Encoder()),
        sql_handler=sql_handler,
    )
    return ConversationalAnalyticsAgent(
        bridge=bridge,
        completers={"agentrouter": completer},
        get_state=lambda session_id: None,
        get_last_response=lambda session_id, db_id: None,
        sql_handler=sql_handler,
    )


def test_agent_answers_greeting_with_one_model_call_and_no_sql(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[str] = []
    completer = Completer([{
        "action": "ANSWER", "intent": "DIRECT_ANSWER", "tool": None,
        "instruction": None, "answer": "Hello! Ask me about your data.",
    }])
    agent = _agent(registry, tmp_path, completer, calls)
    response = asyncio.run(
        agent.run(ChatRequest(session_id="hello", db_id="shop", message="Hello"))
    )
    assert "Hello" in response.message
    assert completer.calls == 1
    assert calls == []


def test_agent_inspects_historical_reference_without_sql(registry: Any, tmp_path: Any) -> None:
    calls: list[str] = []
    completer = Completer([
        {"action":"CALL_TOOL", "intent":"EXPLAIN", "tool":"inspect_result",
         "tool_arguments":{"objective":"result_old"}, "answer":None},
        {"action":"ANSWER", "intent":"DIRECT_ANSWER", "tool":None,
         "answer":"The retained historical preview contains five rows."},
    ])
    agent = _agent(registry, tmp_path, completer, calls)
    references: list[tuple[str, str, str]] = []

    def inspect(session: str, db: str, reference: str) -> dict[str, Any]:
        references.append((session, db, reference))
        return {"result_id":reference, "complete":False, "page_complete":False,
                "calculation_allowed":False, "rows":[[1], [2], [3], [4], [5]]}

    agent.history_inspector = inspect
    response = asyncio.run(agent.run(ChatRequest(
        session_id="history", db_id="shop", message="Inspect result_old",
    )))
    assert references == [("history", "shop", "result_old")]
    assert calls == []
    assert "five rows" in response.message
    assert completer.calls == 2


def test_provider_dns_failure_has_specific_safe_message() -> None:
    from semantic_text2sql.llm import ModelError

    message = ConversationalAnalyticsAgent._provider_failure_message(
        ModelError("The Groq request failed: [Errno 8] nodename nor servname provided")
    )
    assert "could not reach the model provider over the network" in message
    assert "Errno" not in message


def test_agent_can_inspect_then_answer_without_sql(registry: Any, tmp_path: Any) -> None:
    calls: list[str] = []
    completer = Completer(
        [
            {"action": "CALL_TOOL", "intent": "EXPLORE_SCHEMA", "tool": "list_tables",
             "instruction": "List available tables", "answer": None},
            {"action": "ANSWER", "intent": "EXPLORE_SCHEMA", "tool": None,
             "instruction": None, "answer": "The database contains a customers table."},
        ]
    )
    agent = _agent(registry, tmp_path, completer, calls)
    response = asyncio.run(
        agent.run(
            ChatRequest(session_id="schema", db_id="shop", message="Which tables are available?")
        )
    )
    assert response.message == "The database contains a customers table."
    assert completer.calls == 2
    assert calls == []


def test_agent_policy_and_structured_tool_objective(registry: Any, tmp_path: Any) -> None:
    calls: list[str] = []
    completer = Completer([
        {"action": "CALL_TOOL", "intent": "EXPLORE_SCHEMA", "tool": "search_schema",
         "tool_arguments": {"objective": "Find customer tables and their columns"},
         "answer": None},
        {"action": "ANSWER", "intent": "EXPLORE_SCHEMA", "tool": None,
         "tool_arguments": None, "answer": "The schema search is complete."},
    ])
    agent = _agent(registry, tmp_path, completer, calls)
    response = asyncio.run(agent.run(ChatRequest(
        session_id="structured", db_id="shop", message="Which customer tables exist?"
    )))
    assert response.message == "The schema search is complete."
    assert calls == []
    assert "single client-facing AI Business Analytics Assistant" in completer.prompts[0]
    assert "Do not regenerate SQL when verified existing data is sufficient" in completer.prompts[0]
    assert "tool_arguments" in completer.prompts[0]
    assert "search_schema" in completer.prompts[1]


def test_agent_delegates_analytical_question_and_explains_it(registry: Any, tmp_path: Any) -> None:
    calls: list[str] = []
    completer = Completer(
        [
            {
                "action": "CALL_TOOL",
                "intent": "NEW_QUERY",
                "tool": "ask_data",
                "instruction": "Count customers",
                "answer": None,
            },
            {
                "action": "ANSWER",
                "intent": "NEW_QUERY",
                "tool": None,
                "instruction": None,
                "answer": "The query ran.",
            },
        ]
    )
    agent = _agent(registry, tmp_path, completer, calls)
    response = asyncio.run(
        agent.run(
            ChatRequest(session_id="sql", db_id="shop", message="How many customers are there?")
        )
    )
    assert response.message == "The query ran."
    assert calls == ["Count customers"]
    assert completer.calls == 2


def test_final_model_failure_preserves_governed_result(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[str] = []
    completer = Completer([
        {"action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
         "instruction": "Count customers", "answer": None},
    ])
    agent = _agent(registry, tmp_path, completer, calls)
    result = asyncio.run(agent.run(ChatRequest(
        session_id="fallback", db_id="shop", message="Count customers"
    )))
    assert calls == ["Count customers"]
    assert result.operation == "NEW_QUERY"
    assert result.message == "The query ran."
    assert "assistant_response_unavailable" in result.provenance
    assert result.token_usage.total_tokens == 15
    assert any(item.status == "FAILED" for item in result.call_ledger)


def test_analytical_wording_is_not_mistaken_for_a_definition() -> None:
    assert _route("What does revenue look like over time?") == "sql"
    assert _route("What does retention mean?") == "business"
    assert _stored_grain_capability(["year_month", "revenue"]) > 0


def test_used_columns_are_read_from_accepted_sql(registry: Any, tmp_path: Any) -> None:
    calls: list[str] = []
    completer = Completer([
        {"action": "CALL_TOOL", "intent": "EXPLAIN", "tool": "explain_existing_result",
         "instruction": "Inspect columns referenced by the accepted SQL", "answer": None},
        {"action": "ANSWER", "intent": "EXPLAIN", "tool": None,
         "instruction": None, "answer": "The accepted SQL used c.country and c.customer_id."},
    ])
    agent = _agent(registry, tmp_path, completer, calls)
    previous = ConversationState(
        session_id="prior",
        db_id="shop",
        root_question="Count customers",
        resolved_question="Count customers",
        last_sql="SELECT c.country, COUNT(c.customer_id) FROM customers AS c GROUP BY c.country",
    )
    agent.get_state = lambda session_id: previous
    response = asyncio.run(
        agent.run(
            ChatRequest(session_id="prior", db_id="shop", message="Which columns did you use?")
        )
    )
    assert "c.country" in response.message
    assert "c.customer_id" in response.message
    assert completer.calls == 2
    assert calls == []


def test_opt_in_agent_uses_existing_api_contract(
    registry: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    class SQLModel:
        async def generate(self, **kwargs: Any) -> tuple[str, int]:
            return "SELECT COUNT(*) FROM customers", 1

    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    monkeypatch.setenv("TEXT2SQL_CONVERSATIONAL_AGENT_ENABLED", "true")
    completer = Completer([{
        "action": "ANSWER", "intent": "DIRECT_ANSWER", "tool": None,
        "instruction": None, "answer": "Hello!",
    }])
    client = TestClient(
        create_app(
            TextToSQLAgent(registry, SQLModel()),
            conversation_completers={"agentrouter": completer},
            run_history=RunHistoryStore(None),
        )
    )
    response = client.post(
        "/api/chat",
        json={
            "session_id": "agent-api",
            "db_id": "shop",
            "model": "test-model",
            "message": "Hello",
        },
    )
    assert response.status_code == 200
    assert response.json()["operation"] == "EXPLAIN_CONTEXT"
    assert completer.calls == 1


def test_invalid_agent_decision_fails_closed(registry: Any, tmp_path: Any) -> None:
    calls: list[str] = []
    completer = Completer([{"action": "CALL_TOOL", "tool": "run_arbitrary_sql"}])
    agent = _agent(registry, tmp_path, completer, calls)
    response = asyncio.run(
        agent.run(ChatRequest(session_id="invalid", db_id="shop", message="Inspect this database"))
    )
    assert "No query was run" in response.message
    assert calls == []


def test_assistant_failure_is_marked_not_presented_as_model_answer(
    registry: Any, tmp_path: Any
) -> None:
    agent = _agent(registry, tmp_path, Completer([]), [])
    result = asyncio.run(
        agent.run(ChatRequest(session_id="down", db_id="shop", message="Hello"))
    )
    assert "model provider" in result.message.lower()
    assert result.provenance == ["assistant_service_failure"]


def _accepted_result() -> RecentResult:
    response = ChatResponse(
        session_id="analysis",
        operation="NEW_QUERY",
        resolved_question="Show monthly revenue",
        message="The query ran.",
        generation=GenerateResponse(
            db_id="shop",
            question="Show monthly revenue",
            provider="agentrouter",
            model="test-model",
            strategy=StrategyHints(mode="exact"),
            sql="SELECT month, revenue FROM revenue_by_month",
            accepted=True,
            termination_reason="accepted",
            columns=["month", "revenue"],
            rows=[["2025-01", 10], ["2025-02", 15]],
            row_count=2,
        ),
    )
    return RecentResult.from_response(response)


def test_analyze_result_calculates_verified_change() -> None:
    analysis = ConversationalAnalyticsAgent._analyze_result(_accepted_result())
    assert analysis["columns"]["revenue"]["mean"] == 12.5
    assert analysis["observed_trend"]["absolute_change"] == 5
    assert analysis["observed_trend"]["percent_change"] == 50
    assert analysis["observed_trend"]["decline_observed"] is False
    assert analysis["rankings"]["revenue"]["highest_rows"] == [("2025-02", 15.0)]


def test_metadata_capabilities_are_distinct(registry: Any, tmp_path: Any) -> None:
    agent = _agent(registry, tmp_path, Completer([]), [])
    tables = asyncio.run(agent.bridge.inspect_metadata("shop", "list_tables"))
    detail = asyncio.run(agent.bridge.inspect_metadata("shop", "describe_table", "customers"))
    relationships = asyncio.run(
        agent.bridge.inspect_metadata("shop", "get_relationships", "orders")
    )
    assert any("customers:" in fact for fact in tables.facts)
    assert any("customer_id" in fact for fact in detail.facts)
    assert all("→" in fact for fact in relationships.facts) or not relationships.found
    assert tables.evidence_id != detail.evidence_id


def test_false_decline_premise_can_stop_without_sql(registry: Any, tmp_path: Any) -> None:
    calls: list[str] = []
    completer = Completer(
        [
            {"action": "CALL_TOOL", "intent": "INVESTIGATE", "tool": "analyze_result",
             "instruction": "Check whether revenue declined", "answer": None},
            {"action": "ANSWER", "intent": "INVESTIGATE", "tool": None,
             "instruction": None, "answer": "Revenue rose from 10 to 15; no decline is shown."},
        ]
    )
    agent = _agent(registry, tmp_path, completer, calls)
    agent.get_last_response = lambda session_id, db_id: _accepted_result()
    result = asyncio.run(
        agent.run(
            ChatRequest(session_id="analysis", db_id="shop", message="Why did revenue decline?")
        )
    )
    assert "no decline" in result.message
    assert calls == []
    assert completer.calls == 2


def test_agent_investigation_uses_distinct_objectives_and_preserves_session(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[tuple[str, str]] = []

    async def sql_handler(request: ChatRequest) -> ChatResponse:
        calls.append((request.session_id, request.message))
        return ChatResponse(
            session_id=request.session_id,
            operation="NEW_QUERY",
            resolved_question=request.message,
            message="The query ran.",
            generation=GenerateResponse(
                db_id=request.db_id, question=request.message, provider=request.provider,
                model=request.model, strategy=StrategyHints(mode="exact"), sql="SELECT 1",
                accepted=True, execution_status="ACCEPTED", termination_reason="accepted",
                rows=[[1]], columns=["count"], row_count=1,
            ),
            token_usage=TokenUsage(input_tokens=7, output_tokens=3),
            call_ledger=[
                CallLedgerEntry(
                    component="sql_generation", kind="MODEL", status="SUCCEEDED",
                    input_tokens=7, output_tokens=3,
                )
            ],
        )

    completer = Completer(
        [
            {"action": "CALL_TOOL", "intent": "INVESTIGATE", "tool": "ask_data",
             "instruction": "Revenue by month in 2025", "answer": None},
            {"action": "CALL_TOOL", "intent": "INVESTIGATE", "tool": "ask_data",
             "instruction": "Revenue by country in September 2025", "answer": None},
            {"action": "ANSWER", "intent": "INVESTIGATE", "tool": None,
             "instruction": None, "answer": "The available breakdowns do not establish a cause."},
        ]
    )
    agent = _agent(registry, tmp_path, completer, [])
    agent.sql_handler = sql_handler
    result = asyncio.run(
        agent.run(ChatRequest(session_id="analysis", db_id="shop", message="Why did revenue fall?"))
    )
    assert [message for _, message in calls] == [
        "Revenue by month in 2025", "Revenue by country in September 2025"
    ]
    assert len({session for session, _ in calls}) == 2
    assert result.session_id == "analysis"
    assert completer.calls == 3
    assert result.token_usage.total_tokens == 65
    assert sum(item.component == "sql_generation" for item in result.call_ledger) == 2


def test_repeated_governed_tools_count_each_model_call_once(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[str] = []

    async def sql_handler(request: ChatRequest) -> ChatResponse:
        calls.append(request.message)
        return ChatResponse(
            session_id=request.session_id,
            operation="NEW_QUERY",
            resolved_question=request.message,
            message="Executed.",
            token_usage=TokenUsage(input_tokens=7, output_tokens=3),
            call_ledger=[CallLedgerEntry(
                component="sql_generation", kind="MODEL", status="SUCCEEDED",
                input_tokens=7, output_tokens=3,
            )],
        )

    completer = Completer([
        {"action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
         "instruction": "First query", "answer": None},
        {"action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
         "instruction": "Second query", "answer": None},
        {"action": "ANSWER", "intent": "NEW_QUERY", "tool": None,
         "instruction": None, "answer": "The second result is available."},
    ])
    agent = _agent(registry, tmp_path, completer, calls)
    agent.sql_handler = sql_handler
    result = asyncio.run(agent.run(
        ChatRequest(session_id="repeated", db_id="shop", message="Compare the data")
    ))
    assert len(calls) == 2
    assert result.token_usage.total_tokens == 65
    assert sum(item.component == "sql_generation" for item in result.call_ledger) == 2


def test_present_existing_result_does_not_call_sql(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[str] = []
    completer = Completer([
        {"action": "CALL_TOOL", "intent": "EXPLAIN", "tool": "present_result",
         "instruction": "Show the accepted result", "answer": None},
        {"action": "ANSWER", "intent": "EXPLAIN", "tool": None,
         "instruction": None, "answer": "The stored result has two monthly rows."},
    ])
    agent = _agent(registry, tmp_path, completer, calls)
    agent.get_last_response = lambda session_id, db_id: _accepted_result()
    result = asyncio.run(agent.run(
        ChatRequest(session_id="analysis", db_id="shop", message="Show that result")
    ))
    assert result.message == "The stored result has two monthly rows."
    assert result.operation == "EXPLAIN_RESULT"
    assert result.generation is not None
    assert result.generation.rows == [["2025-01", 10], ["2025-02", 15]]
    assert calls == []


def test_correction_and_reset_use_the_existing_conversation_service(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[str] = []
    completer = Completer([
        {"action": "ANSWER", "intent": "CORRECT", "tool": None,
         "instruction": None, "answer": "Corrected."},
        {"action": "ANSWER", "intent": "CORRECT", "tool": None,
         "instruction": None, "answer": "The correction was processed."},
        {"action": "ANSWER", "intent": "RESET_CONTEXT", "tool": None,
         "instruction": None, "answer": "Reset."},
        {"action": "ANSWER", "intent": "RESET_CONTEXT", "tool": None,
         "instruction": None, "answer": "The conversation was reset."},
    ])
    agent = _agent(registry, tmp_path, completer, calls)
    correction = asyncio.run(agent.run(ChatRequest(
        session_id="routing", db_id="shop", message="No, only paying customers"
    )))
    reset = asyncio.run(agent.run(ChatRequest(
        session_id="routing", db_id="shop", message="Start over"
    )))
    assert calls == ["No, only paying customers", "Start over"]
    assert correction.message == "The correction was processed."
    assert reset.message == "The conversation was reset."


def test_shared_budget_context_is_restored() -> None:
    first = RequestBudget(10, max_model_calls=1)
    assert active_request_budget() is None
    with use_request_budget(first):
        assert active_request_budget() is first
    assert active_request_budget() is None


def test_delegated_tool_cannot_bypass_parent_model_budget(
    registry: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    monkeypatch.setenv("TEXT2SQL_ASSISTANT_MAX_MODEL_CALLS", "1")
    calls: list[str] = []

    async def governed_handler(request: ChatRequest) -> ChatResponse:
        calls.append(request.message)
        budget = active_request_budget()
        assert budget is not None
        budget.claim("model")
        return ChatResponse(
            session_id=request.session_id, operation="NEW_QUERY",
            resolved_question=request.message, message="Unexpected execution",
        )

    completer = Completer(
        [{"action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
          "instruction": "Count customers", "answer": None}]
    )
    agent = _agent(registry, tmp_path, completer, [])
    agent.sql_handler = governed_handler
    response = asyncio.run(
        agent.run(ChatRequest(session_id="budget", db_id="shop", message="Count customers"))
    )
    assert calls == ["Count customers"]
    assert "budget" in response.message.casefold()


def test_plot_followup_is_selected_by_top_model_and_uses_no_new_sql(
    registry: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    class SQLModel:
        calls = 0

        async def generate(self, **kwargs: Any) -> tuple[str, int]:
            self.calls += 1
            return "SELECT country, COUNT(*) AS total FROM customers GROUP BY country", 1

    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    monkeypatch.setenv("TEXT2SQL_CONVERSATIONAL_AGENT_ENABLED", "true")
    monkeypatch.setenv("TEXT2SQL_PLOTTING_PROVIDER", "groq")
    model = SQLModel()
    plot_completer = Completer([{
        "primary_chart": {
            "chart_type": "bar", "x": "country", "x_components": [],
            "y": "total", "y_measures": ["total"], "series": None,
            "reason": "Compare customer counts by country.",
        },
        "alternative_charts": [],
    }])
    completer = Completer(
        [
            {"action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
             "instruction": "Count customers by country", "answer": None},
            {"action": "ANSWER", "intent": "NEW_QUERY", "tool": None,
             "instruction": None, "answer": "Here are customer counts by country."},
            {"action": "CALL_TOOL", "intent": "VISUALIZE", "tool": "change_visualization",
             "instruction": "Plot the accepted customer counts", "answer": None},
            {"action": "ANSWER", "intent": "VISUALIZE", "tool": None,
             "instruction": None, "answer": "Here is a chart of those customer counts."},
            {"action": "CALL_TOOL", "intent": "VISUALIZE", "tool": "change_visualization",
             "instruction": "Plot the accepted customer counts", "answer": None},
            {"action": "ANSWER", "intent": "VISUALIZE", "tool": None,
             "instruction": None, "answer": "The stored customer counts remain charted."},
            {"action": "CALL_TOOL", "intent": "RESET_CONTEXT", "tool": "reset_context",
             "instruction": "Reset this conversation", "answer": None},
            {"action": "ANSWER", "intent": "RESET_CONTEXT", "tool": None,
             "instruction": None, "answer": "The conversation was reset."},
        ]
    )
    history = RunHistoryStore(None)
    client = TestClient(
        create_app(
            TextToSQLAgent(registry, model),
            conversation_completers={"agentrouter": completer, "groq": plot_completer},
            run_history=history,
        )
    )
    base = {
        "session_id": "agent-chart", "db_id": "shop", "provider": "agentrouter",
        "model": "test-model",
    }
    initial = client.post(
        "/api/chat", json={**base, "message": "Count customers by country."}
    )
    assert initial.status_code == 200
    assert initial.json()["generation"]["accepted"] is True
    saved_initial = history.get(initial.json()["run_id"], "agent-chart")
    assert saved_initial is not None
    assert saved_initial.response_payload["message"] == "Here are customer counts by country."
    assert saved_initial.token_usage["input_tokens"] == (
        initial.json()["token_usage"]["input_tokens"]
    )
    assert saved_initial.token_usage["output_tokens"] == (
        initial.json()["token_usage"]["output_tokens"]
    )
    assert model.calls == 1
    assert completer.calls == 2
    plotted = client.post("/api/chat", json={**base, "message": "Plot it."})
    assert plotted.status_code == 200
    assert plotted.json()["presentation"]["chart_spec"] is not None
    assert plotted.json()["result_id"] == initial.json()["result_id"]
    assert plotted.json()["chart_id"] != initial.json()["chart_id"]
    assert plotted.json()["presentation_state"]["source_result_id"] == initial.json()["result_id"]
    assert plotted.json()["presentation_state"]["source_run_id"] == initial.json()["run_id"]
    assert plotted.json()["message"] == "Here is a chart of those customer counts."
    assert any(
        item["component"] == "result_cache_reuse" for item in plotted.json()["call_ledger"]
    )
    assert model.calls == 1
    assert completer.calls == 4
    assert plot_completer.calls == 1
    saved_plot = history.get(plotted.json()["run_id"], "agent-chart")
    assert saved_plot is not None
    assert saved_plot.response_payload["message"] == plotted.json()["message"]
    bare_plot = client.post("/api/chat", json={**base, "message": "Plot"})
    assert bare_plot.status_code == 200
    assert bare_plot.json()["presentation"]["chart_spec"] is not None
    assert model.calls == 1
    assert completer.calls == 6
    assert plot_completer.calls == 1
    reset = client.post("/api/chat", json={**base, "message": "Start over"})
    assert reset.status_code == 200
    assert reset.json()["operation"] == "RESET_CONTEXT"
    assert reset.json()["message"] == "The conversation was reset."
    assert model.calls == 1
    assert completer.calls == 8


def test_first_turn_sql_then_plot_preserves_result_identity_without_intermediate_run(
    registry: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    class SQLModel:
        calls = 0

        async def generate(self, **kwargs: Any) -> tuple[str, int]:
            self.calls += 1
            return "SELECT country, COUNT(*) AS total FROM customers GROUP BY country", 1

    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    monkeypatch.setenv("TEXT2SQL_CONVERSATIONAL_AGENT_ENABLED", "true")
    monkeypatch.setenv("TEXT2SQL_PLOTTING_PROVIDER", "groq")
    plot_completer = Completer([{
        "primary_chart": {
            "chart_type": "bar", "x": "country", "x_components": [],
            "y": "total", "y_measures": ["total"], "series": None,
            "reason": "Compare customers by country.",
        },
        "alternative_charts": [],
    }])
    top = Completer([
        {"action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
         "instruction": "Count customers by country", "answer": None},
        {"action": "CALL_TOOL", "intent": "VISUALIZE", "tool": "change_visualization",
         "instruction": "Plot those customer counts", "answer": None},
        {"action": "ANSWER", "intent": "VISUALIZE", "tool": None,
         "instruction": None, "answer": "Here are the customer counts by country."},
    ])
    sql_model = SQLModel()
    history = RunHistoryStore(None)
    client = TestClient(create_app(
        TextToSQLAgent(registry, sql_model),
        conversation_completers={"agentrouter": top, "groq": plot_completer},
        run_history=history,
    ))
    result = client.post("/api/chat", json={
        "session_id": "first-sql-plot", "db_id": "shop", "provider": "agentrouter",
        "model": "test-model", "message": "Count customers by country and plot it.",
    })
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["generation"]["accepted"] is True
    assert body["presentation"]["chart_spec"] is not None
    assert body["result_id"]
    assert body["chart_id"]
    assert body["presentation_state"]["source_result_id"] == body["result_id"]
    assert body["presentation_state"]["source_run_id"] == body["run_id"]
    assert len(history.list_sessions("first-sql-plot")) == 1
    assert sql_model.calls == 1


def test_investigation_uses_governed_sql_for_distinct_breakdowns(
    registry: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    class BreakdownSQL:
        def __init__(self) -> None:
            self.questions: list[str] = []

        async def generate(self, **kwargs: Any) -> tuple[str, int]:
            question = str(kwargs.get("question", ""))
            self.questions.append(question)
            if "country" in question.casefold():
                return "SELECT country, COUNT(*) AS n FROM customers GROUP BY country", 1
            return "SELECT COUNT(*) AS n FROM orders", 1

    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    monkeypatch.setenv("TEXT2SQL_CONVERSATIONAL_AGENT_ENABLED", "true")
    model = BreakdownSQL()
    completer = Completer(
        [
            {"action": "CALL_TOOL", "intent": "INVESTIGATE", "tool": "ask_data",
             "instruction": "Count orders", "answer": None},
            {"action": "CALL_TOOL", "intent": "INVESTIGATE", "tool": "ask_data",
             "instruction": "Count customers by country", "answer": None},
            {"action": "ANSWER", "intent": "INVESTIGATE", "tool": None,
             "instruction": None,
             "answer": "These breakdowns are descriptive and do not establish a cause."},
        ]
    )
    client = TestClient(
        create_app(
            TextToSQLAgent(registry, model),
            conversation_completers={"agentrouter": completer},
            run_history=RunHistoryStore(None),
        )
    )
    response = client.post(
        "/api/chat",
        json={"session_id": "investigate", "db_id": "shop", "provider": "agentrouter",
              "model": "test-model", "message": "Why did orders change?"},
    )
    assert response.status_code == 200
    assert response.json()["message"].startswith("These breakdowns")
    assert model.questions == ["Count orders", "Count customers by country"]
    assert completer.calls == 3


class ScriptedStructuredCompleter:
    """Plays raw scripted outputs through ``complete_structured`` and records calls."""

    def __init__(self, items: list[str]) -> None:
        self.items = iter(items)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.prompts: list[str] = []

    async def complete_structured(
        self, model: str, prompt: str, schema_name: str, schema: dict[str, Any]
    ) -> tuple[str, TokenUsage]:
        self.calls.append((schema_name, schema))
        self.prompts.append(prompt)
        return next(self.items), TokenUsage(input_tokens=10, output_tokens=5)


def _decision_agent(
    registry: Any, tmp_path: Any, completer: Any, calls: list[str]
) -> ConversationalAnalyticsAgent:
    async def sql_handler(request: ChatRequest) -> ChatResponse:
        calls.append(request.message)
        return ChatResponse(
            session_id=request.session_id,
            operation="NEW_QUERY",
            resolved_question=request.message,
            message="The query ran.",
        )

    bridge = ConversationalAnalyticsBridge(
        database=registry,
        postgres=None,
        profiles=ProfileStore(tmp_path / "profiles"),
        glossaries=GlossaryStore(tmp_path / "glossaries"),
        retriever=HybridSchemaRetriever(Encoder()),
        sql_handler=sql_handler,
    )
    agent = ConversationalAnalyticsAgent(
        bridge=bridge,
        completers={"agentrouter": completer},
        get_state=lambda session_id: None,
        get_last_response=lambda session_id, db_id: None,
        sql_handler=sql_handler,
    )
    return agent


def test_structured_schema_enumerates_every_allowed_value() -> None:
    """The schema carries every action/intent/tool value and concrete examples."""
    action_spec = AGENT_DECISION_SCHEMA["properties"]["action"]
    intent_spec = AGENT_DECISION_SCHEMA["properties"]["intent"]
    tool_spec = AGENT_DECISION_SCHEMA["properties"]["tool"]
    arguments_spec = AGENT_DECISION_SCHEMA["properties"]["tool_arguments"]

    action_enum = action_spec["enum"]  # type: ignore[index]
    assert action_enum == ["ANSWER", "CALL_TOOL"]

    intent_enum = intent_spec["enum"]  # type: ignore[index]
    for expected in (
        "DIRECT_ANSWER", "NEW_QUERY", "MODIFY_QUERY", "VISUALIZE",
        "EXPLORE_SCHEMA", "BUSINESS_CONTEXT", "EXPLAIN", "INVESTIGATE",
        "CORRECT", "REVIEW_SQL", "OPTIMIZE", "RESET_CONTEXT",
    ):
        assert expected in intent_enum

    tool_enum = [
        value
        for alternative in tool_spec["anyOf"]  # type: ignore[index]
        for value in alternative.get("enum", [None])  # type: ignore[union-attr]
    ]
    assert None in tool_enum
    for expected in (
        "list_tables", "search_schema", "describe_table", "get_relationships",
        "get_column_profile", "get_available_metrics", "get_business_context",
        "ask_data", "analyze_result", "check_sql_correctness", "change_visualization",
        "present_result", "explain_existing_result", "reset_context",
    ):
        assert expected in tool_enum
    assert "run_arbitrary_sql" not in tool_enum

    assert arguments_spec["anyOf"][0]["required"] == [  # type: ignore[index]
        "objective", "operation", "changes", "source_run_id", "dashboard_goal", "source_result_ids"
    ]
    assert AGENT_DECISION_SCHEMA["required"] == [
        "action", "intent", "tool", "tool_arguments", "instruction", "answer",
        "finish_after_tool", "chart_proposal",
    ]

    from semantic_text2sql.conversational_agent import _DECISION_EXAMPLES, AgentDecision, ToolInput
    for example in _DECISION_EXAMPLES:
        decision = AgentDecision.model_validate(example)
        if decision.tool_arguments is not None:
            assert ToolInput.model_validate(decision.tool_arguments) is not None


def test_agent_prompt_carries_schema_rules_and_examples(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[str] = []
    completer = ScriptedStructuredCompleter([
        json.dumps({
            "action": "ANSWER", "intent": "DIRECT_ANSWER", "tool": None,
            "tool_arguments": None, "instruction": None, "answer": "Hello!",
        })
    ])
    agent = _decision_agent(registry, tmp_path, completer, calls)
    asyncio.run(agent.run(ChatRequest(session_id="prompt", db_id="shop", message="Hello")))
    prompt = completer.prompts[0]
    assert "action must be exactly one of: ANSWER, CALL_TOOL." in prompt
    assert "NEW_QUERY" in prompt and "MODIFY_QUERY" in prompt
    assert "tool must be exactly one of: list_tables" in prompt
    assert "tool_arguments contains objective, operation and typed changes" in prompt
    assert '"intent":"NEW_QUERY"' in prompt  # NEW_QUERY example
    assert '"intent":"VISUALIZE"' in prompt  # VISUALIZE example
    assert '"action":"ANSWER"' in prompt  # ANSWER example
    assert "primary decision-maker for conversational analytics" in prompt
    assert "Preserve all unaffected chart settings" in prompt
    assert "Preserve unaffected SemanticContract constraints" in prompt
    assert "REQUERY_REQUIRED" in prompt
    assert "Return only a valid AgentDecision JSON" in prompt


def test_top_layer_receives_accepted_analytical_contract(
    registry: Any, tmp_path: Any,
) -> None:
    calls: list[str] = []
    completer = Completer([{
        "action": "ANSWER", "intent": "DIRECT_ANSWER", "tool": None,
        "tool_arguments": None, "answer": "I have the accepted context.",
    }])
    agent = _agent(registry, tmp_path, completer, calls)
    accepted = ConversationState(
        session_id="contract", db_id="shop", root_question="Revenue",
        resolved_question="Revenue in 2025", last_sql="SELECT 1",
        semantic_contract=SemanticContract(
            measures=["revenue"], aggregation="sum", active_period="2025",
            comparison_periods=["2024"], output_grain=["country"],
            ranked_metric="revenue", ranking_direction="DESC",
            filters=[SemanticFilter(operand="status", operator="=", value="paid")],
        ),
    )
    agent.get_state = lambda session_id: accepted
    asyncio.run(agent.run(ChatRequest(session_id="contract", db_id="shop", message="Explain it")))
    prompt = completer.prompts[0]
    for expected in ('"active_period": "2025"', '"comparison_periods": ["2024"]',
                     '"ranked_metric": "revenue"', '"output_grain": ["country"]',
                     '"aggregation": "sum"', '"value": "paid"'):
        assert expected in prompt
    assert "accepted SemanticContract" in prompt


def test_prose_intent_is_repaired_not_converted_for_total_consumption(
    registry: Any, tmp_path: Any
) -> None:
    """'total consumption' first-turn bug: prose intent triggers one bounded repair."""
    calls: list[str] = []
    completer = ScriptedStructuredCompleter([
        json.dumps({
            "action": "CALL_TOOL",
            "intent": "Sum the total consumption recorded in the database",
            "tool": "ask_data", "instruction": "Total consumption", "answer": None,
        }),
        json.dumps({
            "action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
            "tool_arguments": {"objective": "Show total consumption"}, "answer": None,
        }),
        json.dumps({
            "action": "ANSWER", "intent": "NEW_QUERY", "tool": None,
            "tool_arguments": None, "instruction": None,
            "answer": "The query ran.",
        }),
    ])
    agent = _decision_agent(registry, tmp_path, completer, calls)
    response = asyncio.run(
        agent.run(ChatRequest(session_id="prose", db_id="shop", message="total consumption"))
    )
    assert calls == ["Show total consumption"]  # the resolved objective was forwarded
    assert response.message == "The query ran."
    # Measured model-call count: 1 rejected attempt + 1 repair + 1 final answer, no more.
    assert len(completer.calls) == 3
    statuses = [
        item.status
        for item in response.call_ledger
        if item.kind == "MODEL" and item.component == "conversational_agent"
    ]
    assert statuses == ["FAILED", "SUCCEEDED", "SUCCEEDED"]
    assert response.token_usage.total_tokens == 45  # 10+5 tokens preserved per attempt
    repair_prompt = completer.prompts[1]
    assert "Validation issues:" in repair_prompt
    assert (
        "intent must be one of the allowed values exactly" in repair_prompt
        or "must be one of" in repair_prompt
    )
    assert "|".join(("DIRECT_ANSWER", "NEW_QUERY")) in repair_prompt


def test_malformed_output_fails_closed_after_exactly_one_repair(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[str] = []
    completer = ScriptedStructuredCompleter([
        "I think the right answer is to look at the database.",  # no JSON at all
        json.dumps({"action": "CALL_TOOL", "tool": "run_arbitrary_sql"}),  # still invalid
        json.dumps({
            "action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
            "tool_arguments": {"objective": "Show total consumption"}, "answer": None,
        }),
        json.dumps({
            "action": "ANSWER", "intent": "NEW_QUERY", "tool": None,
            "tool_arguments": None, "instruction": None, "answer": "Executed anyway.",
        }),
    ])
    agent = _decision_agent(registry, tmp_path, completer, calls)
    response = asyncio.run(
        agent.run(ChatRequest(session_id="bad", db_id="shop", message="total consumption"))
    )
    assert calls == []  # never executed a tool from an invalid decision
    assert len(completer.calls) == 2  # exactly one bounded repair attempt
    assert "couldn't interpret that request safely" in response.message
    assert "Invalid JSON" not in response.message  # raw pydantic text stays out
    assert response.provenance == ["assistant_service_failure"]
    statuses = [
        item.status
        for item in response.call_ledger
        if item.kind == "MODEL" and item.component == "conversational_agent"
    ]
    assert statuses == ["FAILED", "FAILED"]
    assert response.token_usage.total_tokens == 30  # usage preserved despite failures
    # The repair prompt repeats the base prompt plus the exact validation issues.
    assert completer.prompts[1].startswith(completer.prompts[0])
    assert "Validation issues:" in completer.prompts[1]
    # Items 3/4 are valid responses that must never be consumed after failing closed.


def test_prose_wrapped_json_never_executes_before_repair(registry: Any, tmp_path: Any) -> None:
    calls: list[str] = []
    valid = json.dumps({
        "action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
        "tool_arguments": {"objective": "Show total consumption"}, "answer": None,
    })
    completer = ScriptedStructuredCompleter([
        "Here is the decision: " + valid,
        valid,
        json.dumps({"action": "ANSWER", "intent": "NEW_QUERY", "answer": "Done."}),
    ])
    response = asyncio.run(_decision_agent(registry, tmp_path, completer, calls).run(
        ChatRequest(session_id="wrapped", db_id="shop", message="total consumption")
    ))
    assert calls == ["Show total consumption"]
    assert len(completer.calls) == 3
    statuses = [
        entry.status for entry in response.call_ledger
        if entry.component == "conversational_agent"
    ]
    assert statuses == [
        "FAILED", "SUCCEEDED", "SUCCEEDED"
    ]


def test_incompatible_intent_tool_is_repaired_before_execution(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[str] = []
    completer = ScriptedStructuredCompleter([
        json.dumps({
            "action": "CALL_TOOL", "intent": "VISUALIZE", "tool": "ask_data",
            "tool_arguments": {"objective": "Plot it"}, "answer": None,
        }),
        json.dumps({
            "action": "CALL_TOOL", "intent": "VISUALIZE", "tool": "change_visualization",
            "tool_arguments": {"objective": "Plot the current result"}, "answer": None,
        }),
    ])
    response = asyncio.run(_decision_agent(registry, tmp_path, completer, calls).run(
        ChatRequest(session_id="wrong-tool", db_id="shop", message="plot it")
    ))
    assert calls == []  # No prior accepted question may be regenerated.
    assert len(completer.calls) == 2  # Invalid decision plus one bounded repair.
    assert response.message


def test_removed_clarify_action_repairs_to_query_for_total_consumption(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[str] = []
    completer = ScriptedStructuredCompleter([
        json.dumps({
            "action": "CLARIFY", "intent": "CLARIFY", "tool": None,
            "tool_arguments": None, "answer": "Which date range?",
        }),
        json.dumps({
            "action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
            "tool_arguments": {"objective": "Show total consumption"}, "answer": None,
        }),
        json.dumps({
            "action": "ANSWER", "intent": "NEW_QUERY", "tool": None,
            "tool_arguments": None, "answer": "The total is available in the result.",
        }),
    ])
    response = asyncio.run(_decision_agent(registry, tmp_path, completer, calls).run(
        ChatRequest(session_id="no-clarify", db_id="shop", message="total consumption")
    ))
    assert calls == ["Show total consumption"]
    assert len(completer.calls) == 3
    assert "Which date range" not in response.message


def test_provider_failure_has_safe_cause_and_failed_ledger(
    registry: Any, tmp_path: Any
) -> None:
    from semantic_text2sql.llm import ModelError

    class RateLimitedCompleter:
        async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
            raise ModelError("The Groq request failed: rate_limit_exceeded 429; secret=hidden")

    calls: list[str] = []
    response = asyncio.run(
        _decision_agent(registry, tmp_path, RateLimitedCompleter(), calls).run(
            ChatRequest(session_id="rate-limited", db_id="shop", message="total consumption")
        )
    )
    assert "rate-limiting" in response.message
    assert "secret" not in response.message
    assert calls == []
    assert [
        entry.status for entry in response.call_ledger
        if entry.component == "conversational_agent"
    ] == ["FAILED"]


def test_consistency_repair_when_call_tool_lacks_a_tool(registry: Any, tmp_path: Any) -> None:
    calls: list[str] = []
    completer = ScriptedStructuredCompleter([
        json.dumps({
            "action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": None,
            "tool_arguments": {"objective": "Count customers"}, "answer": None,
        }),
        json.dumps({
            "action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
            "tool_arguments": {"objective": "Count customers"}, "answer": None,
        }),
        json.dumps({
            "action": "ANSWER", "intent": "NEW_QUERY", "tool": None,
            "tool_arguments": None, "instruction": None,
            "answer": "The query ran.",
        }),
    ])
    agent = _decision_agent(registry, tmp_path, completer, calls)
    response = asyncio.run(
        agent.run(ChatRequest(session_id="consistency", db_id="shop", message="Count customers"))
    )
    assert calls == ["Count customers"]  # executed once, not twice
    assert len(completer.calls) == 3
    assert "tool is required when action is CALL_TOOL" in completer.prompts[1]
    assert "could not complete" not in response.message


@pytest.mark.parametrize("code,message", [
    ("json_validate_failed", "Failed to generate JSON"),
    ("invalid_request_error", "response_format unsupported"),
    ("invalid_request_error", "json_schema unsupported keyword maxLength"),
])
def test_qwen_native_failure_uses_bounded_json_mode_then_validates_tool(
    registry: Any, tmp_path: Any, code: str, message: str
) -> None:
    bodies: list[dict[str, Any]] = []
    decision = {
        "action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
        "tool_arguments": {"objective": "Count customers", "operation": "NEW_QUERY", "changes": []},
        "instruction": None, "answer": None,
    }
    answer = {
        "action": "ANSWER", "intent": "NEW_QUERY", "tool": None,
        "tool_arguments": None, "instruction": None, "answer": "The query ran.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(400, json={"error": {"code": code, "message": message}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(
                decision if len(bodies) == 2 else answer
            )}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 80},
        })

    model = GroqSQLModel("test-key", transport=httpx.MockTransport(handler))
    calls: list[str] = []
    response = asyncio.run(_decision_agent(registry, tmp_path, model, calls).run(
        ChatRequest(session_id="qwen-fallback", db_id="shop", message="Count customers")
    ))
    assert calls == ["Count customers"]
    assert response.message == "The query ran."
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert bodies[1]["response_format"] == {"type": "json_object"}
    assert bodies[2]["response_format"] == {"type": "json_object"}
    assert bodies[1]["max_completion_tokens"] >= 1_000
    assert response.token_usage.total_tokens == 220
    assert [entry.status for entry in response.call_ledger
            if entry.component == "conversational_agent"] == [
                "FAILED", "SUCCEEDED", "SUCCEEDED",
            ]


def test_qwen_unrelated_bad_request_never_executes_sql(registry: Any, tmp_path: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {
            "code": "invalid_request_error", "message": "invalid temperature parameter",
        }})

    model = GroqSQLModel("test-key", transport=httpx.MockTransport(handler))
    calls: list[str] = []
    response = asyncio.run(_decision_agent(registry, tmp_path, model, calls).run(
        ChatRequest(session_id="qwen-bad-request", db_id="shop", message="Count customers")
    ))
    assert calls == []
    assert "invalid_request_error" in response.message
    assert response.call_ledger[-1].status == "FAILED"


def test_plot_it_valid_visualize_decision_is_not_repaired(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[str] = []
    completer = ScriptedStructuredCompleter([
        json.dumps({
            "action": "CALL_TOOL", "intent": "VISUALIZE", "tool": "change_visualization",
            "tool_arguments": {"objective": "Plot the accepted result"}, "answer": None,
        }),
        json.dumps({
            "action": "ANSWER", "intent": "VISUALIZE", "tool": None,
            "tool_arguments": None, "instruction": None,
            "answer": "Here is a chart of the stored result.",
        }),
    ])
    agent = _decision_agent(registry, tmp_path, completer, calls)
    agent.get_state = lambda session_id: ConversationState(
        session_id="viz",
        db_id="shop",
        root_question="Show monthly consumption",
        resolved_question="Show monthly consumption",
        last_sql="SELECT yearmonth, SUM(Consumption) FROM yearmonth GROUP BY yearmonth",
    )
    response = asyncio.run(
        agent.run(ChatRequest(session_id="viz", db_id="shop", message="plot it"))
    )
    assert calls == ["Show monthly consumption"]  # Missing rows trigger governed SQL.
    assert response.presentation is None
    assert response.message
    assert "chart" not in response.message.casefold()
    assert len(completer.calls) == 2  # one decision per round; zero repair calls


def test_add_currency_category_prose_intent_is_repaired_to_modify_query(
    registry: Any, tmp_path: Any
) -> None:
    calls: list[str] = []
    completer = ScriptedStructuredCompleter([
        json.dumps({
            "action": "CALL_TOOL",
            "intent": "break the consumption down by currency category",
            "tool": "ask_data", "instruction": "Add currency category", "answer": None,
        }),
        json.dumps({
            "action": "CALL_TOOL", "intent": "MODIFY_QUERY", "tool": "ask_data",
            "tool_arguments": {"objective": "Add a currency category breakdown"}, "answer": None,
        }),
        json.dumps({
            "action": "ANSWER", "intent": "MODIFY_QUERY", "tool": None,
            "tool_arguments": None, "instruction": None,
            "answer": "The query ran with the currency breakdown.",
        }),
    ])
    agent = _decision_agent(registry, tmp_path, completer, calls)
    response = asyncio.run(
        agent.run(
            ChatRequest(session_id="modify", db_id="shop", message="add currency category")
        )
    )
    assert calls == ["Add a currency category breakdown"]
    assert response.message == "The query ran."
    assert len(completer.calls) == 3  # Invalid + repaired + final decision.
    assert "Validation issues:" in completer.prompts[1]


def test_structured_completer_transport_failure_keeps_ledger_and_usage(
    registry: Any, tmp_path: Any
) -> None:
    agent_fallback: list[str] = []

    class DenyingCompleter:
        calls = 0

        async def complete_structured(
            self, model: str, prompt: str, schema_name: str, schema: dict[str, Any]
        ) -> tuple[str, TokenUsage]:
            self.calls += 1
            raise RuntimeError("gateway down")

    async def sql_handler(request: ChatRequest) -> ChatResponse:
        agent_fallback.append(request.message)
        return ChatResponse(
            session_id=request.session_id,
            operation="NEW_QUERY",
            resolved_question=request.message,
            message="The query ran.",
        )

    bridge = ConversationalAnalyticsBridge(
        database=registry,
        postgres=None,
        profiles=ProfileStore(tmp_path / "profiles"),
        glossaries=GlossaryStore(tmp_path / "glossaries"),
        retriever=HybridSchemaRetriever(Encoder()),
        sql_handler=sql_handler,
    )
    completer = DenyingCompleter()
    agent = ConversationalAnalyticsAgent(
        bridge=bridge,
        completers={"agentrouter": completer},
        get_state=lambda session_id: None,
        get_last_response=lambda session_id, db_id: None,
        sql_handler=sql_handler,
    )
    response = asyncio.run(
        agent.run(ChatRequest(session_id="down", db_id="shop", message="Hello"))
    )
    assert agent_fallback == []  # no tool ever executed
    assert completer.calls == 1  # a transport failure is not retried as a repair
    assert "model provider" in response.message.lower()
    statuses = [
        item.status
        for item in response.call_ledger
        if item.kind == "MODEL" and item.component == "conversational_agent"
    ]
    assert statuses == ["FAILED"]
    # Zero tokens were consumed because zero model responses arrived; usage is a
    # structured TokenUsage (not raw exception text), so no tokens leak or vanish.
    assert response.token_usage is not None
    assert response.token_usage.total_tokens == 0


def test_final_answer_failure_preserves_accepted_sql_and_records_failed_call(
    registry: Any, tmp_path: Any
) -> None:
    from semantic_text2sql.llm import ModelError

    class FailingFinalAnswer:
        calls = 0

        async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
            self.calls += 1
            if self.calls == 1:
                return json.dumps({
                    "action": "CALL_TOOL", "intent": "NEW_QUERY", "tool": "ask_data",
                    "tool_arguments": {"objective": "Show monthly revenue"}, "answer": None,
                }), TokenUsage(input_tokens=12, output_tokens=8)
            raise ModelError("Gateway unavailable", usage=TokenUsage(input_tokens=7))

    completer = FailingFinalAnswer()
    calls: list[str] = []
    agent = _decision_agent(registry, tmp_path, completer, calls)

    async def accepted_handler(request: ChatRequest) -> ChatResponse:
        calls.append(request.message)
        return ChatResponse(
            session_id=request.session_id, operation="NEW_QUERY",
            resolved_question=request.message, message="The query ran.",
            generation=_accepted_result().generation,
        )

    agent.sql_handler = accepted_handler
    response = asyncio.run(agent.run(ChatRequest(
        session_id="partial-answer", db_id="shop", message="Show monthly revenue",
    )))
    assert calls == ["Show monthly revenue"]
    assert response.generation is not None and response.generation.accepted
    assert response.generation.sql == "SELECT month, revenue FROM revenue_by_month"
    assert "assistant_response_unavailable" in response.provenance
    assert response.token_usage.total_tokens == 27
    assert [item.status for item in response.call_ledger
            if item.component == "conversational_agent"] == ["SUCCEEDED", "FAILED"]
