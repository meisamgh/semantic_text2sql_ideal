from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import httpx
import pytest

from semantic_text2sql.llm import (
    AgentRouterClaudeModel,
    AgentRouterCodexModel,
    AgentRouterModel,
    CodeCraftSQLModel,
    GroqSQLModel,
    JustDoWorkSQLModel,
    ModelError,
    OllamaCompleter,
    OpenRouterSQLModel,
    ProviderRequestError,
    RoutingSQLModel,
    StructuredOutputUnsupported,
    TrueSotaSQLModel,
    _prompt,
)
from semantic_text2sql.models import (
    ColumnInfo,
    SchemaInfo,
    StrategyHints,
    TableInfo,
    TokenUsage,
)
from semantic_text2sql.runtime import RequestBudget, use_request_budget


def test_ollama_understanding_uses_native_schema_and_task_budget(monkeypatch: Any) -> None:
    monkeypatch.setenv("TEXT2SQL_TOP_DECISION_MAX_TOKENS", "321")
    schema: dict[str, object] = {"type": "object", "properties": {"action": {"type": "string"}}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "gemma4:12b"
        assert payload["format"] == schema
        assert payload["options"]["num_predict"] == 321
        assert payload["think"] is False
        assert payload["stream"] is False
        return httpx.Response(200, json={
            "message": {"content": '{"action":"ANSWER"}'},
            "prompt_eval_count": 20, "eval_count": 5,
        })

    model = OllamaCompleter(transport=httpx.MockTransport(handler))
    result, usage = asyncio.run(model.complete_structured_task(
        "gemma4:12b", "Understand the request", "AgentDecision", schema, "decision",
    ))
    assert json.loads(result)["action"] == "ANSWER"
    assert usage.total_tokens == 25


def test_ollama_grammar_rejection_disables_native_schema_for_that_model() -> None:
    model = OllamaCompleter(transport=httpx.MockTransport(lambda request: httpx.Response(
        400, json={"error": "Failed to initialize samplers: failed to parse grammar"},
    )))
    with pytest.raises(StructuredOutputUnsupported):
        asyncio.run(model.complete_structured_task(
            "gemma4:12b", "Question", "AgentDecision", {"type": "object"}, "decision",
        ))
    assert not model.native_schema_supported("gemma4:12b", "AgentDecision")
    assert model.native_schema_supported("another-model", "AgentDecision")


def test_true_sota_uses_responses_without_codex_cli_or_key_leak() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        payload: dict[str, Any] = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "SELECT 1"},
            ]}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        })

    model = TrueSotaSQLModel(
        "test-key", "https://sota.test", transport=httpx.MockTransport(handler)
    )
    text, usage = asyncio.run(model.complete_detailed("gpt-6-sol", "Return SQL"))
    assert text == "SELECT 1"
    assert usage.total_tokens == 5
    assert requests[0]["model"] == "gpt-6-sol"
    assert requests[0]["store"] is False


def test_task_specific_structured_limits_reach_provider_payloads(monkeypatch: Any) -> None:
    monkeypatch.setenv("TEXT2SQL_TOP_DECISION_MAX_TOKENS", "321")
    captured: dict[str, dict[str, Any]] = {}

    def sota_handler(request: httpx.Request) -> httpx.Response:
        captured["sota"] = json.loads(request.content)
        return httpx.Response(200, json={
            "output_text": '{"action":"ANSWER"}', "usage": {},
        })

    def codex_handler(request: httpx.Request) -> httpx.Response:
        captured["agentrouter"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"action":"ANSWER"}'}}],
            "usage": {},
        })

    def groq_handler(request: httpx.Request) -> httpx.Response:
        captured["groq"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"action":"ANSWER"}'}}],
            "usage": {},
        })

    schema: dict[str, object] = {"type": "object"}
    asyncio.run(TrueSotaSQLModel(
        "key", "https://sota.test", transport=httpx.MockTransport(sota_handler)
    ).complete_structured_task("gpt-6-sol", "decide", "AgentDecision", schema, "decision"))
    asyncio.run(AgentRouterCodexModel(
        "key", "https://agentrouter.test", transport=httpx.MockTransport(codex_handler)
    ).complete_structured_task("gpt-5.6-sol", "decide", "AgentDecision", schema, "decision"))
    asyncio.run(GroqSQLModel(
        "key", transport=httpx.MockTransport(groq_handler)
    ).complete_structured_task("qwen", "decide", "AgentDecision", schema, "decision"))
    assert captured["sota"]["max_output_tokens"] == 321
    assert captured["agentrouter"]["max_completion_tokens"] == 321
    assert captured["groq"]["max_completion_tokens"] == 321


def test_anthropic_and_codecraft_task_limits_reach_provider_payloads(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TEXT2SQL_TOP_ANSWER_MAX_TOKENS", "456")
    captured: dict[str, dict[str, Any]] = {}

    def anthropic_handler(request: httpx.Request) -> httpx.Response:
        captured["anthropic"] = json.loads(request.content)
        return httpx.Response(200, json={
            "content": [{"type": "tool_use", "input": {"answer": "done"}}],
            "usage": {},
        })

    def codecraft_handler(request: httpx.Request) -> httpx.Response:
        captured["codecraft"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"answer":"done"}'}}],
            "usage": {},
        })

    asyncio.run(AgentRouterClaudeModel(
        "key", "https://agentrouter.test", transport=httpx.MockTransport(anthropic_handler)
    ).complete_structured_task(
        "claude-opus-5", "answer", "AgentDecision", {"type": "object"}, "final_answer"
    ))
    asyncio.run(CodeCraftSQLModel(
        "key", transport=httpx.MockTransport(codecraft_handler)
    ).complete_task("model", "answer", "final_answer", json_mode=True))
    assert captured["anthropic"]["max_tokens"] == 456
    assert captured["codecraft"]["max_tokens"] == 456


def test_openrouter_nemotron_sql_completion_and_local_json_policy() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "SELECT 1"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3},
        })

    model = OpenRouterSQLModel(
        "test-key", transport=httpx.MockTransport(handler),
    )
    sql, _, usage = asyncio.run(model.generate(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        question="Return one", evidence=None,
        schema=SchemaInfo(db_id="shop", tables=[]),
        strategy=StrategyHints(mode="exact"), dialect="sqlite",
        profile_context="{}", previous_sql=None, feedback=None,
        rejected_shapes=[], generation_style="reasoning",
    ))
    assert sql == "SELECT 1" and usage.total_tokens == 11
    assert captured["path"] == "/api/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    body = captured["body"]
    assert body["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert "response_format" not in body
    assert model.native_schema_supported(body["model"], "AgentDecision") is False


def test_openrouter_missing_key_and_truncation_fail_safely() -> None:
    with pytest.raises(ModelError, match="OPENROUTER_API_KEY"):
        asyncio.run(OpenRouterSQLModel(None).complete_detailed("nemotron", "SELECT 1"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "SELECT"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3},
        })

    model = OpenRouterSQLModel("test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(ModelError, match="truncated"):
        asyncio.run(model.complete_detailed("nemotron", "SELECT 1"))


def test_groq_rate_limit_retry_claims_shared_model_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})

    monkeypatch.setattr("semantic_text2sql.llm._groq_retry_delay", lambda response: 0)
    model = GroqSQLModel("test-key", transport=httpx.MockTransport(handler))
    budget = RequestBudget(timeout_seconds=5, max_model_calls=1)
    with use_request_budget(budget), pytest.raises(TimeoutError, match="model-call budget"):
        asyncio.run(budget.wait(
            model.complete_detailed("qwen/qwen3.8-27b", "Hello"), kind="model"
        ))
    assert calls == 1
    assert budget.model_calls == 1


def _call(model: AgentRouterClaudeModel) -> tuple[str, int, TokenUsage]:
    return asyncio.run(
        model.generate(
            provider="agentrouter",
            model="claude-opus-5",
            question="List books",
            evidence=None,
            schema=SchemaInfo(db_id="books", tables=[]),
            strategy=StrategyHints(mode="exact"),
            dialect="sqlite",
            profile_context=json.dumps(
                {
                    "question": "List books",
                    "dialect": "sqlite",
                    "tables": {"books": {"columns": {"title": {"type": "TEXT"}}}},
                }
            ),
            previous_sql=None,
            feedback=None,
            rejected_shapes=[],
            generation_style="reasoning",
        )
    )


def test_agentrouter_uses_anthropic_messages_contract() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["key"] = request.headers.get("x-api-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "SELECT title FROM books"}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )

    model = AgentRouterClaudeModel(
        "test-key",
        "https://router.test",
        transport=httpx.MockTransport(handler),
    )

    sql, _, usage = _call(model)

    assert sql == "SELECT title FROM books"
    assert usage.total_tokens == 12
    assert captured["path"] == "/v1/messages"
    assert captured["key"] == "test-key"
    assert captured["body"]["model"] == "claude-opus-5"  # type: ignore[index]
    prompt = captured["body"]["messages"][0]["content"]  # type: ignore[index]
    assert "CORRECTNESS-FIRST EFFICIENCY RULES" in prompt
    assert "Preserve approved business definitions and all unaffected constraints" in prompt
    assert "Do not guess unsupported database facts" in prompt
    assert "never use SELECT * unless" in prompt
    assert "concise, stable, meaningful `AS` alias" in prompt
    assert "prefer EXISTS" in prompt
    assert "Retrieved live" not in prompt
    assert "Strategy hint:" not in prompt
    assert "storage_type" not in prompt
    assert "observed_format" not in prompt
    assert "METRIC DEPENDENCY RULES" not in prompt
    assert "APPROVED FORMULAS:" not in prompt


def test_prompt_adds_formula_guidance_only_when_context_contains_a_formula() -> None:
    context = json.dumps(
        {
            "question": "Show unit price",
            "dialect": "sqlite",
            "tables": {
                "transactions": {
                    "columns": {
                        "price": {"type": "REAL"},
                        "amount": {"type": "INTEGER"},
                    }
                }
            },
            "approved_formulas": [
                {
                    "id": "unit_price",
                    "operator": "DIVIDE",
                    "arguments": ["transactions.price", "transactions.amount"],
                    "zero_safe": True,
                }
            ],
        }
    )

    prompt = _prompt(
        "Show unit price",
        None,
        SchemaInfo(db_id="shop", tables=[]),
        StrategyHints(mode="exact"),
        "sqlite",
        context,
        None,
        None,
        [],
        "reasoning",
    )

    assert "APPROVED FORMULAS:" in prompt
    assert "zero-safe DIVIDE(a,b)" in prompt


def test_prompt_builds_compact_schema_fallback_for_invalid_context() -> None:
    schema = SchemaInfo(
        db_id="books",
        tables=[
            TableInfo(
                name="books",
                create_sql="CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT)",
                columns=[
                    ColumnInfo(name="id", data_type="INTEGER", primary_key=True),
                    ColumnInfo(name="title", data_type="TEXT"),
                ],
            )
        ],
    )

    prompt = _prompt(
        "List books",
        None,
        schema,
        StrategyHints(mode="exact"),
        "sqlite",
        "No verified context available.",
        None,
        None,
        [],
        "reasoning",
    )

    assert '"books"' in prompt
    assert '"title":{"type":"TEXT"}' in prompt
    assert "Retrieved live" not in prompt


def test_optimization_uses_dedicated_prompt_and_excludes_generation_context() -> None:
    context = json.dumps(
        {
            "question": "List customers",
            "dialect": "sqlite",
            "tables": {
                "customers": {
                    "primary_key": ["customer_id"],
                    "columns": {"customer_id": {"type": "INTEGER"}},
                }
            },
            "business_context": "Customer-specific glossary prose",
            "historical_examples": [
                {"question": "Old example", "sql": "SELECT * FROM old_customers"}
            ],
        }
    )

    prompt = _prompt(
        "Optimize the previous query",
        "Unused evidence",
        SchemaInfo(db_id="shop", tables=[]),
        StrategyHints(mode="exact"),
        "sqlite",
        context,
        "SELECT customer_id FROM customers",
        (
            "Optimize the previously accepted SQL without changing its columns, row semantics, "
            "filters, aggregation, ordering, or result values. The replacement must be "
            "result-equivalent and measurably faster.\nOriginal EXPLAIN plan:\nSCAN customers"
        ),
        ["unused-fingerprint"],
        "reasoning",
    )

    assert prompt.startswith("You are a SQL query optimizer.")
    assert "ACCEPTED_SQL:\nSELECT customer_id FROM customers" in prompt
    assert "ORIGINAL_EXPLAIN:\nSCAN customers" in prompt
    assert '"customers"' in prompt
    assert "Return the accepted SQL unchanged" in prompt
    assert "Preserve existing output aliases" in prompt
    assert "Question:" not in prompt
    assert "CORRECTNESS-FIRST EFFICIENCY RULES" not in prompt
    assert "HISTORICAL EXAMPLES" not in prompt
    assert "old_customers" not in prompt
    assert "Customer-specific glossary prose" not in prompt
    assert "unused-fingerprint" not in prompt


def test_agentrouter_fails_without_environment_key() -> None:
    try:
        _call(AgentRouterClaudeModel(None))
    except ModelError as exc:
        assert "AGENTROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("Missing AgentRouter key was accepted")


def test_agentrouter_dispatches_gpt_to_codex() -> None:
    claude = AgentRouterClaudeModel("test-key")
    codex = AgentRouterCodexModel("test-key", executable="/missing/codex")
    router = AgentRouterModel(claude, codex)

    assert router._select("claude-opus-5") is claude
    assert router._select("gpt-5.6-sol") is codex


def _call_codex(model: AgentRouterCodexModel) -> tuple[str, int, TokenUsage]:
    return asyncio.run(
        model.generate(
            provider="agentrouter",
            model="gpt-5.6-sol",
            question="List books",
            evidence=None,
            schema=SchemaInfo(db_id="books", tables=[]),
            strategy=StrategyHints(mode="exact"),
            dialect="sqlite",
            profile_context="No cached value profiles available.",
            previous_sql=None,
            feedback=None,
            rejected_shapes=[],
            generation_style="reasoning",
        )
    )


def test_agentrouter_gpt_uses_chat_completions_over_http() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "SELECT title FROM books"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            },
        )

    model = AgentRouterCodexModel(
        "test-key",
        "https://router.test",
        transport=httpx.MockTransport(handler),
    )

    sql, _, usage = _call_codex(model)

    assert sql == "SELECT title FROM books"
    assert usage.total_tokens == 12
    assert captured["path"] == "/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "gpt-5.6-sol"  # type: ignore[index]


def test_agentrouter_gpt_falls_back_to_the_responses_wire_format() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(404, json={"error": "unsupported endpoint"})
        return httpx.Response(
            200,
            json={
                "output": [{"content": [{"type": "output_text", "text": "SELECT 1"}]}],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    model = AgentRouterCodexModel(
        "test-key",
        "https://router.test",
        transport=httpx.MockTransport(handler),
    )

    sql, _, usage = _call_codex(model)

    assert sql == "SELECT 1"
    assert usage.total_tokens == 4
    assert paths == ["/v1/chat/completions", "/v1/responses"]


def test_agentrouter_gpt_stops_at_a_rejected_key() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(401, json={"error": "invalid key"})

    model = AgentRouterCodexModel(
        "test-key",
        "https://router.test",
        transport=httpx.MockTransport(handler),
    )

    try:
        asyncio.run(model._complete_over_http("gpt-5.6-sol", "List books"))
    except ModelError as exc:
        assert "401" in str(exc)
    else:
        raise AssertionError("A rejected AgentRouter key was accepted")

    assert paths == ["/v1/chat/completions"]


def test_groq_qwen_uses_chat_completions_and_reports_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "SELECT title FROM books"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3},
            },
        )

    model = GroqSQLModel(
        "groq-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    sql, usage = asyncio.run(model.complete_detailed("qwen/qwen3.6-27b", "Return SQL only"))

    assert sql == "SELECT title FROM books"
    assert usage.total_tokens == 12
    assert captured["path"] == "/openai/v1/chat/completions"
    assert captured["authorization"] == "Bearer groq-test-key"
    assert captured["body"]["model"] == "qwen/qwen3.6-27b"  # type: ignore[index]
    assert captured["body"]["reasoning_effort"] == "none"  # type: ignore[index]
    assert captured["body"]["reasoning_format"] == "hidden"  # type: ignore[index]
    assert captured["body"]["max_completion_tokens"] == 2_048  # type: ignore[index]


def test_groq_fails_without_api_key() -> None:
    try:
        asyncio.run(GroqSQLModel(None).complete("qwen/qwen3.6-27b", "Return SQL"))
    except ModelError as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("Missing Groq key was accepted")


def test_codecraft_uses_chat_completions_and_reports_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "SELECT title FROM books"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    codecraft = CodeCraftSQLModel(
        "codecraft-test-key",
        transport=httpx.MockTransport(handler),
    )
    router = RoutingSQLModel(
        AgentRouterModel(AgentRouterClaudeModel(None), AgentRouterCodexModel(None)),
        GroqSQLModel(None),
        JustDoWorkSQLModel(None, None),
        codecraft,
    )
    sql, usage = asyncio.run(
        router.complete_detailed("codecraft", "claude-opus-5", "Return SQL only")
    )

    assert sql == "SELECT title FROM books"
    assert usage.total_tokens == 14
    assert captured["path"] == "/v1/chat/completions"
    assert captured["authorization"] == "Bearer codecraft-test-key"
    assert captured["body"]["model"] == "claude-opus-5"  # type: ignore[index]
    assert "reasoning_effort" not in captured["body"]  # type: ignore[operator]
    assert captured["body"]["stream"] is False  # type: ignore[index]


def test_codecraft_rejects_missing_key_and_empty_output() -> None:
    with pytest.raises(ModelError, match="CODECRAFT_API_KEY"):
        asyncio.run(CodeCraftSQLModel(None).complete("claude-opus-5", "Return SQL"))

    def empty_response(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    model = CodeCraftSQLModel("key", transport=httpx.MockTransport(empty_response))
    with pytest.raises(ModelError, match="no text output"):
        asyncio.run(model.complete("claude-opus-5", "Return SQL"))


def test_justdowork_uses_anthropic_messages_and_reports_usage() -> None:
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "shared-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        payload = json.loads(request.content)
        requested_models.append(payload["model"])
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "SELECT 1"}],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    model = JustDoWorkSQLModel(
        "shared-key",
        "https://justdowork.test/v1",
        transport=httpx.MockTransport(handler),
    )
    for name in ("claude-opus-4-8", "gpt-5.6-sol"):
        sql, usage = asyncio.run(model.complete_detailed(name, "Return SQL only"))
        assert sql == "SELECT 1"
        assert usage.total_tokens == 4
    assert requested_models == ["claude-opus-4-8", "gpt-5.6-sol"]


def test_justdowork_root_url_uses_v1_messages_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "SELECT 1"}]},
        )

    model = JustDoWorkSQLModel(
        "shared-key",
        "https://api.justwoker.icu",
        transport=httpx.MockTransport(handler),
    )
    sql = asyncio.run(model.complete("claude-opus-4-8", "Return SQL only"))
    assert sql == "SELECT 1"


def test_justdowork_fails_without_api_key() -> None:
    with pytest.raises(ModelError, match="JUSTDOWORK_API_KEY"):
        asyncio.run(JustDoWorkSQLModel(None, None).complete("claude-opus-5", "Return SQL"))

    with pytest.raises(ModelError, match="JUSTDOWORK_BASE_URL"):
        asyncio.run(JustDoWorkSQLModel("configured-key", None).complete("claude-opus-5", "SQL"))


def test_groq_retries_transient_rate_limit() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0.001"},
                json={"error": {"code": "rate_limit_exceeded"}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "SELECT 1"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    model = GroqSQLModel(
        "groq-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )

    text, usage = asyncio.run(model.complete_detailed("qwen/qwen3.6-27b", "Return SQL"))

    assert text == "SELECT 1"
    assert usage.total_tokens == 3
    assert usage.provider_retries == 1
    assert calls == 2


def _agent_decision_schema_typed() -> dict[str, Any]:
    from semantic_text2sql.conversational_agent import AGENT_DECISION_SCHEMA

    return cast(dict[str, Any], AGENT_DECISION_SCHEMA)


def test_agent_decision_schema_enumerates_every_allowed_value() -> None:
    schema = _agent_decision_schema_typed()
    properties = schema["properties"]
    assert schema["required"] == [
        "action", "intent", "tool", "tool_arguments", "instruction", "answer",
        "finish_after_tool", "chart_proposal",
    ]
    assert properties["action"]["enum"] == [
        "ANSWER",
        "CALL_TOOL",
    ]
    assert properties["intent"]["enum"] == [
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
    assert properties["tool"]["anyOf"][0]["enum"] == [
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
    assert schema["additionalProperties"] is False
    from semantic_text2sql.conversational_agent import _DECISION_EXAMPLES
    assert len(_DECISION_EXAMPLES) == 3


def test_groq_native_structured_output_carries_full_json_schema() -> None:
    from semantic_text2sql.conversational_agent import AGENT_DECISION_SCHEMA, AgentDecision

    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "action": "ANSWER",
                                    "intent": "DIRECT_ANSWER",
                                    "tool": None,
                                    "tool_arguments": None,
                                    "answer": "Summary",
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 8},
            },
        )

    model = GroqSQLModel(
        "groq-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    text, usage = asyncio.run(
        model.complete_structured(
            "qwen/qwen3.6-27b", "Decide next step", "AgentDecision", AGENT_DECISION_SCHEMA
        )
    )

    decision = AgentDecision.model_validate_json(text)
    assert decision.intent == "DIRECT_ANSWER"
    assert usage.total_tokens == 38
    assert len(requests) == 1
    body = requests[0]
    response_format: dict[str, Any] = body["response_format"]
    assert body["max_completion_tokens"] == 1_536
    assert response_format["type"] == "json_schema"
    json_schema: dict[str, Any] = response_format["json_schema"]
    assert json_schema["name"] == "AgentDecision"
    assert json_schema["strict"] is True
    assert json_schema["schema"] == AGENT_DECISION_SCHEMA


def test_groq_structured_output_reports_unsupported_without_hidden_retry() -> None:
    from semantic_text2sql.conversational_agent import AGENT_DECISION_SCHEMA

    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(
                400, json={"error": {"message": "response_format unsupported"}}
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "action": "ANSWER",
                                    "intent": "DIRECT_ANSWER",
                                    "tool": None,
                                    "tool_arguments": None,
                                    "answer": "Summary",
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5},
            },
        )

    model = GroqSQLModel(
        "groq-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    from semantic_text2sql.llm import StructuredOutputUnsupported
    with pytest.raises(StructuredOutputUnsupported):
        asyncio.run(model.complete_structured(
            "qwen/qwen3.6-27b", "Decide next step", "AgentDecision", AGENT_DECISION_SCHEMA
        ))
    assert len(bodies) == 1


def test_justdowork_structured_output_uses_forced_tool_schema() -> None:
    from semantic_text2sql.conversational_agent import (
        AGENT_DECISION_SCHEMA,
        AgentDecision,
    )

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "tool_use",
                        "name": "AgentDecision",
                        "input": {
                            "action": "ANSWER",
                            "intent": "DIRECT_ANSWER",
                            "tool": None,
                            "tool_arguments": None,
                            "answer": "Summary",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 24, "completion_tokens": 9},
            },
        )

    model = JustDoWorkSQLModel(
        "shared-key",
        "https://justdowork.test/v1",
        transport=httpx.MockTransport(handler),
    )
    text, usage = asyncio.run(
        model.complete_structured(
            "claude-opus-4-8", "Decide next step", "AgentDecision", AGENT_DECISION_SCHEMA
        )
    )

    assert AgentDecision.model_validate_json(text).intent == "DIRECT_ANSWER"
    assert usage.total_tokens == 33
    body: dict[str, Any] = captured["body"]
    tool_choice = body["tool_choice"]
    assert tool_choice == {"type": "tool", "name": "AgentDecision"}
    forced_tool = next(
        tool for tool in body["tools"] if tool["name"] == "AgentDecision"
    )
    schema: dict[str, Any] = forced_tool["input_schema"]
    assert schema["required"] == [
        "action", "intent", "tool", "tool_arguments", "instruction", "answer",
        "finish_after_tool", "chart_proposal",
    ]
    assert schema["properties"]["action"]["enum"] == [
        "ANSWER",
        "CALL_TOOL",
    ]


def test_agentrouter_routes_structured_output_and_falls_back() -> None:  # noqa: C901
    class StructuredClaude:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str, dict[str, object]]] = []

        async def complete_structured(
            self,
            model: str,
            prompt: str,
            schema_name: str,
            schema: dict[str, object],
        ) -> tuple[str, TokenUsage]:
            self.calls.append((model, prompt, schema_name, schema))
            return json.dumps(
                {"action": "ANSWER", "intent": "DIRECT_ANSWER"}
            ), TokenUsage(input_tokens=5, output_tokens=3)

        async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
            raise AssertionError("structured path should not need plain completions")

    class UnsupportedClaude:
        async def complete_structured(
            self,
            model: str,
            prompt: str,
            schema_name: str,
            schema: dict[str, object],
        ) -> tuple[str, TokenUsage]:
            raise ModelError("gateway rejected the response format")

        async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
            return "SELECT 1", TokenUsage(input_tokens=2, output_tokens=1)

    router = AgentRouterModel(
        StructuredClaude(),  # type: ignore[arg-type]
        AgentRouterCodexModel(None),
    )
    text, usage = asyncio.run(router.complete_structured("claude-x", "p", "AgentDecision", {}))
    assert json.loads(text) == {"action": "ANSWER", "intent": "DIRECT_ANSWER"}
    assert usage.total_tokens == 8

    router = AgentRouterModel(
        UnsupportedClaude(),  # type: ignore[arg-type]
        AgentRouterCodexModel(None),
    )
    with pytest.raises(ModelError):
        asyncio.run(router.complete_structured("claude-x", "p", "AgentDecision", {}))


def test_agentrouter_complete_task_prefers_task_aware_json_transports() -> None:
    class TaskAware:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str, bool]] = []

        async def complete_task(
            self, model: str, prompt: str, task: str, *, json_mode: bool = False,
        ) -> tuple[str, TokenUsage]:
            self.calls.append((model, prompt, task, json_mode))
            return json.dumps({"done": True}), TokenUsage(input_tokens=4, output_tokens=2)

        async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
            raise AssertionError("complete_task should be delegated directly")

    class StructuredOnly:
        def __init__(self) -> None:
            self.structured: list[tuple[str, str, str, dict[str, object]]] = []
            self.plain = 0

        async def complete_structured(
            self, model: str, prompt: str, schema_name: str, schema: dict[str, object],
        ) -> tuple[str, TokenUsage]:
            self.structured.append((model, prompt, schema_name, schema))
            return json.dumps({"done": True}), TokenUsage(input_tokens=1, output_tokens=1)

        async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
            self.plain += 1
            return json.dumps({"done": True}), TokenUsage(input_tokens=1, output_tokens=1)

    class UnsupportedStructured(StructuredOnly):
        async def complete_structured(
            self, model: str, prompt: str, schema_name: str, schema: dict[str, object],
        ) -> tuple[str, TokenUsage]:
            raise StructuredOutputUnsupported("gateway rejects response_format")

    task_aware = TaskAware()
    router = AgentRouterModel(task_aware, AgentRouterCodexModel(None))  # type: ignore[arg-type]
    text, usage = asyncio.run(
        router.complete_task("claude-x", "prompt", "decision", json_mode=True)
    )
    assert json.loads(text) == {"done": True}
    assert usage.total_tokens == 6
    assert task_aware.calls == [("claude-x", "prompt", "decision", True)]

    structured_only = StructuredOnly()
    router = AgentRouterModel(structured_only, AgentRouterCodexModel(None))  # type: ignore[arg-type]
    text, usage = asyncio.run(
        router.complete_task("claude-x", "prompt", "decision", json_mode=True)
    )
    assert json.loads(text) == {"done": True}
    assert usage.total_tokens == 2
    assert structured_only.structured[0][2] == "BoundedTaskOutput"
    assert structured_only.structured[0][3] == {"type": "object"}
    assert structured_only.plain == 0

    unsupported = UnsupportedStructured()
    router = AgentRouterModel(unsupported, AgentRouterCodexModel(None))  # type: ignore[arg-type]
    text, usage = asyncio.run(
        router.complete_task("claude-x", "prompt", "decision", json_mode=True)
    )
    assert json.loads(text) == {"done": True}
    assert usage.total_tokens == 2
    assert unsupported.plain == 1

    # Without JSON mode the plain completion path is preserved even when a
    # structured transport exists.
    asyncio.run(router.complete_task("claude-x", "prompt", "decision", json_mode=True))
    assert unsupported.plain == 2


def test_agentrouter_native_schema_capability_delegates_to_the_backend() -> None:
    class Structured:
        async def complete_structured(
            self, model: str, prompt: str, schema_name: str, schema: dict[str, object],
        ) -> tuple[str, TokenUsage]:
            return "{}", TokenUsage()

    class Bookkeeping(Structured):
        def native_schema_supported(self, model: str, schema_name: str) -> bool:
            return False

    class Bare:
        pass

    router = AgentRouterModel(Structured(), AgentRouterCodexModel(None))  # type: ignore[arg-type]
    assert router.native_schema_supported("claude-x", "AgentDecision") is True
    assert router.native_schema_supported("some-gpt-model", "AgentDecision") is True

    router = AgentRouterModel(Bookkeeping(), AgentRouterCodexModel(None))  # type: ignore[arg-type]
    assert router.native_schema_supported("claude-x", "AgentDecision") is False

    router = AgentRouterModel(Bare(), Bare())  # type: ignore[arg-type]
    assert router.native_schema_supported("claude-x", "AgentDecision") is False


def test_groq_max_completion_tokens_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(int(json.loads(request.content)["max_completion_tokens"]))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "SELECT 1"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    model = GroqSQLModel(
        "groq-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
        max_completion_tokens=1_024,
    )
    asyncio.run(model.complete_detailed("qwen/qwen3.6-27b", "Return SQL"))
    asyncio.run(
        model.complete_structured("qwen/qwen3.6-27b", "Decide", "AgentDecision", {})
    )

    # Env fallback when no explicit value is passed.
    monkeypatch.setenv("TEXT2SQL_MAX_COMPLETION_TOKENS", "640")
    env_model = GroqSQLModel(
        "groq-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(env_model.complete_detailed("qwen/qwen3.6-27b", "Return SQL"))

    # The default no longer truncates a decision fallback at the old 180-token cap.
    monkeypatch.delenv("TEXT2SQL_MAX_COMPLETION_TOKENS", raising=False)
    default_model = GroqSQLModel(
        "groq-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    asyncio.run(default_model.complete_detailed("qwen/qwen3.6-27b", "Return SQL"))

    assert captured == [1_024, 1_536, 640, 2_048]


@pytest.mark.parametrize("status,code", [
    (400, "invalid_request_error"), (401, "authentication_error"),
    (403, "permission_error"), (404, "model_not_found"),
    (429, "rate_limit_exceeded"),
])
def test_groq_http_failure_is_typed_and_redacted(
    status: int, code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("semantic_text2sql.llm._groq_retry_delay", lambda response: 0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers={"x-request-id": "req-safe"}, json={
            "error": {"code": code, "message": "private prompt and secret-key"},
        })

    model = GroqSQLModel("test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderRequestError) as raised:
        asyncio.run(model.complete_detailed("qwen/qwen3.8-27b", "Decide"))
    error = raised.value
    assert (error.status, error.code, error.request_id) == (status, code, "req-safe")
    assert error.model == "qwen/qwen3.8-27b"
    assert error.latency_ms >= 0
    assert "private prompt" not in str(error)
    assert "secret-key" not in str(error)


def test_groq_length_finish_is_not_accepted_and_retains_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"action":'}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 180},
        })

    model = GroqSQLModel("test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(ModelError, match="truncated") as raised:
        asyncio.run(model.complete_detailed("qwen/qwen3.8-27b", "Decide"))
    assert raised.value.usage.total_tokens == 200
