from __future__ import annotations

import asyncio
import json

import httpx

from semantic_text2sql.llm import (
    AgentRouterClaudeModel,
    AgentRouterCodexModel,
    AgentRouterModel,
    GroqSQLModel,
    JustDoWorkSQLModel,
    ModelError,
    SotaSQLModel,
    _prompt,
    _responses_stream_result,
    ollama_model_status,
)
from semantic_text2sql.models import ColumnInfo, SchemaInfo, StrategyHints, TableInfo, TokenUsage


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
    assert "never use SELECT * unless" in prompt
    assert "concise, stable, meaningful `AS` alias" in prompt
    assert "prefer EXISTS" in prompt
    assert "Retrieved live" not in prompt
    assert "Strategy hint:" not in prompt
    assert "storage_type" not in prompt
    assert "observed_format" not in prompt
    assert "METRIC DEPENDENCY RULES" not in prompt
    assert "APPROVED FORMULAS:" not in prompt


def test_sota_uses_responses_api_without_storage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output_text": "SELECT title AS book_title FROM books",
                "usage": {"input_tokens": 9, "output_tokens": 4},
            },
        )

    model = SotaSQLModel(
        "project-key",
        "https://sota.test",
        transport=httpx.MockTransport(handler),
    )
    content, usage = asyncio.run(model.complete_detailed("gpt-5.5", "Generate SQL"))

    assert content == "SELECT title AS book_title FROM books"
    assert usage.total_tokens == 13
    assert captured["path"] == "/responses"
    assert captured["authorization"] == "Bearer project-key"
    assert captured["body"] == {
        "model": "gpt-5.5",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Generate SQL"}],
            }
        ],
        "reasoning": {"effort": "xhigh"},
        "store": False,
        "stream": True,
    }


def test_sota_extracts_streamed_response_text_and_usage() -> None:
    payload = "\n".join(
        [
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","delta":"SELECT "}',
            'data: {"type":"response.output_text.delta","delta":"1"}',
            'data: {"type":"response.completed","response":{"usage":'
            '{"input_tokens":8,"output_tokens":2}}}',
            'data: [DONE]',
        ]
    )

    content, usage = _responses_stream_result(payload)

    assert content == "SELECT 1"
    assert usage.total_tokens == 10


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
    assert captured["body"]["max_completion_tokens"] == 180  # type: ignore[index]


def test_justdowork_supports_claude_and_gpt_over_one_openai_compatible_api() -> None:
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer shared-key"
        assert "temperature" not in body
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "SELECT 1"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    model = JustDoWorkSQLModel(
        "shared-key",
        "https://justdowork.test/v1",
        transport=httpx.MockTransport(handler),
    )

    for name in ("claude-opus-5", "gpt-5.6-sol"):
        sql, usage = asyncio.run(model.complete_detailed(name, "Return SQL only"))
        assert sql == "SELECT 1"
        assert usage.total_tokens == 4
    assert requested_models == ["claude-opus-5", "gpt-5.6-sol"]


def test_justdowork_fails_without_api_key() -> None:
    try:
        asyncio.run(JustDoWorkSQLModel(None).complete("claude-opus-5", "Return SQL"))
    except ModelError as exc:
        assert "JUSTDOWORK_API_KEY" in str(exc)
    else:
        raise AssertionError("Missing JustDoWork key was accepted")


def test_groq_fails_without_api_key() -> None:
    try:
        asyncio.run(GroqSQLModel(None).complete("qwen/qwen3.6-27b", "Return SQL"))
    except ModelError as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("Missing Groq key was accepted")


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
    assert calls == 2


def test_ollama_status_flags_a_model_that_is_not_installed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "llama3:latest", "size": 4_000}]})

    reason = asyncio.run(
        ollama_model_status(
            "qwen3.8:27b",
            base_url="http://ollama.test",
            transport=httpx.MockTransport(handler),
        )
    )

    assert reason is not None
    assert "not installed" in reason


def test_ollama_status_flags_a_model_larger_than_system_memory() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "huge:latest", "size": 10**15}]})

    reason = asyncio.run(
        ollama_model_status(
            "huge:latest",
            base_url="http://ollama.test",
            transport=httpx.MockTransport(handler),
        )
    )

    assert reason is not None
    assert "RAM" in reason


def test_ollama_status_accepts_an_installed_model_that_fits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "small:latest", "size": 1_000_000}]})

    reason = asyncio.run(
        ollama_model_status(
            "small:latest",
            base_url="http://ollama.test",
            transport=httpx.MockTransport(handler),
        )
    )

    assert reason is None


def test_ollama_status_reports_an_unreachable_daemon() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    reason = asyncio.run(
        ollama_model_status(
            "small:latest",
            base_url="http://ollama.test",
            transport=httpx.MockTransport(handler),
        )
    )

    assert reason is not None
    assert "not reachable" in reason
