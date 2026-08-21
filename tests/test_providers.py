from __future__ import annotations

import asyncio
import json

import httpx

from semantic_text2sql.llm import (
    AgentRouterClaudeModel,
    AgentRouterCodexModel,
    AgentRouterModel,
    GroqSQLModel,
    ModelError,
    ollama_model_status,
)
from semantic_text2sql.models import SchemaInfo, StrategyHints, TokenUsage


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
            profile_context="No cached value profiles available.",
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
