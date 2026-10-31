"""Mockable project-scoped model provider boundaries."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol, cast

import httpx

from semantic_text2sql.dialects import DatabaseDialect
from semantic_text2sql.models import ModelProvider, SchemaInfo, StrategyHints, TokenUsage
from semantic_text2sql.runtime import active_request_budget

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    def __init__(self, message: str, *, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or TokenUsage()


class ProviderRequestError(ModelError):
    """Safe provider metadata; never contains a prompt, key or raw body."""

    def __init__(
        self, provider: str, model: str, phase: str, status: int,
        code: str, request_id: str | None, latency_ms: int,
        capability_rejection: bool = False,
        retry_count: int = 0,
    ) -> None:
        super().__init__(
            f"{provider} HTTP {status} ({code or 'unknown_error'})",
            usage=TokenUsage(provider_retries=retry_count),
        )
        self.provider = provider
        self.model = model
        self.phase = phase
        self.status = status
        self.code = code
        self.request_id = request_id
        self.latency_ms = latency_ms
        self.capability_rejection = capability_rejection
        self.retry_count = retry_count


class StructuredOutputFallbackRequired(ModelError):
    """Native structured decoding failed; local validation is still mandatory."""


class StructuredOutputUnsupported(StructuredOutputFallbackRequired):
    """The provider explicitly rejected native schema-constrained output."""


class OllamaCompleter:
    """Local conversation completion using Ollama's native structured chat API."""

    def __init__(
        self, base_url: str = "http://127.0.0.1:11434", timeout_seconds: float = 180,
        *, transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self._schema_unsupported_models: set[str] = set()

    def native_schema_supported(self, model: str, schema_name: str) -> bool:
        return model not in self._schema_unsupported_models

    async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        return await self.complete_task(model, prompt, "decision")

    async def complete_task(
        self, model: str, prompt: str, task: str, *, json_mode: bool = False,
    ) -> tuple[str, TokenUsage]:
        return await self._chat(model, prompt, task, "json" if json_mode else None)

    async def complete_structured(
        self, model: str, prompt: str, schema_name: str, schema: dict[str, object],
    ) -> tuple[str, TokenUsage]:
        return await self.complete_structured_task(model, prompt, schema_name, schema, "decision")

    async def complete_structured_task(
        self, model: str, prompt: str, schema_name: str, schema: dict[str, object], task: str,
    ) -> tuple[str, TokenUsage]:
        return await self._chat(model, prompt, task, schema)

    async def _chat(
        self, model: str, prompt: str, task: str, output_format: str | dict[str, object] | None,
    ) -> tuple[str, TokenUsage]:
        payload: dict[str, object] = {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "think": False, "keep_alive": "5m",
            "options": {
                "temperature": 0, "num_predict": _task_token_limit(task),
                "num_ctx": max(2048, int(os.environ.get("OLLAMA_NUM_CTX", "16384"))),
            },
        }
        if output_format is not None:
            payload["format"] = output_format
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport,
            ) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
            if response.is_error:
                if (response.status_code == 400 and isinstance(output_format, dict)
                        and "failed to parse grammar" in response.text.lower()):
                    self._schema_unsupported_models.add(model)
                    raise StructuredOutputUnsupported(
                        "Ollama cannot compile this structured-output schema."
                    )
                raise ModelError(f"Ollama request failed (HTTP {response.status_code}).")
            body = response.json()
            usage = TokenUsage(
                input_tokens=body.get("prompt_eval_count"), output_tokens=body.get("eval_count"),
            )
            message = body.get("message") or {}
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ModelError("Ollama returned no text output.", usage=usage)
            return content.strip(), usage
        except httpx.HTTPError as exc:
            raise ModelError("The local Ollama server could not complete the request.") from exc
        except (ValueError, TypeError, AttributeError) as exc:
            raise ModelError("Ollama returned an invalid completion response.") from exc


def _structured_output_unsupported(error: ModelError) -> bool:
    """Only a documented schema-capability rejection permits plain JSON fallback."""
    if isinstance(error, ProviderRequestError):
        return error.capability_rejection
    message = str(error).lower()
    if any(term in message for term in (
        "response_format is not supported", "response_format unsupported",
        "json_schema is not supported", "json_schema unsupported",
        "tool_choice is not supported", "tool_choice unsupported",
    )):
        return True
    # Some gateways describe the same capability failure as an invalid/unknown
    # parameter. Do not treat unrelated 400s (invalid model, bad schema) as such.
    mentions_format = any(
        term in message for term in ("response_format", "json_schema", "tool_choice")
    )
    rejects_format = any(
        term in message
        for term in ("unsupported", "not supported", "unknown parameter", "unrecognized parameter")
    )
    return mentions_format and rejects_format


CLAUDE_CODE_EXECUTABLE = Path.home() / ".local/bin/claude"
"""AgentRouter drives Claude through the Claude Code CLI installed at this fixed path."""

_CODEX_PREAMBLE = "Do not use tools or inspect files. Return only the requested response.\n\n"

AGENTROUTER_USER_AGENT = "codex_cli_rs/0.20.0"
"""Client identifier AgentRouter accepts on its OpenAI-compatible HTTP API.

The gateway authenticates the *client* as well as the key: it answers requests carrying a
supported CLI's ``User-Agent`` and rejects every other one with HTTP 401
``unauthorized_client_error``, even when the API key is valid. A default ``python-httpx``
request is therefore refused. Override with ``AGENTROUTER_USER_AGENT`` if the gateway's
accepted client set changes.
"""


def _openai_schema_format(
    schema_name: str, schema: dict[str, object]
) -> dict[str, object]:
    """OpenAI-compatible chat-completions JSON-schema response format."""
    return {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "schema": schema, "strict": True},
    }


def _forced_schema_tools(
    schema_name: str, schema: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Anthropic Messages forced-tool pair for native structured output."""
    tools: list[dict[str, object]] = [
        {
            "name": schema_name,
            "description": "Return the structured decision.",
            "input_schema": schema,
        }
    ]
    return tools, {"type": "tool", "name": schema_name}


class SQLModel(Protocol):
    async def generate(
        self,
        *,
        provider: ModelProvider,
        model: str,
        question: str,
        evidence: str | None,
        schema: SchemaInfo,
        strategy: StrategyHints,
        dialect: DatabaseDialect,
        profile_context: str,
        previous_sql: str | None,
        feedback: str | None,
        rejected_shapes: list[str],
        generation_style: Literal["reasoning", "icl", "alternative"],
    ) -> tuple[str, int] | tuple[str, int, TokenUsage]: ...


class GroqSQLModel:
    """Groq OpenAI-compatible client used for both bounded model calls."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_completion_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self._native_schema_unavailable: set[tuple[str, str]] = set()
        # Shared fallback for callers without a task-specific completion path.
        self.max_completion_tokens = int(
            max_completion_tokens
            if max_completion_tokens is not None
            else os.environ.get("TEXT2SQL_MAX_COMPLETION_TOKENS", "2048")
        )

    async def generate(self, **kwargs: object) -> tuple[str, int, TokenUsage]:
        from time import perf_counter

        prompt = _prompt(
            str(kwargs["question"]),
            cast(str | None, kwargs.get("evidence")),
            cast(SchemaInfo, kwargs["schema"]),
            cast(StrategyHints, kwargs["strategy"]),
            cast(DatabaseDialect, kwargs["dialect"]),
            str(kwargs["profile_context"]),
            cast(str | None, kwargs.get("previous_sql")),
            cast(str | None, kwargs.get("feedback")),
            cast(list[str], kwargs["rejected_shapes"]),
            cast(Literal["reasoning", "icl", "alternative"], kwargs["generation_style"]),
        )
        started = perf_counter()
        content, usage = await self._complete_detailed(
            str(kwargs["model"]), prompt,
            int(os.environ.get("TEXT2SQL_SQL_MAX_TOKENS", str(self.max_completion_tokens))),
        )
        return content, round((perf_counter() - started) * 1_000), usage

    async def complete(self, model: str, prompt: str) -> str:
        content, _ = await self.complete_detailed(model, prompt)
        return content

    async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        return await self._complete_detailed(model, prompt, self.max_completion_tokens)

    async def complete_task(
        self, model: str, prompt: str, task: str, *, json_mode: bool = False,
    ) -> tuple[str, TokenUsage]:
        budget_name = {
            "decision": "TEXT2SQL_TOP_DECISION_MAX_TOKENS",
            "decision_repair": "TEXT2SQL_TOP_REPAIR_MAX_TOKENS",
            "final_answer": "TEXT2SQL_TOP_ANSWER_MAX_TOKENS",
            "context": "TEXT2SQL_CONTEXT_MAX_TOKENS",
            "plotting": "TEXT2SQL_PLOTTING_MAX_TOKENS",
            "dashboard": "TEXT2SQL_DASHBOARD_MAX_TOKENS",
        }.get(task, "TEXT2SQL_MAX_COMPLETION_TOKENS")
        default = "2048" if task == "decision_repair" else "1536"
        return await self._complete_detailed(
            model, prompt, int(os.environ.get(budget_name, default)),
            response_format={"type": "json_object"} if json_mode else None,
        )

    async def complete_structured(
        self,
        model: str,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> tuple[str, TokenUsage]:
        """Prefer Groq-native JSON-schema output; one plain retry keeps support for
        models whose gateway rejects ``response_format``."""
        if not self.native_schema_supported(model, schema_name):
            raise StructuredOutputFallbackRequired(
                "Native structured output was rejected earlier for this model and schema."
            )
        try:
            return await self._complete_detailed(
                model,
                prompt,
                int(os.environ.get("TEXT2SQL_STRUCTURED_MAX_COMPLETION_TOKENS", "1536")),
                response_format=_openai_schema_format(schema_name, schema),
            )
        except ProviderRequestError as exc:
            if exc.code == "json_validate_failed":
                self._native_schema_unavailable.add((model, schema_name))
                raise StructuredOutputFallbackRequired(
                    "Groq strict JSON generation failed; retrying in JSON mode."
                ) from exc
            if not _structured_output_unsupported(exc):
                raise
            self._native_schema_unavailable.add((model, schema_name))
            raise StructuredOutputUnsupported(str(exc)) from exc

    async def complete_structured_task(
        self, model: str, prompt: str, schema_name: str,
        schema: dict[str, object], task: str,
    ) -> tuple[str, TokenUsage]:
        if not self.native_schema_supported(model, schema_name):
            raise StructuredOutputFallbackRequired(
                "Native structured output was rejected earlier for this model and schema."
            )
        try:
            return await self._complete_detailed(
                model, prompt, _task_token_limit(task),
                response_format=_openai_schema_format(schema_name, schema),
            )
        except ProviderRequestError as exc:
            if exc.code == "json_validate_failed":
                self._native_schema_unavailable.add((model, schema_name))
                raise StructuredOutputFallbackRequired(
                    "Groq strict JSON generation failed; retrying in JSON mode."
                ) from exc
            if not _structured_output_unsupported(exc):
                raise
            self._native_schema_unavailable.add((model, schema_name))
            raise StructuredOutputUnsupported(str(exc)) from exc

    def native_schema_supported(self, model: str, schema_name: str) -> bool:
        return (model, schema_name) not in self._native_schema_unavailable

    async def _complete_detailed(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        response_format: dict[str, object] | None = None,
    ) -> tuple[str, TokenUsage]:
        if not self.api_key:
            raise ModelError("GROQ_API_KEY is not configured.")
        started = perf_counter()
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                for attempt in range(3):
                    if attempt:
                        budget = active_request_budget()
                        if budget is not None:
                            budget.claim("model")
                    response = await client.post(
                        "chat/completions",
                        headers={
                            "authorization": f"Bearer {self.api_key}",
                            "content-type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.2,
                            "max_completion_tokens": max_tokens,
                            "reasoning_effort": "none",
                            "reasoning_format": "hidden",
                            **(
                                {"response_format": response_format}
                                if response_format is not None
                                else {}
                            ),
                        },
                    )
                    if response.status_code != 429 or attempt == 2:
                        break
                    await asyncio.sleep(_groq_retry_delay(response))
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                content = choice["message"]["content"]
        except httpx.HTTPStatusError as exc:
            try:
                error = exc.response.json().get("error") or {}
                code = str(error.get("code") or error.get("type") or "")[:80]
                detail = str(error.get("message") or "").lower()
            except (ValueError, TypeError, AttributeError):
                code = ""
                detail = ""
            capability_rejection = (
                any(term in detail for term in ("response_format", "json_schema", "schema"))
                and any(term in detail for term in (
                    "unsupported", "not supported", "unknown parameter",
                    "unrecognized parameter", "unsupported keyword",
                ))
            )
            request_id = exc.response.headers.get("x-request-id")
            phase = "structured" if response_format is not None else "completion"
            problem = ProviderRequestError(
                "groq", model, phase, exc.response.status_code,
                code, request_id, round((perf_counter() - started) * 1000),
                capability_rejection=capability_rejection,
                retry_count=attempt,
            )
            logger.warning(
                "provider_request_failed provider=%s model=%s phase=%s status=%s "
                "code=%s request_id=%s latency_ms=%s response_format=%s retry_count=%s",
                problem.provider, problem.model, problem.phase, problem.status,
                problem.code, problem.request_id, problem.latency_ms,
                response_format.get("type") if response_format else "text", attempt,
            )
            raise problem from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelError(f"The Groq request failed: {exc}") from exc
        if choice.get("finish_reason") == "length":
            raise ModelError(
                "Groq output was truncated at the completion-token limit.",
                usage=_token_usage(body.get("usage") or {}).model_copy(
                    update={"provider_retries": attempt}
                ),
            )
        if not isinstance(content, str) or not content.strip():
            raise ModelError("Groq returned no text output.")
        return content, _token_usage(body.get("usage") or {}).model_copy(
            update={"provider_retries": attempt}
        )


class JustDoWorkSQLModel:
    """Anthropic Messages-compatible client for project-scoped JustDoWork models."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        configured_base = (base_url or "").rstrip("/")
        if configured_base and not configured_base.endswith("/v1"):
            configured_base = f"{configured_base}/v1"
        self.base_url = configured_base
        self.timeout = timeout
        self.transport = transport

    async def generate(self, **kwargs: object) -> tuple[str, int, TokenUsage]:
        from time import perf_counter

        prompt = _prompt(
            str(kwargs["question"]),
            cast(str | None, kwargs.get("evidence")),
            cast(SchemaInfo, kwargs["schema"]),
            cast(StrategyHints, kwargs["strategy"]),
            cast(DatabaseDialect, kwargs["dialect"]),
            str(kwargs["profile_context"]),
            cast(str | None, kwargs.get("previous_sql")),
            cast(str | None, kwargs.get("feedback")),
            cast(list[str], kwargs["rejected_shapes"]),
            cast(Literal["reasoning", "icl", "alternative"], kwargs["generation_style"]),
        )
        started = perf_counter()
        content, usage = await self.complete_detailed(str(kwargs["model"]), prompt)
        return content, round((perf_counter() - started) * 1_000), usage

    async def complete(self, model: str, prompt: str) -> str:
        content, _ = await self.complete_detailed(model, prompt)
        return content

    async def complete_structured(
        self,
        model: str,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> tuple[str, TokenUsage]:
        """Use Anthropic's forced-tool JSON-schema output when the gateway allows it,
        with one plain completion as fallback."""
        try:
            return await self._structured_messages_completion(
                model, prompt, schema_name, schema
            )
        except ModelError as exc:
            if not _structured_output_unsupported(exc):
                raise
            raise StructuredOutputUnsupported(str(exc)) from exc

    async def complete_structured_task(
        self, model: str, prompt: str, schema_name: str,
        schema: dict[str, object], task: str,
    ) -> tuple[str, TokenUsage]:
        try:
            return await self._structured_messages_completion(
                model, prompt, schema_name, schema, max_tokens=_task_token_limit(task)
            )
        except ModelError as exc:
            if not _structured_output_unsupported(exc):
                raise
            raise StructuredOutputUnsupported(str(exc)) from exc

    async def complete_task(
        self, model: str, prompt: str, task: str, *, json_mode: bool = False,
    ) -> tuple[str, TokenUsage]:
        if json_mode:
            try:
                return await self.complete_structured_task(
                    model, prompt, "BoundedTaskOutput", {"type": "object"}, task
                )
            except StructuredOutputUnsupported:
                pass
        return await self.complete_detailed(
            model, prompt, max_tokens=_task_token_limit(task)
        )

    async def _structured_messages_completion(
        self,
        model: str,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 4_000,
    ) -> tuple[str, TokenUsage]:
        if not self.api_key:
            raise ModelError("JUSTDOWORK_API_KEY is not configured.")
        if not self.base_url:
            raise ModelError("JUSTDOWORK_BASE_URL is not configured.")
        tools, tool_choice = _forced_schema_tools(schema_name, schema)
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    "messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0,
                        "tools": tools,
                        "tool_choice": tool_choice,
                    },
                )
                response.raise_for_status()
                body = response.json()
                blocks = body["content"]
                payload = next(
                    (
                        block["input"]
                        for block in blocks
                        if block.get("type") == "tool_use"
                        and isinstance(block.get("input"), dict)
                    ),
                    None,
                )
        except httpx.HTTPStatusError as exc:
            detail = _response_detail(exc.response)
            raise ModelError(
                f"The JustDoWork structured request failed: "
                f"{detail or f'HTTP {exc.response.status_code}'}"
            ) from exc
        except (httpx.HTTPError, KeyError, StopIteration, TypeError, ValueError) as exc:
            raise ModelError(f"The JustDoWork structured request failed: {exc}") from exc
        if payload is None:
            raise ModelError("JustDoWork returned no structured decision.")
        return json.dumps(payload, separators=(",", ":")), _token_usage(body.get("usage") or {})

    async def complete_detailed(
        self, model: str, prompt: str, max_tokens: int = 4_000,
    ) -> tuple[str, TokenUsage]:
        if not self.api_key:
            raise ModelError("JUSTDOWORK_API_KEY is not configured.")
        if not self.base_url:
            raise ModelError("JUSTDOWORK_BASE_URL is not configured.")
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    "messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0,
                    },
                )
                response.raise_for_status()
                body = response.json()
                blocks = body["content"]
                content = "\n".join(
                    str(block["text"])
                    for block in blocks
                    if block.get("type") == "text" and block.get("text")
                )
        except httpx.HTTPStatusError as exc:
            detail = _response_detail(exc.response)
            raise ModelError(
                f"The JustDoWork request failed: {detail or f'HTTP {exc.response.status_code}'}"
            ) from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ModelError(f"The JustDoWork request failed: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelError("JustDoWork returned no text output.")
        return content, _token_usage(body.get("usage") or {})


class CodeCraftSQLModel:
    """CodeCraft chat-completions client for SQL and bounded reasoning calls."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://codecraftapi.com/v1",
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    async def generate(self, **kwargs: object) -> tuple[str, int, TokenUsage]:
        from time import perf_counter

        prompt = _prompt(
            str(kwargs["question"]),
            cast(str | None, kwargs.get("evidence")),
            cast(SchemaInfo, kwargs["schema"]),
            cast(StrategyHints, kwargs["strategy"]),
            cast(DatabaseDialect, kwargs["dialect"]),
            str(kwargs["profile_context"]),
            cast(str | None, kwargs.get("previous_sql")),
            cast(str | None, kwargs.get("feedback")),
            cast(list[str], kwargs["rejected_shapes"]),
            cast(Literal["reasoning", "icl", "alternative"], kwargs["generation_style"]),
        )
        started = perf_counter()
        content, usage = await self.complete_detailed(str(kwargs["model"]), prompt)
        return content, round((perf_counter() - started) * 1_000), usage

    async def complete(self, model: str, prompt: str) -> str:
        content, _ = await self.complete_detailed(model, prompt)
        return content

    async def complete_structured(
        self,
        model: str,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> tuple[str, TokenUsage]:
        """Prefer native JSON-schema output; one plain retry keeps support for
        gateways that reject ``response_format``."""
        try:
            return await self._complete_detailed(
                model, prompt, response_format=_openai_schema_format(schema_name, schema)
            )
        except ModelError as exc:
            if not _structured_output_unsupported(exc):
                raise
            raise StructuredOutputUnsupported(str(exc)) from exc

    async def complete_structured_task(
        self, model: str, prompt: str, schema_name: str,
        schema: dict[str, object], task: str,
    ) -> tuple[str, TokenUsage]:
        try:
            return await self._complete_detailed(
                model, prompt,
                response_format=_openai_schema_format(schema_name, schema),
                max_tokens=_task_token_limit(task),
            )
        except ModelError as exc:
            if not _structured_output_unsupported(exc):
                raise
            raise StructuredOutputUnsupported(str(exc)) from exc

    async def complete_task(
        self, model: str, prompt: str, task: str, *, json_mode: bool = False,
    ) -> tuple[str, TokenUsage]:
        limit = _task_token_limit(task)
        if json_mode:
            try:
                return await self._complete_detailed(
                    model, prompt, response_format={"type": "json_object"},
                    max_tokens=limit,
                )
            except ModelError as exc:
                if not _structured_output_unsupported(exc):
                    raise
        return await self._complete_detailed(model, prompt, max_tokens=limit)

    async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        return await self._complete_detailed(model, prompt)

    async def _complete_detailed(
        self,
        model: str,
        prompt: str,
        response_format: dict[str, object] | None = None,
        *,
        max_tokens: int = 4_096,
    ) -> tuple[str, TokenUsage]:
        if not self.api_key:
            raise ModelError("CODECRAFT_API_KEY is not configured.")
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
            **(
                {"response_format": response_format} if response_format is not None else {}
            ),
        }
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    "chat/completions",
                    headers={
                        "authorization": f"Bearer {self.api_key}",
                        "content-type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise ModelError(
                f"The CodeCraft request failed: {_response_detail(exc.response)}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelError(f"The CodeCraft request failed: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelError("CodeCraft returned no text output.")
        return content, _token_usage(body.get("usage") or {})


class OpenRouterSQLModel(CodeCraftSQLModel):
    """OpenRouter chat-completions adapter with local structured-output validation."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 180.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(api_key, base_url, timeout, transport)

    def native_schema_supported(self, model: str, schema_name: str) -> bool:
        return False

    async def complete_structured(
        self, model: str, prompt: str, schema_name: str, schema: dict[str, object],
    ) -> tuple[str, TokenUsage]:
        raise StructuredOutputUnsupported(
            "This OpenRouter free model does not enforce JSON-schema responses."
        )

    async def complete_task(
        self, model: str, prompt: str, task: str, *, json_mode: bool = False,
    ) -> tuple[str, TokenUsage]:
        # The model does not support response_format; callers validate JSON locally.
        setting = {
            "decision": "TEXT2SQL_TOP_DECISION_MAX_TOKENS",
            "decision_repair": "TEXT2SQL_TOP_REPAIR_MAX_TOKENS",
            "final_answer": "TEXT2SQL_TOP_ANSWER_MAX_TOKENS",
            "context": "TEXT2SQL_CONTEXT_MAX_TOKENS",
            "plotting": "TEXT2SQL_PLOTTING_MAX_TOKENS",
            "dashboard": "TEXT2SQL_DASHBOARD_MAX_TOKENS",
        }.get(task, "TEXT2SQL_MAX_COMPLETION_TOKENS")
        return await self._complete_detailed(
            model, prompt, max_tokens=int(os.environ.get(setting, "4096"))
        )

    async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        return await self._complete_detailed(
            model, prompt,
            max_tokens=int(os.environ.get("TEXT2SQL_SQL_MAX_TOKENS", "4096")),
        )

    async def _complete_detailed(
        self,
        model: str,
        prompt: str,
        response_format: dict[str, object] | None = None,
        *,
        max_tokens: int = 4_096,
    ) -> tuple[str, TokenUsage]:
        if not self.api_key:
            raise ModelError("OPENROUTER_API_KEY is not configured.")
        if response_format is not None:
            raise StructuredOutputUnsupported(
                "This OpenRouter free model does not enforce JSON-schema responses."
            )
        started = perf_counter()
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, transport=self.transport,
            ) as client:
                response = await client.post(
                    "chat/completions",
                    headers={
                        "authorization": f"Bearer {self.api_key}",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                content = choice["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise ProviderRequestError(
                "openrouter", model, "completion", exc.response.status_code,
                "", exc.response.headers.get("x-request-id"),
                round((perf_counter() - started) * 1000),
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelError(
                "The OpenRouter request failed or returned an invalid response."
            ) from exc
        usage = _token_usage(body.get("usage") or {})
        if choice.get("finish_reason") == "length":
            raise ModelError("OpenRouter output was truncated at the token limit.", usage=usage)
        if not isinstance(content, str) or not content.strip():
            raise ModelError("OpenRouter returned no text output.", usage=usage)
        return content, usage


def _response_detail(response: httpx.Response) -> str:
    """Extract a bounded, non-secret provider error message."""
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    error = payload.get("error", payload) if isinstance(payload, dict) else payload
    if isinstance(error, dict):
        detail = error.get("message") or error.get("detail") or error.get("type")
    else:
        detail = error
    return str(detail or f"HTTP {response.status_code}").replace("\n", " ")[:500]


def _groq_retry_delay(response: httpx.Response) -> float:
    """Return Groq's bounded retry delay for a transient token-rate rejection."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(max(float(header), 0.1), 30.0)
        except ValueError:
            pass
    match = re.search(r"try again in\s+([0-9.]+)s", response.text, re.IGNORECASE)
    if match:
        return min(max(float(match.group(1)) + 0.25, 0.1), 30.0)
    return 2.0


class AgentRouterClaudeModel:
    """Claude Code client for the AgentRouter gateway.

    AgentRouter requires Claude Code-compatible request metadata. A raw HTTP
    transport remains injectable only for deterministic contract tests.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://agentrouter.org",
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    async def generate(
        self,
        *,
        provider: ModelProvider,
        model: str,
        question: str,
        evidence: str | None,
        schema: SchemaInfo,
        strategy: StrategyHints,
        dialect: DatabaseDialect,
        profile_context: str,
        previous_sql: str | None,
        feedback: str | None,
        rejected_shapes: list[str],
        generation_style: Literal["reasoning", "icl", "alternative"],
    ) -> tuple[str, int, TokenUsage]:
        from time import perf_counter

        if not self.api_key:
            raise ModelError("AGENTROUTER_API_KEY is not configured.")
        prompt = _prompt(
            question,
            evidence,
            schema,
            strategy,
            dialect,
            profile_context,
            previous_sql,
            feedback,
            rejected_shapes,
            generation_style,
        )
        started = perf_counter()
        if self.transport is None and self.base_url == "https://agentrouter.org":
            content, usage = await self._generate_with_claude_code_detailed(model, prompt)
            return content, round((perf_counter() - started) * 1_000), usage
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    "/v1/messages?beta=true",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 2_000,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                body = response.json()
                blocks = body["content"]
                content = next(
                    block["text"]
                    for block in blocks
                    if block.get("type") == "text" and block.get("text")
                )
        except (httpx.HTTPError, KeyError, StopIteration, TypeError, ValueError) as exc:
            raise ModelError("The AgentRouter Claude request failed.") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelError("AgentRouter returned no SQL text.")
        return (
            content,
            round((perf_counter() - started) * 1_000),
            _token_usage(body.get("usage") or {}),
        )

    async def _generate_with_claude_code(self, model: str, prompt: str) -> str:
        content, _ = await self._generate_with_claude_code_detailed(model, prompt)
        return content

    async def _generate_with_claude_code_detailed(
        self, model: str, prompt: str
    ) -> tuple[str, TokenUsage]:
        executable = CLAUDE_CODE_EXECUTABLE
        if not executable.is_file():
            raise ModelError(f"Claude Code is not installed at {CLAUDE_CODE_EXECUTABLE}.")
        environment = os.environ.copy()
        environment.update(
            {
                "ANTHROPIC_BASE_URL": self.base_url,
                "ANTHROPIC_AUTH_TOKEN": self.api_key or "",
                "ANTHROPIC_API_KEY": self.api_key or "",
                "CLAUDE_CODE_USE_AUTH_TOKEN": "true",
                "ANTHROPIC_MODEL": model,
            }
        )
        process = await asyncio.create_subprocess_exec(
            str(executable),
            "-p",
            "--model",
            model,
            "--tools",
            "",
            "--max-turns",
            "1",
            "--output-format",
            "json",
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except asyncio.CancelledError:
            process.kill()
            await process.communicate()
            raise
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise ModelError("Claude Code generation timed out.") from exc
        if process.returncode != 0:
            message = stderr.decode(errors="replace")[-500:].strip()
            raise ModelError(f"Claude Code failed: {message or 'unknown error'}")
        try:
            body = json.loads(stdout)
            content = body["result"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelError("Claude Code returned malformed JSON output.") from exc
        if body.get("is_error") or not isinstance(content, str) or not content.strip():
            errors = body.get("errors") or []
            detail = str(errors[-1])[:500] if errors else "no SQL text"
            raise ModelError(f"Claude Code generation failed: {detail}")
        return content, _token_usage(body.get("usage") or {})

    async def complete(self, model: str, prompt: str) -> str:
        """Return a bounded free-form Claude completion for the controlled agent layer."""
        content, _ = await self.complete_detailed(model, prompt)
        return content

    async def complete_structured(
        self,
        model: str,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> tuple[str, TokenUsage]:
        """Native structured output over theMessages API; the Claude Code CLI
        transport cannot constrain JSON, so it validates locally instead."""
        if self.transport is None and self.base_url == "https://agentrouter.org":
            return await self.complete_detailed(model, prompt)
        try:
            return await self._structured_messages_completion(
                model, prompt, schema_name, schema
            )
        except ModelError as exc:
            if not _structured_output_unsupported(exc):
                raise
            raise StructuredOutputUnsupported(str(exc)) from exc

    async def complete_structured_task(
        self, model: str, prompt: str, schema_name: str,
        schema: dict[str, object], task: str,
    ) -> tuple[str, TokenUsage]:
        if self.transport is None and self.base_url == "https://agentrouter.org":
            return await self.complete_detailed(model, prompt)
        try:
            return await self._structured_messages_completion(
                model, prompt, schema_name, schema, max_tokens=_task_token_limit(task)
            )
        except ModelError as exc:
            if not _structured_output_unsupported(exc):
                raise
            raise StructuredOutputUnsupported(str(exc)) from exc

    async def _structured_messages_completion(
        self,
        model: str,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 4_000,
    ) -> tuple[str, TokenUsage]:
        if not self.api_key:
            raise ModelError("AGENTROUTER_API_KEY is not configured.")
        tools, tool_choice = _forced_schema_tools(schema_name, schema)
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    "/v1/messages?beta=true",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}],
                        "tools": tools,
                        "tool_choice": tool_choice,
                    },
                )
                response.raise_for_status()
                body = response.json()
                blocks = body["content"]
                payload = next(
                    (
                        block["input"]
                        for block in blocks
                        if block.get("type") == "tool_use"
                        and isinstance(block.get("input"), dict)
                    ),
                    None,
                )
        except (httpx.HTTPError, KeyError, StopIteration, TypeError, ValueError) as exc:
            raise ModelError("The AgentRouter Claude structured request failed.") from exc
        if payload is None:
            raise ModelError("AgentRouter Claude returned no structured decision.")
        return json.dumps(payload, separators=(",", ":")), _token_usage(body.get("usage") or {})

    async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        """Return a completion and provider-reported token usage."""
        if not self.api_key:
            raise ModelError("AGENTROUTER_API_KEY is not configured.")
        if self.transport is None and self.base_url == "https://agentrouter.org":
            return await self._generate_with_claude_code_detailed(model, prompt)
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    "/v1/messages?beta=true",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 4_000,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                body = response.json()
                blocks = body["content"]
                content = next(
                    block["text"]
                    for block in blocks
                    if block.get("type") == "text" and block.get("text")
                )
        except (httpx.HTTPError, KeyError, StopIteration, TypeError, ValueError) as exc:
            raise ModelError("The AgentRouter Claude completion failed.") from exc
        return str(content), _token_usage(body.get("usage") or {})


class AgentRouterCodexModel:
    """Use AgentRouter GPT over the gateway's OpenAI-compatible HTTP API.

    AgentRouter fronts GPT at ``{base_url}/v1``, so no local CLI is required. The
    gateway may speak either the chat-completions or the responses wire format, so
    both are attempted; a local Codex CLI is used only as a fallback when installed.
    """

    _HTTP_PATHS = ("/v1/chat/completions", "/v1/responses")
    _WRONG_WIRE_FORMAT = frozenset({400, 404, 405, 415, 422, 501})

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://agentrouter.org",
        timeout: float = 180.0,
        executable: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        user_agent: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.executable = executable
        self.transport = transport
        self.user_agent = (
            user_agent or os.environ.get("AGENTROUTER_USER_AGENT") or AGENTROUTER_USER_AGENT
        )

    async def generate(self, **kwargs: object) -> tuple[str, int, TokenUsage]:
        from time import perf_counter

        evidence = cast(str | None, kwargs.get("evidence"))
        previous_sql = cast(str | None, kwargs.get("previous_sql"))
        feedback = cast(str | None, kwargs.get("feedback"))
        prompt = _prompt(
            str(kwargs["question"]),
            evidence,
            cast(SchemaInfo, kwargs["schema"]),
            cast(StrategyHints, kwargs["strategy"]),
            cast(DatabaseDialect, kwargs["dialect"]),
            str(kwargs["profile_context"]),
            previous_sql,
            feedback,
            cast(list[str], kwargs["rejected_shapes"]),
            cast(Literal["reasoning", "icl", "alternative"], kwargs["generation_style"]),
        )
        started = perf_counter()
        content, usage = await self._complete_detailed(str(kwargs["model"]), prompt)
        return content, round((perf_counter() - started) * 1_000), usage

    async def complete(self, model: str, prompt: str) -> str:
        content, _ = await self._complete_detailed(model, prompt)
        return content

    async def complete_task(
        self, model: str, prompt: str, task: str, *, json_mode: bool = False,
    ) -> tuple[str, TokenUsage]:
        """Use task-specific output limits on the HTTP transport."""
        if self.executable is not None:
            # The optional Codex CLI transport does not expose a per-request
            # completion-token limit; the shared request budget still applies.
            return await self._complete_with_codex_cli(model, prompt)
        return await self._complete_over_http(
            model, prompt,
            schema_name="BoundedTaskOutput" if json_mode else None,
            schema={"type": "object"} if json_mode else None,
            max_tokens=_task_token_limit(task),
        )

    async def complete_structured(
        self,
        model: str,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> tuple[str, TokenUsage]:
        """Native structured output over either GPT wire format; the Codex CLI
        transport cannot constrain JSON, so it validates locally instead."""
        if self.executable is not None:
            return await self._complete_with_codex_cli(model, prompt)
        try:
            return await self._complete_over_http(
                model, prompt, schema_name=schema_name, schema=schema
            )
        except ModelError as exc:
            if not _structured_output_unsupported(exc):
                raise
            raise StructuredOutputUnsupported(str(exc)) from exc

    async def complete_structured_task(
        self, model: str, prompt: str, schema_name: str,
        schema: dict[str, object], task: str,
    ) -> tuple[str, TokenUsage]:
        if self.executable is not None:
            return await self._complete_with_codex_cli(model, prompt)
        try:
            return await self._complete_over_http(
                model, prompt, schema_name=schema_name, schema=schema,
                max_tokens=_task_token_limit(task),
            )
        except ModelError as exc:
            if not _structured_output_unsupported(exc):
                raise
            raise StructuredOutputUnsupported(str(exc)) from exc

    async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        return await self._complete_detailed(model, prompt)

    async def _complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        """Complete over HTTP, or through an explicitly configured Codex CLI.

        An injected ``executable`` selects the CLI transport outright, mirroring how an
        injected transport selects raw HTTP for the Claude client.
        """
        if not self.api_key:
            raise ModelError("AGENTROUTER_API_KEY is not configured.")
        if self.executable is not None:
            return await self._complete_with_codex_cli(model, prompt)
        try:
            return await self._complete_over_http(model, prompt)
        except ModelError as http_error:
            if codex_cli_path() is None:
                raise
            try:
                return await self._complete_with_codex_cli(model, prompt)
            except ModelError as cli_error:
                raise ModelError(f"{http_error} Codex CLI fallback: {cli_error}") from cli_error

    async def _complete_over_http(
        self,
        model: str,
        prompt: str,
        *,
        schema_name: str | None = None,
        schema: dict[str, object] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, TokenUsage]:
        failures: list[str] = []
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            for path in self._HTTP_PATHS:
                try:
                    response = await client.post(
                        path,
                        headers={
                            "authorization": f"Bearer {self.api_key}",
                            "content-type": "application/json",
                            "user-agent": self.user_agent,
                        },
                        json=_gpt_payload(
                            path,
                            model,
                            _CODEX_PREAMBLE + prompt,
                            schema_name=schema_name,
                            schema=schema,
                            max_tokens=max_tokens,
                        ),
                    )
                except httpx.HTTPError as exc:
                    failures.append(f"{path}: {type(exc).__name__}")
                    continue
                if response.status_code >= 400:
                    failures.append(f"{path}: HTTP {response.status_code} {_gpt_error(response)}")
                    if response.status_code in self._WRONG_WIRE_FORMAT:
                        continue
                    break
                try:
                    body = response.json()
                except ValueError:
                    failures.append(f"{path}: malformed JSON body")
                    continue
                content = _gpt_text(body)
                if content:
                    usage = body.get("usage") if isinstance(body, dict) else None
                    return content, _token_usage(usage or {})
                failures.append(f"{path}: no text in the response body")
        raise ModelError("The AgentRouter GPT request failed (" + "; ".join(failures) + ").")

    async def _complete_with_codex_cli(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        executable = codex_cli_path(self.executable)
        if executable is None:
            raise ModelError("Codex CLI is not installed or available on PATH.")
        environment = os.environ.copy()
        environment["AGENT_ROUTER_TOKEN"] = self.api_key or ""
        with tempfile.NamedTemporaryFile(prefix="text2sql-codex-", delete=False) as output_file:
            output_path = Path(output_file.name)
        command = (
            executable,
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--model",
            model,
            "-c",
            'model_provider="agentrouter-responses"',
            "-c",
            'preferred_auth_method="apikey"',
            "-c",
            'model_providers.agentrouter-responses.name="AgentRouter Responses"',
            "-c",
            f'model_providers.agentrouter-responses.base_url="{self.base_url}/v1"',
            "-c",
            'model_providers.agentrouter-responses.env_key="AGENT_ROUTER_TOKEN"',
            "-c",
            'model_providers.agentrouter-responses.wire_api="responses"',
            "-o",
            str(output_path),
            _CODEX_PREAMBLE + prompt,
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            if process.returncode != 0:
                detail = stderr.decode(errors="replace")[-500:].strip()
                raise ModelError(f"AgentRouter Codex failed: {detail or 'unknown error'}")
            content = output_path.read_text(encoding="utf-8").strip()
        except asyncio.CancelledError:
            process.kill()
            await process.communicate()
            raise
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise ModelError("AgentRouter Codex generation timed out.") from exc
        finally:
            output_path.unlink(missing_ok=True)
        if not content:
            raise ModelError("AgentRouter Codex returned no text output.")
        return content, TokenUsage()


class AgentRouterModel:
    """Dispatch AgentRouter model names to Claude Code or Codex."""

    def __init__(self, claude: AgentRouterClaudeModel, codex: AgentRouterCodexModel) -> None:
        self.claude = claude
        self.codex = codex

    def _select(self, model: str) -> AgentRouterClaudeModel | AgentRouterCodexModel:
        return self.claude if model.casefold().startswith("claude") else self.codex

    async def generate(self, **kwargs: object) -> tuple[str, int, TokenUsage]:
        return await self._select(str(kwargs["model"])).generate(**kwargs)  # type: ignore[arg-type]

    async def complete(self, model: str, prompt: str) -> str:
        return await self._select(model).complete(model, prompt)

    async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        return await self._select(model).complete_detailed(model, prompt)

    async def complete_task(
        self, model: str, prompt: str, task: str, *, json_mode: bool = False,
    ) -> tuple[str, TokenUsage]:
        """Route task reasoning through the backend's best available JSON transport.

        A backend implementing ``complete_task`` handles the call directly;
        otherwise JSON mode rides the backend's native structured transport and
        only a documented structured-output rejection degrades to plain text
        (whose JSON validity is then checked by the caller).
        """
        selected = self._select(model)
        task_completion = getattr(selected, "complete_task", None)
        if callable(task_completion):
            return cast(
                "tuple[str, TokenUsage]",
                await task_completion(model, prompt, task, json_mode=json_mode),
            )
        if json_mode:
            structured = getattr(selected, "complete_structured", None)
            if callable(structured):
                try:
                    return cast(
                        "tuple[str, TokenUsage]",
                        await structured(model, prompt, "BoundedTaskOutput", {"type": "object"}),
                    )
                except StructuredOutputUnsupported:
                    pass  # JSON output unsupported on this backend; validate locally.
        return await selected.complete_detailed(model, prompt)

    def native_schema_supported(self, model: str, schema_name: str) -> bool:
        """Delegate structured-output capability bookkeeping to the selected backend."""
        selected = self._select(model)
        capability = getattr(selected, "native_schema_supported", None)
        if callable(capability):
            return bool(capability(model, schema_name))
        return callable(getattr(selected, "complete_structured", None))

    async def complete_structured(
        self,
        model: str,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> tuple[str, TokenUsage]:
        selected = self._select(model)
        structured = getattr(selected, "complete_structured", None)
        if not callable(structured):
            return await selected.complete_detailed(model, prompt)
        try:
            return cast(
                "tuple[str, TokenUsage]",
                await structured(model, prompt, schema_name, schema),
            )
        except ModelError:
            raise

    async def complete_structured_task(
        self, model: str, prompt: str, schema_name: str,
        schema: dict[str, object], task: str,
    ) -> tuple[str, TokenUsage]:
        selected = self._select(model)
        task_completion = getattr(selected, "complete_structured_task", None)
        if callable(task_completion):
            return cast(tuple[str, TokenUsage], await task_completion(
                model, prompt, schema_name, schema, task
            ))
        return await self.complete_structured(model, prompt, schema_name, schema)


class TrueSotaSQLModel(AgentRouterCodexModel):
    """OpenAI-compatible Responses transport, isolated from Codex CLI settings."""

    def __init__(
        self, api_key: str | None, base_url: str = "https://true-sota.com",
        timeout: float = 180.0, transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(api_key, base_url, timeout, transport=transport)

    async def _complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        if not self.api_key:
            raise ModelError("TRUE_SOTA_API_KEY is not configured.")
        return await self._complete_over_http(model, prompt)

    async def complete_structured(
        self, model: str, prompt: str, schema_name: str, schema: dict[str, object],
    ) -> tuple[str, TokenUsage]:
        if not self.api_key:
            raise ModelError("TRUE_SOTA_API_KEY is not configured.")
        try:
            return await self._complete_over_http(
                model, prompt, schema_name=schema_name, schema=schema
            )
        except ModelError as exc:
            if _structured_output_unsupported(exc):
                raise StructuredOutputUnsupported(str(exc)) from exc
            raise

    async def _complete_over_http(
        self, model: str, prompt: str, *, schema_name: str | None = None,
        schema: dict[str, object] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, TokenUsage]:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, transport=self.transport,
            ) as client:
                response = await client.post(
                    "/v1/responses",
                    headers={"authorization": f"Bearer {self.api_key}"},
                    json=_gpt_payload(
                        "/v1/responses", model, _CODEX_PREAMBLE + prompt,
                        schema_name=schema_name, schema=schema,
                        max_tokens=max_tokens,
                    ),
                )
        except httpx.HTTPError as exc:
            raise ModelError(f"True SOTA request failed: {type(exc).__name__}.") from exc
        if response.status_code >= 400:
            raise ModelError(
                f"True SOTA request failed: HTTP {response.status_code} "
                f"{_gpt_error(response)}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ModelError("True SOTA returned malformed JSON.") from exc
        content = _gpt_text(body)
        if not content:
            raise ModelError("True SOTA returned no text output.")
        usage = body.get("usage") if isinstance(body, dict) else None
        return content, _token_usage(usage)


class RoutingSQLModel:
    def __init__(
        self,
        agentrouter: SQLModel,
        groq: GroqSQLModel,
        justdowork: JustDoWorkSQLModel,
        codecraft: CodeCraftSQLModel,
        true_sota: TrueSotaSQLModel | None = None,
        openrouter: OpenRouterSQLModel | None = None,
    ) -> None:
        self.providers: dict[str, SQLModel] = {
            "agentrouter": agentrouter,
            "groq": groq,
            "justdowork": justdowork,
            "codecraft": codecraft,
        }
        if true_sota is not None:
            self.providers["true_sota"] = true_sota
        if openrouter is not None:
            self.providers["openrouter"] = openrouter

    async def generate(
        self,
        *,
        provider: ModelProvider,
        model: str,
        question: str,
        evidence: str | None,
        schema: SchemaInfo,
        strategy: StrategyHints,
        dialect: DatabaseDialect,
        profile_context: str,
        previous_sql: str | None,
        feedback: str | None,
        rejected_shapes: list[str],
        generation_style: Literal["reasoning", "icl", "alternative"],
    ) -> tuple[str, int] | tuple[str, int, TokenUsage]:
        selected = self.providers.get(provider)
        if selected is None:
            raise ModelError(f"Unsupported model provider: {provider}")
        return await selected.generate(
            provider=provider,
            model=model,
            question=question,
            evidence=evidence,
            schema=schema,
            strategy=strategy,
            dialect=dialect,
            profile_context=profile_context,
            previous_sql=previous_sql,
            feedback=feedback,
            rejected_shapes=rejected_shapes,
            generation_style=generation_style,
        )

    async def complete_detailed(
        self, provider: ModelProvider, model: str, prompt: str
    ) -> tuple[str, TokenUsage]:
        """Route bounded non-SQL reasoning through the user-selected provider."""
        selected = self.providers.get(provider)
        detailed = getattr(selected, "complete_detailed", None)
        if selected is None or not callable(detailed):
            raise ModelError(f"Provider {provider} does not support bounded reasoning calls.")
        return cast(tuple[str, TokenUsage], await detailed(model, prompt))


def _token_usage(value: object) -> TokenUsage:
    usage = value if isinstance(value, dict) else {}
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return TokenUsage(
        input_tokens=int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        cache_read_tokens=int(
            usage.get("cache_read_input_tokens") or usage.get("cache_read_tokens") or cached or 0
        ),
        cache_creation_tokens=int(
            usage.get("cache_creation_input_tokens") or usage.get("cache_creation_tokens") or 0
        ),
    )


def _gpt_error(response: httpx.Response) -> str:
    """Quote the gateway's own refusal so a 401 names its cause instead of only its code.

    AgentRouter reports a rejected *client* and a rejected *key* with the same status, and
    only the body distinguishes them.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text[:120].strip()
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str):
        message = body.get("message") if isinstance(body, dict) else None
    return str(message)[:120].strip() if message else ""


def _gpt_payload(
    path: str,
    model: str,
    instruction: str,
    *,
    schema_name: str | None = None,
    schema: dict[str, object] | None = None,
    max_tokens: int | None = None,
) -> dict[str, object]:
    """Build the body for whichever OpenAI-compatible wire format ``path`` speaks.

    Sampling controls are omitted deliberately: reasoning-class GPT models reject
    ``temperature``, and that rejection is indistinguishable from a wrong wire format.
    When ``schema`` is set, native JSON-schema structured output is requested on the
    matching wire format.
    """
    if path.endswith("/responses"):
        payload: dict[str, object] = {"model": model, "input": instruction, "store": False}
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if schema is not None:
            payload["text"] = {
                "format": {"type": "json_schema", "name": schema_name, "schema": schema}
            }
        return payload
    payload = {"model": model, "messages": [{"role": "user", "content": instruction}]}
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        }
    return payload


def _task_token_limit(task: str) -> int:
    setting = {
        "decision": "TEXT2SQL_TOP_DECISION_MAX_TOKENS",
        "decision_repair": "TEXT2SQL_TOP_REPAIR_MAX_TOKENS",
        "final_answer": "TEXT2SQL_TOP_ANSWER_MAX_TOKENS",
        "context": "TEXT2SQL_CONTEXT_MAX_TOKENS",
        "plotting": "TEXT2SQL_PLOTTING_MAX_TOKENS",
        "dashboard": "TEXT2SQL_DASHBOARD_MAX_TOKENS",
    }.get(task, "TEXT2SQL_MAX_COMPLETION_TOKENS")
    return max(1, int(os.environ.get(setting, "2048")))


def _gpt_text(body: object) -> str:
    """Extract assistant text from either GPT wire format, tolerating string-or-parts content."""
    if not isinstance(body, dict):
        return ""
    direct = body.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    blocks: list[object] = []
    choices = body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                message = choice.get("message")
                blocks.append(message.get("content") if isinstance(message, dict) else None)
                blocks.append(choice.get("text"))
    output = body.get("output")
    if isinstance(output, list):
        blocks.extend(item.get("content") for item in output if isinstance(item, dict))
    return _flatten_text(blocks)


def _flatten_text(blocks: list[object]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, list):
            parts.append(_flatten_text(cast(list[object], block)))
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part.strip()).strip()


def codex_cli_path(override: str | None = None) -> str | None:
    """Resolve the optional Codex CLI that backs the GPT fallback transport."""
    return override or shutil.which("codex")


def claude_code_available() -> bool:
    """Report whether the Claude Code CLI that AgentRouter Claude shells out to exists."""
    return CLAUDE_CODE_EXECUTABLE.is_file()


def _prompt(
    question: str,
    evidence: str | None,
    schema: SchemaInfo,
    strategy: StrategyHints,
    dialect: DatabaseDialect,
    profile_context: str,
    previous_sql: str | None,
    feedback: str | None,
    rejected_shapes: list[str],
    generation_style: Literal["reasoning", "icl", "alternative"],
) -> str:
    verified_context, context_payload = _verified_or_fallback_context(
        profile_context,
        schema,
        question,
        dialect,
    )
    dialect_rules = _dialect_rules(dialect)
    if previous_sql and _is_optimization_feedback(feedback):
        return _optimization_prompt(
            previous_sql,
            dialect,
            dialect_rules,
            context_payload,
            feedback,
        )
    repair = ""
    if previous_sql or feedback:
        instruction = (
            "Correct the cited error using the live schema. Produce a structurally different "
            "query. Never repeat a rejected SQL structure or invent a replacement identifier."
        )
        repair = f"""

REPAIR REQUIRED
Rejected SQL:
{previous_sql or "None"}

Deterministic context:
{feedback or "None"}

Rejected normalized structures:
{rejected_shapes or "None"}

{instruction}
"""
    style_rules = {
        "reasoning": (
            "Derive the query from the requested outputs, filters, grain, joins, aggregation, "
            "and ordering."
        ),
        "icl": "Use any supplied historical example only as an advisory SQL pattern.",
        "alternative": (
            "Seek a semantically equivalent but structurally different solution and re-check "
            "date, NULL, DISTINCT, and aggregation choices."
        ),
    }[generation_style]
    formula_rules = ""
    if context_payload.get("approved_formulas"):
        formula_rules = """

APPROVED FORMULAS:
- Compile every relevant formula in VERIFIED_CONTEXT exactly from its listed operator and
  arguments; do not substitute a plausible raw column.
- In SQLite, zero-safe DIVIDE(a,b) means CAST(a AS REAL) / NULLIF(b, 0).
"""
    historical_rules = ""
    if context_payload.get("historical_examples"):
        historical_rules = """

HISTORICAL EXAMPLES:
- Examples in VERIFIED_CONTEXT are advisory patterns only. They never override the current
  question, trusted evidence, selected schema, relationships, physical formats, or formulas.
"""
    return f"""You are Model 2, the SQL reasoner and generator. Return exactly one safe {dialect}
SELECT or WITH...SELECT statement. Return SQL only: no JSON, markdown, comments, explanation, or
alternative queries.
Authority order: (1) safety and authorization; (2) approved metric definitions and mandatory
metric filters; (3) resolved analytical contract hard requirements; (4) verified physical schema
and relationships; (5) advisory metadata and historical SQL patterns. Compile the resolved
analytical contract in VERIFIED_CONTEXT into SQL. The original question supplies context but
does not override the resolved contract or approved definitions. Use only tables, columns, and
relationships in VERIFIED_CONTEXT; aliases created inside the SQL are allowed. {dialect_rules}
Preserve approved business definitions and all unaffected constraints when the question is a
follow-up or correction. Treat context and historical examples as grounding, not as permission
to add unrequested filters, metrics, or result columns. Do not guess unsupported database facts,
identifiers, or value mappings. If repairing SQL, fix the cited failure while preserving the
user's analytical objective and the already-verified constraints.
Generation approach: {style_rules}

CORRECTNESS-FIRST EFFICIENCY RULES:
- Preserve the requested outputs, filters, formulas, grain, ordering, and result semantics.
- Project only required columns; never use SELECT * unless every column is explicitly requested.
- Give every computed, aggregated, derived, ranked, or otherwise renamed output expression a
  concise, stable, meaningful `AS` alias that reflects the user's requested business output.
  Preserve natural source-column names for plain projected columns unless renaming adds clarity.
- Apply selective filters early when semantics allow, and avoid unnecessary joins, CTEs,
  DISTINCT operations, repeated scans, and sorting.
- When a related table is needed only to test eligibility, prefer EXISTS if it preserves the
  requested grain; pre-aggregate a many-side table before joining when necessary to prevent
  measure multiplication. DISTINCT is not a general fanout repair.
- Prefer a direct range predicate when the verified physical format supports an equivalent one.

VERIFIED CONTEXT RULES:
- `type` is the physical database type. If `format` is present, it is the authoritative stored
  representation and SQL operations must be compatible with it.
- For SQLite format `YYYYMM`, use SUBSTR(column, 1, 4) for year and SUBSTR(column, 5, 2) for month;
  do not apply STRFTIME, date(), or datetime() to that value.
- `example_values` show observed spelling, case, and storage form; they are not an exhaustive list.
- Preserve compatible explicit user literals. Do not invent a value mapping or physical conversion.
{formula_rules}{historical_rules}

Question:
{question}

Trusted evidence:
{evidence or "None"}

VERIFIED_CONTEXT:
{verified_context}
{repair}
"""


def _verified_or_fallback_context(
    profile_context: str,
    schema: SchemaInfo,
    question: str,
    dialect: DatabaseDialect,
) -> tuple[str, dict[str, object]]:
    """Use the authoritative context once; synthesize a compact fallback for direct callers."""
    try:
        parsed = json.loads(profile_context)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("tables"), dict):
        return profile_context.strip(), cast(dict[str, object], parsed)

    fallback: dict[str, object] = {
        "question": question,
        "dialect": dialect,
        "tables": {
            table.name: {
                "primary_key": [column.name for column in table.columns if column.primary_key],
                "columns": {column.name: {"type": column.data_type} for column in table.columns},
            }
            for table in schema.tables
        },
        "relationships": [
            {
                "left": f"{item.from_table}.{item.from_column}",
                "right": f"{item.to_table}.{item.to_column}",
            }
            for item in schema.relationships
        ],
    }
    return json.dumps(fallback, separators=(",", ":")), fallback


def _is_optimization_feedback(feedback: str | None) -> bool:
    return bool(feedback and feedback.startswith("Optimize the previously accepted SQL"))


def _dialect_rules(dialect: DatabaseDialect) -> str:
    if dialect in {"mysql", "tsql", "snowflake", "bigquery"}:
        return {
            "mysql": "Use MySQL 8 SELECT syntax, backtick identifiers and MySQL date functions.",
            "tsql": "Use SQL Server T-SQL, bracket identifiers and TOP/OFFSET FETCH, not LIMIT.",
            "snowflake": "Use Snowflake SQL and verified schema-qualified identifiers.",
            "bigquery": "Use BigQuery GoogleSQL, backticks and verified project/dataset names.",
        }[dialect] + " Use only read-only queries and the supplied verified schema."
    if dialect == "postgres":
        return "Use PostgreSQL 15 syntax. Never use SQLite-only functions such as IIF or STRFTIME."
    return (
        "Use SQLite syntax. Never use PostgreSQL-only functions. In a compound query "
        "(UNION/UNION ALL/INTERSECT/EXCEPT), do not put branch-level ORDER BY or LIMIT "
        "directly before the compound operator; wrap each ranked branch in a subquery/CTE "
        "or use window functions."
    )


def _optimization_prompt(
    accepted_sql: str,
    dialect: DatabaseDialect,
    dialect_rules: str,
    context_payload: dict[str, object],
    feedback: str | None,
) -> str:
    """Build a narrow optimizer prompt that cannot reinterpret the user's question."""
    optimizer_context = {
        key: context_payload[key]
        for key in ("tables", "relationships", "approved_formulas")
        if context_payload.get(key)
    }
    explain_marker = "Original EXPLAIN plan:\n"
    explain_plan = "Unavailable"
    if feedback and explain_marker in feedback:
        explain_plan = feedback.split(explain_marker, 1)[1].strip() or "Unavailable"
    return f"""You are a SQL query optimizer. Optimize the accepted {dialect} SQL only when a
meaningful physical or structural improvement is possible. Return exactly one SQL statement.
Return SQL only: no JSON, markdown, comments, explanation, alternatives, or surrounding text.
Return the accepted SQL unchanged when no safe improvement exists. {dialect_rules}

INVARIANTS:
- Preserve output column count, names, order, result rows, result values, and meaningful ordering.
- Preserve existing output aliases; add a meaningful `AS` alias when a projected expression has
  no stable output name, without changing the result shape.
- Preserve every filter, formula, aggregation, grouping level, grain, ranking, and LIMIT.
- Use only tables, columns, and relationships in REFERENCED_CONTEXT; SQL aliases are allowed.
- Do not add DISTINCT to conceal join fanout or remove required NULL/zero-division handling.

PREFERRED IMPROVEMENTS:
- Remove unnecessary CTEs, subqueries, repeated scans, sorting, and projections.
- Apply selective filters earlier when doing so preserves exact results.
- Use EXISTS for filter-only relationships when it preserves output and grain.
- Pre-aggregate a many-side table only when it preserves exact results and prevents fanout.
- Prefer index-friendly predicates when compatible with the verified physical type and format.

DIALECT:
{dialect}

ACCEPTED_SQL:
{accepted_sql}

REFERENCED_CONTEXT:
{json.dumps(optimizer_context, separators=(",", ":"))}

ORIGINAL_EXPLAIN:
{explain_plan}
"""
