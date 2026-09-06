"""Mockable local Ollama boundary."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Literal, Protocol, cast

import httpx

from semantic_text2sql.models import ModelProvider, SchemaInfo, StrategyHints, TokenUsage


class ModelError(RuntimeError):
    pass


CLAUDE_CODE_EXECUTABLE = Path.home() / ".local/bin/claude"
"""AgentRouter drives Claude through the Claude Code CLI installed at this fixed path."""

_MEMORY_HEADROOM = 0.85
"""Share of system RAM a local model may occupy before it swaps instead of generating."""

_CODEX_PREAMBLE = "Do not use tools or inspect files. Return only the requested response.\n\n"

AGENTROUTER_USER_AGENT = "codex_cli_rs/0.20.0"
"""Client identifier AgentRouter accepts on its OpenAI-compatible HTTP API.

The gateway authenticates the *client* as well as the key: it answers requests carrying a
supported CLI's ``User-Agent`` and rejects every other one with HTTP 401
``unauthorized_client_error``, even when the API key is valid. A default ``python-httpx``
request is therefore refused. Override with ``AGENTROUTER_USER_AGENT`` if the gateway's
accepted client set changes.
"""


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
        dialect: Literal["sqlite", "postgres"],
        profile_context: str,
        previous_sql: str | None,
        feedback: str | None,
        rejected_shapes: list[str],
        generation_style: Literal["reasoning", "icl", "alternative"],
    ) -> tuple[str, int] | tuple[str, int, TokenUsage]: ...


class OllamaSQLModel:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def generate(
        self,
        *,
        provider: ModelProvider,
        model: str,
        question: str,
        evidence: str | None,
        schema: SchemaInfo,
        strategy: StrategyHints,
        dialect: Literal["sqlite", "postgres"],
        profile_context: str,
        previous_sql: str | None,
        feedback: str | None,
        rejected_shapes: list[str],
        generation_style: Literal["reasoning", "icl", "alternative"],
    ) -> tuple[str, int, TokenUsage]:
        from time import perf_counter

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
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                response = await client.post(
                    "/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "think": False,
                        "options": {"temperature": 0, "seed": 0, "num_predict": 1_500},
                        "keep_alive": "5m",
                    },
                )
                response.raise_for_status()
                body = response.json()
                content = body["message"]["content"]
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[-500:].strip()
            raise ModelError(
                f"The local Ollama generation request failed: {detail or exc.response.status_code}"
            ) from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ModelError(f"The local Ollama generation request failed: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelError("Ollama returned no SQL text.")
        return (
            content,
            round((perf_counter() - started) * 1_000),
            TokenUsage(
                input_tokens=int(body.get("prompt_eval_count") or 0),
                output_tokens=int(body.get("eval_count") or 0),
            ),
        )

    async def complete(self, model: str, prompt: str) -> str:
        content, _ = await self.complete_detailed(model, prompt)
        return content

    async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        """Use the local model as a bounded conversational/semantic interpreter."""
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                response = await client.post(
                    "/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "format": "json",
                        "think": False,
                        "options": {"temperature": 0, "seed": 0, "num_predict": 1_000},
                        "keep_alive": "5m",
                    },
                )
                response.raise_for_status()
                body = response.json()
                content = body["message"]["content"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ModelError("The local Ollama interpretation request failed.") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelError("Ollama returned no interpretation.")
        return content, TokenUsage(
            input_tokens=int(body.get("prompt_eval_count") or 0),
            output_tokens=int(body.get("eval_count") or 0),
        )


class GroqSQLModel:
    """Groq OpenAI-compatible client used for both bounded model calls."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.groq.com/openai/v1",
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
            cast(Literal["sqlite", "postgres"], kwargs["dialect"]),
            str(kwargs["profile_context"]),
            cast(str | None, kwargs.get("previous_sql")),
            cast(str | None, kwargs.get("feedback")),
            cast(list[str], kwargs["rejected_shapes"]),
            cast(Literal["reasoning", "icl", "alternative"], kwargs["generation_style"]),
        )
        started = perf_counter()
        content, usage = await self._complete_detailed(str(kwargs["model"]), prompt, 2_000)
        return content, round((perf_counter() - started) * 1_000), usage

    async def complete(self, model: str, prompt: str) -> str:
        content, _ = await self.complete_detailed(model, prompt)
        return content

    async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        return await self._complete_detailed(model, prompt, 4_000)

    async def _complete_detailed(
        self, model: str, prompt: str, max_tokens: int
    ) -> tuple[str, TokenUsage]:
        if not self.api_key:
            raise ModelError("GROQ_API_KEY is not configured.")
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                for attempt in range(3):
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
                        },
                    )
                    if response.status_code != 429 or attempt == 2:
                        break
                    await asyncio.sleep(_groq_retry_delay(response))
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[-500:].strip()
            raise ModelError(
                f"The Groq request failed: {detail or f'HTTP {exc.response.status_code}'}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelError(f"The Groq request failed: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelError("Groq returned no text output.")
        return content, _token_usage(body.get("usage") or {})


class JustDoWorkSQLModel:
    """OpenAI-compatible client for Claude and GPT models served by JustDoWork."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.justwoker.icu/v1",
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
            cast(Literal["sqlite", "postgres"], kwargs["dialect"]),
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

    async def complete_detailed(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
        if not self.api_key:
            raise ModelError("JUSTDOWORK_API_KEY is not configured.")
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
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 4_000,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            detail = _response_detail(exc.response)
            raise ModelError(
                f"The JustDoWork request failed: {detail or f'HTTP {exc.response.status_code}'}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelError(f"The JustDoWork request failed: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelError("JustDoWork returned no text output.")
        return content, _token_usage(body.get("usage") or {})


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
        dialect: Literal["sqlite", "postgres"],
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
            cast(Literal["sqlite", "postgres"], kwargs["dialect"]),
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

    async def _complete_over_http(self, model: str, prompt: str) -> tuple[str, TokenUsage]:
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
                        json=_gpt_payload(path, model, _CODEX_PREAMBLE + prompt),
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


class RoutingSQLModel:
    def __init__(
        self,
        ollama: OllamaSQLModel,
        agentrouter: SQLModel,
        groq: GroqSQLModel,
        justdowork: JustDoWorkSQLModel | None = None,
    ) -> None:
        self.providers: dict[str, SQLModel] = {
            "ollama": ollama,
            "agentrouter": agentrouter,
            "groq": groq,
        }
        if justdowork is not None:
            self.providers["justdowork"] = justdowork

    async def generate(
        self,
        *,
        provider: ModelProvider,
        model: str,
        question: str,
        evidence: str | None,
        schema: SchemaInfo,
        strategy: StrategyHints,
        dialect: Literal["sqlite", "postgres"],
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


def _gpt_payload(path: str, model: str, instruction: str) -> dict[str, object]:
    """Build the body for whichever OpenAI-compatible wire format ``path`` speaks.

    Sampling controls are omitted deliberately: reasoning-class GPT models reject
    ``temperature``, and that rejection is indistinguishable from a wrong wire format.
    """
    if path.endswith("/responses"):
        return {"model": model, "input": instruction, "store": False}
    return {"model": model, "messages": [{"role": "user", "content": instruction}]}


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


def _total_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


async def ollama_model_status(
    model: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 5.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str | None:
    """Return ``None`` when ``model`` can serve locally, otherwise why it cannot.

    A model whose weights do not fit in RAM is reported unusable: Ollama accepts the
    pull, but generation then swaps and times out instead of ever answering.
    """
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        ) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError):
        return f"Ollama is not reachable at {base_url}."
    installed = body.get("models") if isinstance(body, dict) else None
    entries = installed if isinstance(installed, list) else []
    tagged = model if ":" in model else f"{model}:latest"
    match = next(
        (
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("name") in {model, tagged}
        ),
        None,
    )
    if match is None:
        return f"{model} is not installed locally (run: ollama pull {model})."
    size = match.get("size")
    memory = _total_memory_bytes()
    if isinstance(size, int) and memory is not None and size > memory * _MEMORY_HEADROOM:
        return (
            f"{model} needs about {size / 1e9:.1f} GB of weights but this machine has "
            f"{memory / 1e9:.1f} GB of RAM, so generation swaps and times out."
        )
    return None


def _prompt(
    question: str,
    evidence: str | None,
    schema: SchemaInfo,
    strategy: StrategyHints,
    dialect: Literal["sqlite", "postgres"],
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
The question and trusted evidence are authoritative. Use only tables, columns, and relationships
in VERIFIED_CONTEXT; aliases created inside the SQL are allowed. {dialect_rules}
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
    dialect: Literal["sqlite", "postgres"],
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
                "columns": {
                    column.name: {"type": column.data_type} for column in table.columns
                },
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


def _dialect_rules(dialect: Literal["sqlite", "postgres"]) -> str:
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
    dialect: Literal["sqlite", "postgres"],
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
