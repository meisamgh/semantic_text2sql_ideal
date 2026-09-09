"""Evidence-producing adapter over the authoritative Text-to-SQL service."""

from __future__ import annotations

import hashlib
import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from autonomous_analytics.models.evidence import Evidence
from autonomous_analytics.models.tools import QueryResultSummary, ToolError, ToolResult
from semantic_text2sql.database import DatabaseError
from semantic_text2sql.models import GenerateResponse, ModelProvider, TokenUsage
from semantic_text2sql.service import ContextMode, QuestionExecution, TextToSQLService

logger = logging.getLogger(__name__)


class TextToSQLTool:
    """Ask an analytical question without exposing arbitrary SQL execution."""

    name = "text_to_sql"

    def __init__(
        self,
        service: TextToSQLService,
        *,
        provider: ModelProvider = "ollama",
        model: str = "qwen3.5:9b",
        context_mode: ContextMode = "retrieval",
        evidence_row_limit: int = 20,
        max_attempts: int = 3,
    ) -> None:
        if not 1 <= evidence_row_limit <= 500:
            raise ValueError("evidence_row_limit must be between 1 and 500")
        self.service = service
        self.provider = provider
        self.model = model
        self.context_mode = context_mode
        self.evidence_row_limit = evidence_row_limit
        self.max_attempts = max_attempts

    async def ask(
        self,
        *,
        question: str,
        db_id: str,
        context: dict[str, Any] | None = None,
        provider: ModelProvider | None = None,
        model: str | None = None,
        case_id: str | None = None,
    ) -> ToolResult:
        """Run one bounded, grounded question and convert its lineage to Evidence."""

        started = perf_counter()
        selected_provider = provider or self.provider
        selected_model = model or self.model
        logger.info(
            "analytics_tool_start tool=%s case_id=%s db_id=%s question=%r",
            self.name,
            case_id,
            db_id,
            question,
        )
        try:
            execution = await self.service.execute_question(
                question=question,
                db_id=db_id,
                provider=selected_provider,
                model=selected_model,
                context_mode=self.context_mode,
                execute=True,
                max_rows=self.evidence_row_limit,
                max_attempts=self.max_attempts,
            )
        except DatabaseError as exc:
            latency_ms = round((perf_counter() - started) * 1_000)
            error = ToolError(
                code=exc.code,
                category="database",
                message=str(exc),
                retryable=exc.code not in {"DATABASE_ID_INVALID", "DATABASE_NOT_FOUND"},
            )
            evidence = self._failure_evidence(
                question=question,
                db_id=db_id,
                context=context,
                provider=selected_provider,
                model=selected_model,
                latency_ms=latency_ms,
                error=error,
            )
            self._log_finish(case_id, db_id, False, 0, None, 0, latency_ms, error.category)
            return ToolResult(success=False, evidence=evidence, error=error)

        latency_ms = round((perf_counter() - started) * 1_000)
        response = execution.generated
        if response.accepted and response.sql is not None:
            evidence = self._success_evidence(execution, context, latency_ms)
            self._log_finish(
                case_id,
                db_id,
                True,
                len(response.attempts),
                _sql_fingerprint(response.sql),
                response.row_count,
                latency_ms,
                None,
            )
            return ToolResult(success=True, evidence=evidence)

        error = _response_error(response)
        evidence = self._failure_evidence(
            question=question,
            db_id=db_id,
            context=context,
            provider=response.provider,
            model=response.model,
            latency_ms=latency_ms,
            error=error,
            response=response,
            execution=execution,
        )
        self._log_finish(
            case_id,
            db_id,
            False,
            len(response.attempts),
            None,
            response.row_count,
            latency_ms,
            error.category,
        )
        return ToolResult(success=False, evidence=evidence, error=error)

    def _success_evidence(
        self,
        execution: QuestionExecution,
        context: dict[str, Any] | None,
        latency_ms: int,
    ) -> Evidence:
        response = execution.generated
        summary = QueryResultSummary(
            columns=response.columns,
            row_count=response.row_count,
            truncated=response.truncated,
            rows=_rows_as_records(response.columns, response.rows[: self.evidence_row_limit]),
        )
        return Evidence(
            evidence_id=f"sql-{uuid4().hex}",
            source_type="sql",
            question=response.question,
            sql=response.sql,
            result_summary=summary.model_dump(mode="json"),
            source_reference=f"database:{response.db_id}",
            metadata=self._lineage(execution, context, latency_ms),
        )

    def _failure_evidence(
        self,
        *,
        question: str,
        db_id: str,
        context: dict[str, Any] | None,
        provider: ModelProvider,
        model: str,
        latency_ms: int,
        error: ToolError,
        response: GenerateResponse | None = None,
        execution: QuestionExecution | None = None,
    ) -> Evidence:
        metadata: dict[str, Any] = {
            "db_id": db_id,
            "resolved_question": question,
            "provider": provider,
            "model": model,
            "latency_ms": latency_ms,
            "error": error.model_dump(mode="json"),
            "investigation_context": _compact_context(context),
        }
        if response is not None and execution is not None:
            metadata.update(self._lineage(execution, context, latency_ms))
        return Evidence(
            evidence_id=f"sql-failure-{uuid4().hex}",
            source_type="sql",
            question=question,
            sql=response.sql if response else None,
            result_summary={"columns": [], "row_count": 0, "truncated": False, "rows": []},
            source_reference=f"database:{db_id}",
            metadata=metadata,
        )

    def _lineage(
        self,
        execution: QuestionExecution,
        context: dict[str, Any] | None,
        latency_ms: int,
    ) -> dict[str, Any]:
        response = execution.generated
        return {
            "db_id": response.db_id,
            "dialect": response.dialect,
            "resolved_question": response.question,
            "selected_tables": execution.approved_tables,
            "selected_columns": (
                response.context_request.columns if response.context_request is not None else {}
            ),
            "verified_model_context": response.model_context,
            "attempts": [item.model_dump(mode="json") for item in response.attempts],
            "attempt_count": len(response.attempts),
            "provider": response.provider,
            "model": response.model,
            "latency_ms": latency_ms,
            "pipeline_timings_ms": {
                "routing": execution.routing_ms,
                "planning": execution.planning_ms,
                "generation": execution.generation_ms,
            },
            "token_usage": _usage_or_unavailable(response.token_usage),
            "termination_reason": response.termination_reason,
            "execution_status": response.execution_status,
            "investigation_context": _compact_context(context),
        }

    @staticmethod
    def _log_finish(
        case_id: str | None,
        db_id: str,
        success: bool,
        attempt_count: int,
        sql_fingerprint: str | None,
        row_count: int,
        latency_ms: int,
        error_category: str | None,
    ) -> None:
        logger.info(
            "analytics_tool_finish tool=text_to_sql case_id=%s db_id=%s success=%s "
            "attempt_count=%s sql_fingerprint=%s row_count=%s latency_ms=%s error_category=%s",
            case_id,
            db_id,
            success,
            attempt_count,
            sql_fingerprint,
            row_count,
            latency_ms,
            error_category,
        )


def _rows_as_records(columns: list[str], rows: list[list[Any]]) -> list[dict[str, Any]]:
    return [dict(zip(columns, row, strict=False)) for row in rows]


def _usage_or_unavailable(usage: TokenUsage) -> dict[str, int | None] | None:
    if all(
        value is None
        for value in (
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cache_creation_tokens,
        )
    ):
        return None
    return usage.model_dump(mode="json")


def _compact_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not context:
        return None
    allowed = {"primary_kpi", "time_range", "segment_filters", "hypothesis_ids"}
    return {key: context[key] for key in sorted(context.keys() & allowed)}


def _response_error(response: GenerateResponse) -> ToolError:
    if response.grounding_issue is not None:
        return ToolError(
            code="VALUE_GROUNDING_REQUIRED",
            category="clarification",
            message=response.grounding_issue.reason,
            retryable=False,
        )
    if response.model_error:
        return ToolError(
            code="MODEL_UNAVAILABLE",
            category="model",
            message=response.model_error,
            retryable=True,
        )
    if response.attempts:
        validation = response.attempts[-1].validation
        category = (
            "safety"
            if validation.code in {"SQL_NOT_READ_ONLY", "SQL_MULTIPLE_STATEMENTS"}
            else "database"
            if validation.code in {"DATABASE_ERROR", "SQL_EXECUTION_FAILED"}
            else "validation"
        )
        return ToolError(
            code=validation.code,
            category=category,
            message=validation.message,
            retryable=category in {"database", "validation"},
        )
    return ToolError(
        code=response.termination_reason.upper(),
        category="generation",
        message="Text-to-SQL did not produce an accepted executable query.",
        retryable=response.termination_reason != "database_error",
    )


def _sql_fingerprint(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()[:12]
