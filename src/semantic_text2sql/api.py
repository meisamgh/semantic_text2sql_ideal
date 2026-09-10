"""FastAPI application for interactive generation and SQL checking."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.conversation import (
    EXPLANATION_OPERATIONS,
    STATE_REQUIRED_OPERATIONS,
    ConversationConflict,
    ConversationStore,
    classify_operation,
    intent_target,
    interpret_turn_detailed,
    requires_model_interpretation,
    resolve_turn,
)
from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.explanation import deterministic_explanation, explain_turn_detailed
from semantic_text2sql.glossary import GlossaryStore
from semantic_text2sql.historical import HistoricalQueryStore
from semantic_text2sql.hybrid_retrieval import (
    FastEmbedEncoder,
    HybridSchemaRetriever,
    LightGBMSchemaReranker,
)
from semantic_text2sql.llm import (
    GroqSQLModel,
    JustDoWorkSQLModel,
    ModelError,
    OllamaSQLModel,
    RoutingSQLModel,
    SotaSQLModel,
)
from semantic_text2sql.models import (
    CallLedgerEntry,
    ChatRequest,
    ChatResponse,
    CheckRequest,
    CheckResponse,
    ConversationState,
    DatabaseOption,
    GenerateResponse,
    HumanReviewRequest,
    ModelOption,
    ModelProvider,
    TokenUsage,
    TurnInterpretation,
)
from semantic_text2sql.postgres import PostgresRegistry, postgres_databases_from_environment
from semantic_text2sql.profiling import ProfileStore
from semantic_text2sql.provider_registry import configured_model_options
from semantic_text2sql.service import TextToSQLService

logger = logging.getLogger(__name__)


def create_app(
    agent: TextToSQLAgent | None = None,
    conversation_completers: dict[str, Any] | None = None,
) -> FastAPI:
    postgres_databases = postgres_databases_from_environment()
    postgres = PostgresRegistry(postgres_databases) if postgres_databases else None
    sqlite = DatabaseRegistry(Path(os.environ.get("TEXT2SQL_DATABASE_ROOT", "data")))
    profiles = ProfileStore(Path(os.environ.get("TEXT2SQL_PROFILE_ROOT", "profiles")))
    glossaries = GlossaryStore(
        Path(os.environ.get("TEXT2SQL_GLOSSARY_ROOT", "data/business_glossaries"))
    )
    history = HistoricalQueryStore(
        Path(
            os.environ.get("TEXT2SQL_HISTORY_PATH", "benchmarks/data/bird_history_seed42_400.json")
        )
    )
    ollama = OllamaSQLModel(os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    groq = GroqSQLModel(
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    )
    justdowork = JustDoWorkSQLModel(
        os.environ.get("JUSTDOWORK_API_KEY"),
        os.environ.get("JUSTDOWORK_BASE_URL", "https://api.justwoker.icu/v1"),
        float(os.environ.get("JUSTDOWORK_TIMEOUT_SECONDS", "120")),
    )
    sota = SotaSQLModel(
        os.environ.get("SOTA_API_KEY"),
        os.environ.get("SOTA_BASE_URL", "https://true-sota.com"),
        float(os.environ.get("SOTA_TIMEOUT_SECONDS", "180")),
    )
    active_agent = agent or TextToSQLAgent(
        sqlite,
        RoutingSQLModel(ollama, justdowork, groq, justdowork, sota),
        postgres,
        profiles,
    )
    turn_completers = conversation_completers or {
        "ollama": ollama,
        "agentrouter": justdowork,
        "groq": groq,
        "justdowork": justdowork,
        "sota": sota,
    }
    app = FastAPI(
        title="Semantic Text-to-SQL v5",
        version="0.5.0",
        description=(
            "Bounded context selection, compact grounding, SQL-only generation, "
            "safety-first validation, and read-only execution."
        ),
    )
    web_root = Path(__file__).resolve().parents[2] / "web"
    if web_root.is_dir():
        app.mount("/static", StaticFiles(directory=web_root), name="static")
    reranker = None
    reranker_path = os.environ.get("TEXT2SQL_SCHEMA_RERANKER_MODEL")
    if os.environ.get("TEXT2SQL_SCHEMA_RERANKER_ENABLED", "false").casefold() == "true":
        try:
            reranker = LightGBMSchemaReranker(Path(reranker_path or ""))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            logger.warning("schema_reranker_disabled=%s", exc)
    hybrid_retriever = HybridSchemaRetriever(
        FastEmbedEncoder(os.environ.get("TEXT2SQL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")),
        max_tables=int(os.environ.get("TEXT2SQL_RETRIEVAL_TABLES", "5")),
        max_columns_per_table=int(os.environ.get("TEXT2SQL_RETRIEVAL_COLUMNS", "5")),
        reranker=reranker,
        reranker_pool=int(os.environ.get("TEXT2SQL_SCHEMA_RERANKER_POOL", "30")),
    )
    question_service = TextToSQLService(
        database=sqlite,
        postgres=postgres,
        profiles=profiles,
        glossaries=glossaries,
        history=history,
        retriever=hybrid_retriever,
        agent=active_agent,
        context_completers=cast(dict[ModelProvider, Any], turn_completers),
    )
    conversation_store_value = os.environ.get("TEXT2SQL_CONVERSATION_STORE")
    conversation_path = Path(conversation_store_value) if conversation_store_value else None
    conversations = ConversationStore(
        conversation_path,
        ttl_seconds=int(os.environ.get("TEXT2SQL_CONVERSATION_TTL_SECONDS", "86400")),
    )
    chat_jobs: dict[str, dict[str, Any]] = {}
    job_ttl_seconds = max(60, int(os.environ.get("TEXT2SQL_JOB_TTL_SECONDS", "3600")))
    max_jobs = max(10, int(os.environ.get("TEXT2SQL_MAX_JOBS", "500")))

    def cleanup_jobs() -> None:
        now = perf_counter()
        expired = [
            job_id
            for job_id, job in chat_jobs.items()
            if job.get("status") in {"completed", "failed", "cancelled"}
            and now - float(job.get("finished_at") or job["started_at"]) > job_ttl_seconds
        ]
        for job_id in expired:
            chat_jobs.pop(job_id, None)
            chat_tasks.pop(job_id, None)
        if len(chat_jobs) > max_jobs:
            terminal = sorted(
                (
                    (job_id, float(job.get("finished_at") or job["started_at"]))
                    for job_id, job in chat_jobs.items()
                    if job.get("status") in {"completed", "failed", "cancelled"}
                ),
                key=lambda item: item[1],
            )
            for job_id, _ in terminal[: len(chat_jobs) - max_jobs]:
                chat_jobs.pop(job_id, None)
                chat_tasks.pop(job_id, None)

    def set_session_stage(session_id: str, stage: str) -> None:
        for job in chat_jobs.values():
            if job.get("session_id") == session_id and job.get("status") == "running":
                job["stage"] = stage

    chat_tasks: dict[str, asyncio.Task[None]] = {}

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "online"}

    @app.get("/", include_in_schema=False)
    async def web_application() -> FileResponse:
        return FileResponse(web_root / "index.html")

    @app.get("/api/models", response_model=list[ModelOption])
    async def models(response: Response) -> list[ModelOption]:
        response.headers["Cache-Control"] = "no-store"
        return configured_model_options()

    @app.get("/api/databases", response_model=list[DatabaseOption])
    async def databases() -> list[DatabaseOption]:
        options = [
            DatabaseOption(db_id=db_id, dialect="sqlite", configured=True)
            for db_id in sqlite.list_ids()
        ]
        if postgres is not None:
            options.extend(
                DatabaseOption(db_id=db_id, dialect="postgres", configured=True)
                for db_id in postgres.configured_ids()
            )
        elif not postgres_databases:
            options.append(
                DatabaseOption(db_id="books_postgres", dialect="postgres", configured=False)
            )
        return options

    @app.get("/api/databases/{db_id}/analytics-capabilities")
    async def analytics_capabilities(db_id: str) -> dict[str, Any]:
        """Expose a compact, read-only analytical capability summary."""
        if db_id in sqlite.list_ids():
            dialect = "sqlite"
            schema = sqlite.inspect(db_id)
        elif postgres is not None and db_id in postgres.configured_ids():
            dialect = "postgres"
            schema = postgres.inspect(db_id)
        else:
            raise HTTPException(status_code=404, detail="Database was not found or configured.")

        profile = profiles.load(dialect, db_id)
        profile_by_column = (
            {(item.table, item.column): item for item in profile.columns} if profile else {}
        )
        measures: list[dict[str, Any]] = []
        dimensions: list[dict[str, Any]] = []
        time_columns: list[dict[str, Any]] = []
        business_entities: set[str] = set()
        numeric_types = {"integer", "real", "numeric", "decimal", "float", "double"}
        for table in schema.tables:
            business_entities.add(table.name)
            for column in table.columns:
                column_profile = profile_by_column.get((table.name, column.name))
                semantic_type = column_profile.semantic_type if column_profile else None
                field = {
                    "name": column.name,
                    "table": table.name,
                    "data_type": column.data_type,
                    "semantic_type": semantic_type,
                    "description": column.description,
                }
                physical_type = column.data_type.casefold().split("(", 1)[0]
                if semantic_type in {"date", "datetime"} or any(
                    token in column.name.casefold() for token in ("date", "time", "month", "year")
                ):
                    time_columns.append(field)
                elif semantic_type == "numeric" or any(
                    numeric_type in physical_type for numeric_type in numeric_types
                ):
                    if not column.primary_key and semantic_type != "identifier":
                        measures.append(field)
                elif semantic_type not in {"secret", "email", "phone", "postal_address"}:
                    dimensions.append(field)

        glossary = glossaries.load(db_id)
        available_kpis = []
        if glossary is not None:
            for term in glossary.terms:
                if term.formula or term.structural_formula:
                    available_kpis.append(
                        {
                            "name": term.term,
                            "description": term.definition,
                            "synonyms": term.synonyms,
                            "columns": term.columns,
                            "source": "glossary",
                        }
                    )
                if term.core:
                    business_entities.add(term.term)
        return {
            "db_id": db_id,
            "dialect": dialect,
            "tables": [table.name for table in schema.tables],
            "business_entities": sorted(business_entities),
            "measures": measures[:200],
            "dimensions": dimensions[:200],
            "time_columns": time_columns[:100],
            "available_kpis": available_kpis[:100],
        }

    @app.post("/api/check", response_model=CheckResponse)
    async def check(request: CheckRequest) -> CheckResponse:
        return await asyncio.to_thread(active_agent.check, request)

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        started = perf_counter()
        set_session_stage(request.session_id, "conversation")
        previous = conversations.get(request.session_id)
        has_matching_state = previous is not None and previous.db_id == request.db_id
        conversation_started = perf_counter()
        conversation_usage = TokenUsage()
        interpretation: TurnInterpretation | None = None
        if previous is not None and requires_model_interpretation(
            request.message,
            has_matching_state,
            request.feedback_category,
        ):
            try:
                interpretation, conversation_usage = await interpret_turn_detailed(
                    turn_completers[request.provider],
                    provider=request.provider,
                    model=request.model,
                    message=request.message,
                    previous=previous,
                )
            except (ModelError, ValueError):
                fallback_operation = classify_operation(
                    request.message,
                    has_matching_state,
                    request.feedback_category,
                )
                interpretation = TurnInterpretation(
                    operation=fallback_operation,
                    depends_on_previous=fallback_operation not in {"NEW_QUERY", "RESET_CONTEXT"},
                    resolved_instruction=request.message,
                    correction_type=request.feedback_category,
                    target=intent_target(request.message, fallback_operation),
                    confidence=0.0,
                    source="fallback",
                    provider=request.provider,
                    model=request.model,
                )
        if interpretation is None:
            rule_operation = classify_operation(
                request.message,
                has_matching_state,
                request.feedback_category,
            )
            interpretation = TurnInterpretation(
                operation=rule_operation,
                depends_on_previous=rule_operation not in {"NEW_QUERY", "RESET_CONTEXT"},
                resolved_instruction=request.message,
                correction_type=request.feedback_category,
                target=intent_target(request.message, rule_operation),
                confidence=1.0,
                source="rules",
            )
        conversation_ms = round((perf_counter() - conversation_started) * 1_000)
        conversation_ledger = (
            [
                CallLedgerEntry(
                    component="conversation_interpretation",
                    kind="MODEL",
                    status="SUCCEEDED",
                    provider=request.provider,
                    requested_model=request.model,
                    effective_model=request.model,
                    input_tokens=conversation_usage.input_tokens,
                    output_tokens=conversation_usage.output_tokens,
                    latency_ms=conversation_ms,
                )
            ]
            if interpretation.source == "model"
            else []
        )
        if interpretation.operation in STATE_REQUIRED_OPERATIONS and not has_matching_state:
            clarification = _missing_state_message(interpretation.operation)
            return ChatResponse(
                session_id=request.session_id,
                operation=interpretation.operation,
                resolved_question=request.message,
                conversation_interpretation=interpretation,
                state=previous,
                message=clarification,
                clarification_required=True,
                clarification_question=clarification,
                token_usage=conversation_usage,
                call_ledger=conversation_ledger,
                timings_ms={
                    "conversation_interpretation": conversation_ms,
                    "total": round((perf_counter() - started) * 1_000),
                },
            )
        if interpretation.source in {"model", "fallback"} and interpretation.confidence < 0.65:
            clarification = (
                "Should I modify the previous query, or treat your message as a new question?"
            )
            return ChatResponse(
                session_id=request.session_id,
                operation=interpretation.operation,
                resolved_question=previous.resolved_question if previous else request.message,
                conversation_interpretation=interpretation,
                state=previous,
                generation=None,
                message=clarification,
                clarification_required=True,
                clarification_question=clarification,
                token_usage=conversation_usage,
                call_ledger=conversation_ledger,
                timings_ms={
                    "conversation_interpretation": conversation_ms,
                    "total": round((perf_counter() - started) * 1_000),
                },
            )
        operation, pending = resolve_turn(
            request.session_id,
            request.db_id,
            request.message,
            previous,
            request.feedback_category,
            interpretation,
        )
        if operation == "RESET_CONTEXT":
            conversations.reset(request.session_id)
            return ChatResponse(
                session_id=request.session_id,
                operation="RESET_CONTEXT",
                resolved_question="",
                conversation_interpretation=interpretation,
                message="Conversation context was reset.",
                token_usage=conversation_usage,
                call_ledger=conversation_ledger,
                timings_ms={
                    "conversation_interpretation": conversation_ms,
                    "total": round((perf_counter() - started) * 1_000),
                },
            )
        assert pending is not None
        if operation in EXPLANATION_OPERATIONS:
            explanation_started = perf_counter()
            explanation_usage = TokenUsage()
            explanation_model_succeeded = False
            try:
                explanation, explanation_usage = await explain_turn_detailed(
                    turn_completers[request.provider],
                    model=request.model,
                    operation=operation,
                    client_message=request.message,
                    state=pending,
                    dialect=("sqlite" if request.db_id in sqlite.list_ids() else "postgres"),
                )
                explanation_model_succeeded = True
            except (ModelError, ValueError):
                explanation = deterministic_explanation(
                    operation,
                    pending,
                    "sqlite" if request.db_id in sqlite.list_ids() else "postgres",
                )
            return ChatResponse(
                session_id=request.session_id,
                operation=operation,
                resolved_question=pending.resolved_question,
                conversation_interpretation=interpretation,
                state=pending,
                message="Explained the previous query without generating or changing SQL.",
                explanation=explanation,
                provenance=[
                    "accepted SQL",
                    "SQLGlot-extracted SQL facts",
                    "verified schema and relationship context",
                    "recorded result or failure metadata",
                ],
                token_usage=_add_usage(conversation_usage, explanation_usage),
                call_ledger=[
                    *conversation_ledger,
                    CallLedgerEntry(
                        component=(
                            "explanation" if explanation_model_succeeded else "explanation_fallback"
                        ),
                        kind="MODEL" if explanation_model_succeeded else "TOOL",
                        status="SUCCEEDED",
                        provider=request.provider if explanation_model_succeeded else None,
                        requested_model=request.model if explanation_model_succeeded else None,
                        effective_model=request.model if explanation_model_succeeded else None,
                        input_tokens=explanation_usage.input_tokens,
                        output_tokens=explanation_usage.output_tokens,
                        latency_ms=round((perf_counter() - explanation_started) * 1_000),
                    ),
                ],
                timings_ms={
                    "conversation_interpretation": conversation_ms,
                    "explanation": round((perf_counter() - explanation_started) * 1_000),
                    "total": round((perf_counter() - started) * 1_000),
                },
            )
        execution = await question_service.execute_question(
            question=pending.resolved_question,
            db_id=request.db_id,
            evidence=request.evidence,
            provider=request.provider,
            model=request.model,
            context_mode=request.context_mode,
            context_provider=request.context_provider,
            context_model=request.context_model,
            execute=request.execute,
            max_rows=request.max_rows,
            semantic_contract=(
                previous.semantic_contract
                if operation == "OPTIMIZE" and previous and previous.semantic_contract
                else None
            ),
            previous_intent=(
                previous.semantic_contract
                if operation != "NEW_QUERY" and previous and previous.semantic_contract
                else None
            ),
            previous_sql=(
                previous.last_sql
                if operation in {"OPTIMIZE", "CHECK_CORRECTNESS"} and previous
                else None
            ),
            previous_approved_tables=(
                previous.approved_tables
                if operation in {"OPTIMIZE", "CHECK_CORRECTNESS"} and previous
                else None
            ),
            optimization_required=operation == "OPTIMIZE",
            correctness_review=operation == "CHECK_CORRECTNESS",
            progress=lambda stage: set_session_stage(request.session_id, stage),
        )
        generated = execution.generated
        proposed_tables = execution.approved_tables
        planner_usage = execution.planner_usage
        routing_ms = execution.routing_ms
        planning_ms = execution.planning_ms
        generation_ms = execution.generation_ms
        response_state: ConversationState | None
        human_review: HumanReviewRequest | None = None
        if generated.accepted:
            pending = pending.model_copy(
                update={
                    "semantic_contract": generated.semantic_contract,
                    "approved_tables": proposed_tables,
                    "last_sql": generated.sql,
                    "last_columns": generated.columns,
                    "last_row_count": generated.row_count,
                    "last_truncated": generated.truncated,
                    "last_model_context": generated.model_context or {},
                    "last_failure": None,
                    "last_failed_sql": None,
                }
            )
            if operation == "CHECK_CORRECTNESS" or (
                generated.recovery and generated.recovery.agent_action == "ESCALATE"
            ):
                # A review candidate is evidence, not accepted conversation state. Keep the
                # prior SQL authoritative unless a later explicit correction replaces it.
                response_state = previous
            else:
                try:
                    response_state = conversations.put(
                        pending,
                        expected_version=previous.version if previous is not None else None,
                    )
                except ConversationConflict as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                pending = response_state
            message = _conversational_answer(operation, generated)
            if generated.recovery and generated.recovery.mode in {
                "ZERO_RESULT",
                "NULL_RESULT",
            }:
                message = generated.recovery.diagnosis_summary or message
                if generated.recovery.agent_action == "ESCALATE":
                    human_review = HumanReviewRequest(
                        reason=message,
                        question="The result requires review before it is treated as accepted.",
                        options=[],
                        evidence=generated.recovery.evidence,
                    )
            if operation == "CHECK_CORRECTNESS" and previous and previous.last_sql:
                database = sqlite if request.db_id in sqlite.list_ids() else postgres
                equivalent = False
                if database is not None:
                    try:
                        old_columns, old_rows, old_truncated = database.execute(
                            request.db_id, previous.last_sql, max_rows=request.max_rows
                        )
                        equivalent = (
                            not old_truncated
                            and not generated.truncated
                            and old_columns == generated.columns
                            and old_rows == generated.rows
                        )
                    except Exception:  # pragma: no cover - review must not break chat
                        equivalent = False
                if equivalent:
                    message = (
                        "An independently generated candidate returned the same bounded result. "
                        "This increases confidence but is not formal proof of business correctness."
                    )
                else:
                    human_review = HumanReviewRequest(
                        reason=(
                            "The independent correctness candidate did not reproduce the previous "
                            "bounded output exactly."
                        ),
                        question="Which interpretation should be trusted before replacing the SQL?",
                        options=[],
                        evidence=generated.recovery.evidence if generated.recovery else [],
                    )
        else:
            message = _failure_message(generated)
            failure_state = (previous or pending).model_copy(
                update={
                    "last_failure": message,
                    "last_failed_sql": generated.attempts[-1].sql if generated.attempts else None,
                }
            )
            try:
                response_state = conversations.put(
                    failure_state,
                    expected_version=previous.version if previous is not None else None,
                )
            except ConversationConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if generated.termination_reason != "model_error":
                recovery = generated.recovery
                human_review = HumanReviewRequest(
                    reason=(
                        recovery.diagnosis_summary
                        if recovery and recovery.diagnosis_summary
                        else "Automated recovery could not establish a safe correction."
                    ),
                    question="Please clarify the intended request in the chat.",
                    options=[],
                    evidence=recovery.evidence if recovery else [],
                )
        return ChatResponse(
            session_id=request.session_id,
            operation=operation,
            resolved_question=pending.resolved_question,
            conversation_interpretation=interpretation,
            state=response_state,
            generation=generated,
            message=message,
            human_review=human_review,
            provenance=[
                "current question and trusted evidence",
                "session-scoped conversation contract",
                "business glossary",
                "live schema and column profiles",
                "SQLGlot read-only safety validation",
            ],
            token_usage=_add_usage(
                conversation_usage,
                _add_usage(planner_usage, generated.token_usage),
            ),
            call_ledger=[*conversation_ledger, *generated.telemetry.call_ledger],
            timings_ms={
                "conversation_interpretation": conversation_ms,
                "routing": routing_ms,
                "planning": planning_ms,
                "generation_validation_execution": generation_ms,
                "total": round((perf_counter() - started) * 1_000),
            },
        )

    async def run_chat_job(job_id: str, request: ChatRequest) -> None:
        job = chat_jobs[job_id]
        job["status"] = "running"
        job["stage"] = "conversation"
        try:
            response = await chat(request)
            job.update(
                status="completed",
                stage="completed",
                response=response.model_dump(mode="json"),
                finished_at=perf_counter(),
            )
        except asyncio.CancelledError:
            job.update(status="cancelled", stage="cancelled", finished_at=perf_counter())
            raise
        except Exception as exc:  # pragma: no cover - defensive job boundary
            job.update(status="failed", stage="failed", error=str(exc), finished_at=perf_counter())

    @app.post("/api/chat/jobs", status_code=202)
    async def start_chat_job(request: ChatRequest) -> dict[str, str]:
        cleanup_jobs()
        job_id = uuid4().hex
        chat_jobs[job_id] = {
            "job_id": job_id,
            "session_id": request.session_id,
            "status": "queued",
            "stage": "queued",
            "started_at": perf_counter(),
            "response": None,
            "error": None,
        }
        chat_tasks[job_id] = asyncio.create_task(run_chat_job(job_id, request))
        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/chat/jobs/{job_id}")
    async def get_chat_job(job_id: str) -> dict[str, Any]:
        cleanup_jobs()
        job = chat_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Chat job was not found.")
        return {
            **job,
            "elapsed_ms": round((perf_counter() - float(job["started_at"])) * 1_000),
        }

    @app.delete("/api/chat/jobs/{job_id}")
    async def cancel_chat_job(job_id: str) -> dict[str, str]:
        cleanup_jobs()
        job = chat_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Chat job was not found.")
        if job["status"] in {"completed", "failed", "cancelled"}:
            return {"job_id": job_id, "status": str(job["status"])}
        job.update(status="cancelled", stage="cancelled")
        task = chat_tasks.get(job_id)
        if task is not None:
            task.cancel()
        return {"job_id": job_id, "status": "cancelled"}

    return app


def _add_usage(first: TokenUsage, second: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=_add_known(first.input_tokens, second.input_tokens),
        output_tokens=_add_known(first.output_tokens, second.output_tokens),
        cache_read_tokens=_add_known(first.cache_read_tokens, second.cache_read_tokens),
        cache_creation_tokens=_add_known(first.cache_creation_tokens, second.cache_creation_tokens),
    )


def _add_known(first: int | None, second: int | None) -> int | None:
    if first is None and second is None:
        return None
    return (first or 0) + (second or 0)


def _missing_state_message(operation: str) -> str:
    if operation == "OPTIMIZE":
        return "I do not have a previous accepted query to optimize. Run a query first."
    if operation == "EXPLAIN_FAILURE":
        return "I do not have a recorded query failure to explain in this conversation."
    if operation in EXPLANATION_OPERATIONS:
        return "I do not have a previous query or result to explain. Run a query first."
    if operation == "CORRECTION":
        return "I do not have a previous request to correct. Ask the complete question first."
    return "Should I treat this as a new analytical question?"


def _conversational_answer(operation: str, generated: GenerateResponse) -> str:
    if operation == "OPTIMIZE":
        if generated.optimization and generated.optimization.status == "optimized":
            prefix = "I verified that the replacement is equivalent and measurably faster."
        elif generated.optimization and generated.optimization.status == "equivalent_not_faster":
            prefix = "The rewrite was equivalent but not faster, so I retained the original query."
        else:
            prefix = "No optimization passed the acceptance gate; I retained the previous query."
    elif operation == "CHECK_CORRECTNESS":
        prefix = "I ran an evidence-based correctness review and executed an independent candidate."
    elif operation == "CORRECTION":
        prefix = "I applied your correction and reran the query."
    elif operation == "NEW_QUERY":
        prefix = (
            "The SQL passed read-only safety checks and executed successfully; "
            "execution does not prove business correctness."
        )
    else:
        prefix = "I applied your follow-up to the previous request and reran the query."
    if len(generated.columns) == 1 and len(generated.rows) == 1:
        value = generated.rows[0][0]
        rendered = "NULL" if value is None else str(value)
        return f"{prefix} {generated.columns[0]}: {rendered}."
    return f"{prefix} I found {generated.row_count} result rows."


def _failure_message(generated: GenerateResponse) -> str:
    """Explain a failed turn, separating an unreachable model from rejected SQL."""
    if generated.termination_reason == "model_error":
        detail = generated.model_error or "the model returned no usable response"
        if detail.startswith("Model 2A"):
            return detail
        return f"The {generated.model} model could not be reached: {detail}"
    if generated.termination_reason == "context_error":
        return generated.model_error or "Required database context could not be verified."
    return "Query failed; previous conversation state was preserved."


app = create_app()
