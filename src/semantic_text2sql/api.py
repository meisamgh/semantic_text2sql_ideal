"""FastAPI application for interactive generation and SQL checking."""

from __future__ import annotations

import asyncio
import logging
import os
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, perf_counter, time
from typing import Any, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from sqlglot import parse_one
from starlette.middleware.gzip import GZipMiddleware

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.ai_dashboard_designer import DashboardRevisions, design_dashboard
from semantic_text2sql.analytics_planner import (
    PlannerDecisionRecord,
    metrics_ledger,
    plan_analytics,
    plan_fields_known,
    plan_needs_refinement,
    planner_enabled,
    planner_llm_enabled,
    refine_plan_with_model,
)
from semantic_text2sql.conversation import (
    ANALYTICAL_CORRECTION_OPERATIONS,
    EXPLANATION_OPERATIONS,
    STATE_REQUIRED_OPERATIONS,
    ConversationConflict,
    ConversationStore,
    classify_operation,
    intent_target,
    interpret_turn_detailed,
    requires_model_interpretation,
    resolve_semantic_contract,
    resolve_turn,
)
from semantic_text2sql.conversational_agent import ConversationalAnalyticsAgent
from semantic_text2sql.conversational_bridge import ConversationalAnalyticsBridge
from semantic_text2sql.dashboard_service import (
    DashboardError,
    DashboardRequest,
    DashboardSpec,
    build_dashboard,
)
from semantic_text2sql.dashboard_store import DashboardStore, SavedDashboard
from semantic_text2sql.dashboard_studio import studio_router
from semantic_text2sql.data_requirement_validator import validate_reusability
from semantic_text2sql.database import DatabaseError, DatabaseRegistry
from semantic_text2sql.explanation import deterministic_explanation, explain_turn_detailed
from semantic_text2sql.external_connections import external_connections_from_environment
from semantic_text2sql.filter_discovery import (
    _SENSITIVE_TYPES as _FILTER_FORBIDDEN_SEMANTIC_TYPES,
)
from semantic_text2sql.filter_discovery import (
    _UNSAFE_NAME as _FORBIDDEN_NAME_PATTERN,
)
from semantic_text2sql.filter_discovery import discover_filters
from semantic_text2sql.glossary import GlossaryStore, GlossaryTerm
from semantic_text2sql.historical import HistoricalQueryStore
from semantic_text2sql.hybrid_retrieval import (
    FastEmbedEncoder,
    HybridSchemaRetriever,
    LightGBMSchemaReranker,
)
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
    RoutingSQLModel,
    TrueSotaSQLModel,
)
from semantic_text2sql.models import (
    DEFAULT_MODEL,
    Attempt,
    CallLedgerEntry,
    ChatRequest,
    ChatResponse,
    CheckRequest,
    CheckResponse,
    ConversationState,
    DashboardTarget,
    DatabaseOption,
    FilterValuesResponse,
    GenerateResponse,
    HumanReviewRequest,
    ModelOption,
    ModelProvider,
    PresentationState,
    ResultPresentation,
    StrictModel,
    TokenUsage,
    TurnInterpretation,
    ValidationResult,
)
from semantic_text2sql.postgres import PostgresRegistry, postgres_databases_from_environment
from semantic_text2sql.presentation import present_result
from semantic_text2sql.profiling import ProfileStore
from semantic_text2sql.provider_registry import configured_model_options, supports_model
from semantic_text2sql.recovery_events import recovery_events
from semantic_text2sql.remote_databases import RemoteDatabaseRegistry
from semantic_text2sql.result_cache import RecentResult
from semantic_text2sql.run_history import RunHistoryStore, RunRecord, build_run_record
from semantic_text2sql.runtime import (
    RequestBudget,
    RequestBudgetExceeded,
    active_request_budget,
)
from semantic_text2sql.semantic_catalog import SemanticCatalogStore, compose_document
from semantic_text2sql.semantic_catalog_validation import (
    postgres_runner,
    sqlite_runner,
    verify_catalog,
)
from semantic_text2sql.service import TextToSQLService
from semantic_text2sql.sql_diffs import (
    SqlDiffRequest,
    SqlDiffResponse,
    sql_changelog,
)
from semantic_text2sql.strategy import route_question
from semantic_text2sql.visualization import (
    VisualizationError,
    parse_intent,
    plan_chart,
    sql_output_requirement,
)
from semantic_text2sql.visualization_recommender import (
    materialize_recommendation,
    merge_recommendations,
    recommend_charts,
    refine_charts_with_model,
    viz_recommender_llm_enabled,
)
from semantic_text2sql.voice import (
    MAX_AUDIO_BYTES,
    MAX_AUDIO_SECONDS,
    GroqVoiceTranscriber,
    QwenVoiceTranscriber,
    TranscriptionResult,
    VoiceInputError,
    VoiceProviderError,
)

logger = logging.getLogger(__name__)


class ResultPageRequest(StrictModel):
    session_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    db_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    result_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    offset: int = Field(ge=0, le=9_900)
    page_size: int = Field(default=20, ge=1, le=100)


def _planner_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def create_app(
    agent: TextToSQLAgent | None = None,
    conversation_completers: dict[str, Any] | None = None,
    voice_transcriber: QwenVoiceTranscriber | GroqVoiceTranscriber | None = None,
    run_history: RunHistoryStore | None = None,
) -> FastAPI:
    enforce_model_catalog = agent is None
    if voice_transcriber is not None:
        voice = voice_transcriber
    else:
        qwen_voice = QwenVoiceTranscriber.from_environment()
        voice = (
            qwen_voice
            if qwen_voice.unavailable_reason() is None
            else GroqVoiceTranscriber.from_environment()
        )
    runs = run_history or RunHistoryStore.from_environment()
    postgres_databases = postgres_databases_from_environment()
    external_connections = external_connections_from_environment()
    postgres = (
        RemoteDatabaseRegistry(postgres_databases, external_connections)
        if external_connections
        else PostgresRegistry(postgres_databases) if postgres_databases else None
    )
    sqlite = DatabaseRegistry(Path(os.environ.get("TEXT2SQL_DATABASE_ROOT", "data")))
    if runs.path is not None and any(
        sqlite.resolve(identity).resolve() == runs.path.resolve() for identity in sqlite.list_ids()
    ):
        raise ValueError("History/dashboard storage must not be an exposed business database.")
    saved_dashboards = DashboardStore(runs.path) if runs.path is not None else None
    if set(sqlite.list_ids()) & set(external_connections):
        raise ValueError("Database IDs must be unique across SQLite and remote connections.")
    profiles = ProfileStore(Path(os.environ.get("TEXT2SQL_PROFILE_ROOT", "profiles")))
    glossaries = GlossaryStore(
        Path(os.environ.get("TEXT2SQL_GLOSSARY_ROOT", "data/business_glossaries"))
    )
    semantic_catalogs = SemanticCatalogStore(
        Path(os.environ.get("TEXT2SQL_SEMANTIC_CATALOG_ROOT", "data/semantic_catalogs"))
    )
    history = HistoricalQueryStore(
        Path(
            os.environ.get("TEXT2SQL_HISTORY_PATH", "benchmarks/data/bird_history_seed42_400.json")
        )
    )
    agentrouter_key = os.environ.get("AGENTROUTER_API_KEY")
    agentrouter_base_url = os.environ.get("AGENTROUTER_BASE_URL", "https://agentrouter.org")
    agentrouter_timeout = float(os.environ.get("AGENTROUTER_TIMEOUT_SECONDS", "180"))
    agentrouter = AgentRouterModel(
        AgentRouterClaudeModel(
            agentrouter_key,
            agentrouter_base_url,
            agentrouter_timeout,
        ),
        AgentRouterCodexModel(
            agentrouter_key,
            agentrouter_base_url,
            agentrouter_timeout,
        ),
    )
    groq = GroqSQLModel(
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    )
    justdowork = JustDoWorkSQLModel(
        os.environ.get("JUSTDOWORK_API_KEY"),
        os.environ.get("JUSTDOWORK_BASE_URL"),
        float(os.environ.get("JUSTDOWORK_TIMEOUT_SECONDS", "120")),
    )
    codecraft = CodeCraftSQLModel(
        os.environ.get("CODECRAFT_API_KEY"),
        os.environ.get("CODECRAFT_BASE_URL", "https://codecraftapi.com/v1"),
        float(os.environ.get("CODECRAFT_TIMEOUT_SECONDS", "120")),
    )
    true_sota = TrueSotaSQLModel(
        os.environ.get("TRUE_SOTA_API_KEY"),
        os.environ.get("TRUE_SOTA_BASE_URL", "https://true-sota.com"),
        float(os.environ.get("TRUE_SOTA_TIMEOUT_SECONDS", "180")),
    )
    openrouter = OpenRouterSQLModel(
        os.environ.get("OPENROUTER_API_KEY"),
        os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "180")),
    )
    active_agent = agent or TextToSQLAgent(
        sqlite,
        RoutingSQLModel(agentrouter, groq, justdowork, codecraft, true_sota, openrouter),
        postgres,
        profiles,
    )
    turn_completers = conversation_completers or {
        "agentrouter": agentrouter,
        "groq": groq,
        "justdowork": justdowork,
        "codecraft": codecraft,
        "true_sota": true_sota,
        "openrouter": openrouter,
        "ollama": OllamaCompleter(
            os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180")),
        ),
    }
    def conversational_agent_enabled() -> bool:
        return (
            os.environ.get("TEXT2SQL_CONVERSATIONAL_AGENT_ENABLED", "false").casefold()
            == "true"
        )
    app = FastAPI(
        title="Semantic Text-to-SQL v5",
        version="0.5.0",
        description=(
            "Bounded context selection, compact grounding, SQL-only generation, "
            "safety-first validation, and read-only execution."
        ),
    )
    if isinstance(postgres, RemoteDatabaseRegistry):
        app.router.add_event_handler("shutdown", postgres.close)

    @app.exception_handler(DatabaseError)
    async def database_error_response(request: Request, exc: DatabaseError) -> JSONResponse:
        del request
        safe_messages = {
            "DATABASE_DRIVER_MISSING": "Install the selected database extra and its system driver.",
            "READ_ONLY_ROLE_REQUIRED": "Provision and confirm a read-only database role first.",
            "DATABASE_CONNECTION_FAILED": "Check the local connection settings and credentials.",
            "DATABASE_INSPECTION_FAILED": "Check schema access and table permissions.",
            "CONNECTION_NOT_CONFIGURED": "Set the configured connection environment variable.",
        }
        return JSONResponse(status_code=503, content={
            "code": exc.code,
            "detail": safe_messages.get(exc.code, "The database operation could not be completed."),
        })
    if os.environ.get("TEXT2SQL_GZIP_ENABLED", "true").casefold() == "true":
        app.add_middleware(
            GZipMiddleware,
            minimum_size=int(os.environ.get("TEXT2SQL_GZIP_MIN_BYTES", "2048")),
            compresslevel=int(os.environ.get("TEXT2SQL_GZIP_LEVEL", "5")),
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
    dense_encoder = FastEmbedEncoder(
        os.environ.get("TEXT2SQL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    async def preload_encoder() -> None:
        if os.environ.get("TEXT2SQL_EMBEDDING_PRELOAD", "true").casefold() != "true":
            return
        timeout = float(os.environ.get("TEXT2SQL_EMBEDDING_STARTUP_TIMEOUT_SECONDS", "15"))
        try:
            await asyncio.wait_for(asyncio.to_thread(dense_encoder.initialize), timeout=timeout)
        except (TimeoutError, RuntimeError, ImportError) as exc:
            dense_encoder.disable(f"Startup initialization failed: {exc}")
            logger.warning("dense_encoder_disabled=%s", exc)

    app.router.add_event_handler("startup", preload_encoder)
    hybrid_retriever = HybridSchemaRetriever(
        dense_encoder,
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
    # Complete bounded results are kept only for the active process. Persisted
    # history retains a smaller preview and is reused only when it is complete.
    recent_results: dict[tuple[str, str], RecentResult] = {}
    # Display-only index of the same typed snapshots, never agent memory.
    # Allows reopening earlier chart turns after the active result changes.
    chart_snapshots: dict[tuple[str, str, str], RecentResult] = {}
    branch_context: ContextVar[
        tuple[str, ConversationState, RecentResult | None, int | None] | None
    ] = ContextVar("queryflow_branch_context", default=None)
    provisional_context: ContextVar[tuple[str, ConversationState] | None] = ContextVar(
        "queryflow_provisional_context", default=None
    )
    history_anchor: ContextVar[str | None] = ContextVar("queryflow_history_anchor", default=None)
    studio_context: ContextVar[SavedDashboard | None] = ContextVar(
        "queryflow_dashboard_studio", default=None
    )

    def accepted_state(session_id: str) -> ConversationState | None:
        provisional = provisional_context.get()
        if provisional and provisional[0] == session_id:
            return provisional[1]
        branch = branch_context.get()
        return branch[1] if branch and branch[0] == session_id else conversations.get(session_id)

    def accepted_result(session_id: str, db_id: str) -> RecentResult | None:
        branch = branch_context.get()
        if branch and branch[0] == session_id:
            return branch[2] if branch[1].db_id == db_id else None
        key = (session_id, db_id)
        cached = recent_results.get(key)
        if cached is not None:
            return cached
        # The bounded in-process cache is lost on restart. Reuse history only
        # when it contains every executed row for the currently accepted SQL;
        # a preview or a run from another branch must never stand in for data.
        state = accepted_state(session_id)
        if state is None or state.db_id != db_id or not state.last_sql:
            return None
        for record in runs.list_sessions(session_id):
            if record.db_id != db_id or record.sql != state.last_sql:
                continue
            if (
                record.state_snapshot is None
                or record.state_snapshot.resolved_question != state.resolved_question
            ):
                continue
            restored = result_from_history(record)
            if restored is not None:
                recent_results[key] = restored
                return restored
            break
        return None

    dashboard_revisions = DashboardRevisions()
    dashboard_bundles: dict[tuple[str, str, str], dict[str, RecentResult]] = {}

    def dashboard_source(session: str, db: str, identity: str) -> RecentResult | None:
        current = recent_results.get((session, db))
        if current is not None and current.result_id == identity:
            return current
        return next((snapshot for (owner, database, _), snapshot
                     in reversed(chart_snapshots.items())
                     if owner == session and database == db
                     and snapshot.result_id == identity), None)

    @app.post("/api/dashboards", response_model=DashboardSpec)
    async def create_dashboard(request: DashboardRequest, response: Response) -> DashboardSpec:
        response.headers["Cache-Control"] = "no-store"
        configured = request.db_id in sqlite.list_ids() or (
            postgres is not None and request.db_id in postgres.configured_ids()
        )
        if not configured:
            raise HTTPException(404, "Database is unavailable.")

        try:
            source = dashboard_source(request.session_id, request.db_id, request.result_id)
            previous = dashboard_revisions.get(request.session_id, request.db_id, request.result_id)
            expected = previous.revision_id if previous else None
            if request.design_mode == "ai":
                if source is None:
                    raise DashboardError("The accepted result is unavailable.", 404)
                provider = os.environ.get("TEXT2SQL_DASHBOARD_PROVIDER", "")
                model = os.environ.get("TEXT2SQL_DASHBOARD_MODEL", "")
                outcome = await design_dashboard(
                    source=source, request=request, completer=turn_completers.get(provider),
                    provider=provider, model=model, previous=previous,
                    budget=active_request_budget() or RequestBudget(
                        timeout_seconds=15, max_model_calls=1, max_database_calls=0
                    ),
                )
                if outcome.spec is None:
                    raise HTTPException(409, {"code": outcome.status, "message": outcome.reason})
                spec = outcome.spec
            else:
                spec = await asyncio.to_thread(build_dashboard, request, lambda *_: source)
            return dashboard_revisions.commit(request.session_id, spec, expected)
        except DashboardError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.post("/api/dashboards/undo", response_model=DashboardSpec)
    async def undo_dashboard(request: DashboardRequest, response: Response) -> DashboardSpec:
        response.headers["Cache-Control"] = "no-store"
        source = dashboard_source(request.session_id, request.db_id, request.result_id)
        configured = request.db_id in sqlite.list_ids() or (
            postgres is not None and request.db_id in postgres.configured_ids()
        )
        if source is None or not configured:
            raise HTTPException(404, "The accepted result is unavailable.")
        if (source.execution_status != "ACCEPTED" or source.truncated
                or source.row_count != len(source.rows)):
            raise HTTPException(409, "Undo requires the complete accepted source result.")
        try:
            return dashboard_revisions.undo(request.session_id, request.db_id, request.result_id)
        except DashboardError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.post("/api/results/page")
    async def result_page(request: ResultPageRequest) -> dict[str, Any]:
        """Page an accepted result without accepting SQL from the browser.

        The first 500 rows come from the exact in-process accepted snapshot. A
        later page re-executes that SQL only when it has an outer ORDER BY and
        the saved prefix still matches, avoiding silent unstable page changes.
        """
        source = accepted_result(request.session_id, request.db_id)
        if source is None or source.result_id != request.result_id:
            raise HTTPException(404, "The accepted result is unavailable or has changed.")
        end = request.offset + request.page_size
        if end <= len(source.rows):
            page_rows = source.rows[request.offset:end]
            has_more = source.truncated or end < len(source.rows)
        else:
            if not source.truncated:
                page_rows = source.rows[request.offset:end]
                has_more = False
            else:
                try:
                    statement = parse_one(source.sql, read=source.dialect)
                except Exception as exc:
                    raise HTTPException(409, "Saved SQL cannot be paged safely.") from exc
                if statement is None or statement.args.get("order") is None:
                    raise HTTPException(
                        409, "More rows require an explicit outer ORDER BY for stable paging."
                    )
                if end > 10_000:
                    raise HTTPException(422, "Paging is limited to the first 10,000 rows.")
                database = (
                    sqlite if source.dialect == "sqlite" and request.db_id in sqlite.list_ids()
                    else postgres if postgres is not None
                    and request.db_id in postgres.configured_ids() else None
                )
                if database is None:
                    raise HTTPException(404, "The selected database is unavailable.")
                try:
                    columns, fetched, has_more = await asyncio.to_thread(
                        database.execute, request.db_id, source.sql,
                        max_rows=end, timeout_seconds=5.0,
                    )
                except DatabaseError as exc:
                    raise HTTPException(409, "The saved result could not be paged.") from exc
                if columns != source.columns or fetched[:len(source.rows)] != source.rows:
                    raise HTTPException(
                        409, "The result changed since the first page. Run the question again."
                    )
                page_rows = fetched[request.offset:end]
        limit_reached = end >= 10_000 and has_more
        return {
            "result_id": source.result_id,
            "columns": source.columns,
            "rows": page_rows,
            "offset": request.offset,
            "displayed_count": len(page_rows),
            "returned_count": source.row_count,
            "total_available": None if source.truncated else source.row_count,
            "has_more": has_more and not limit_reached,
            "limit_reached": limit_reached,
        }

    def result_from_history(record: RunRecord) -> RecentResult | None:
        if not record.accepted or record.preview_truncated or record.truncated:
            return None
        if record.row_count != len(record.rows_preview) or not record.sql:
            return None
        if not record.state_snapshot or record.state_snapshot.last_sql != record.sql:
            return None
        try:
            payload = dict(record.response_payload)
            if isinstance(payload.get("generation"), dict):
                payload["generation"] = {
                    key: value for key, value in payload["generation"].items()
                    if key != "formatted_sql"
                }
            response = ChatResponse.model_validate(payload)
            if not response.result_id:
                response = response.model_copy(update={"result_id": record.run_id})
            return RecentResult.from_response(response)
        except (ValueError, TypeError):
            logger.warning("history_result_restore_unavailable run_id=%s", record.run_id)
            return None

    def restore_cached_result(record: RunRecord) -> None:
        """Keep the in-process result reference aligned with restored history."""
        key = (record.session_id, record.db_id)
        recent_results.pop(key, None)
        restored = result_from_history(record)
        if restored is not None:
            recent_results[key] = restored
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
                if job.get("stage") != stage:
                    job.setdefault("events", []).append(
                        {
                            "stage": stage,
                            "elapsed_ms": round((perf_counter() - float(job["started_at"])) * 1000),
                        }
                    )
                job["stage"] = stage

    chat_tasks: dict[str, asyncio.Task[None]] = {}

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "online"}

    @app.get("/", include_in_schema=False)
    async def web_application() -> FileResponse:
        return FileResponse(web_root / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/models", response_model=list[ModelOption])
    async def models(response: Response) -> list[ModelOption]:
        response.headers["Cache-Control"] = "no-store"
        return configured_model_options()

    @app.get("/api/ui-config")
    async def ui_config(response: Response) -> dict[str, str | None]:
        response.headers["Cache-Control"] = "no-store"
        sql_provider = os.environ.get("TEXT2SQL_SQL_PROVIDER") or "agentrouter"
        sql_model = os.environ.get("TEXT2SQL_SQL_MODEL") or DEFAULT_MODEL
        context_mode = os.environ.get("TEXT2SQL_CONTEXT_MODE") or "retrieval"
        return {
            "sql_provider": sql_provider,
            "sql_model": sql_model,
            "context_mode": context_mode,
            "context_provider": (
                os.environ.get("TEXT2SQL_CONTEXT_PROVIDER") or sql_provider
                if context_mode == "model1" else None
            ),
            "context_model": (
                os.environ.get("TEXT2SQL_CONTEXT_MODEL") or sql_model
                if context_mode == "model1" else None
            ),
            "assistant_provider": os.environ.get("TEXT2SQL_ASSISTANT_PROVIDER") or sql_provider,
            "assistant_model": os.environ.get("TEXT2SQL_ASSISTANT_MODEL") or sql_model,
            "plotting_provider": os.environ.get("TEXT2SQL_PLOTTING_PROVIDER") or sql_provider,
            "plotting_model": os.environ.get("TEXT2SQL_PLOTTING_MODEL") or sql_model,
            "default_db_id": os.environ.get("TEXT2SQL_DEFAULT_DB_ID") or "debit_card_specializing",
        }

    @app.get("/api/voice")
    async def voice_configuration(response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        reason = voice.unavailable_reason()
        return {
            "configured": reason is None,
            "provider": voice.provider,
            "model": voice.model,
            "max_duration_seconds": MAX_AUDIO_SECONDS,
            "unavailable_reason": reason,
        }

    @app.post("/api/voice/transcribe", response_model=TranscriptionResult)
    async def transcribe_voice(request: Request, response: Response) -> TranscriptionResult:
        response.headers["Cache-Control"] = "no-store"
        if reason := voice.unavailable_reason():
            raise HTTPException(status_code=503, detail=reason)
        if request.headers.get("content-type", "").split(";")[0].strip() != "audio/wav":
            raise HTTPException(status_code=415, detail="Send a WAV recording from the microphone.")
        audio = bytearray()
        async for chunk in request.stream():
            if len(audio) + len(chunk) > MAX_AUDIO_BYTES:
                raise HTTPException(status_code=413, detail="Record up to 60 seconds of audio.")
            audio.extend(chunk)
        try:
            return await voice.transcribe(bytes(audio))
        except VoiceInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except VoiceProviderError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.get("/api/history")
    async def history_list(response: Response, session_id: str, limit: int = 50) -> list[Any]:
        response.headers["Cache-Control"] = "no-store"
        return [
            item.model_dump(exclude={"response_payload", "state_snapshot"})
            for item in runs.list_sessions(session_id, limit=min(limit, 200))
        ]

    @app.get("/api/history/{run_id}")
    async def history_run(response: Response, run_id: str, session_id: str) -> RunRecord:
        response.headers["Cache-Control"] = "no-store"
        record = runs.get(run_id, session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return record

    @app.post("/api/results/inspect")
    async def inspect_historical_result(
        request: ResultPageRequest, response: Response,
    ) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        current = accepted_state(request.session_id)
        configured = request.db_id in sqlite.list_ids() or (
            postgres is not None and request.db_id in postgres.configured_ids()
        )
        if not configured:
            raise HTTPException(404, "Database is unavailable.")
        if current is None or current.db_id != request.db_id:
            raise HTTPException(404, "No active analytical context for this historical result.")
        data = await asyncio.to_thread(
            runs.inspect_result, request.session_id, request.db_id, request.result_id,
            offset=request.offset, page_size=request.page_size,
        )
        if data is None:
            raise HTTPException(404, "Historical result is unavailable in this context.")
        return data

    @app.get("/api/history/{run_id}/turns", response_model=list[RunRecord])
    async def history_turns(response: Response, run_id: str, session_id: str) -> list[RunRecord]:
        response.headers["Cache-Control"] = "no-store"
        if runs.get(run_id, session_id) is None:
            raise HTTPException(status_code=404, detail="Conversation turn not found.")
        records = runs.lineage(session_id, run_id)
        configured = set(sqlite.list_ids())
        if postgres is not None:
            configured.update(postgres.configured_ids())
        restored = []
        for record in records:
            cached = chart_snapshots.get((session_id, record.db_id, record.run_id))
            payload = record.response_payload
            if (
                cached is not None and record.db_id in configured and record.accepted
                and cached.result_id == payload.get("result_id")
                and cached.chart_id == payload.get("chart_id")
                and cached.sql == record.sql and not cached.truncated
                and cached.row_count == len(cached.rows) and cached.chart_spec is not None
            ):
                payload = {
                    **payload,
                    "generation": {**(payload.get("generation") or {}), "rows": cached.rows},
                    "presentation": {
                        **(payload.get("presentation") or {}),
                        "chart_spec": cached.chart_spec.model_dump(mode="json"),
                    },
                }
                record = record.model_copy(update={"response_payload": payload})
            restored.append(record)
        return restored

    @app.post("/api/history/{run_id}/continue", response_model=ConversationState)
    async def continue_history(run_id: str, session_id: str) -> ConversationState:
        record = runs.get(run_id, session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Conversation turn not found.")
        if record.state_snapshot is None:
            raise HTTPException(status_code=409, detail="This older turn has no restorable state.")
        current = conversations.get(session_id)
        try:
            restored = conversations.put(
                record.state_snapshot,
                expected_version=current.version if current else None,
            )
            for key in [key for key in recent_results if key[0] == session_id]:
                recent_results.pop(key, None)
            restore_cached_result(record)
            return restored
        except ConversationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/sql/diff", response_model=SqlDiffResponse)
    async def sql_diff(payload: SqlDiffRequest, response: Response) -> SqlDiffResponse:
        response.headers["Cache-Control"] = "no-store"
        dialect = payload.dialect
        return sql_changelog(payload.previous_sql, payload.updated_sql, dialect=dialect)

    @app.delete("/api/history", status_code=204)
    async def history_clear(response: Response, session_id: str) -> Response:
        response.headers["Cache-Control"] = "no-store"
        runs.drop_session(session_id)
        dashboard_revisions.clear(session_id)
        for bundle_key in [key for key in dashboard_bundles if key[0] == session_id]:
            dashboard_bundles.pop(bundle_key, None)
        for snapshot_key in [key for key in chart_snapshots if key[0] == session_id]:
            chart_snapshots.pop(snapshot_key, None)
        return Response(status_code=204)

    @app.post("/api/conversations/{session_id}/reset", status_code=204)
    async def reset_conversation(response: Response, session_id: str) -> Response:
        """Start a fresh conversation without invoking any model or deleting audit history."""
        response.headers["Cache-Control"] = "no-store"
        conversations.reset(session_id)
        dashboard_revisions.clear(session_id)
        for bundle_key in [key for key in dashboard_bundles if key[0] == session_id]:
            dashboard_bundles.pop(bundle_key, None)
        await asyncio.to_thread(runs.mark_memory_reset, session_id)
        for key in [key for key in recent_results if key[0] == session_id]:
            recent_results.pop(key, None)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @app.get("/api/databases/{db_id}/glossary")
    async def database_glossary(response: Response, db_id: str) -> list[GlossaryTerm]:
        """Approved business definitions, for the contextual schema inspector."""
        response.headers["Cache-Control"] = "no-store"
        glossary = glossaries.load(db_id)
        return glossary.terms if glossary else []

    @app.get("/api/databases", response_model=list[DatabaseOption])
    async def databases() -> list[DatabaseOption]:
        options = [
            DatabaseOption(db_id=db_id, dialect="sqlite", configured=True)
            for db_id in sqlite.list_ids()
        ]
        if postgres is not None:
            options.extend(
                DatabaseOption(db_id=db_id, dialect=postgres.dialect_for(db_id), configured=True)
                for db_id in postgres.configured_ids()
            )
        elif not postgres_databases:
            options.append(
                DatabaseOption(db_id="books_postgres", dialect="postgres", configured=False)
            )
        return options

    @app.get("/api/databases/{db_id}/schema")
    async def database_schema(db_id: str) -> dict[str, Any]:
        """Browse live schema and labeled profile relationships without exposing rows."""
        if db_id in sqlite.list_ids():
            schema = await asyncio.to_thread(sqlite.inspect, db_id)
        elif postgres is not None and db_id in postgres.configured_ids():
            schema = await asyncio.to_thread(postgres.inspect, db_id)
        else:
            raise HTTPException(status_code=404, detail="Database was not found or configured.")
        live_columns = {
            (table.name, column.name) for table in schema.tables for column in table.columns
        }
        relationships = [
            {
                "from_table": item.from_table,
                "from_column": item.from_column,
                "to_table": item.to_table,
                "to_column": item.to_column,
                "source": "declared_foreign_key",
                "cardinality": None,
            }
            for item in schema.relationships
        ]
        declared = {
            (item.from_table, item.from_column, item.to_table, item.to_column)
            for item in schema.relationships
        }
        try:
            profile = profiles.load(schema.dialect, db_id)
        except (OSError, ValueError):
            profile = None
            logger.warning("Could not read schema profile for %s", db_id, exc_info=True)
        if profile is not None:
            for item in profile.relationships:
                key = (
                    item.child_table,
                    item.child_column,
                    item.parent_table,
                    item.parent_column,
                )
                if key in declared or any(
                    part not in live_columns
                    for part in (
                        (item.child_table, item.child_column),
                        (item.parent_table, item.parent_column),
                    )
                ):
                    continue
                relationships.append(
                    {
                        "from_table": item.child_table,
                        "from_column": item.child_column,
                        "to_table": item.parent_table,
                        "to_column": item.parent_column,
                        "source": "profile_inference",
                        "cardinality": item.type,
                    }
                )
        return {
            "db_id": db_id,
            "dialect": schema.dialect,
            "tables": [
                {
                    "name": table.name,
                    "columns": [
                        {"name": col.name, "type": col.data_type, "primary_key": col.primary_key}
                        for col in table.columns
                    ],
                }
                for table in schema.tables
            ],
            "relationships": relationships,
        }

    filter_values_cache: dict[tuple[str, str, str, str], tuple[float, list[str], bool]] = {}

    @app.get("/api/databases/{db_id}/filter-values", response_model=FilterValuesResponse)
    async def filter_values(
        db_id: str, table: str, column: str, search: str | None = None,
        exact: bool = False,
    ) -> FilterValuesResponse:
        """Bounded, read-only distinct values for one verified filter field.

        Authorization is fail-closed: the database must exist, the field must
        be a verified non-key, non-sensitive textual/date column of the live
        schema, and per-column semantic metadata may not mark it sensitive.
        High-cardinality fields use a bounded server-side ``search`` instead of
        presenting the first values as the complete domain. Values are cached
        for at most one minute in a bounded, process-local map.
        """
        import re as _re

        if db_id not in sqlite.list_ids() and (
            postgres is None or db_id not in postgres.configured_ids()
        ):
            raise HTTPException(status_code=404, detail="Database was not found or configured.")
        if not _re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table) or not _re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", column
        ):
            raise HTTPException(status_code=400, detail="Unsupported table or column name.")
        search_term = (search or "").strip()
        if len(search_term) > 64:
            raise HTTPException(status_code=400, detail="Search term is too long.")
        database = (
            sqlite if db_id in sqlite.list_ids() else (postgres if postgres is not None else None)
        )
        if database is None:
            raise HTTPException(status_code=404, detail="Database was not found or configured.")
        try:
            schema = await asyncio.to_thread(database.inspect, db_id)
        except DatabaseError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        table_info = next((item for item in schema.tables if item.name == table), None)
        column_info = (
            next((item for item in table_info.columns if item.name == column), None)
            if table_info is not None
            else None
        )
        if column_info is None or column_info.primary_key:
            raise HTTPException(status_code=404, detail="Unknown filter field.")
        physical = column_info.data_type.casefold().split("(", 1)[0]
        if not (
            physical.startswith(("text", "varchar", "char", "nvarchar", "nchar", "string"))
            or physical in {"date", "datetime", "timestamp"}
        ):
            raise HTTPException(status_code=400, detail="Only text or date fields support values.")
        # Authorization: verified column metadata (profiles) may retire a
        # field from value discovery even when the raw schema still lists it.
        profile = None
        try:
            profile = await asyncio.to_thread(profiles.load, schema.dialect, db_id)
        except (DatabaseError, OSError, ValueError):
            profile = None
        for item in profile.columns if profile else []:
            if item.table == table and item.column == column:
                if item.semantic_type in _FILTER_FORBIDDEN_SEMANTIC_TYPES:
                    raise HTTPException(
                        status_code=403,
                        detail="This field is not available for value discovery.",
                    )
                break
        if _re.search(_FORBIDDEN_NAME_PATTERN, (table + "." + column).casefold()):
            raise HTTPException(
                status_code=403,
                detail="This field is not available for value discovery.",
            )
        # Profile/schema changes invalidate domains even inside the cache TTL.
        version = str(profile.profiled_at) if profile else schema.model_dump_json()
        cache_key = (db_id, table, column, search_term + "\x00" + str(exact) + version)
        cached = filter_values_cache.get(cache_key)
        if cached is not None and monotonic() - cached[0] < 60:
            _, cached_values, cached_more = cached
            return FilterValuesResponse(
                db_id=db_id,
                table=table,
                column=column,
                search=search_term or None,
                values=cached_values,
                more_available=cached_more,
                cached=True,
                expires_in_seconds=max(0, int(60 - (monotonic() - cached[0]))),
                complete_domain=not search_term and not cached_more,
                ambiguous=bool(search_term and len(cached_values) > 1),
            )
        if search_term:
            # Treat the user's text literally; it is never SQL or a LIKE pattern.
            placeholder = "?" if schema.dialect == "sqlite" else "%s"
            predicate = (
                f'instr(lower(CAST("{column}" AS TEXT)), lower({placeholder})) > 0'
                if schema.dialect == "sqlite"
                else f'strpos(lower(CAST("{column}" AS TEXT)), lower({placeholder})) > 0'
            )
            if exact:
                predicate = f'lower(CAST("{column}" AS TEXT)) = lower({placeholder})'
            sql = (
                f'SELECT DISTINCT CAST("{column}" AS TEXT) FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL AND {predicate} ORDER BY 1 LIMIT 101'
            )
            parameters: tuple[str, ...] = (search_term,)
        else:
            sql = (
                f'SELECT DISTINCT CAST("{column}" AS TEXT) FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL ORDER BY 1 LIMIT 101'
            )
            parameters = ()
        try:
            if isinstance(database, RemoteDatabaseRegistry) and db_id in database.connections:
                _, rows, _ = await asyncio.to_thread(
                    database.lookup_values, db_id, table, column, search_term, exact=exact,
                )
            else:
                _, rows, _ = await asyncio.to_thread(
                    database.execute, db_id, sql, max_rows=101,
                    timeout_seconds=2.0, parameters=parameters,
                )
        except DatabaseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        values: list[str] = [str(row[0]) for row in rows if row and row[0] is not None]
        more_available = len(values) > 100
        if len(filter_values_cache) >= 256:
            filter_values_cache.clear()
        filter_values_cache[cache_key] = (monotonic(), values[:100], more_available)
        return FilterValuesResponse(
            db_id=db_id,
            table=table,
            column=column,
            search=search_term or None,
            values=values[:100],
            complete_domain=not search_term and not more_available,
            ambiguous=bool(search_term and len(values) > 1),
            more_available=more_available,
            cached=False,
        )

    @app.get("/api/analytics-metrics")
    async def analytics_metrics() -> dict[str, int]:
        """Aggregate planner decision counts for this process (no turns recorded)."""
        return metrics_ledger().summarize()

    @app.get("/api/databases/{db_id}/analytics-capabilities")
    async def analytics_capabilities(db_id: str) -> dict[str, Any]:
        """Expose a compact, read-only analytical capability summary."""
        if db_id in sqlite.list_ids():
            dialect = "sqlite"
            schema = sqlite.inspect(db_id)
        elif postgres is not None and db_id in postgres.configured_ids():
            dialect = postgres.dialect_for(db_id)
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

    @app.get("/api/databases/{db_id}/semantic-catalog")
    async def semantic_catalog(db_id: str) -> dict[str, Any]:
        """Governed semantic catalog composed with live schema and profile status."""
        if db_id in sqlite.list_ids():
            schema = await asyncio.to_thread(sqlite.inspect, db_id)
        elif postgres is not None and db_id in postgres.configured_ids():
            schema = await asyncio.to_thread(postgres.inspect, db_id)
        else:
            raise HTTPException(status_code=404, detail="Database was not found or configured.")
        try:
            profile = profiles.load(schema.dialect, db_id)
        except (OSError, ValueError):
            profile = None
            logger.warning("Could not read schema profile for %s", db_id, exc_info=True)
        document = semantic_catalogs.load(db_id)
        if document is None:
            raise HTTPException(
                status_code=404,
                detail="No semantic catalog document is registered for this database.",
            )
        return compose_document(document, schema, profile)

    @app.get("/api/databases/{db_id}/semantic-catalog/verification")
    async def semantic_catalog_verification(db_id: str) -> dict[str, Any]:
        """Deterministic evidence probes for the registered catalog document."""
        if db_id in sqlite.list_ids():
            schema = await asyncio.to_thread(sqlite.inspect, db_id)
            runner = sqlite_runner(sqlite, db_id)
        elif postgres is not None and db_id in postgres.configured_ids():
            schema = await asyncio.to_thread(postgres.inspect, db_id)
            runner = postgres_runner(postgres, db_id)
        else:
            raise HTTPException(status_code=404, detail="Database was not found or configured.")
        document = semantic_catalogs.load(db_id)
        if document is None:
            raise HTTPException(
                status_code=404,
                detail="No semantic catalog document is registered for this database.",
            )
        report = await asyncio.to_thread(verify_catalog, document, schema, runner, schema.dialect)
        return report.model_dump()

    @app.post("/api/check", response_model=CheckResponse)
    async def check(request: CheckRequest) -> CheckResponse:
        if (
            request.execute
            and os.environ.get("TEXT2SQL_ENABLE_ADMIN_CHECK_EXECUTION", "false").casefold()
            != "true"
        ):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Direct SQL execution is disabled. Use /api/chat or explicitly enable the "
                    "administrative check endpoint."
                ),
            )
        return await asyncio.to_thread(active_agent.check, request)

    async def process_chat(
        request: ChatRequest, *, persist_state: bool = True
    ) -> ChatResponse:
        # Studio invokes the same Top layer even when legacy chat compatibility
        # is enabled; never reinterpret its already-typed SQL objective.
        assistant_enabled = conversational_agent_enabled() or studio_context.get() is not None
        if enforce_model_catalog and not supports_model(request.provider, request.model):
            raise HTTPException(
                status_code=422,
                detail=f"Model {request.model!r} is not allowed for {request.provider}.",
            )
        if (
            enforce_model_catalog
            and request.context_provider
            and request.context_model
            and not supports_model(request.context_provider, request.context_model)
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Context model {request.context_model!r} is not allowed for "
                    f"{request.context_provider}."
                ),
            )
        started = perf_counter()
        analytics_usage = TokenUsage()
        analytics_ledger: list[CallLedgerEntry] = []
        visualization_usage = TokenUsage()
        governed_metric_evidence: str | None = None
        request_budget = active_request_budget() or RequestBudget(
            timeout_seconds=float(os.environ.get("TEXT2SQL_REQUEST_TIMEOUT_SECONDS", "180")),
            max_model_calls=int(os.environ.get("TEXT2SQL_REQUEST_MAX_MODEL_CALLS", "6")),
            max_database_calls=int(os.environ.get("TEXT2SQL_REQUEST_MAX_DATABASE_CALLS", "24")),
        )
        set_session_stage(request.session_id, "conversation")
        previous = accepted_state(request.session_id)
        has_matching_state = previous is not None and previous.db_id == request.db_id
        viz_intent = parse_intent(request.message, has_previous=has_matching_state)
        if request.visualization is False:
            viz_intent = viz_intent.model_copy(update={"requested": False})
        if request.visualization is True and not viz_intent.requested:
            viz_intent = viz_intent.model_copy(update={"requested": True})
        if viz_intent.requested and viz_intent.followup and previous and previous.last_sql:
            prior_response = accepted_result(request.session_id, request.db_id)
            if (
                request.parent_run_id
                and prior_response
                and (prior_response.run_id != request.parent_run_id)
            ):
                prior_response = None
            source = prior_response.generation if prior_response else None
            source_record: RunRecord | None = None
            if source is None or source.sql != previous.last_sql or source.truncated:
                candidates = (
                    [runs.get(request.parent_run_id, request.session_id)]
                    if request.parent_run_id
                    else runs.list_sessions(request.session_id)
                )
                for record in candidates:
                    if record is None:
                        continue
                    if record.db_id != request.db_id or record.sql != previous.last_sql:
                        continue
                    source_record = record
                    if record.truncated or record.row_count != len(record.rows_preview):
                        break
                    raw = record.response_payload.get("generation")
                    if raw:
                        source = GenerateResponse.model_validate({
                            key: value for key, value in raw.items()
                            if key != "formatted_sql"
                        })
                    break
            if source and source.accepted and not source.truncated:
                explanation = (
                    prior_response.presentation.summary
                    if prior_response and prior_response.presentation
                    else f"The stored result contains {source.row_count} rows."
                )
                chart_presentation = ResultPresentation(summary=explanation, source="fallback")
                # --- Result Reusability Gate: decided here, before any chart
                # planning. A renderable chart that answers a different
                # analytical request than the one asked is a wrong answer, so
                # metric identities, dimensions, grain and result completeness
                # gate every reuse of the stored result.
                analytics_started = perf_counter()
                proposal = None
                # Governed derived metrics: only definitions approved in the
                # semantic catalog are recognized; formulas are never invented.
                approved_metrics: dict[str, Any] = {}
                try:
                    catalog_document = semantic_catalogs.load(request.db_id)
                except (ValueError, OSError):
                    catalog_document = None
                if catalog_document is not None:
                    approved_metrics = {
                        entry.id: entry
                        for entry in catalog_document.metrics
                        if entry.status == "approved"
                    }
                if planner_enabled():
                    try:
                        proposal = plan_analytics(
                            request.message,
                            viz_intent=viz_intent,
                            previous=previous,
                            db_id=request.db_id,
                            approved_metrics=approved_metrics or None,
                        )
                    except Exception:  # noqa: BLE001 - planner proposals never abort a chart turn
                        proposal = None
                # Ambiguous deterministic interpretation gets one flagged,
                # validated model refinement; obvious chart mechanics never
                # reach the LLM.
                if (
                    planner_enabled()
                    and planner_llm_enabled()
                    and not assistant_enabled
                    and proposal is not None
                    and plan_needs_refinement(proposal)
                ):
                    known_fields = {name.casefold() for name in source.columns or []}
                    known_fields |= {name.casefold() for name in previous.last_columns}
                    deterministic_plan = proposal
                    model_calls_before = request_budget.model_calls
                    proposal, refine_usage, refine_ms = await refine_plan_with_model(
                        turn_completers[request.provider],
                        provider=request.provider,
                        model=request.model,
                        message=request.message,
                        previous=previous,
                        plan=proposal,
                        budget=request_budget,
                    )
                    analytics_usage = _add_usage(analytics_usage, refine_usage)
                    if request_budget.model_calls > model_calls_before:
                        analytics_ledger.append(
                            CallLedgerEntry(
                                component="analytics_planner",
                                kind="MODEL",
                                status="SUCCEEDED" if proposal.source == "model" else "FAILED",
                                provider=request.provider,
                                requested_model=request.model,
                                effective_model=request.model,
                                input_tokens=refine_usage.input_tokens,
                                output_tokens=refine_usage.output_tokens,
                                latency_ms=refine_ms,
                            )
                        )
                    invented_derived = proposal.source == "model" and bool(
                        set(proposal.requested_analysis.derived_measures) - set(approved_metrics)
                    )
                    if proposal.source == "model" and (
                        not plan_fields_known(proposal, known_fields) or invented_derived
                    ):
                        # An invented dimension or unverified derived formula is
                        # a failed refinement, never authorization to query
                        # unverified schema.
                        proposal = deterministic_plan.model_copy(update={"source": "fallback"})
                analytics_planning_ms = round((perf_counter() - analytics_started) * 1000)
                if not planner_enabled():
                    metrics_ledger().record(
                        PlannerDecisionRecord(
                            recorded_at=_planner_now(),
                            session_id=request.session_id,
                            db_id=request.db_id,
                            source="disabled",
                            latency_ms=analytics_planning_ms,
                        )
                    )
                requirement = None
                requery_requested = False
                if proposal is not None:
                    approved_requested = [
                        approved_metrics[metric_id]
                        for metric_id in proposal.requested_analysis.derived_measures
                        if metric_id in approved_metrics
                    ]
                    if approved_requested:
                        governed_metric_evidence = "Approved metric definitions:\n" + "\n".join(
                            f"{metric.id}: {metric.formula}; source={metric.source_relation}; "
                            f"required_fields={', '.join(metric.required_fields)}"
                            for metric in approved_requested
                        )
                    requirement = validate_reusability(
                        proposal,
                        stored_columns=source.columns,
                        stored_rows=source.rows,
                        stored_row_count=source.row_count,
                        stored_truncated=source.truncated,
                        approved_metrics=approved_metrics or None,
                    )
                    if requirement.decision == "requery_database":
                        requery_requested = True
                if planner_enabled():
                    metrics_ledger().record(
                        PlannerDecisionRecord(
                            recorded_at=_planner_now(),
                            session_id=request.session_id,
                            db_id=request.db_id,
                            source=(proposal.source if proposal is not None else "fallback"),
                            operation=proposal.operation if proposal else None,
                            reusability_decision=(
                                requirement.decision if requirement is not None else None
                            ),
                            requery=requery_requested,
                            latency_ms=analytics_planning_ms,
                            planner_llm_enabled=planner_llm_enabled() and not assistant_enabled,
                        )
                    )
                if not requery_requested:
                    if requirement is not None and requirement.decision == "need_clarification":
                        clarification = requirement.reason
                        return ChatResponse(
                            session_id=request.session_id,
                            operation="VISUALIZE",
                            resolved_question=previous.resolved_question,
                            state=previous,
                            message=clarification,
                            clarification_required=True,
                            clarification_question=clarification,
                            timings_ms={
                                "analytics_planning": analytics_planning_ms,
                                "total": round((perf_counter() - started) * 1000),
                            },
                            token_usage=analytics_usage,
                            call_ledger=analytics_ledger,
                        )
                    try:
                        # A visualization-only follow-up reads the complete result held
                        # in RAM. Chart selection is deterministic; no model recalls rows
                        # or regenerates the accepted SQL.
                        spec = None
                        if not viz_intent.explicit_type:
                            recommendations = recommend_charts(
                                request.message,
                                source.columns,
                                source.rows,
                                truncated=False,
                            )
                            if recommendations is not None:
                                spec = materialize_recommendation(
                                    recommendations.primary_chart,
                                    question=request.message,
                                    columns=source.columns,
                                    rows=source.rows,
                                    truncated=False,
                                )
                        if spec is None:
                            # A chart-type-only follow-up changes presentation, not
                            # the analytical axes. Re-inferring axes from a wide
                            # result can incorrectly reject a chart that is already
                            # visible to the user.
                            previous_chart = (
                                prior_response.presentation.chart_spec
                                if prior_response and prior_response.presentation
                                else None
                            )
                            chart_intent = viz_intent
                            if (
                                previous_chart is not None
                                and viz_intent.explicit_type
                                and not any(
                                    (
                                        viz_intent.x,
                                        viz_intent.y,
                                        viz_intent.series,
                                        viz_intent.size,
                                        viz_intent.time_grain,
                                        viz_intent.x_components,
                                        viz_intent.y_measures,
                                    )
                                )
                            ):
                                chart_intent = viz_intent.model_copy(
                                    update={
                                        "x": previous_chart.x,
                                        "y": previous_chart.y,
                                        "x_components": previous_chart.x_components,
                                    }
                                )
                            spec = plan_chart(
                                chart_intent,
                                question=request.message,
                                columns=source.columns,
                                rows=source.rows,
                                truncated=source.truncated,
                            )
                        chart_presentation = chart_presentation.model_copy(
                            update={
                                "chart_type": spec.chart_type,
                                "chart_spec": spec,
                                "x": spec.x,
                                "y": spec.y,
                                "title": spec.title,
                                "x_label": spec.x_label,
                                "y_label": spec.y_label,
                                "number_format": spec.number_format,
                                "time_format": spec.time_format,
                            }
                        )
                        message = "Charted the stored result without running SQL again."
                    except VisualizationError as exc:
                        message = (
                            requirement.reason
                            if requirement is not None
                            and requirement.decision not in {"reuse_result", "unsupported"}
                            else str(exc)
                        )
                        chart_presentation = chart_presentation.model_copy(
                            update={"chart_reason": message}
                        )
                    return ChatResponse(
                        session_id=request.session_id,
                        result_id=(
                            prior_response.result_id if prior_response
                            else source_record.response_payload.get("result_id")
                            if source_record else None
                        ),
                        chart_id=uuid4().hex if chart_presentation.chart_spec else None,
                        operation="VISUALIZE",
                        resolved_question=previous.resolved_question,
                        state=previous,
                        generation=source,
                        presentation=chart_presentation,
                        presentation_state=PresentationState(
                            mode="chart" if chart_presentation.chart_spec else "sql",
                            source_run_id=(
                                prior_response.run_id if prior_response
                                else source_record.run_id if source_record else None
                            ),
                            source_result_id=(
                                prior_response.result_id if prior_response
                                else source_record.response_payload.get("result_id")
                                if source_record else None
                            ),
                        ),
                        message=message,
                        previous_sql=previous.last_sql,
                        timings_ms={
                            "analytics_planning": analytics_planning_ms,
                            "total": round((perf_counter() - started) * 1000),
                        },
                        token_usage=analytics_usage,
                        call_ledger=analytics_ledger,
                    )
                # Fall through: the gate asked for new data, so the proposed
                # semantic changes route through the accepted text-to-SQL
                # pipeline with the chart output hint.
            else:
                # Never turn a bounded preview into a misleading complete chart.
                clarification = (
                    "The previous result exceeded its row limit, so plotting it would omit "
                    "data. Narrow the question and include the plot request."
                    if source is not None and source.truncated
                    else "The previous result is no longer in server memory. Rerun the "
                    "original question with the plot request."
                )
                return ChatResponse(
                    session_id=request.session_id,
                    operation="VISUALIZE",
                    resolved_question=previous.resolved_question,
                    state=previous,
                    message=clarification,
                    clarification_required=True,
                    clarification_question=clarification,
                    timings_ms={"total": round((perf_counter() - started) * 1000)},
                )
        conversation_started = perf_counter()
        conversation_usage = TokenUsage()
        interpretation: TurnInterpretation | None = None
        if not assistant_enabled and previous is not None and requires_model_interpretation(
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
        if interpretation is None and assistant_enabled and request.assistant_operation:
            interpretation = TurnInterpretation(
                operation=request.assistant_operation,
                depends_on_previous=request.assistant_operation != "NEW_QUERY",
                resolved_instruction=request.message,
                correction_type=request.feedback_category,
                target=None,
                confidence=1.0,
                source="model",
                provider=request.provider,
                model=request.model,
                changes=request.assistant_changes,
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
            if interpretation.source == "model" and not assistant_enabled
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
        if pending is not None and operation not in EXPLANATION_OPERATIONS:
            try:
                updated_contract = resolve_semantic_contract(
                    previous.semantic_contract if operation != "NEW_QUERY" and previous else None,
                    original_question=request.assistant_original_question or request.message,
                    resolved_question=pending.resolved_question,
                    changes=request.assistant_changes or interpretation.changes,
                )
            except ValueError as exc:
                return ChatResponse(
                    session_id=request.session_id,
                    operation=operation,
                    resolved_question=pending.resolved_question,
                    state=previous,
                    generation=None,
                    message=str(exc),
                    clarification_required=True,
                    clarification_question=str(exc),
                    token_usage=conversation_usage,
                    call_ledger=conversation_ledger,
                )
            pending = pending.model_copy(update={"semantic_contract": updated_contract})
        if operation == "RESET_CONTEXT":
            conversations.reset(request.session_id)
            await asyncio.to_thread(runs.mark_memory_reset, request.session_id)
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
        # Versioned presentation state: an analytical correction that follows a
        # chart-producing turn keeps the chart as the primary answer. The mode
        # comes from backend conversation state, never from a client heuristic.
        last_accepted = accepted_result(request.session_id, request.db_id)
        last_chart_spec = (
            last_accepted.presentation.chart_spec
            if last_accepted and last_accepted.presentation
            else None
        )
        if (
            has_matching_state
            and last_chart_spec is not None
            and not viz_intent.requested
            and operation in ANALYTICAL_CORRECTION_OPERATIONS
        ):
            stable_types = {"line", "bar", "area", "step_line", "horizontal_bar"}
            viz_intent = viz_intent.model_copy(
                update={
                    "requested": True,
                    "followup": False,
                    "chart_type": (
                        last_chart_spec.chart_type
                        if last_chart_spec.chart_type in stable_types
                        else None
                    ),
                }
            )
        for job in chat_jobs.values():
            if job.get("session_id") == request.session_id and job.get("status") == "running":
                job["understanding"] = {
                    "operation": interpretation.operation,
                    "resolved_instruction": pending.resolved_question,
                    "source": interpretation.source,
                    "confidence": interpretation.confidence,
                    "changes": [item.model_dump() for item in interpretation.changes],
                }
        if operation in EXPLANATION_OPERATIONS:
            explanation_started = perf_counter()
            explanation_usage = TokenUsage()
            explanation_model_succeeded = False
            try:
                if assistant_enabled:
                    raise ValueError("The top-layer assistant owns the final explanation.")
                explanation, explanation_usage = await explain_turn_detailed(
                    turn_completers[request.provider],
                    model=request.model,
                    operation=operation,
                    client_message=request.message,
                    state=pending,
                    dialect=("sqlite" if request.db_id in sqlite.list_ids()
                             else postgres.dialect_for(request.db_id) if postgres else "postgres"),
                )
                explanation_model_succeeded = True
            except (ModelError, ValueError):
                explanation = deterministic_explanation(
                    operation,
                    pending,
                    "sqlite" if request.db_id in sqlite.list_ids()
                    else postgres.dialect_for(request.db_id) if postgres else "postgres",
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
            request_budget=request_budget,
            question=pending.resolved_question,
            db_id=request.db_id,
            evidence="\n\n".join(
                item for item in (request.evidence, governed_metric_evidence) if item
            )
            or None,
            visualization_requirement=sql_output_requirement(viz_intent),
            provider=request.provider,
            model=request.model,
            context_mode=request.context_mode,
            context_provider=request.context_provider,
            context_model=request.context_model,
            execute=request.execute,
            max_rows=request.max_rows,
            semantic_contract=pending.semantic_contract,
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
                if persist_state:
                    try:
                        response_state = conversations.put(
                            pending,
                            expected_version=previous.version if previous is not None else None,
                        )
                    except ConversationConflict as exc:
                        raise HTTPException(status_code=409, detail=str(exc)) from exc
                else:
                    response_state = pending
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
                # None means the comparison could not run (budget exhausted); that is not
                # the same as the candidate disagreeing with the accepted result.
                equivalent: bool | None = None
                if database is not None:
                    try:
                        request_budget.claim("database")
                    except RequestBudgetExceeded:
                        equivalent = None
                    else:
                        timeout_seconds = max(0.5, min(5.0, request_budget.remaining()))
                        try:
                            old_columns, old_rows, old_truncated = await asyncio.to_thread(
                                database.execute,
                                request.db_id,
                                previous.last_sql,
                                max_rows=request.max_rows,
                                timeout_seconds=timeout_seconds,
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
                elif equivalent is None:
                    human_review = HumanReviewRequest(
                        reason=(
                            "The request budget was exhausted before the previous SQL could be "
                            "re-executed, so candidate equivalence is unverified."
                        ),
                        question="Which interpretation should be trusted before replacing the SQL?",
                        options=[],
                        evidence=generated.recovery.evidence if generated.recovery else [],
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
            prior_for_database = previous if has_matching_state else None
            # A rejected candidate is a run-history observation, never accepted state.
            response_state = prior_for_database
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
        presentation: ResultPresentation | None = None
        presentation_usage = TokenUsage()
        presentation_ledger = []
        if (
            generated.accepted
            and generated.rows
            and not human_review
            and operation not in {"CHECK_CORRECTNESS", "OPTIMIZE"}
            and not generated.recovery
        ):
            set_session_stage(request.session_id, "presentation")
            presentation, presentation_usage, presentation_entries = await present_result(
                None if assistant_enabled else turn_completers.get(request.provider),
                question=pending.resolved_question,
                sql=generated.sql or "",
                columns=generated.columns,
                rows=generated.rows,
                truncated=generated.truncated,
                model=request.model,
                provider=request.provider,
                budget=request_budget,
                # Top-layer mode owns chart creation through change_visualization.
                # Keep deterministic chart planning only for the legacy route.
                visualization=None if assistant_enabled else viz_intent,
            )
            presentation_ledger.extend(presentation_entries)
        chart_recommendations = None
        suggested_filters = None
        if (
            generated.accepted
            and generated.rows
            and not generated.truncated
            and operation not in {"CHECK_CORRECTNESS", "OPTIMIZE"}
            and not human_review
        ):
            if not assistant_enabled:
                try:
                    chart_recommendations = recommend_charts(
                        pending.resolved_question,
                        generated.columns,
                        generated.rows,
                        truncated=False,
                    )
                except Exception:
                    logger.warning("Chart recommendation failed", exc_info=True)
                    chart_recommendations = None
            if (
                viz_intent.requested
                and len(generated.columns) > 2
                and viz_recommender_llm_enabled()
                and not assistant_enabled
            ):
                try:
                    model_calls_before = request_budget.model_calls
                    model_recommended, viz_usage, viz_ms = await refine_charts_with_model(
                        turn_completers.get(request.provider),
                        provider=request.provider,
                        model=request.model,
                        question=pending.resolved_question,
                        columns=generated.columns,
                        rows=generated.rows,
                        budget=request_budget,
                        required_chart_type=(
                            viz_intent.chart_type if viz_intent.explicit_type else None
                        ),
                    )
                    client_ledger = (
                        [
                            CallLedgerEntry(
                                component="visualization_recommendation",
                                kind="MODEL",
                                status="SUCCEEDED" if model_recommended is not None else "FAILED",
                                provider=request.provider,
                                requested_model=request.model,
                                effective_model=request.model,
                                input_tokens=viz_usage.input_tokens,
                                output_tokens=viz_usage.output_tokens,
                                latency_ms=viz_ms,
                            )
                        ]
                        if request_budget.model_calls > model_calls_before
                        else []
                    )
                    chart_recommendations = merge_recommendations(
                        model_recommended, chart_recommendations
                    )
                    visualization_usage = _add_usage(visualization_usage, viz_usage)
                    presentation_ledger.extend(client_ledger)
                except Exception:
                    logger.warning("LLM chart recommendation failed", exc_info=True)
            if (
                not assistant_enabled
                and presentation is not None
                and viz_intent.requested
                and chart_recommendations is not None
            ):
                try:
                    winning_spec = materialize_recommendation(
                        chart_recommendations.primary_chart,
                        question=pending.resolved_question,
                        columns=generated.columns,
                        rows=generated.rows,
                        truncated=False,
                    )
                    presentation = presentation.model_copy(
                        update={
                            "chart_spec": winning_spec,
                            "chart_type": winning_spec.chart_type,
                            "x": winning_spec.x,
                            "y": winning_spec.y,
                            "title": winning_spec.title,
                            "x_label": winning_spec.x_label,
                            "y_label": winning_spec.y_label,
                            "number_format": winning_spec.number_format,
                            "time_format": winning_spec.time_format,
                            "chart_reason": "",
                        }
                    )
                except VisualizationError:
                    logger.warning("Primary chart recommendation could not be materialized")
            database = (
                sqlite
                if request.db_id in sqlite.list_ids()
                else (postgres if postgres is not None else None)
            )
            if database is not None:
                try:
                    schema = await asyncio.to_thread(database.inspect, request.db_id)
                    profile = profiles.load(schema.dialect, request.db_id)
                    suggested_filters = discover_filters(
                        schema=schema,
                        columns=generated.columns,
                        rows=generated.rows,
                        truncated=False,
                        approved_tables=(
                            response_state.approved_tables or None if response_state else None
                        ),
                        profile=profile,
                        executed_sql=generated.sql,
                    )
                except (DatabaseError, OSError, ValueError):
                    logger.warning("Filter discovery failed", exc_info=True)
        branch = branch_context.get()
        response = ChatResponse(
            session_id=request.session_id,
            operation=operation,
            resolved_question=pending.resolved_question,
            conversation_interpretation=interpretation,
            state=response_state,
            state_commit_pending=not persist_state and generated.accepted,
            state_expected_version=(
                branch[3] if branch else
                previous.version if previous else None
            ),
            generation=generated,
            presentation=presentation,
            message=message,
            human_review=human_review,
            chart_recommendations=chart_recommendations,
            suggested_filters=suggested_filters,
            provenance=[
                "current question and trusted evidence",
                "session-scoped conversation contract",
                "business glossary",
                "live schema and column profiles",
                "SQLGlot read-only safety validation",
            ],
            token_usage=_add_usage(
                _add_usage(conversation_usage, analytics_usage),
                _add_usage(
                    presentation_usage,
                    _add_usage(
                        visualization_usage, _add_usage(planner_usage, generated.token_usage)
                    ),
                ),
            ),
            call_ledger=[
                *analytics_ledger,
                *conversation_ledger,
                *generated.telemetry.call_ledger,
                *presentation_ledger,
            ],
            timings_ms={
                "conversation_interpretation": conversation_ms,
                "routing": routing_ms,
                "planning": planning_ms,
                "generation_validation_execution": generation_ms,
                "total": round((perf_counter() - started) * 1_000),
            },
        )
        return response

    def build_presentation_state(response: ChatResponse) -> PresentationState | None:
        """Derive the typed presentation state from a completed response.

        A validated chart wins; otherwise an accepted SQL answer is SQL mode.
        Clarifications and failed operations intentionally yield ``None`` so the
        previously accepted state is never overwritten by a bad turn.
        """

        if response.presentation and response.presentation.chart_spec is not None:
            spec = response.presentation.chart_spec
            return PresentationState(
                mode="chart",
                chart_type=spec.chart_type,
                x=spec.x,
                y=spec.y,
                series=spec.series,
                size=spec.size,
                title=spec.title,
                source_run_id=(
                    response.presentation_state.source_run_id
                    if response.presentation_state else response.run_id
                ),
                source_result_id=response.result_id,
                data_complete=not (response.generation and response.generation.truncated),
            )
        if response.generation and response.generation.accepted and response.generation.sql:
            return PresentationState(
                mode="sql", source_run_id=response.run_id,
                source_result_id=response.result_id,
            )
        return response.presentation_state

    def remember_result(request: ChatRequest, response: ChatResponse) -> None:
        """Retain bounded executed rows for chart follow-ups, independent of history."""
        if not response.generation or not response.generation.accepted:
            return
        key = (request.session_id, request.db_id)
        recent_results.pop(key, None)
        snapshot = RecentResult.from_response(response)
        recent_results[key] = snapshot
        if response.run_id and snapshot.chart_spec is not None:
            chart_key = (request.session_id, request.db_id, response.run_id)
            chart_snapshots.pop(chart_key, None)
            chart_snapshots[chart_key] = snapshot
            if len(chart_snapshots) > 100:
                chart_snapshots.pop(next(iter(chart_snapshots)))
        if len(recent_results) > 100:
            recent_results.pop(next(iter(recent_results)))

    async def save_response(request: ChatRequest, response: ChatResponse) -> ChatResponse:
        if response.operation == "RESET_CONTEXT":
            recent_results.pop((request.session_id, request.db_id), None)
            return response
        branch = branch_context.get()
        if (
            branch and response.state is not None and response.generation
            and response.generation.accepted
        ):
            response = response.model_copy(update={
                "state_commit_pending": True,
                "state_expected_version": branch[3],
            })
        if response.state_commit_pending and response.state is not None:
            try:
                committed = conversations.put(
                    response.state,
                    expected_version=response.state_expected_version,
                )
            except ConversationConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            response = response.model_copy(update={
                "state": committed, "state_commit_pending": False,
            })
        run_id = uuid4().hex
        result_id = response.result_id
        if response.generation and response.generation.accepted and result_id is None:
            result_id = uuid4().hex
        chart_id = response.chart_id
        if response.presentation and response.presentation.chart_spec and chart_id is None:
            chart_id = uuid4().hex
        presentation_state = build_presentation_state(response)
        if presentation_state is not None and presentation_state.source_run_id is None:
            presentation_state = presentation_state.model_copy(update={"source_run_id": run_id})
        if presentation_state is not None and presentation_state.source_result_id is None:
            presentation_state = presentation_state.model_copy(
                update={"source_result_id": result_id}
            )
        response = response.model_copy(
            update={
                "run_id": run_id,
                "result_id": result_id,
                "chart_id": chart_id,
                "turn_id": run_id,
                "recovery_events": recovery_events(response.generation, run_id),
                "presentation_state": presentation_state,
            }
        )
        try:
            record = await asyncio.to_thread(
                runs.record,
                build_run_record(
                    {
                        "response": response,
                        "request": request,
                        "message": request.message,
                        "latency_ms": response.timings_ms.get("total", 0),
                    }
                ),
            )
            saved_response = response.model_copy(
                update={
                    "conversation_id": record.conversation_id,
                    "parent_run_id": record.parent_run_id,
                    "sql_version": record.sql_version,
                    "history_saved": True,
                }
            )
            remember_result(request, saved_response)
            return saved_response
        except Exception:  # History failures must not discard a completed query response.
            logger.warning("run_history_record_failed", exc_info=True)
            remember_result(request, response)
            return response

    async def execute_chat(request: ChatRequest, *, persist: bool = True) -> ChatResponse:
        token = None
        if request.parent_run_id and branch_context.get() is None:
            parent = runs.get(request.parent_run_id, request.session_id)
            if parent is None or parent.db_id != request.db_id or parent.state_snapshot is None:
                raise HTTPException(
                    status_code=409, detail="The selected previous turn is unavailable."
                )
            current = conversations.get(request.session_id)
            token = branch_context.set((
                request.session_id, parent.state_snapshot,
                result_from_history(parent), current.version if current else None,
            ))
        try:
            previous = accepted_state(request.session_id)
            response = await process_chat(
                request, persist_state=persist and branch_context.get() is None
            )
            if request.parent_run_id:
                response = response.model_copy(update={"parent_run_id":request.parent_run_id})
            if previous and previous.db_id == request.db_id and response.operation != "NEW_QUERY":
                response = response.model_copy(update={"previous_sql": previous.last_sql})
            return await save_response(request, response) if persist else response
        finally:
            if token is not None:
                branch_context.reset(token)

    async def assistant_sql_handler(request: ChatRequest) -> ChatResponse:
        # The top model still has to produce its final answer. Save the
        # completed turn once, after that answer has been attached.
        if request._history_source_run_id and runs.sql_memory(
            request.session_id, request.db_id, request._history_source_run_id,
            parent_run_id=history_anchor.get(),
        ) is None:
            return ChatResponse(
                session_id=request.session_id, operation="REFINE",
                resolved_question=request.message,
                message="The historical analysis is unavailable in this session and database.",
            )
        token = (
            provisional_context.set((request.session_id, request._turn_state))
            if request._turn_state is not None else None
        )
        try:
            response = await execute_chat(request, persist=False)
            if response.state_commit_pending and request._turn_state is not None:
                response = response.model_copy(update={
                    "state_expected_version": request._turn_expected_version,
                })
            return response
        finally:
            if token is not None:
                provisional_context.reset(token)

    async def assistant_investigation_handler(request: ChatRequest) -> ChatResponse:
        # Investigation state is request-local. Never save an intermediate turn
        # in the client's history or leave its temporary conversation behind.
        try:
            return await execute_chat(request, persist=False)
        finally:
            conversations.reset(request.session_id)
            recent_results.pop((request.session_id, request.db_id), None)

    bridge = ConversationalAnalyticsBridge(
        database=sqlite,
        postgres=postgres,
        profiles=profiles,
        glossaries=glossaries,
        semantic_catalogs=semantic_catalogs,
        retriever=hybrid_retriever,
        sql_handler=execute_chat,
    )
    def reset_assistant_context(session_id: str) -> None:
        conversations.reset(session_id)
        runs.mark_memory_reset(session_id)
        for key in [key for key in recent_results if key[0] == session_id]:
            recent_results.pop(key, None)

    def inspect_assistant_history(session: str, db: str, reference: str) -> dict[str, Any] | None:
        if db not in sqlite.list_ids() and (
            postgres is None or db not in postgres.configured_ids()
        ):
            return None
        if reference.startswith("snapshot:"):
            memory = runs.sql_memory(
                session, db, reference.removeprefix("snapshot:"),
                parent_run_id=history_anchor.get(),
            )
            if memory is None:
                return None
            snapshot = runs.inspect_result(session, db, str(memory["run_id"]), page_size=100)
            if snapshot is None or not snapshot["calculation_allowed"]:
                return {"evidence_kind":"SNAPSHOT_UNAVAILABLE", "run_id":memory["run_id"],
                        "verified_numerical_evidence":False,
                        "limitation":"No complete snapshot or time-travel source is available."}
            return {**snapshot, "evidence_kind":"HISTORICAL_SNAPSHOT",
                    "execution_timestamp":memory["execution_timestamp"],
                    "verified_numerical_evidence":True}
        return runs.sql_memory(session, db, reference, parent_run_id=history_anchor.get())

    async def refresh_assistant_history(request: ChatRequest, reference: str) -> ChatResponse:
        memory = inspect_assistant_history(request.session_id, request.db_id, reference)
        if memory is None:
            return ChatResponse(
                session_id=request.session_id, operation="REFINE",
                resolved_question=request.message,
                message="The historical analysis is unavailable.",
            )
        try:
            current = accepted_state(request.session_id)
            return await refresh_historical_sql(
                str(memory["run_id"]), request.session_id, persist=False,
                budget=active_request_budget(),
                expected_version=(request._turn_expected_version if request._turn_state else
                    (current.version if current else None)),
            )
        except HTTPException:
            return ChatResponse(
                session_id=request.session_id, operation="REFINE",
                resolved_question=request.message,
                message="The historical SQL cannot be refreshed safely.",
            )

    async def assistant_dashboard_handler(
        request: ChatRequest, objective: str, source: RecentResult | ChatResponse, undo: bool
    ) -> ChatResponse:
        retained = (
            source if isinstance(source, RecentResult) else RecentResult.from_response(source)
        )
        if not retained.result_id:
            raise DashboardError("The accepted result has no immutable identity.")
        design_request = DashboardRequest(
            session_id=request.session_id, db_id=request.db_id, result_id=retained.result_id,
            design_mode="ai", objective=objective,
        )
        previous = dashboard_revisions.get(request.session_id, request.db_id, retained.result_id)
        if previous is None:
            for prior_id in request._dashboard_source_ids:
                candidate = dashboard_revisions.get(request.session_id, request.db_id, prior_id)
                if candidate is not None:
                    previous = candidate
                    break
        saved = studio_context.get()
        evidence: dict[str, RecentResult] = dict(dashboard_bundles.get(
            (request.session_id, request.db_id, retained.result_id), {}
        ))
        if previous:
            evidence.update(dashboard_bundles.get(
                (request.session_id, request.db_id, previous.result_id), {}
            ))
        if saved is not None:
            evidence.update({key: value.result()
                             for key, value in saved.document.results().items()})
        for key, item in request._dashboard_sources.items():
            evidence[key] = (item if isinstance(item, RecentResult)
                             else RecentResult.from_response(item))
        evidence[retained.result_id] = retained
        if saved is not None:
            retained = saved.document.source.result()
            assert retained.result_id is not None
            design_request = design_request.model_copy(update={"result_id": retained.result_id})
            previous = saved.document.spec
        if request._dashboard_source_ids:
            selected = set(request._dashboard_source_ids)
            for identity in selected - set(evidence):
                cached_source = dashboard_source(request.session_id, request.db_id, identity)
                if cached_source is not None:
                    evidence[identity] = cached_source
            if (not selected.issubset(evidence) or retained.result_id not in selected
                    or len(selected) != len(request._dashboard_source_ids)):
                return ChatResponse(session_id=request.session_id, operation="VISUALIZE",
                    resolved_question=objective,
                    message="Some requested analysis is unavailable. Reopen the dashboard.")
            evidence = {key: evidence[key] for key in request._dashboard_source_ids}
        if len(evidence) > 4:
            return ChatResponse(session_id=request.session_id, operation="VISUALIZE",
                resolved_question=objective,
                message="This dashboard supports up to four separate analyses. Narrow its scope.")
        assert retained.result_id is not None
        current_revision = dashboard_revisions.get(
            request.session_id, request.db_id, retained.result_id
        )
        expected = current_revision.revision_id if current_revision else None
        usage = TokenUsage()
        ledger: list[CallLedgerEntry] = []
        try:
            if undo:
                if saved is not None:
                    raise DashboardError(
                        "Use Undo in Dashboard Studio to restore a saved revision."
                    )
                # Validate current source authorization/completeness before undo.
                build_dashboard(design_request, lambda *_: retained)
                spec = dashboard_revisions.undo(
                    request.session_id, request.db_id, retained.result_id
                )
            else:
                provider = os.environ.get("TEXT2SQL_DASHBOARD_PROVIDER") or request.provider
                model = os.environ.get("TEXT2SQL_DASHBOARD_MODEL") or request.model
                outcome = await design_dashboard(
                    source=retained, request=design_request,
                    completer=turn_completers.get(provider), provider=provider, model=model,
                    budget=active_request_budget() or RequestBudget(timeout_seconds=15),
                    previous=previous,
                    patch_only=saved is not None,
                    sources=evidence,
                    repair=True,
                    target_widget_id=request._dashboard_widget_id,
                )
                usage, ledger = outcome.usage, list(outcome.ledger)
                if outcome.spec is None:
                    return ChatResponse(
                        session_id=request.session_id, operation="VISUALIZE",
                        resolved_question=objective, message=outcome.reason,
                        token_usage=usage, call_ledger=ledger,
                    )
                spec = (outcome.spec if saved is not None else
                        dashboard_revisions.commit(request.session_id, outcome.spec, expected))
                if saved is None:
                    bundle_key = (request.session_id, request.db_id, retained.result_id)
                    dashboard_bundles[bundle_key] = evidence
                    if len(dashboard_bundles) > 100:
                        dashboard_bundles.pop(next(iter(dashboard_bundles)))
            response = ChatResponse(
                session_id=request.session_id, result_id=retained.result_id,
                operation="VISUALIZE", resolved_question=objective,
                generation=retained.generation, presentation=None,
                message="The dashboard is ready from the accepted result.",
                dashboard=spec.model_dump(mode="json"), token_usage=usage, call_ledger=ledger,
                state_commit_pending=(source.state_commit_pending
                                      if isinstance(source, ChatResponse) else False),
                state_expected_version=(source.state_expected_version
                                        if isinstance(source, ChatResponse) else None),
                state=(source.state if isinstance(source, ChatResponse)
                       else accepted_state(request.session_id)),
            )
            response._dashboard_evidence = evidence
            return response
        except DashboardError as exc:
            return ChatResponse(
                session_id=request.session_id, operation="VISUALIZE", resolved_question=objective,
                message=str(exc), token_usage=usage, call_ledger=ledger,
            )

    def dashboard_summary(session: str, db: str, result: str) -> dict[str, Any] | None:
        saved = studio_context.get()
        spec = (saved.document.spec if saved is not None
                else dashboard_revisions.get(session, db, result))
        if spec is None:
            return None
        return {"result_id": result, "revision_id": spec.revision_id, "title": spec.title,
                "source_result_ids": list(spec.sources) or [spec.result_id],
                "theme": spec.theme, "kpis": [{"id": k.kpi_id, "title": k.title}
                                              for k in spec.kpis],
                "charts": [{"id": w.widget_id, "title": w.chart_spec.title,
                            "type": w.chart_spec.chart_type, "width": w.column_span,
                            "source_result_id": w.result_id}
                           for w in spec.widgets]}

    conversational_agent = ConversationalAnalyticsAgent(
        bridge=bridge,
        completers=turn_completers,
        get_state=accepted_state,
        get_last_response=accepted_result,
        reset_handler=reset_assistant_context,
        sql_handler=assistant_sql_handler,
        investigation_handler=assistant_investigation_handler,
        history_inspector=inspect_assistant_history,
        history_refresher=refresh_assistant_history,
        dashboard_handler=assistant_dashboard_handler,
        dashboard_inspector=dashboard_summary,
    )

    async def edit_saved_dashboard(saved: SavedDashboard, message: str) -> ChatResponse:
        source = saved.document.source.result()
        if saved._selected_widget:
            widget_source = next((w.result_id for w in saved.document.spec.widgets
                                  if w.widget_id == saved._selected_widget), None)
            widget_source = widget_source or next((k.result_id for k in saved.document.spec.kpis
                                                  if k.kpi_id == saved._selected_widget), None)
            if widget_source:
                source = saved.document.results()[widget_source].result()
        session = f"studio-{saved.dashboard_id}"
        state = ConversationState(
            session_id=session, db_id=source.db_id, root_question=source.question,
            resolved_question=source.question, semantic_contract=source.semantic_contract,
            last_sql=source.sql, last_columns=source.columns, last_row_count=source.row_count,
            last_model_context=source.verified_context,
        )
        branch_token = branch_context.set((session, state, source, None))
        studio_token = studio_context.set(saved)
        request = ChatRequest(session_id=session, db_id=source.db_id, message=message)
        request._dashboard_snapshot_only = True
        request._dashboard_widget_id = saved._selected_widget
        request.dashboard_target = DashboardTarget(
            dashboard_id=saved.dashboard_id, revision_id=str(saved.version),
            widget_id=saved._selected_widget,
        )
        request._dashboard_sources = {key: value.result() for key, value
                                     in saved.document.results().items()}
        try:
            # Same LangGraph, no HTTP recursion or intermediate accepted-state save.
            return await conversational_agent.run(request)
        finally:
            studio_context.reset(studio_token)
            branch_context.reset(branch_token)

    def studio_source(session: str, db: str, identity: str) -> RecentResult | None:
        return (dashboard_bundles.get((session, db, identity), {}).get(identity)
                or dashboard_source(session, db, identity))

    app.include_router(studio_router(
        saved_dashboards, dashboard_revisions, studio_source, edit_saved_dashboard,
        source_bundle=lambda session, db, identity: dashboard_bundles.get(
            (session, db, identity), {}
        ),
    ))

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        if conversational_agent_enabled():
            token = None
            if request.parent_run_id:
                parent = runs.get(request.parent_run_id, request.session_id)
                if parent is None or parent.db_id != request.db_id or parent.state_snapshot is None:
                    raise HTTPException(
                        status_code=409, detail="The selected previous turn is unavailable."
                    )
                current = conversations.get(request.session_id)
                token = branch_context.set((
                    request.session_id, parent.state_snapshot, result_from_history(parent),
                    current.version if current else None,
                ))
            anchor_token = history_anchor.set(request.parent_run_id)
            try:
                if request.dashboard_target:
                    current_result = accepted_result(request.session_id, request.db_id)
                    current_design = (dashboard_revisions.get(
                        request.session_id, request.db_id, current_result.result_id
                    ) if current_result and current_result.result_id else None)
                    target = request.dashboard_target
                    if (current_design is None or current_design.dashboard_id != target.dashboard_id
                            or current_design.revision_id != target.revision_id
                            or (target.widget_id is not None and target.widget_id not in
                                {w.widget_id for w in current_design.widgets}
                                | {k.kpi_id for k in current_design.kpis})):
                        raise HTTPException(409, "That dashboard component changed; reopen it.")
                    request._dashboard_widget_id = target.widget_id
                    request._dashboard_snapshot_only = True
                response = await conversational_agent.run(request)
                return (
                    response if response.history_saved
                    else await save_response(request, response)
                )
            finally:
                history_anchor.reset(anchor_token)
                if token is not None:
                    branch_context.reset(token)
        if os.environ.get("TEXT2SQL_CONVERSATIONAL_BRIDGE_ENABLED", "false").casefold() != "true":
            return await execute_chat(request)
        try:
            outcome = await bridge.run(request)
        except DatabaseError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if outcome.route == "sql":
            return outcome.response
        return await save_response(request, outcome.response)

    async def refresh_historical_sql(
        run_id: str, session_id: str, *, persist: bool = True,
        budget: RequestBudget | None = None, expected_version: int | None = None,
    ) -> ChatResponse:
        record = runs.get(run_id, session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Conversation turn not found.")
        if not record.accepted or not record.sql or record.state_snapshot is None:
            raise HTTPException(status_code=409, detail="This turn has no accepted SQL to rerun.")
        if record.state_snapshot.last_sql != record.sql:
            raise HTTPException(status_code=409, detail="Historical SQL/state identity mismatch.")
        context = (record.response_payload.get("generation") or {}).get("model_context") or {}
        tables = context.get("tables") or context.get("execution_context", {}).get("tables") or {}
        allowed_tables = set(tables) or set(record.state_snapshot.approved_tables)
        if not allowed_tables:
            raise HTTPException(
                status_code=409, detail="This older turn has no verified table scope."
            )
        allowed_columns = (
            {table: set(info.get("columns") or []) for table, info in tables.items()}
            if tables
            else None
        )
        original = record.response_payload.get("generation") or {}
        dialect = original.get("dialect", "sqlite")
        budget = budget or RequestBudget(15, max_model_calls=0, max_database_calls=2)
        current = conversations.get(session_id)
        if persist:
            expected_version = current.version if current else None
        started = perf_counter()
        try:
            checked = await budget.wait(
                asyncio.to_thread(
                    active_agent.check,
                    CheckRequest(
                        db_id=record.db_id,
                        sql=record.sql,
                        dialect=dialect,
                        execute=True,
                        max_rows=100,
                    ),
                    budget=budget,
                    allowed_tables=allowed_tables,
                    allowed_columns=allowed_columns,
                ),
                kind="work",
            )
        except (DatabaseError, RequestBudgetExceeded) as exc:
            checked = CheckResponse(
                db_id=record.db_id,
                dialect=dialect,
                validation=ValidationResult(valid=False, code="DATABASE_ERROR", message=str(exc)),
            )
        valid = checked.validation.valid
        elapsed = round((perf_counter() - started) * 1000)
        generated = GenerateResponse(
            db_id=record.db_id,
            question=record.resolved_question,
            dialect=dialect,
            provider=cast(ModelProvider, record.provider),
            model=record.model,
            strategy=route_question(record.resolved_question),
            sql=record.sql if valid else None,
            semantic_contract=record.state_snapshot.semantic_contract,
            accepted=valid,
            execution_status="EXECUTABLE" if valid else "NOT_EXECUTED",
            execution_timestamp=time() if valid else None,
            columns=checked.columns,
            rows=checked.rows,
            row_count=checked.row_count,
            truncated=checked.truncated,
            model_context=context,
            attempts=[
                Attempt(
                    number=1,
                    sql=record.sql,
                    normalized_sql=record.sql,
                    validation=checked.validation,
                    latency_ms=elapsed,
                )
            ],
            termination_reason="accepted" if valid else "database_error",
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )
        response_state = record.state_snapshot
        if valid:
            response_state = response_state.model_copy(
                    update={
                        "last_sql": record.sql,
                        "last_columns": checked.columns,
                        "last_row_count": checked.row_count,
                        "last_truncated": checked.truncated,
                    }
            )
        refresh_request = ChatRequest(
                session_id=session_id,
                db_id=record.db_id,
                message="Rerun the saved SQL.",
                parent_run_id=record.run_id,
                provider=cast(ModelProvider, record.provider),
                model=record.model,
            )
        response = ChatResponse(
                session_id=session_id,
                operation="REFINE",
                resolved_question=record.resolved_question,
                state=response_state,
                generation=generated,
                parent_run_id=record.run_id,
                result_id=uuid4().hex if valid else None,
                state_commit_pending=valid,
                state_expected_version=expected_version,
                provenance=[f"historical_sql_source_run:{record.run_id}",
                            "refreshed_current_data_not_historical_snapshot"],
                message=(
                    f"Reran the saved SQL against current data: {checked.row_count} rows returned."
                    if valid
                    else "The saved SQL could not be rerun: " + checked.validation.message
                ),
                timings_ms={"total": elapsed},
                token_usage=TokenUsage(input_tokens=0, output_tokens=0),
            )
        return await save_response(refresh_request, response) if persist else response

    @app.post("/api/history/{run_id}/rerun", response_model=ChatResponse)
    async def rerun_history(run_id: str, session_id: str) -> ChatResponse:
        return await refresh_historical_sql(run_id, session_id)

    async def run_chat_job(job_id: str, request: ChatRequest) -> None:
        job = chat_jobs[job_id]
        job["status"] = "running"
        job["stage"] = "conversation"
        job["events"].append({"stage": "conversation", "elapsed_ms": 0})
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
            "events": [],
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
            "elapsed_ms": round(
                (float(job.get("finished_at", perf_counter())) - float(job["started_at"])) * 1_000
            ),
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
