"""FastAPI application for interactive generation and SQL checking."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.context_planner import (
    fallback_context_request,
    plan_context_detailed,
    reconcile_context_contract,
    selection_to_context_request,
    verify_context_request,
)
from semantic_text2sql.conversation import (
    ConversationStore,
    classify_operation,
    interpret_turn_detailed,
    requires_model_interpretation,
    resolve_turn,
)
from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.glossary import GlossaryStore
from semantic_text2sql.historical import HistoricalQueryStore
from semantic_text2sql.hybrid_retrieval import (
    FastEmbedEncoder,
    HybridSchemaRetriever,
    LightGBMSchemaReranker,
    metadata_requests,
)
from semantic_text2sql.llm import (
    CLAUDE_CODE_EXECUTABLE,
    AgentRouterClaudeModel,
    AgentRouterCodexModel,
    AgentRouterModel,
    GroqSQLModel,
    ModelError,
    OllamaSQLModel,
    RoutingSQLModel,
    claude_code_available,
    ollama_model_status,
)
from semantic_text2sql.models import (
    DEFAULT_OLLAMA_MODEL,
    GROQ_QWEN_MODEL,
    ChatRequest,
    ChatResponse,
    CheckRequest,
    CheckResponse,
    ContextRequest,
    ConversationState,
    DatabaseOption,
    GenerateRequest,
    GenerateResponse,
    HistoricalExample,
    ModelOption,
    ModelProvider,
    PlannerMetadataRequirement,
    SemanticContract,
    TokenUsage,
    TurnInterpretation,
)
from semantic_text2sql.postgres import PostgresRegistry
from semantic_text2sql.profiling import ProfileStore
from semantic_text2sql.semantic import (
    apply_structural_formulas,
    resolution_report,
)
from semantic_text2sql.validator import validate_sql

logger = logging.getLogger(__name__)


def create_app(
    agent: TextToSQLAgent | None = None,
    conversation_completers: dict[str, Any] | None = None,
) -> FastAPI:
    postgres_dsn = os.environ.get("POSTGRES_BOOKS_DSN")
    postgres = PostgresRegistry({"books_postgres": postgres_dsn}) if postgres_dsn else None
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
    claude = AgentRouterClaudeModel(
        os.environ.get("AGENTROUTER_API_KEY"),
        os.environ.get("AGENTROUTER_BASE_URL", "https://agentrouter.org"),
    )
    agentrouter = AgentRouterModel(
        claude,
        AgentRouterCodexModel(
            os.environ.get("AGENTROUTER_API_KEY"),
            os.environ.get("AGENTROUTER_BASE_URL", "https://agentrouter.org"),
        ),
    )
    ollama = OllamaSQLModel(os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    groq = GroqSQLModel(
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    )
    active_agent = agent or TextToSQLAgent(
        sqlite,
        RoutingSQLModel(ollama, agentrouter, groq),
        postgres,
        profiles,
    )
    turn_completers = conversation_completers or {
        "ollama": ollama,
        "agentrouter": agentrouter,
        "groq": groq,
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
    conversations = ConversationStore()
    chat_jobs: dict[str, dict[str, Any]] = {}
    chat_tasks: dict[str, asyncio.Task[None]] = {}

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "online"}

    @app.get("/", include_in_schema=False)
    async def web_application() -> FileResponse:
        return FileResponse(web_root / "index.html")

    @app.get("/api/models", response_model=list[ModelOption])
    async def models() -> list[ModelOption]:
        missing_key = (
            None
            if os.environ.get("AGENTROUTER_API_KEY")
            else "AGENTROUTER_API_KEY is not set in the environment."
        )
        missing_groq_key = (
            None
            if os.environ.get("GROQ_API_KEY")
            else "GROQ_API_KEY is not set in the environment."
        )
        local_reason = await ollama_model_status(DEFAULT_OLLAMA_MODEL, base_url=ollama.base_url)
        claude_reason = missing_key or (
            None
            if claude_code_available()
            else f"Claude Code is not installed at {CLAUDE_CODE_EXECUTABLE}."
        )
        return [
            _model_option("ollama", DEFAULT_OLLAMA_MODEL, local=True, reason=local_reason),
            _model_option("agentrouter", "gpt-5.6-sol", local=False, reason=missing_key),
            _model_option("agentrouter", "claude-opus-5", local=False, reason=claude_reason),
            _model_option("agentrouter", "claude-opus-4-7", local=False, reason=claude_reason),
            _model_option("groq", GROQ_QWEN_MODEL, local=False, reason=missing_groq_key),
        ]

    @app.get("/api/databases", response_model=list[DatabaseOption])
    async def databases() -> list[DatabaseOption]:
        options = [
            DatabaseOption(db_id=db_id, dialect="sqlite", configured=True)
            for db_id in sqlite.list_ids()
        ]
        options.append(
            DatabaseOption(
                db_id="books_postgres",
                dialect="postgres",
                configured=postgres is not None,
            )
        )
        return options

    @app.post("/api/check", response_model=CheckResponse)
    async def check(request: CheckRequest) -> CheckResponse:
        return active_agent.check(request)

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        started = perf_counter()
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
                confidence=1.0,
                source="rules",
            )
        conversation_ms = round((perf_counter() - conversation_started) * 1_000)
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
                timings_ms={
                    "conversation_interpretation": conversation_ms,
                    "total": round((perf_counter() - started) * 1_000),
                },
            )
        assert pending is not None
        if operation == "EXPLAIN":
            contract = pending.semantic_contract
            explanation = (
                "The last accepted query interpreted the request with "
                f"aggregation={contract.aggregation if contract else None}, "
                f"grain={contract.grain if contract else []}, "
                f"filters={contract.proposed_filters if contract else []}, and "
                f"tables={pending.approved_tables}."
            )
            return ChatResponse(
                session_id=request.session_id,
                operation="EXPLAIN",
                resolved_question=pending.resolved_question,
                conversation_interpretation=interpretation,
                state=pending,
                message="Explained the last accepted query without generating new SQL.",
                explanation=explanation,
                provenance=[
                    "current question and trusted evidence",
                    "business glossary",
                    "live schema and PK/FK relationships",
                    "semantic contract",
                    "SQLGlot read-only safety validation",
                ],
                token_usage=conversation_usage,
                timings_ms={
                    "conversation_interpretation": conversation_ms,
                    "total": round((perf_counter() - started) * 1_000),
                },
            )
        routing_started = perf_counter()
        schema = sqlite.inspect(request.db_id)
        database_profile = profiles.load("sqlite", request.db_id)
        retrieval_trace = None
        if operation == "OPTIMIZE" and previous and previous.approved_tables:
            proposed_tables = previous.approved_tables
            planner_schema = schema.model_copy(
                update={
                    "tables": [table for table in schema.tables if table.name in proposed_tables],
                    "relationships": [
                        item
                        for item in schema.relationships
                        if item.from_table in proposed_tables and item.to_table in proposed_tables
                    ],
                }
            )
            context_request = fallback_context_request(
                previous.semantic_contract or SemanticContract()
            )
        else:
            historical_schema_evidence = history.schema_evidence(
                pending.resolved_question,
                request.db_id,
                schema,
                top_k=3,
                min_score=float(os.environ.get("TEXT2SQL_HISTORY_ML_MIN_SCORE", "0.65")),
            )
            planner_schema, retrieval_selection, retrieval_trace = hybrid_retriever.retrieve(
                pending.resolved_question,
                request.evidence,
                schema,
                database_profile,
                glossaries.load(request.db_id),
                historical_schema_evidence,
            )
            proposed_tables = retrieval_selection.tables
            metadata = metadata_requests(
                pending.resolved_question, planner_schema, database_profile
            )
            context_request = ContextRequest(
                tables=retrieval_selection.tables,
                columns=retrieval_selection.columns,
                business_concepts=sorted(
                    glossaries.relevant_concept_ids(request.db_id, pending.resolved_question)
                ),
                metadata_requirements=[
                    PlannerMetadataRequirement(kind=cast(Any, kind), targets=[target])
                    for kind, target in metadata
                ],
            )
        routing_ms = round((perf_counter() - routing_started) * 1_000)
        planning_started = perf_counter()
        contract = (
            previous.semantic_contract
            if operation == "OPTIMIZE" and previous and previous.semantic_contract
            else SemanticContract()
        )
        contract = apply_structural_formulas(
            contract,
            glossaries.structural_formulas(request.db_id, pending.resolved_question),
        )
        candidate_names = set(proposed_tables)
        candidate_relationships = [
            {
                "left": f"{item.from_table}.{item.from_column}",
                "right": f"{item.to_table}.{item.to_column}",
                "state": "VERIFIED_FK",
            }
            for item in schema.relationships
            if item.from_table in candidate_names and item.to_table in candidate_names
        ]
        if database_profile is not None:
            candidate_relationships.extend(
                {
                    "left": f"{item.parent_table}.{item.parent_column}",
                    "right": f"{item.child_table}.{item.child_column}",
                    "state": "INFERRED_KEY_RELATIONSHIP" if item.inferred else "VERIFIED_FK",
                }
                for item in database_profile.relationships
                if item.parent_table in candidate_names and item.child_table in candidate_names
            )
        planner_usage = TokenUsage()
        planner_call_used = False
        planner_enabled = (
            request.context_mode == "model1"
            and os.environ.get(
                "TEXT2SQL_CONTEXT_PLANNER_ENABLED", "true" if agent is None else "false"
            ).casefold()
            == "true"
        )
        planner_provider = request.context_provider or request.provider
        planner_model = (
            request.context_model or os.environ.get("TEXT2SQL_CONTEXT_MODEL") or request.model
        )
        if planner_enabled and operation != "OPTIMIZE":
            try:
                selection, planner_usage = await plan_context_detailed(
                    turn_completers[planner_provider],
                    planner_model,
                    pending.resolved_question,
                    request.evidence,
                    planner_schema,
                    glossaries.load(request.db_id),
                    previous.semantic_contract
                    if previous is not None and operation != "NEW_QUERY"
                    else None,
                    candidate_relationships,
                )
                context_request = selection_to_context_request(selection)
                planner_call_used = True
            except (ModelError, ValueError):
                pass
        approved_concepts = glossaries.relevant_concept_ids(
            request.db_id, pending.resolved_question
        )
        context_request = verify_context_request(
            context_request,
            schema,
            contract,
            database_profile,
            approved_concepts,
        )
        logger.info("verified_context_request=%s", context_request.model_dump_json())
        contract = reconcile_context_contract(contract, context_request)
        generation_business_context = (
            glossaries.retrieve(request.db_id, pending.resolved_question, top_k=5)
            if context_request.business_concepts
            else None
        )
        historical_examples: list[HistoricalExample] = []
        if os.environ.get("TEXT2SQL_HISTORY_ENABLED", "false").casefold() == "true":
            historical_examples = [
                item
                for item in history.search(
                    pending.resolved_question,
                    request.db_id,
                    top_k=2,
                    min_score=float(os.environ.get("TEXT2SQL_HISTORY_MIN_SCORE", "0.85")),
                    bm25_pool=20,
                    semantic_pool=5,
                    candidate_tables=set(context_request.tables),
                )
                if validate_sql(item.sql, schema, dialect="sqlite").valid
            ][:2]
        report = resolution_report(pending.resolved_question, contract)
        if context_request.tables:
            proposed_tables = context_request.tables
        planning_ms = round((perf_counter() - planning_started) * 1_000)
        generation_started = perf_counter()
        generation_request = GenerateRequest(
            db_id=request.db_id,
            question=pending.resolved_question,
            evidence=request.evidence,
            provider=request.provider,
            model=os.environ.get("TEXT2SQL_SQL_MODEL") or request.model,
            execute=request.execute,
            max_rows=request.max_rows,
            approved_tables=proposed_tables,
            semantic_contract=contract,
            business_context=generation_business_context,
            previous_sql=previous.last_sql if operation == "OPTIMIZE" and previous else None,
            optimization_required=operation == "OPTIMIZE",
            resolution_report=report,
            semantic_call_used=False,
            semantic_token_usage=TokenUsage(),
            context_request=context_request,
            planner_call_used=planner_call_used,
            planner_token_usage=planner_usage,
            historical_examples=historical_examples,
        )
        generated = await active_agent.generate(generation_request)
        generation_ms = round((perf_counter() - generation_started) * 1_000)
        historical_attempted = (
            os.environ.get("TEXT2SQL_HISTORY_ENABLED", "false").casefold() == "true"
        )
        generated = generated.model_copy(
            update={
                "telemetry": generated.telemetry.model_copy(
                    update={
                        "model1_latency_ms": planning_ms,
                        "ab_context_mode": request.context_mode,
                        "retrieval": retrieval_trace,
                        "selected_table_count": len(context_request.tables),
                        "selected_column_count": sum(
                            len(columns) for columns in context_request.columns.values()
                        ),
                        "metadata_request_count": len(context_request.metadata_requirements),
                        "historical_attempted": historical_attempted,
                        "historical_candidates_retrieved": (
                            len(historical_examples) if historical_attempted else None
                        ),
                        "historical_examples_admitted": len(historical_examples),
                        "historical_similarity_scores": [
                            item.score for item in historical_examples
                        ],
                    }
                )
            }
        )
        response_state: ConversationState | None
        if generated.accepted:
            pending = pending.model_copy(
                update={
                    "semantic_contract": generated.semantic_contract,
                    "approved_tables": proposed_tables,
                    "last_sql": generated.sql,
                }
            )
            conversations.put(pending)
            response_state = pending
            message = _conversational_answer(operation, generated)
        else:
            response_state = previous
            message = _failure_message(generated)
        return ChatResponse(
            session_id=request.session_id,
            operation=operation,
            resolved_question=pending.resolved_question,
            conversation_interpretation=interpretation,
            state=response_state,
            generation=generated,
            message=message,
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
        job["stage"] = "interpreting_and_generating"
        try:
            response = await chat(request)
            job.update(
                status="completed",
                stage="completed",
                response=response.model_dump(mode="json"),
            )
        except asyncio.CancelledError:
            job.update(status="cancelled", stage="cancelled")
            raise
        except Exception as exc:  # pragma: no cover - defensive job boundary
            job.update(status="failed", stage="failed", error=str(exc))

    @app.post("/api/chat/jobs", status_code=202)
    async def start_chat_job(request: ChatRequest) -> dict[str, str]:
        job_id = uuid4().hex
        chat_jobs[job_id] = {
            "job_id": job_id,
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
        job = chat_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Chat job was not found.")
        return {
            **job,
            "elapsed_ms": round((perf_counter() - float(job["started_at"])) * 1_000),
        }

    @app.delete("/api/chat/jobs/{job_id}")
    async def cancel_chat_job(job_id: str) -> dict[str, str]:
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


def _model_option(
    provider: ModelProvider,
    model: str,
    *,
    local: bool,
    reason: str | None,
) -> ModelOption:
    """Advertise a catalog model, treating a stated reason as "cannot serve requests"."""
    return ModelOption(
        provider=provider,
        model=model,
        local=local,
        configured=reason is None,
        unavailable_reason=reason,
    )


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


def _conversational_answer(operation: str, generated: GenerateResponse) -> str:
    if operation == "OPTIMIZE":
        if generated.optimization and generated.optimization.status == "optimized":
            prefix = "I verified that the replacement is equivalent and measurably faster."
        elif generated.optimization and generated.optimization.status == "equivalent_not_faster":
            prefix = "The rewrite was equivalent but not faster, so I retained the original query."
        else:
            prefix = "No optimization passed the acceptance gate; I retained the previous query."
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
    return "Query failed; previous conversation state was preserved."


app = create_app()
