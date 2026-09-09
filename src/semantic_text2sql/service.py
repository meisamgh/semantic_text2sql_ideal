"""Reusable non-conversation orchestration for grounded Text-to-SQL questions."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.context_planner import (
    ContextPlannerCompleter,
    fallback_context_request,
    plan_context_detailed,
    reconcile_context_contract,
    selection_to_context_request,
    verify_context_request,
)
from semantic_text2sql.database import DatabaseError, DatabaseRegistry
from semantic_text2sql.formulas import apply_structural_formulas
from semantic_text2sql.glossary import GlossaryStore
from semantic_text2sql.historical import HistoricalQueryStore
from semantic_text2sql.hybrid_retrieval import HybridSchemaRetriever, metadata_requests
from semantic_text2sql.llm import ModelError
from semantic_text2sql.models import (
    ContextRequest,
    GenerateRequest,
    GenerateResponse,
    HistoricalExample,
    ModelProvider,
    PlannerMetadataRequirement,
    RetrievalTrace,
    SemanticContract,
    TokenUsage,
)
from semantic_text2sql.postgres import PostgresRegistry
from semantic_text2sql.profiling import ProfileStore
from semantic_text2sql.strategy import route_question
from semantic_text2sql.validator import validate_sql
from semantic_text2sql.value_grounding import ground_question_values

logger = logging.getLogger(__name__)
ContextMode = Literal["model1", "retrieval"]


@dataclass(frozen=True)
class QuestionExecution:
    generated: GenerateResponse
    approved_tables: list[str]
    semantic_contract: SemanticContract
    planner_usage: TokenUsage
    routing_ms: int
    planning_ms: int
    generation_ms: int


class TextToSQLService:
    """One shared retrieval-to-execution path for chat and analytical tools."""

    def __init__(
        self,
        *,
        database: DatabaseRegistry,
        postgres: PostgresRegistry | None = None,
        profiles: ProfileStore,
        glossaries: GlossaryStore,
        history: HistoricalQueryStore,
        retriever: HybridSchemaRetriever,
        agent: TextToSQLAgent,
        context_completers: Mapping[ModelProvider, ContextPlannerCompleter],
    ) -> None:
        self.database = database
        self.postgres = postgres
        self.profiles = profiles
        self.glossaries = glossaries
        self.history = history
        self.retriever = retriever
        self.agent = agent
        self.context_completers = context_completers

    async def execute_question(
        self,
        *,
        question: str,
        db_id: str,
        provider: ModelProvider,
        model: str,
        evidence: str | None = None,
        context_mode: ContextMode = "retrieval",
        context_provider: ModelProvider | None = None,
        context_model: str | None = None,
        execute: bool = True,
        max_rows: int = 100,
        max_attempts: int = 3,
        semantic_contract: SemanticContract | None = None,
        previous_intent: SemanticContract | None = None,
        previous_sql: str | None = None,
        previous_approved_tables: list[str] | None = None,
        optimization_required: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> QuestionExecution:
        routing_started = perf_counter()
        if progress:
            progress("retrieval")
        database = self._database(db_id)
        schema = database.inspect(db_id)
        dialect = schema.dialect
        database_profile = self.profiles.load(dialect, db_id)
        contract = semantic_contract or SemanticContract()
        retrieval_trace: RetrievalTrace | None = None

        if optimization_required and previous_approved_tables:
            proposed_tables = previous_approved_tables
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
            context_request = fallback_context_request(contract)
        else:
            historical_schema_evidence = self.history.schema_evidence(
                question,
                db_id,
                schema,
                top_k=3,
                min_score=float(os.environ.get("TEXT2SQL_HISTORY_ML_MIN_SCORE", "0.65")),
            )
            planner_schema, retrieval_selection, retrieval_trace = self.retriever.retrieve(
                question,
                evidence,
                schema,
                database_profile,
                self.glossaries.load(db_id),
                historical_schema_evidence,
            )
            proposed_tables = retrieval_selection.tables
            metadata = metadata_requests(question, planner_schema, database_profile)
            context_request = ContextRequest(
                tables=retrieval_selection.tables,
                columns=retrieval_selection.columns,
                business_concepts=sorted(
                    self.glossaries.relevant_concept_ids(db_id, question)
                ),
                metadata_requirements=[
                    PlannerMetadataRequirement(kind=cast(Any, kind), targets=[target])
                    for kind, target in metadata
                ],
            )
        routing_ms = round((perf_counter() - routing_started) * 1_000)

        planning_started = perf_counter()
        if progress:
            progress("context_selection" if context_mode == "model1" else "grounding")
        contract = apply_structural_formulas(
            contract, self.glossaries.structural_formulas(db_id, question)
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
        if context_mode == "model1" and not optimization_required:
            planner_provider = context_provider or provider
            planner_model = (
                context_model or os.environ.get("TEXT2SQL_CONTEXT_MODEL") or model
            )
            try:
                selection, planner_usage = await plan_context_detailed(
                    self.context_completers[planner_provider],
                    planner_model,
                    question,
                    evidence,
                    planner_schema,
                    self.glossaries.load(db_id),
                    previous_intent,
                    candidate_relationships,
                )
                context_request = selection_to_context_request(selection)
                planner_call_used = True
            except (KeyError, ModelError, ValueError):
                pass

        approved_concepts = self.glossaries.relevant_concept_ids(db_id, question)
        if progress:
            progress("grounding")
        context_request = verify_context_request(
            context_request,
            schema,
            contract,
            database_profile,
            approved_concepts,
        )
        logger.info("verified_context_request=%s", context_request.model_dump_json())
        value_grounding = ground_question_values(
            question,
            context_request,
            database_profile,
            self.glossaries.load(db_id),
        )
        if value_grounding.issue is not None:
            planning_ms = round((perf_counter() - planning_started) * 1_000)
            generated = GenerateResponse(
                db_id=db_id,
                question=question,
                provider=provider,
                model=model,
                dialect=dialect,
                strategy=route_question(question),
                semantic_contract=contract,
                sql=None,
                accepted=False,
                termination_reason="grounding_clarification",
                grounding_issue=value_grounding.issue,
                context_request=context_request,
            )
            return QuestionExecution(
                generated=generated,
                approved_tables=context_request.tables,
                semantic_contract=contract,
                planner_usage=planner_usage,
                routing_ms=routing_ms,
                planning_ms=planning_ms,
                generation_ms=0,
            )
        if value_grounding.evidence:
            evidence = "\n".join(
                value for value in (evidence or "", *value_grounding.evidence) if value
            )
        contract = reconcile_context_contract(contract, context_request)
        business_context = (
            self.glossaries.retrieve(db_id, question, top_k=5)
            if context_request.business_concepts
            else None
        )
        historical_examples: list[HistoricalExample] = []
        history_enabled = os.environ.get("TEXT2SQL_HISTORY_ENABLED", "false").casefold() == "true"
        if history_enabled:
            historical_examples = [
                item
                for item in self.history.search(
                    question,
                    db_id,
                    top_k=2,
                    min_score=float(os.environ.get("TEXT2SQL_HISTORY_MIN_SCORE", "0.85")),
                    bm25_pool=20,
                    semantic_pool=5,
                    candidate_tables=set(context_request.tables),
                )
                if validate_sql(item.sql, schema, dialect=dialect).valid
            ][:2]
        if context_request.tables:
            proposed_tables = context_request.tables
        planning_ms = round((perf_counter() - planning_started) * 1_000)

        generation_started = perf_counter()
        if progress:
            progress("generation")
        generated = await self.agent.generate(
            GenerateRequest(
                db_id=db_id,
                question=question,
                evidence=evidence,
                provider=provider,
                model=os.environ.get("TEXT2SQL_SQL_MODEL") or model,
                dialect=dialect,
                execute=execute,
                max_rows=max_rows,
                max_attempts=max_attempts,
                approved_tables=proposed_tables,
                semantic_contract=contract,
                business_context=business_context,
                previous_sql=previous_sql,
                optimization_required=optimization_required,
                context_request=context_request,
                planner_call_used=planner_call_used,
                planner_token_usage=planner_usage,
                historical_examples=historical_examples,
            ),
            progress=progress,
        )
        generation_ms = round((perf_counter() - generation_started) * 1_000)
        generated = generated.model_copy(
            update={
                "telemetry": generated.telemetry.model_copy(
                    update={
                        "model1_latency_ms": planning_ms,
                        "ab_context_mode": context_mode,
                        "retrieval": retrieval_trace,
                        "selected_table_count": len(context_request.tables),
                        "selected_column_count": sum(
                            len(columns) for columns in context_request.columns.values()
                        ),
                        "metadata_request_count": len(context_request.metadata_requirements),
                        "historical_attempted": history_enabled,
                        "historical_candidates_retrieved": (
                            len(historical_examples) if history_enabled else None
                        ),
                        "historical_examples_admitted": len(historical_examples),
                        "historical_similarity_scores": [
                            item.score for item in historical_examples
                        ],
                    }
                )
            }
        )
        return QuestionExecution(
            generated=generated,
            approved_tables=proposed_tables,
            semantic_contract=contract,
            planner_usage=planner_usage,
            routing_ms=routing_ms,
            planning_ms=planning_ms,
            generation_ms=generation_ms,
        )

    def _database(self, db_id: str) -> DatabaseRegistry | PostgresRegistry:
        """Resolve an allowlisted ID without exposing a caller-supplied DSN."""

        if db_id in self.database.list_ids():
            return self.database
        if self.postgres is not None and db_id in self.postgres.configured_ids():
            return self.postgres
        raise DatabaseError("DATABASE_NOT_FOUND", f"Database {db_id} was not found or configured.")
