"""Bounded generation, deterministic validation, and validator-assisted repair."""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from collections.abc import Callable, Iterable
from statistics import median
from time import perf_counter_ns

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError
from sqlglot.optimizer.simplify import simplify

from semantic_text2sql.context import (
    MAX_CONTEXT_EXPANSIONS,
    build_context_plan,
    estimate_tokens,
    failure_context_category,
    model_context_payload,
    reactive_context,
    record_context_expansion,
    render_context,
)
from semantic_text2sql.database import DatabaseError, DatabaseRegistry
from semantic_text2sql.linker import select_schema
from semantic_text2sql.llm import ModelError, SQLModel
from semantic_text2sql.models import (
    Attempt,
    CheckRequest,
    CheckResponse,
    GenerateRequest,
    GenerateResponse,
    OptimizationEvidence,
    PipelineTelemetry,
    RecoveryTrace,
    SemanticContract,
    TokenUsage,
    ValidationResult,
)
from semantic_text2sql.postgres import PostgresRegistry
from semantic_text2sql.profiling import ProfileStore
from semantic_text2sql.recovery import RecoveryCoordinator, RecoveryTools, recovery_feedback
from semantic_text2sql.strategy import route_question
from semantic_text2sql.validator import (
    clean_model_sql,
    normalize_sql,
    repair_context,
    validate_sql,
)


class TextToSQLAgent:
    def __init__(
        self,
        registry: DatabaseRegistry,
        model: SQLModel,
        postgres: PostgresRegistry | None = None,
        profiles: ProfileStore | None = None,
    ) -> None:
        self.registry = registry
        self.model = model
        self.postgres = postgres
        self.profiles = profiles

    def _database(self, dialect: str) -> DatabaseRegistry | PostgresRegistry:
        if dialect == "sqlite":
            return self.registry
        if dialect == "postgres" and self.postgres is not None:
            return self.postgres
        raise DatabaseError("DATABASE_DIALECT_UNAVAILABLE", f"{dialect} is not configured.")

    def check(self, request: CheckRequest) -> CheckResponse:
        database = self._database(request.dialect)
        schema = database.inspect(request.db_id)
        sql = clean_model_sql(request.sql)
        validation = validate_sql(sql, schema, dialect=request.dialect)
        if not validation.valid or not request.execute:
            return CheckResponse(
                db_id=request.db_id,
                dialect=request.dialect,
                validation=validation,
            )
        columns, rows, truncated = database.execute(request.db_id, sql, max_rows=request.max_rows)
        return CheckResponse(
            db_id=request.db_id,
            dialect=request.dialect,
            validation=validation,
            rows=rows,
            columns=columns,
            row_count=len(rows),
            truncated=truncated,
        )

    async def generate(
        self,
        request: GenerateRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> GenerateResponse:
        strategy = route_question(request.question)
        semantic_contract = request.semantic_contract or SemanticContract()
        try:
            database = self._database(request.dialect)
            schema = database.inspect(request.db_id)
        except DatabaseError:
            return GenerateResponse(
                db_id=request.db_id,
                question=request.question,
                provider=request.provider,
                model=request.model,
                dialect=request.dialect,
                strategy=strategy,
                sql=None,
                accepted=False,
                termination_reason="database_error",
            )
        generation_schema = schema
        approved = set(request.approved_tables or [])
        if approved:
            live_tables = {table.name for table in schema.tables}
            if not approved <= live_tables:
                return GenerateResponse(
                    db_id=request.db_id,
                    question=request.question,
                    provider=request.provider,
                    model=request.model,
                    dialect=request.dialect,
                    strategy=strategy,
                    sql=None,
                    accepted=False,
                    termination_reason="database_error",
                )
            generation_schema = schema.model_copy(
                update={
                    "tables": [table for table in schema.tables if table.name in approved],
                    "relationships": [
                        item
                        for item in schema.relationships
                        if item.from_table in approved and item.to_table in approved
                    ],
                }
            )
        profile = self.profiles.load(request.dialect, request.db_id) if self.profiles else None
        semantic_scope = " ".join(
            [
                *semantic_contract.outputs,
                *semantic_contract.entities,
                *semantic_contract.proposed_tables,
                *semantic_contract.required_columns,
            ]
        )
        selection_evidence = "\n".join(
            value for value in (request.evidence or "", semantic_scope) if value
        )
        retrieved_schema, schema_selection = select_schema(
            request.question,
            selection_evidence or None,
            generation_schema,
            profile,
            required_columns=semantic_contract.required_columns,
            required_tables=semantic_contract.proposed_tables,
        )
        context_plan = build_context_plan(
            retrieved_schema,
            profile,
            semantic_contract,
            request.question,
            request.evidence,
            request.context_request,
        )
        if not context_plan.coverage_complete:
            return GenerateResponse(
                db_id=request.db_id,
                question=request.question,
                provider=request.provider,
                model=request.model,
                dialect=request.dialect,
                strategy=strategy,
                schema_selection=schema_selection,
                semantic_contract=semantic_contract,
                sql=None,
                accepted=False,
                termination_reason="context_error",
                model_error=(
                    "Grounded context is incomplete: "
                    + ", ".join(context_plan.missing_requirements)
                )[:500],
                context_plan=context_plan,
                context_request=request.context_request,
            )
        profile_context = render_context(
            context_plan,
            retrieved_schema,
            profile,
            semantic_contract,
            request.question,
            request.evidence,
            request.business_context,
            request.historical_examples,
        )
        final_model_context = model_context_payload(
            context_plan,
            profile,
            semantic_contract,
            request.business_context,
            request.question,
            request.dialect,
            request.historical_examples,
            retrieved_schema,
        )
        authorized_tables = {table.name.casefold() for table in retrieved_schema.tables}
        available_context_tokens = (
            estimate_tokens(schema.model_dump_json())
            + estimate_tokens(profile.model_dump_json() if profile else "")
            + estimate_tokens(semantic_contract.model_dump_json())
            + estimate_tokens(request.business_context or "")
        )
        retrieved_categories: list[str] = []
        attempts: list[Attempt] = []
        seen: set[str] = set()
        rejected_shapes: list[str] = []
        previous_sql = request.previous_sql
        baseline_validation: ValidationResult | None = None
        if request.optimization_required and request.previous_sql:
            baseline_validation = validate_sql(
                request.previous_sql,
                schema,
                dialect=request.dialect,
                allowed_tables=authorized_tables,
            )
        feedback = None
        if request.optimization_required:
            baseline_plan = (
                baseline_validation.explain_plan if baseline_validation is not None else []
            )
            feedback = (
                "Optimize the previously accepted SQL without changing its columns, row semantics, "
                "filters, aggregation, ordering, or result values. The replacement must be "
                "result-equivalent and measurably faster.\nOriginal EXPLAIN plan:\n"
                + ("\n".join(baseline_plan) or "Unavailable")
            )
        final_sql = None
        optimization_evidence: OptimizationEvidence | None = None
        model_failed = False
        model_error_detail: str | None = None
        executed_columns: list[str] = []
        executed_rows: list[list[object]] = []
        executed_truncated = False
        recovery_trace: RecoveryTrace | None = None

        # In explicit optimization mode, try a deterministic SQLGlot rewrite before
        # spending another model call. Formatting-only changes are ignored. The rewrite
        # is accepted only through the same equivalence and measured-performance gates
        # used for model-generated optimization candidates.
        if (
            request.optimization_required
            and request.previous_sql
            and baseline_validation is not None
            and baseline_validation.valid
        ):
            sqlglot_candidate = _sqlglot_optimization_candidate(
                request.previous_sql, dialect=request.dialect
            )
            if sqlglot_candidate and normalize_sql(
                sqlglot_candidate, dialect=request.dialect
            ) != normalize_sql(request.previous_sql, dialect=request.dialect):
                validation = validate_sql(
                    sqlglot_candidate,
                    schema,
                    dialect=request.dialect,
                    allowed_tables=authorized_tables,
                )
                if validation.valid:
                    try:
                        validation = await asyncio.to_thread(
                            _validate_result_equivalence,
                            database,
                            request.db_id,
                            request.previous_sql,
                            sqlglot_candidate,
                            validation,
                            max_rows=max(request.max_rows, 1_000),
                        )
                        if validation.valid:
                            optimization_evidence = (
                                await asyncio.to_thread(
                                    _benchmark_optimization,
                                    database,
                                    request.db_id,
                                    request.previous_sql,
                                    sqlglot_candidate,
                                    baseline_validation.explain_plan,
                                    validation.explain_plan,
                                    max_rows=max(request.max_rows, 1_000),
                                )
                            ).model_copy(update={"optimizer": "sqlglot"})
                            if optimization_evidence.status == "optimized":
                                final_sql = sqlglot_candidate
                                if request.execute:
                                    (
                                        executed_columns,
                                        executed_rows,
                                        executed_truncated,
                                    ) = await asyncio.to_thread(
                                        database.execute,
                                        request.db_id,
                                        final_sql,
                                        max_rows=request.max_rows,
                                    )
                    except DatabaseError:
                        # A deterministic candidate is optional. The bounded Model 2
                        # optimizer remains the fallback when it cannot be evaluated.
                        pass

        for number in range(1, request.max_attempts + 1) if final_sql is None else ():
            if progress:
                progress("generation")
            try:
                generated = await self.model.generate(
                    model=request.model,
                    provider=request.provider,
                    question=request.question,
                    evidence=request.evidence,
                    schema=retrieved_schema,
                    strategy=strategy,
                    dialect=request.dialect,
                    profile_context=profile_context,
                    previous_sql=previous_sql,
                    feedback=feedback,
                    rejected_shapes=rejected_shapes,
                    generation_style=request.generation_style,
                )
                raw_sql, latency = generated[:2]
                usage = generated[2] if len(generated) == 3 else TokenUsage()
            except ModelError as error:
                model_failed = True
                model_error_detail = str(error)[:500]
                break
            sql = clean_model_sql(raw_sql)
            normalized = normalize_sql(sql, dialect=request.dialect)
            fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
            if progress:
                progress("validation")
            validation = validate_sql(
                sql,
                schema,
                dialect=request.dialect,
                allowed_tables=authorized_tables,
            )
            if validation.valid:
                validation = validation.model_copy(
                    update={
                        "code": "SQL_SAFETY_VALID",
                        "message": (
                            "SQL passed syntax and read-only safety validation. "
                            "This does not prove executability or business correctness."
                        ),
                    }
                )
            if validation.valid and request.optimization_required and request.previous_sql:
                validation = await asyncio.to_thread(
                    _validate_result_equivalence,
                    database,
                    request.db_id,
                    request.previous_sql,
                    sql,
                    validation,
                    max_rows=max(request.max_rows, 1_000),
                )
            if validation.valid and request.optimization_required and request.previous_sql:
                measured_optimization = await asyncio.to_thread(
                    _benchmark_optimization,
                    database,
                    request.db_id,
                    request.previous_sql,
                    sql,
                    baseline_validation.explain_plan if baseline_validation else [],
                    validation.explain_plan,
                    max_rows=max(request.max_rows, 1_000),
                )
                optimization_evidence = measured_optimization
                if measured_optimization.status != "optimized":
                    validation = validation.model_copy(
                        update={
                            "valid": False,
                            "code": "OPTIMIZATION_NOT_FASTER",
                            "message": (
                                "The candidate was result-equivalent but was not measurably "
                                "faster than the accepted SQL."
                            ),
                        }
                    )

            if validation.valid and not request.optimization_required:
                if progress:
                    progress("optimization")
                sqlglot_candidate = _sqlglot_optimization_candidate(sql, dialect=request.dialect)
                if not sqlglot_candidate or normalize_sql(
                    sqlglot_candidate, dialect=request.dialect
                ) == normalize_sql(sql, dialect=request.dialect):
                    optimization_evidence = OptimizationEvidence(
                        optimizer="sqlglot",
                        status="no_structural_change",
                        result_equivalent=True,
                        selected_sql="baseline",
                    )
                elif request.execute:
                    candidate_validation = validate_sql(
                        sqlglot_candidate,
                        schema,
                        dialect=request.dialect,
                        allowed_tables=authorized_tables,
                    )
                    try:
                        if candidate_validation.valid:
                            candidate_validation = await asyncio.to_thread(
                                _validate_result_equivalence,
                                database,
                                request.db_id,
                                sql,
                                sqlglot_candidate,
                                candidate_validation,
                                max_rows=max(request.max_rows, 1_000),
                            )
                        if candidate_validation.valid:
                            candidate_evidence = (
                                await asyncio.to_thread(
                                    _benchmark_optimization,
                                    database,
                                    request.db_id,
                                    sql,
                                    sqlglot_candidate,
                                    validation.explain_plan,
                                    candidate_validation.explain_plan,
                                    max_rows=max(request.max_rows, 1_000),
                                )
                            ).model_copy(update={"optimizer": "sqlglot"})
                            optimization_evidence = candidate_evidence
                            if candidate_evidence.status == "optimized":
                                sql = sqlglot_candidate
                                normalized = normalize_sql(sql, dialect=request.dialect)
                                fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
                                validation = candidate_validation
                        else:
                            optimization_evidence = OptimizationEvidence(
                                optimizer="sqlglot",
                                status="rejected",
                                result_equivalent=False,
                                selected_sql="baseline",
                            )
                    except DatabaseError:
                        optimization_evidence = OptimizationEvidence(
                            optimizer="sqlglot",
                            status="rejected",
                            result_equivalent=False,
                            selected_sql="baseline",
                        )
                else:
                    optimization_evidence = OptimizationEvidence(
                        optimizer="sqlglot",
                        status="rejected",
                        result_equivalent=False,
                        selected_sql="baseline",
                    )
            if validation.valid and request.execute:
                if progress:
                    progress("execution")
                try:
                    executed_columns, executed_rows, executed_truncated = await asyncio.to_thread(
                        database.execute, request.db_id, sql, max_rows=request.max_rows
                    )
                except DatabaseError as error:
                    validation = validation.model_copy(
                        update={
                            "valid": False,
                            "code": "DATABASE_ERROR",
                            "message": f"Read-only execution failed: {error}",
                        }
                    )
            attempts.append(
                Attempt(
                    number=number,
                    sql=sql,
                    normalized_sql=normalized,
                    validation=validation,
                    latency_ms=latency,
                    token_usage=usage,
                )
            )
            if validation.valid:
                final_sql = sql
                break
            if validation.code == "OPTIMIZATION_NOT_FASTER" and optimization_evidence is not None:
                break
            repeated = fingerprint in seen
            seen.add(fingerprint)
            rejected_shapes.append(fingerprint)
            previous_sql = sql
            feedback = repair_context(validation)
            if number == 1 and request.max_attempts > 1:
                if progress:
                    progress("recovery")
                recovery_trace = await RecoveryCoordinator(
                    RecoveryTools(database, request.db_id, schema, profile)
                ).ainvestigate(
                    question=request.question,
                    failed_sql=sql,
                    failure_code=validation.code,
                    failure_message=validation.message,
                    allowed_tables=[table.name for table in retrieved_schema.tables],
                    completer=self.model,
                    provider=request.provider,
                    model=request.model,
                )
                if recovery_trace.agent_action in {"INFORM", "ESCALATE"}:
                    break
                if recovery_trace.failure_category != "provider":
                    feedback += "\n\n" + recovery_feedback(recovery_trace)
            category = failure_context_category(validation.code)
            if (
                category
                and category not in retrieved_categories
                and len(retrieved_categories) < MAX_CONTEXT_EXPANSIONS
            ):
                profile_context += "\n\n" + reactive_context(category, retrieved_schema, profile)
                retrieved_categories.append(category)
                context_plan = record_context_expansion(context_plan, category)
            if repeated:
                feedback += "\nThe previous query repeated a rejected normalized structure."

        if final_sql is None:
            if (
                request.optimization_required
                and request.previous_sql
                and optimization_evidence is not None
                and optimization_evidence.status == "equivalent_not_faster"
            ):
                baseline_columns, baseline_rows, baseline_truncated = await asyncio.to_thread(
                    database.execute,
                    request.db_id,
                    request.previous_sql,
                    max_rows=request.max_rows,
                )
                return GenerateResponse(
                    db_id=request.db_id,
                    question=request.question,
                    provider=request.provider,
                    model=request.model,
                    dialect=request.dialect,
                    strategy=strategy,
                    schema_selection=schema_selection,
                    semantic_contract=semantic_contract,
                    sql=request.previous_sql,
                    accepted=True,
                    execution_status="ACCEPTED",
                    attempts=attempts,
                    rows=baseline_rows,
                    columns=baseline_columns,
                    row_count=len(baseline_rows),
                    truncated=baseline_truncated,
                    termination_reason="accepted",
                    token_usage=_sum_usage(attempts),
                    optimization=optimization_evidence,
                    context_plan=context_plan,
                    model_context=final_model_context,
                    context_request=request.context_request,
                    telemetry=_telemetry(
                        available_context_tokens,
                        profile_context,
                        request.planner_call_used,
                        request.planner_token_usage,
                        retrieved_categories,
                        attempts,
                    ),
                )
            if (
                request.optimization_required
                and request.previous_sql
                and baseline_validation is not None
                and baseline_validation.valid
            ):
                baseline_columns, baseline_rows, baseline_truncated = await asyncio.to_thread(
                    database.execute,
                    request.db_id,
                    request.previous_sql,
                    max_rows=request.max_rows,
                )
                return GenerateResponse(
                    db_id=request.db_id,
                    question=request.question,
                    provider=request.provider,
                    model=request.model,
                    dialect=request.dialect,
                    strategy=strategy,
                    schema_selection=schema_selection,
                    semantic_contract=semantic_contract,
                    sql=request.previous_sql,
                    accepted=True,
                    execution_status="ACCEPTED",
                    attempts=attempts,
                    rows=baseline_rows,
                    columns=baseline_columns,
                    row_count=len(baseline_rows),
                    truncated=baseline_truncated,
                    termination_reason="accepted",
                    token_usage=_sum_usage(attempts),
                    optimization=OptimizationEvidence(
                        status="rejected",
                        baseline_explain=baseline_validation.explain_plan,
                        selected_sql="baseline",
                    ),
                    context_plan=context_plan,
                    model_context=final_model_context,
                    context_request=request.context_request,
                    telemetry=_telemetry(
                        available_context_tokens,
                        profile_context,
                        request.planner_call_used,
                        request.planner_token_usage,
                        retrieved_categories,
                        attempts,
                    ),
                )
            return GenerateResponse(
                db_id=request.db_id,
                question=request.question,
                provider=request.provider,
                model=request.model,
                dialect=request.dialect,
                strategy=strategy,
                schema_selection=schema_selection,
                semantic_contract=semantic_contract,
                sql=None,
                accepted=False,
                attempts=attempts,
                termination_reason="model_error" if model_failed else "attempt_limit",
                model_error=model_error_detail,
                token_usage=_sum_usage(attempts),
                context_plan=context_plan,
                model_context=final_model_context,
                context_request=request.context_request,
                telemetry=_telemetry(
                    available_context_tokens,
                    profile_context,
                    request.planner_call_used,
                    request.planner_token_usage,
                    retrieved_categories,
                    attempts,
                ),
                recovery=recovery_trace,
            )
        rows = executed_rows if request.execute else []
        columns = executed_columns if request.execute else []
        truncated = executed_truncated if request.execute else False
        return GenerateResponse(
            db_id=request.db_id,
            question=request.question,
            provider=request.provider,
            model=request.model,
            dialect=request.dialect,
            strategy=strategy,
            schema_selection=schema_selection,
            semantic_contract=semantic_contract,
            sql=final_sql,
            accepted=True,
            execution_status="ACCEPTED" if request.execute else "SAFETY_VALIDATED",
            attempts=attempts,
            rows=rows,
            columns=columns,
            row_count=len(rows),
            truncated=truncated,
            termination_reason="accepted",
            token_usage=_sum_usage(attempts),
            optimization=optimization_evidence,
            context_plan=context_plan,
            model_context=final_model_context,
            context_request=request.context_request,
            telemetry=_telemetry(
                available_context_tokens,
                profile_context,
                request.planner_call_used,
                request.planner_token_usage,
                retrieved_categories,
                attempts,
            ),
            recovery=recovery_trace,
        )


def _sum_usage(attempts: list[Attempt]) -> TokenUsage:
    return TokenUsage(
        input_tokens=_sum_known(item.token_usage.input_tokens for item in attempts),
        output_tokens=_sum_known(item.token_usage.output_tokens for item in attempts),
        cache_read_tokens=_sum_known(item.token_usage.cache_read_tokens for item in attempts),
        cache_creation_tokens=_sum_known(
            item.token_usage.cache_creation_tokens for item in attempts
        ),
    )


def _sum_known(values: Iterable[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _telemetry(
    available_context_tokens: int,
    context: str,
    planner_call_used: bool,
    planner_usage: TokenUsage,
    categories: list[str],
    attempts: list[Attempt],
) -> PipelineTelemetry:
    selected = estimate_tokens(context)
    available = max(available_context_tokens, selected)
    pruned = max(0, available - selected)
    return PipelineTelemetry(
        available_context_tokens=available,
        selected_context_tokens=selected,
        selected_model_context_tokens=selected,
        pruned_tokens=pruned,
        pruning_percentage=(pruned / available * 100) if available else 0,
        context_expansions=len(categories),
        retrieved_context_categories=categories,
        generation_attempts=len(attempts),
        planner_call_used=planner_call_used,
        planner_call=planner_usage if planner_call_used else TokenUsage(),
        generation_call=(attempts[0].token_usage if attempts else TokenUsage()),
        repair_calls=[item.token_usage for item in attempts[1:]],
        generation_system_tokens=None,
        generation_context_tokens_estimated=selected,
        model2_latency_ms=sum(item.latency_ms for item in attempts) if attempts else None,
    )


def _validate_result_equivalence(
    database: DatabaseRegistry | PostgresRegistry,
    db_id: str,
    baseline_sql: str,
    candidate_sql: str,
    validation: ValidationResult,
    *,
    max_rows: int,
) -> ValidationResult:
    baseline_columns, baseline_rows, baseline_truncated = database.execute(
        db_id, baseline_sql, max_rows=max_rows
    )
    candidate_columns, candidate_rows, candidate_truncated = database.execute(
        db_id, candidate_sql, max_rows=max_rows
    )
    if baseline_truncated or candidate_truncated:
        return validation.model_copy(
            update={
                "valid": False,
                "code": "OPTIMIZATION_EQUIVALENCE_UNPROVEN",
                "message": (
                    "Result equivalence could not be proven because an output was truncated."
                ),
            }
        )
    ordered = _has_meaningful_order(baseline_sql, dialect=validation_dialect(database))
    same_rows = (
        _normalized_rows(baseline_rows) == _normalized_rows(candidate_rows)
        if ordered
        else Counter(_normalized_rows(baseline_rows)) == Counter(_normalized_rows(candidate_rows))
    )
    if baseline_columns != candidate_columns or not same_rows:
        return validation.model_copy(
            update={
                "valid": False,
                "code": "OPTIMIZATION_CHANGED_RESULT",
                "message": (
                    "The optimized query changed the output columns or result values. Preserve "
                    "the exact behavior of the previously accepted SQL."
                ),
            }
        )
    return validation


def _sqlglot_optimization_candidate(sql: str, *, dialect: str) -> str | None:
    """Return a conservative deterministic rewrite, or ``None`` when parsing fails.

    SQLGlot's full optimizer performs identifier qualification and dialect-sensitive
    rewrites that are inappropriate without a complete catalog. ``simplify`` provides
    the safe first stage here: constant/boolean simplification while preserving names,
    joins, projections, grouping, and query grain.
    """
    try:
        expression = parse_one(sql, read=dialect)
        return str(simplify(expression).sql(dialect=dialect, pretty=True))
    except (ParseError, ValueError):
        return None


def _benchmark_optimization(
    database: DatabaseRegistry | PostgresRegistry,
    db_id: str,
    baseline_sql: str,
    candidate_sql: str,
    baseline_explain: list[str],
    candidate_explain: list[str],
    *,
    max_rows: int,
    repetitions: int = 3,
) -> OptimizationEvidence:
    """Benchmark equivalent queries alternately; retain baseline unless improvement is material."""
    database.execute(db_id, baseline_sql, max_rows=max_rows)
    database.execute(db_id, candidate_sql, max_rows=max_rows)
    baseline_times: list[float] = []
    candidate_times: list[float] = []

    def measure(sql: str) -> float:
        started = perf_counter_ns()
        database.execute(db_id, sql, max_rows=max_rows)
        return (perf_counter_ns() - started) / 1_000_000

    for index in range(repetitions):
        if index % 2 == 0:
            baseline_times.append(measure(baseline_sql))
            candidate_times.append(measure(candidate_sql))
        else:
            candidate_times.append(measure(candidate_sql))
            baseline_times.append(measure(baseline_sql))
    baseline_median = median(baseline_times)
    candidate_median = median(candidate_times)
    improvement_ms = baseline_median - candidate_median
    improvement_percent = improvement_ms / baseline_median * 100 if baseline_median > 0 else 0.0
    measurably_faster = improvement_ms >= 0.1 and improvement_percent >= 5.0
    return OptimizationEvidence(
        status="optimized" if measurably_faster else "equivalent_not_faster",
        baseline_explain=baseline_explain,
        candidate_explain=candidate_explain,
        baseline_timings_ms=baseline_times,
        candidate_timings_ms=candidate_times,
        baseline_median_ms=baseline_median,
        candidate_median_ms=candidate_median,
        improvement_percent=improvement_percent,
        result_equivalent=True,
        selected_sql="candidate" if measurably_faster else "baseline",
    )


def _has_meaningful_order(sql: str, *, dialect: str) -> bool:
    try:
        return parse_one(sql, read=dialect).find(exp.Order) is not None
    except ParseError:
        return True


def validation_dialect(database: DatabaseRegistry | PostgresRegistry) -> str:
    return "postgres" if isinstance(database, PostgresRegistry) else "sqlite"


def _normalized_rows(rows: list[list[object]]) -> list[tuple[object, ...]]:
    return [tuple(_normalized_value(value) for value in row) for row in rows]


def _normalized_value(value: object) -> object:
    if isinstance(value, float):
        return round(value, 9)
    return value
