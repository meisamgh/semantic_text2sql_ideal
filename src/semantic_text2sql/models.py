"""Strict public and internal data contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from semantic_text2sql.sql_formatting import format_sql_for_display

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
GROQ_QWEN_MODEL = "qwen/qwen3.6-27b"
SOTA_GPT_MODEL = "gpt-5.5"
ModelProvider = Literal["ollama", "agentrouter", "groq", "justdowork", "sota"]
"""Single source of truth for the locally installed Ollama generation model.

Sized to fit in system RAM: a model whose weights exceed memory pages to disk and
never completes a generation, which ``/api/models`` reports as unavailable.
"""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColumnInfo(StrictModel):
    name: str
    data_type: str
    primary_key: bool = False
    nullable: bool | None = None
    description: str | None = None
    semantic_type: str | None = None
    aliases: list[str] = Field(default_factory=list)


class ForeignKeyInfo(StrictModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    constraint_id: str | None = None
    ordinal: int = 0


class TableInfo(StrictModel):
    name: str
    create_sql: str
    columns: list[ColumnInfo]


class SchemaInfo(StrictModel):
    db_id: str
    dialect: Literal["sqlite", "postgres"] = "sqlite"
    tables: list[TableInfo]
    relationships: list[ForeignKeyInfo] = Field(default_factory=list)


class SuspectedSentinel(StrictModel):
    value: str
    confidence: float = Field(ge=0, le=1)
    reason: str


class ValueFrequency(StrictModel):
    value: str
    count: int = Field(ge=0)


class ColumnProfile(StrictModel):
    table: str
    column: str
    database_type: str
    semantic_type: str
    nullable: bool | None = None
    primary_key: bool = False
    foreign_key_target: str | None = None
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    row_count: int = Field(ge=0)
    null_count: int = Field(ge=0)
    null_ratio: float = Field(ge=0, le=1)
    empty_string_count: int = Field(default=0, ge=0)
    suspected_sentinels: list[SuspectedSentinel] = Field(default_factory=list)
    distinct_count: int | None = Field(default=None, ge=0)
    minimum: str | None = None
    maximum: str | None = None
    range_exact: bool = False
    median: float | None = None
    quantiles: dict[str, float] = Field(default_factory=dict)
    observed_format: str | None = None
    timezone: str | None = None
    top_values: list[ValueFrequency] = Field(default_factory=list)
    allowed_values: list[str] = Field(default_factory=list)
    text_length_min: int | None = Field(default=None, ge=0)
    text_length_median: float | None = Field(default=None, ge=0)
    text_length_max: int | None = Field(default=None, ge=0)
    examples: list[str] = Field(default_factory=list)
    detected_pattern: str | None = None
    exact: bool = True


class DateCoverage(StrictModel):
    column: str
    minimum: str | None = None
    maximum: str | None = None
    observed_format: str | None = None
    exact: bool = False


class TableProfile(StrictModel):
    table: str
    summary: str
    grain: str
    metrics: dict[str, str] = Field(default_factory=dict)
    dimensions: dict[str, str] = Field(default_factory=dict)
    date_coverage: list[DateCoverage] = Field(default_factory=list)
    supported_terms: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)


class RelationshipProfile(StrictModel):
    parent_table: str
    parent_column: str
    child_table: str
    child_column: str
    parent_columns: list[str] = Field(default_factory=list)
    child_columns: list[str] = Field(default_factory=list)
    type: Literal["ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY", "UNKNOWN"]
    parent_key_unique: bool
    child_key_unique: bool
    inferred: bool = False
    confidence: float = Field(default=1.0, ge=0, le=1)


class DatabaseProfile(StrictModel):
    db_id: str
    dialect: Literal["sqlite", "postgres"]
    profiled_at: str
    columns: list[ColumnProfile]
    tables: list[TableProfile] = Field(default_factory=list)
    relationships: list[RelationshipProfile] = Field(default_factory=list)


class SchemaSelection(StrictModel):
    tables: list[str]
    columns: dict[str, list[str]]
    table_scores: dict[str, float]
    column_scores: dict[str, dict[str, float]]


class StrategyHints(StrictModel):
    mode: Literal["exact", "fuzzy", "semantic", "hybrid"]
    fuzzy_terms: list[str] = Field(default_factory=list)
    semantic_terms: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ValidationResult(StrictModel):
    valid: bool
    code: str
    message: str
    tables: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    alias_map: dict[str, str] = Field(default_factory=dict)
    available_columns: dict[str, list[str]] = Field(default_factory=dict)
    explain_plan: list[str] = Field(default_factory=list)
    semantic_checks: list[str] = Field(default_factory=list)


class TokenUsage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_creation_tokens: int | None = Field(default=None, ge=0)

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


class AggregationStage(StrictModel):
    name: str
    function: Literal["count", "sum", "average", "min", "max"]
    input: str
    group_by: list[str] = Field(default_factory=list)
    output_grain: list[str] = Field(default_factory=list)


class MetricSelector(StrictModel):
    name: str
    function: Literal["min", "max", "argmin", "argmax"]
    metric: str
    partition_by: list[str] = Field(default_factory=list)
    scope: Literal["within_partition", "across_groups"] = "within_partition"
    tie_policy: Literal["return_all", "deterministic_one"] = "return_all"


class OutputOperation(StrictModel):
    name: str
    operator: Literal["subtract", "add", "divide", "multiply"]
    left: str
    right: str


class StructuralFormula(StrictModel):
    id: str
    operator: Literal["DIVIDE", "MULTIPLY", "ADD", "SUBTRACT"]
    arguments: list[str] = Field(min_length=2, max_length=2)
    zero_safe: bool = False
    source: Literal[
        "APPROVED_GLOSSARY",
        "TRUSTED_EVIDENCE",
        "DETERMINISTIC_RULE",
        "SEMANTIC_INTERPRETER",
    ]
    usage: Literal["FILTER_OPERAND", "OUTPUT", "DERIVED_METRIC"] = "DERIVED_METRIC"


class ContextRequirement(StrictModel):
    kind: Literal[
        "COLUMN_SCHEMA",
        "PHYSICAL_TYPE",
        "PHYSICAL_DATE_FORMAT",
        "ALLOWED_VALUES",
        "FORMULA_DEFINITION",
        "JOIN_PATH",
        "JOIN_CARDINALITY",
        "TABLE_GRAIN",
        "OUTPUT_GRAIN",
        "COLUMN_SEMANTICS",
        "NULL_REPRESENTATION",
        "TEXT_PATTERN",
        "TEXT_EXAMPLES",
        "NUMERIC_PROFILE",
        "COLUMN_DESCRIPTION",
    ]
    target: str
    required_by: str
    priority: Literal["HARD", "SOFT"] = "HARD"
    resolved: bool = False
    source: str | None = None


class SemanticFilter(StrictModel):
    operand: str
    operator: Literal["=", "!=", ">", ">=", "<", "<=", "IN", "BETWEEN"]
    value: str | int | float | list[str] | list[int] | list[float]


class PlannerMetadataRequirement(StrictModel):
    kind: Literal[
        "RELATIONSHIP",
        "JOIN_CARDINALITY",
        "TABLE_GRAIN",
        "DATE_FORMAT",
        "CATEGORY_VALUES",
        "FORMULA_DEFINITION",
        "PHYSICAL_TYPE",
        "NULL_SEMANTICS",
        "TEXT_EXAMPLES",
        "TEXT_PATTERN",
        "NUMERIC_PROFILE",
        "COLUMN_DESCRIPTION",
    ]
    targets: list[str] = Field(min_length=1, max_length=20)


class MetadataRequest(StrictModel):
    type: Literal[
        "RELATIONSHIP",
        "JOIN_CARDINALITY",
        "TABLE_GRAIN",
        "DATE_FORMAT",
        "CATEGORY_VALUES",
        "FORMULA_DEFINITION",
        "PHYSICAL_TYPE",
        "NULL_SEMANTICS",
        "TEXT_EXAMPLES",
        "TEXT_PATTERN",
        "NUMERIC_PROFILE",
        "COLUMN_DESCRIPTION",
    ]
    target: str


class PlannerAggregation(StrictModel):
    function: Literal["count", "sum", "average", "min", "max"]
    input: str
    output: str | None = None
    group_by: list[str] = Field(default_factory=list, max_length=20)


class PlannerRanking(StrictModel):
    metric: str
    direction: Literal["ASC", "DESC"]
    limit: int | None = Field(default=None, ge=1, le=10_000)


class ContextRequest(StrictModel):
    """Structured WHAT-only request emitted by the context-planner model."""

    outputs: list[str] = Field(default_factory=list, max_length=30)
    tables: list[str] = Field(default_factory=list, max_length=20)
    columns: dict[str, list[str]] = Field(default_factory=dict)
    filters: list[SemanticFilter] = Field(default_factory=list, max_length=30)
    measures: list[str] = Field(default_factory=list, max_length=20)
    aggregations: list[PlannerAggregation] = Field(default_factory=list, max_length=20)
    group_by: list[str] = Field(default_factory=list, max_length=20)
    ranking: list[PlannerRanking] = Field(default_factory=list, max_length=10)
    temporal_operations: list[str] = Field(default_factory=list, max_length=20)
    business_concepts: list[str] = Field(default_factory=list, max_length=20)
    metadata_requirements: list[PlannerMetadataRequirement] = Field(
        default_factory=list, max_length=40
    )


class ContextManifest(StrictModel):
    """Bounded Model 1 output; contains context choices, never analytical semantics."""

    selected_tables: list[str] = Field(default_factory=list, max_length=20)
    selected_columns: dict[str, list[str]] = Field(default_factory=dict)
    business_concepts: list[str] = Field(default_factory=list, max_length=20)
    metadata_requests: list[MetadataRequest] = Field(default_factory=list, max_length=40)


ContextSelection = ContextManifest


class ContextRelationship(StrictModel):
    left_table: str
    right_table: str
    left_column: str
    right_column: str
    state: Literal["VERIFIED_FK", "INFERRED_KEY_RELATIONSHIP", "UNRESOLVED"]
    join_cardinality: Literal["ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY", "UNKNOWN"]
    left_key_unique: bool | None = None
    right_key_unique: bool | None = None
    fanout_risk: bool
    filter_side: str | None = None
    recommended_strategy: Literal["JOIN", "EXISTS", "PRE_AGGREGATE", "UNRESOLVED"]


class VerifiedColumn(StrictModel):
    type: str
    observed_nulls: bool | None = None
    example_values: list[str] = Field(default_factory=list, max_length=5)
    format: str | None = None


class VerifiedTable(StrictModel):
    grain: str | None = None
    primary_key: list[str] = Field(default_factory=list)
    relationship_keys: list[str] = Field(default_factory=list)
    unique_keys: list[list[str]] = Field(default_factory=list)
    columns: dict[str, VerifiedColumn] = Field(default_factory=dict)


class VerifiedRelationship(StrictModel):
    left: str
    right: str
    state: Literal["VERIFIED_FK", "INFERRED_KEY_RELATIONSHIP", "UNRESOLVED"]
    cardinality: Literal["ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY", "UNKNOWN"]
    left_key_unique: bool | None = None
    right_key_unique: bool | None = None
    fanout_risk: bool


class VerifiedContext(StrictModel):
    question: str
    dialect: Literal["sqlite", "postgres"]
    tables: dict[str, VerifiedTable] = Field(default_factory=dict)
    relationships: list[VerifiedRelationship] = Field(default_factory=list)
    business_context: str | None = None
    optional_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SemanticContract(StrictModel):
    """High-confidence question requirements checked independently of the LLM."""

    aggregation: Literal["count", "sum", "average", "min", "max"] | None = None
    requires_grouping: bool = False
    requires_distinct: bool = False
    requires_ordering: bool = False
    limit: int | None = Field(default=None, ge=1, le=10_000)
    required_literals: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    filters: list[SemanticFilter] = Field(default_factory=list)
    advisory_outputs: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    proposed_tables: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)
    proposed_joins: list[str] = Field(default_factory=list)
    proposed_filters: list[str] = Field(default_factory=list)
    grain: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    interpreter_confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: list[str] = Field(default_factory=list)
    measures: list[str] = Field(default_factory=list)
    aggregation_stages: list[AggregationStage] = Field(default_factory=list)
    selectors: list[MetricSelector] = Field(default_factory=list)
    output_operations: list[OutputOperation] = Field(default_factory=list)
    named_outputs: list[str] = Field(default_factory=list)
    structural_formulas: list[StructuralFormula] = Field(default_factory=list)


class ContextTable(StrictModel):
    name: str
    role: Literal["MEASURE", "FILTER", "DIMENSION", "BRIDGE", "UNKNOWN"] = "UNKNOWN"
    grain: str | None = None
    primary_key: list[str] = Field(default_factory=list)
    unique_keys: list[list[str]] = Field(default_factory=list)
    reason: str


class ContextMetadata(StrictModel):
    formulas: list[str] = Field(default_factory=list)
    date_profiles: list[str] = Field(default_factory=list)
    relationships: bool = False
    cardinality: bool = False
    categorical_values: bool = False
    numeric_profile: bool = False
    missing_profile: bool = False
    text_profile: bool = False


class ContextPlan(StrictModel):
    tables: list[ContextTable] = Field(default_factory=list)
    columns: dict[str, list[str]] = Field(default_factory=dict)
    relationships: list[ContextRelationship] = Field(default_factory=list)
    metadata: ContextMetadata = Field(default_factory=ContextMetadata)
    excluded: list[str] = Field(default_factory=list)
    inclusion_reasons: dict[str, str] = Field(default_factory=dict)
    requirements: list[ContextRequirement] = Field(default_factory=list)
    coverage_complete: bool = True
    missing_requirements: list[str] = Field(default_factory=list)
    token_budget: int = Field(default=1_800, ge=500, le=8_000)


class RetrievalTrace(StrictModel):
    bm25_ranks: dict[str, int] = Field(default_factory=dict)
    embedding_ranks: dict[str, int] = Field(default_factory=dict)
    value_match_ranks: dict[str, int] = Field(default_factory=dict)
    rrf_scores: dict[str, float] = Field(default_factory=dict)
    ml_reranker_applied: bool = False
    ml_reranker_scores: dict[str, float] = Field(default_factory=dict)
    selected_tables: list[str] = Field(default_factory=list)
    selected_columns: dict[str, list[str]] = Field(default_factory=dict)
    bridge_tables_added: list[str] = Field(default_factory=list)
    metadata_supplied: list[str] = Field(default_factory=list)


class PipelineTelemetry(StrictModel):
    available_context_tokens: int = Field(default=0, ge=0)
    selected_context_tokens: int = Field(default=0, ge=0)
    selected_model_context_tokens: int = Field(default=0, ge=0)
    pruned_tokens: int = Field(default=0, ge=0)
    pruning_percentage: float = Field(default=0, ge=0, le=100)
    context_expansions: int = Field(default=0, ge=0, le=2)
    retrieved_context_categories: list[str] = Field(default_factory=list)
    generation_attempts: int = Field(default=0, ge=0, le=4)
    planner_call_used: bool = False
    planner_call: TokenUsage = Field(default_factory=TokenUsage)
    generation_call: TokenUsage = Field(default_factory=TokenUsage)
    repair_calls: list[TokenUsage] = Field(default_factory=list)
    generation_system_tokens: int | None = Field(default=None, ge=0)
    generation_context_tokens_estimated: int = Field(default=0, ge=0)
    model1_latency_ms: int | None = Field(default=None, ge=0)
    model2_latency_ms: int | None = Field(default=None, ge=0)
    selected_table_count: int = Field(default=0, ge=0)
    selected_column_count: int = Field(default=0, ge=0)
    metadata_request_count: int = Field(default=0, ge=0)
    historical_attempted: bool = False
    historical_candidates_retrieved: int | None = Field(default=None, ge=0)
    historical_examples_admitted: int = Field(default=0, ge=0, le=2)
    historical_similarity_scores: list[float] = Field(default_factory=list, max_length=2)
    ab_context_mode: Literal["model1", "retrieval"] | None = None
    retrieval: RetrievalTrace | None = None


class ContractDelta(StrictModel):
    """A conversational change request; it is not a second semantic contract."""

    operation: str
    instruction: str
    feedback_category: str | None = None


class Attempt(StrictModel):
    number: int = Field(ge=1, le=4)
    sql: str
    normalized_sql: str
    validation: ValidationResult
    latency_ms: int = Field(ge=0)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class RecoveryToolCall(StrictModel):
    tool: str
    purpose: str
    result_summary: str


class RecoveryTrace(StrictModel):
    activated: bool = True
    mode: Literal["FAILURE", "CORRECTNESS", "ZERO_RESULT", "NULL_RESULT", "FILTER"] = (
        "FAILURE"
    )
    failure_code: str
    failure_category: str
    diagnosis_code: str | None = None
    diagnosis_summary: str | None = None
    filter_checks: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    tool_calls: list[RecoveryToolCall] = Field(default_factory=list, max_length=6)
    evidence: list[str] = Field(default_factory=list, max_length=20)
    requires_human_review: bool = False


class ValueGroundingIssue(StrictModel):
    user_value: str
    column: str
    available_values: list[str] = Field(default_factory=list, max_length=20)
    reason: str


class HumanReviewRequest(StrictModel):
    reason: str
    question: str
    options: list[str] = Field(default_factory=list, max_length=5)
    evidence: list[str] = Field(default_factory=list, max_length=20)


class OptimizationEvidence(StrictModel):
    optimizer: Literal["sqlglot", "model2"] = "model2"
    status: Literal[
        "optimized", "equivalent_not_faster", "no_structural_change", "rejected"
    ]
    baseline_explain: list[str] = Field(default_factory=list)
    candidate_explain: list[str] = Field(default_factory=list)
    baseline_timings_ms: list[float] = Field(default_factory=list)
    candidate_timings_ms: list[float] = Field(default_factory=list)
    baseline_median_ms: float | None = Field(default=None, ge=0)
    candidate_median_ms: float | None = Field(default=None, ge=0)
    improvement_percent: float | None = None
    result_equivalent: bool = False
    selected_sql: Literal["baseline", "candidate"]


class HistoricalExample(StrictModel):
    db_id: str
    question: str
    sql: str
    score: float = Field(ge=0)


class GenerateRequest(StrictModel):
    db_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    question: str = Field(min_length=1, max_length=4_000)
    evidence: str | None = Field(default=None, max_length=8_000)
    dialect: Literal["sqlite", "postgres"] = "sqlite"
    provider: ModelProvider = "ollama"
    model: str = Field(default=DEFAULT_OLLAMA_MODEL, min_length=1, max_length=200)
    max_attempts: int = Field(default=3, ge=1, le=3)
    execute: bool = False
    max_rows: int = Field(default=100, ge=1, le=500)
    generation_style: Literal["reasoning", "icl", "alternative"] = "reasoning"
    approved_tables: list[str] | None = Field(default=None, max_length=20)
    semantic_contract: SemanticContract | None = None
    business_context: str | None = Field(default=None, max_length=12_000)
    previous_sql: str | None = Field(default=None, max_length=20_000)
    optimization_required: bool = False
    context_request: ContextRequest | None = None
    planner_call_used: bool = False
    planner_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    historical_examples: list[HistoricalExample] = Field(default_factory=list, max_length=3)


class ChatRequest(StrictModel):
    session_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    db_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=4_000)
    evidence: str | None = Field(default=None, max_length=8_000)
    provider: ModelProvider = "ollama"
    model: str = Field(default=DEFAULT_OLLAMA_MODEL, min_length=1, max_length=200)
    context_provider: ModelProvider | None = None
    context_model: str | None = Field(default=None, min_length=1, max_length=200)
    execute: bool = True
    max_rows: int = Field(default=100, ge=1, le=500)
    context_mode: Literal["model1", "retrieval"] = "retrieval"
    feedback_category: (
        Literal[
            "wrong_result",
            "wrong_interpretation",
            "missing_filter",
            "wrong_table_column",
            "wrong_aggregation",
            "wrong_datetime",
            "missing_rows",
            "duplicate_rows",
            "wrong_output",
            "wrong_grain",
            "wrong_ranking",
            "wrong_metric",
            "other",
        ]
        | None
    ) = None


class ConversationState(StrictModel):
    session_id: str
    db_id: str
    root_question: str
    modifications: list[str] = Field(default_factory=list, max_length=30)
    resolved_question: str
    semantic_contract: SemanticContract | None = None
    approved_tables: list[str] = Field(default_factory=list)
    last_sql: str | None = None
    last_columns: list[str] = Field(default_factory=list)
    last_row_count: int | None = Field(default=None, ge=0)
    last_truncated: bool = False
    last_model_context: dict[str, Any] = Field(default_factory=dict)
    last_failure: str | None = None
    last_failed_sql: str | None = None
    turn_count: int = Field(default=0, ge=0)
    corrections: list[str] = Field(default_factory=list, max_length=30)
    contract_deltas: list[ContractDelta] = Field(default_factory=list, max_length=30)


class TurnInterpretation(StrictModel):
    operation: Literal[
        "NEW_QUERY",
        "REFINE",
        "OPTIMIZE",
        "ADD_FILTER",
        "REMOVE_FILTER",
        "CHANGE_METRIC",
        "CHANGE_GRAIN",
        "COMPARE",
        "CORRECTION",
        "EXPLAIN",
        "EXPLAIN_SQL",
        "EXPLAIN_RESULT",
        "EXPLAIN_INTERPRETATION",
        "EXPLAIN_CONTEXT",
        "EXPLAIN_FAILURE",
        "CHECK_CORRECTNESS",
        "RESET_CONTEXT",
    ]
    depends_on_previous: bool
    resolved_instruction: str = Field(min_length=1, max_length=4_000)
    correction_type: str | None = Field(default=None, max_length=100)
    target: str | None = Field(default=None, max_length=100)
    confidence: float = Field(ge=0, le=1)
    source: Literal["rules", "model", "fallback"] = "model"
    provider: ModelProvider | None = None
    model: str | None = None


class ChatResponse(StrictModel):
    session_id: str
    operation: Literal[
        "NEW_QUERY",
        "REFINE",
        "OPTIMIZE",
        "ADD_FILTER",
        "REMOVE_FILTER",
        "CHANGE_METRIC",
        "CHANGE_GRAIN",
        "COMPARE",
        "CORRECTION",
        "EXPLAIN",
        "EXPLAIN_SQL",
        "EXPLAIN_RESULT",
        "EXPLAIN_INTERPRETATION",
        "EXPLAIN_CONTEXT",
        "EXPLAIN_FAILURE",
        "CHECK_CORRECTNESS",
        "RESET_CONTEXT",
    ]
    resolved_question: str
    conversation_interpretation: TurnInterpretation | None = None
    state: ConversationState | None = None
    generation: GenerateResponse | None = None
    message: str
    explanation: str | None = None
    clarification_required: bool = False
    clarification_question: str | None = None
    human_review: HumanReviewRequest | None = None
    provenance: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    timings_ms: dict[str, int] = Field(default_factory=dict)


class GenerateResponse(StrictModel):
    db_id: str
    question: str
    provider: ModelProvider
    model: str
    dialect: Literal["sqlite", "postgres"] = "sqlite"
    strategy: StrategyHints
    schema_selection: SchemaSelection | None = None
    semantic_contract: SemanticContract | None = None
    sql: str | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def formatted_sql(self) -> str | None:
        """Pretty SQL for presentation; ``sql`` remains the executed statement."""
        return format_sql_for_display(self.sql, self.dialect)

    accepted: bool
    execution_status: Literal["NOT_EXECUTED", "EXECUTABLE", "ACCEPTED"] = "NOT_EXECUTED"
    attempts: list[Attempt] = Field(default_factory=list, max_length=4)
    rows: list[list[Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    truncated: bool = False
    termination_reason: Literal[
        "accepted", "attempt_limit", "model_error", "database_error", "grounding_clarification"
    ]
    model_error: str | None = Field(default=None, max_length=500)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    optimization: OptimizationEvidence | None = None
    context_plan: ContextPlan | None = None
    model_context: dict[str, Any] | None = None
    context_request: ContextRequest | None = None
    telemetry: PipelineTelemetry = Field(default_factory=PipelineTelemetry)
    recovery: RecoveryTrace | None = None
    grounding_issue: ValueGroundingIssue | None = None


class CheckRequest(StrictModel):
    db_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    sql: str = Field(min_length=1, max_length=50_000)
    dialect: Literal["sqlite", "postgres"] = "sqlite"
    execute: bool = False
    max_rows: int = Field(default=100, ge=1, le=500)


class CheckResponse(StrictModel):
    db_id: str
    dialect: Literal["sqlite", "postgres"]
    validation: ValidationResult
    rows: list[list[Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    truncated: bool = False


class ModelOption(StrictModel):
    provider: ModelProvider
    model: str
    local: bool
    configured: bool
    unavailable_reason: str | None = Field(default=None, max_length=300)


class DatabaseOption(StrictModel):
    db_id: str
    dialect: Literal["sqlite", "postgres"]
    configured: bool
