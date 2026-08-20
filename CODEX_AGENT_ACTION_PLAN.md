# Codex-Style Agent Plan for the Frozen 50 BIRD Questions

## Objective

Generate one safe SQLite query for each of the same 50 frozen BIRD questions by investigating the
question, evidence, approved historical corpus, live schema, relationships, descriptions, profiles,
and representative values. Store the complete action trace for every prediction. Only after all
predictions are frozen may a separate evaluator access gold SQL.

## Non-leakage contract

Generation may access:

- `question`, `evidence`, `db_id`, and non-answer metadata for the target;
- the live target SQLite database through allowlisted read-only tools;
- BIRD table and column description CSV files;
- cached profiles generated without target SQL;
- the 399 leakage-screened historical examples in `data/bird_history_seed42_400.json`;
- SQL patterns from those 399 examples, restricted to the same database.

Generation may not access:

- target `SQL` or any field derived from it;
- target gold tables, columns, query plan, execution result, or answer;
- results from earlier benchmark runs for the protected target;
- any artifact that labels a candidate correct or incorrect for the target.

The prediction artifact is written without gold SQL. A separate evaluation command joins frozen
predictions to gold SQL by `dataset_index` only after prediction generation has completed.

## Per-question action policy

The controller provides at most 12 investigation actions and 2 SQL repairs:

1. `interpret_question`
   - Extract requested outputs, metric, dimensions, filters, time logic, aggregation, ranking,
     expected grain, ordering, limit, and unresolved ambiguity.
2. `search_similar_queries`
   - Retrieve up to 3 same-database examples from the approved 399-record corpus.
3. `search_tables`
   - Rank tables from question, evidence, descriptions, profile terms, and historical patterns.
4. `describe_table`
   - Inspect candidate table grain, primary key, columns, and relationships.
5. `search_columns`
   - Rank columns only inside verified tables; preserve PK/FK join keys.
6. `describe_column`
   - Inspect description, aliases, type, semantic type, nullability, and key role.
7. `profile_column`
   - Inspect cached missingness, numeric/date range, categorical frequencies, and safe patterns.
8. `sample_values`
   - Use only when a filter literal cannot be grounded from cached safe values; return at most 10
     values from one allowlisted column.
9. `get_relationships`
   - Find real join paths among selected tables and necessary bridge tables.
10. `submit_semantic_ir`
    - Freeze the interpreted request before SQL generation.
11. `submit_plan`
    - Freeze tables, joins, select expressions, predicates, group-by, having, ordering, and limit.
12. `submit_sql`
    - Generate exactly one read-only query from the validated plan.

## Deterministic gates

Before execution:

- Semantic IR preserves question outputs, filters, time logic, aggregation, ranking, and grain.
- Logical-plan tables and columns exist in the full live schema.
- Every join follows a live FK relationship or has an explicitly justified equality relationship.
- SQLGlot accepts exactly one SQLite `SELECT` or `WITH ... SELECT`.
- Writes, DDL, pragmas, attachments, multiple statements, and row locks are rejected.
- SQLite `EXPLAIN QUERY PLAN` succeeds.

Execution:

- immutable/query-only database connection;
- 30-second evaluation timeout and 5-second interactive execution timeout;
- maximum 100 prompt-visible rows;
- no target gold result is visible during generation or repair.

Repair is allowed only for a concrete deterministic failure:

- parse or schema-validation error;
- database `EXPLAIN` or execution error;
- duplicated output columns;
- an all-NULL output column;
- output shape inconsistent with the frozen Semantic IR.

An empty result is recorded but is not automatically considered erroneous.

## Trace format

One JSONL record is stored per action:

```json
{
  "run_id": "...",
  "dataset_index": 0,
  "step": 3,
  "action": "search_tables",
  "reason": "Locate transaction and customer entities",
  "arguments": {"query": "..."},
  "result_summary": {"tables": ["transactions", "customers"]},
  "timestamp": "..."
}
```

One prediction record per target stores:

- target index, question ID, database, question, and evidence;
- Semantic IR and logical plan;
- retrieved historical example IDs and scores;
- selected tables and columns;
- final SQL and rejected SQL fingerprints;
- validation, EXPLAIN, execution status, latency, and termination reason;
- trace-file path and action count;
- no gold SQL or correctness label.

## Frozen run configuration

- Dataset: BIRD Mini-Dev, 500 records.
- Protected IDs: first 50 IDs from the existing frozen 100-ID test split.
- Historical corpus: 399 RAG IDs; one duplicate excluded.
- Generator: `claude-opus-5` through the configured AgentRouter Claude adapter.
- Temperature: 0.
- Maximum action calls: 12.
- Maximum SQL repairs: 2.
- Checkpoint: after every target.
- Resume: exact run signature required.

## Release report

After all 50 predictions are frozen, the separate evaluator reports:

- completed/50 and execution accuracy;
- accepted, executable, empty-result, and timeout rates;
- table and column retrieval selections, without deriving them from gold during generation;
- average actions, repairs, latency, and provider failures;
- failure categories by database and difficulty;
- paired fixed/harmed cases versus the 31/50 baseline and 30/50 v2 result.

No partial checkpoint is called a final accuracy result.
