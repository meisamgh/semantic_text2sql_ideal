# Recovery Agent Improvement Plan

This document records verified improvements for the current architecture. The normal Text-to-SQL
path remains retrieval, context assembly, Model 2 generation, SQL safety validation and read-only
execution. The bounded recovery agent activates only for a failed attempt, zero rows, unexpected
`NULL`, or an explicit correctness review.

## Current recovery contract

```text
Anomaly or correctness request
        |
        v
One recovery agent
   |              |
inspect_schema  query_database
   |              |
   +------ evidence
             |
       reason again
             |
 REPAIR / INFORM / ESCALATE
```

The model never controls SQL authorization. Diagnostic SQL remains one statement, SELECT-only,
restricted to approved tables, row-bounded and timeout-bounded. Profile examples are advisory;
the agent must use `query_database` when a conclusion depends on whether an identifier or category
actually exists.

## P0: make the recovery contract true

### 1. Run deterministic fallback only after agent failure

`ainvestigate()` currently calls the complete deterministic recovery graph before starting the
agent, then resets the deadline and starts again. This duplicates schema/filter work and causes
reported latency and probe counts to omit part of the actual work.

Required flow:

```text
agent recovery
    |
    +-- valid terminal decision --> return agent trace
    |
    +-- model/tool/format failure --> run deterministic fallback once
```

Acceptance criteria:

- No database probe occurs before the agent requests `query_database`.
- Deterministic fallback runs only if the agent cannot complete safely.
- One deadline covers the active recovery path.
- Returned telemetry includes all work actually performed.

### 2. Make agent decisions control execution

The labels currently do not fully control the pipeline. Failed SQL continues into the normal retry
loop regardless of `INFORM` or `ESCALATE`, while zero-row/NULL handling displays a diagnosis even if
the agent requested `REPAIR`.

Required transitions:

```text
REPAIR
  -> send repair_instruction to Model 2
  -> generate exactly one bounded candidate
  -> validate and execute

INFORM
  -> display the evidence-based diagnosis
  -> stop without another SQL attempt

ESCALATE
  -> display the unresolved uncertainty
  -> stop without changing the request or running more SQL
```

Acceptance criteria:

- `INFORM` and `ESCALATE` never trigger another Model 2 attempt.
- `REPAIR` triggers one focused regeneration, subject to the three-total-attempt ceiling.
- The action, repair instruction and resulting transition are visible in the recovery trace.

### 3. Use one SQL safety boundary

`RecoveryTools.query_database()` duplicates part of the main validator and does not mirror every
blocked construct (`MERGE`, locks, `SELECT INTO`, and root-query validation). Security logic should
not drift between normal execution and recovery.

Required flow:

```text
agent diagnostic SQL
        |
        v
existing validate_sql()
        |
validation.tables subset of approved tables
        |
diagnostic row/time limits
        |
read-only execution
```

Acceptance criteria:

- Normal and diagnostic SQL share the same AST safety rules.
- Diagnostic SQL rejects writes, DDL, administrative commands, locks, `MERGE`, `SELECT INTO`,
  multiple statements and non-query roots.
- Recovery adds only the stricter table allowlist and diagnostic resource limits.

### 4. Enforce the PostgreSQL recovery timeout

`PostgresRegistry.execute(..., timeout_seconds=...)` currently discards the supplied timeout and
uses a fixed five-second statement timeout. A requested two-second diagnostic probe can therefore
violate the recovery budget.

Acceptance criteria:

- PostgreSQL sets `SET LOCAL statement_timeout` from the bounded `timeout_seconds` argument.
- The configured value cannot exceed the normal database ceiling.
- A slow diagnostic query is cancelled within the expected tolerance.

### 5. Separate tool-call and reasoning-call budgets

Four loop iterations cannot support four tool calls plus a final model decision. The fourth tool can
run without a remaining model turn to interpret its result.

Use explicit constants:

```text
MAX_RECOVERY_TOOL_CALLS = 3
MAX_RECOVERY_MODEL_CALLS = 4
MAX_DIAGNOSTIC_ROWS = 20
MAX_DATABASE_PROBES = 8
```

Three tools are sufficient for the intended workflow and leave one final reasoning call.

## P1: make evaluation and telemetry trustworthy

### 6. Account for all recovery work

Record every reasoning call and token, every tool and database probe, total recovery latency,
fallback usage, budget exhaustion and estimated model cost when pricing is configured. The API
response token total must include recovery tokens for failure, correctness, zero-row and NULL paths.

### 7. Preserve duplicate multiplicity in every benchmark

`benchmark.compare_sql()` currently compares `set(predicted) == set(gold)`, so `[A, A, B]` and
`[A, B]` are incorrectly equivalent. Use the same `Counter`-based comparison already used by the
newer hard-query benchmarks. Duplicate inflation caused by join fanout must fail equivalence.

### 8. Remove failed dense retrieval from RRF

When FastEmbed fails, all dense scores become zero but still receive ranks because dense ranking is
calculated with `positive_only=False`. An unavailable retriever must contribute no RRF score:

```text
dense retrieval succeeds -> include dense ranks
dense retrieval fails    -> dense_ranks = {}
```

BM25 and value matching remain deterministic fallbacks.

## P2: operational consistency

### 9. Expose configured PostgreSQL databases in Query Room

The backend lists configured PostgreSQL databases, but the web application filters the selector to
SQLite. Display every configured database and retain its dialect in the submitted request.

### 10. Bound process-local state

Completed, failed and cancelled jobs remain indefinitely in process-local dictionaries. Add TTL/LRU
cleanup for jobs and conversations. Durable shared state remains a later production requirement.

### 11. Clarify configuration semantics

- `TEXT2SQL_HISTORY_ENABLED=false` controls examples sent to Model 2, not every use of successful
  history in schema ranking.
- A context token budget is an estimate, not a hard tokenizer-enforced ceiling.
- Date format is supplied when an observed format exists; missing profile coverage is not currently
  a generation blocker.

## Explicitly retained product decision

Do not restore pre-generation categorical/date rejection. Before Model 2, profile values and date
metadata remain advisory examples. Actual value existence is checked by the recovery agent only
after an anomaly or during correctness review.

Known consequence: if Model 2 silently replaces an explicit value and the altered query returns a
normal-looking result, anomaly recovery may not activate. Preserve explicit literals in the prompt,
measure this failure mode, and expose correctness review; never treat sampled profile values as an
exhaustive domain.

## Test gate

Add tests for:

- agent failure triggers fallback only afterward;
- three tools still allow a final decision;
- `INFORM` and `ESCALATE` stop retries;
- `REPAIR` performs exactly one focused regeneration;
- diagnostic SQL rejects everything blocked by the main validator;
- outside-table probes are rejected;
- PostgreSQL diagnostic timeout is enforced;
- telemetry includes all reasoning calls, tokens, probes and latency;
- result equivalence preserves duplicates;
- failed dense retrieval contributes no RRF rank;
- configured PostgreSQL databases appear in the web selector.

## Recommended implementation order

1. Lazy deterministic fallback and accurate telemetry.
2. Real `REPAIR` / `INFORM` / `ESCALATE` transitions.
3. Shared SQL validator and PostgreSQL timeout enforcement.
4. Explicit three-tool/four-model-call budget.
5. Benchmark multiplicity and dense-RRF fixes.
6. PostgreSQL UI exposure and process-state cleanup.

Do not add more tools, another model, a vector database or proactive agent routing until these
correctness and observability gaps are closed and measured.
