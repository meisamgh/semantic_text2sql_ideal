# QueryGPT-Inspired Text-to-SQL v5

An independent local-first service inspired by Uber QueryGPT, XiYan-SQL, and ReFoRCE. It combines
conversation resolution, bounded context selection, deterministic grounding, compact model
context, deterministic validation, and read-only execution.
It reimplements public design patterns and does not copy Uber internal code, prompts, schemas, or
data. Source: <https://www.uber.com/de/en/blog/query-gpt/>.

## Architecture

```text
Question + evidence
        |
        v
conversation resolver + deterministic high-recall schema retrieval (up to 5 candidates)
        |
        v
Model 1 bounded context agent (tables, columns, glossary concepts, metadata requests only)
        |
        v
deterministic grounding (identifiers, join keys, formulas, relationships, required metadata)
        |
        v
verified minimal context (PK/FK/UK, grain, types, missing semantics, conditional metadata)
        |
        v
Model 2 returns one SQL statement only
        |
        v
SQLGlot read-only + schema validation -> EXPLAIN -> read-only execution
        |
        v
parse/schema/database error? -> focused SQL repair
        |
        +-- no --> targeted SQL repair
        |
        v
retry, maximum 3 generation attempts -> final SQL/result
```

The context agent finds information but does not classify operations or generate SQL. The
controller guarantees dependencies and context sufficiency. Model 2 generates SQL directly from
the verified context; deterministic code validates its safety and schema references. Historical examples
default to zero and remain disabled in the main path until a paired benchmark proves benefit.
Responses expose the typed resolution report, context plan, and telemetry: available/selected/
pruned context tokens, pruning percentage, semantic-call use, context expansions, and attempts.

### Validation boundary

The active generation path intentionally uses only high-confidence gates: one parsed statement,
read-only `SELECT` or `WITH...SELECT`, no write/DDL nodes, valid live tables and columns, resolvable
aliases/CTEs, approved retrieved tables, database `EXPLAIN`, and read-only execution. Heavy formula,
aggregation-shape, join-strategy, CTE-shape, grain, and AST-pattern rejection rules are disabled.
`EXECUTABLE` means the SQL passed safety/schema checks and `EXPLAIN`; `ACCEPTED` means read-only
execution also succeeded. Neither status proves business correctness, and non-empty rows are never
used as correctness evidence.

### Frozen Phase 1 boundary

Phase 1 uses lexical table-first and column-first retrieval, exact identifiers, glossary aliases,
literal/value matches, and the FK graph. Its authoritative semantic contract contains named
aggregation stages and structural formulas such as `DIVIDE(Price, Amount)`. Each semantic node
emits typed context requirements (`COLUMN_SCHEMA`, `PHYSICAL_DATE_FORMAT`, `ALLOWED_VALUES`,
`FORMULA_DEFINITION`, `JOIN_PATH`, `JOIN_CARDINALITY`, and related capabilities). The deterministic
coverage checker resolves these from live schema, approved glossary facts, and offline profiles.
Only resolved, node-required facts enter the SQL prompt; diagnostic inclusion reasons and excluded
metadata remain response telemetry rather than model context.

The following remain intentionally outside normal chat generation:

- Dense/embedding retrieval, adopted only after a paired BM25 comparison.
- Deterministic IR-to-SQL compilation for selected high-frequency patterns.
- Multi-candidate generation for high-risk contracts.
- Dense historical-query retrieval. After the post-grounding semantic plan is verified, normal chat
  may run BM25 top-20 retrieval followed by strict operation and business-concept compatibility.
  The combined acceptance score must be at least 0.65. Zero examples is the default, one strong
  analogue is preferred, and a second is allowed only when it adds a relevant complementary table
  or temporal pattern. Three or more examples are never sent. Table count alone never triggers
  retrieval. A local cross-encoder reranker remains an optional disabled experiment boundary.

## Public API surface

The current application exposes only the active conversation workflow and SQL checker. Workspace
intent, table proposal, interpretation, and schema reduction remain internal steps of chat rather
than separate public POST endpoints.

- `POST /api/chat/jobs` starts the current cancellable conversational workflow.
- `POST /api/chat` provides the same workflow synchronously and handles reset requests.
- `POST /api/check` validates and optionally executes user-supplied read-only SQL.
- `GET /api/chat/jobs/{job_id}` reports job progress and results.
- `DELETE /api/chat/jobs/{job_id}` cancels a running request.
- `GET /api/health`, `/api/models`, and `/api/databases` provide discovery.

## Conversational API

`POST /api/chat` maintains explicit session-scoped state. A turn is classified as a new query,
refinement, filter addition/removal, metric/grain change, comparison, or reset. Each follow-up is
resolved into a standalone cumulative request before glossary/schema retrieval and SQL generation.
Failed turns do not overwrite the last successful state.

```json
{"session_id":"demo-1","db_id":"books","message":"Count books for each category"}
```

Follow up using the same `session_id`:

```json
{"session_id":"demo-1","db_id":"books","message":"Now only books published after 2020"}
```

Reset with:

```json
{"session_id":"demo-1","db_id":"books","message":"Reset context"}
```

Conversation memory and schema memory are separate. Conversation state stores the root request,
ordered modifications, resolved request, semantic contract, approved tables, and last accepted SQL.
The built-in store is process-local; a multi-worker production deployment should replace it with a
durable session store.

The chat interface also supports `CORRECTION` and `EXPLAIN`. A correction carries a feedback
category, becomes a session-scoped contract modification, and regenerates through the same bounded
agent. Failed corrections preserve the previous successful state. Explanation turns do not call the
SQL model; they return the last contract and provenance. The UI reports provider-supplied input,
output, cache-read, and cache-creation token counts across the Claude interpreter and all SQL repair
attempts. Ollama token counts cover SQL generation because local turns use the deterministic
interpreter.

The model selector lists local `qwen3.5:9b` plus the AgentRouter GPT and Claude models.
The selected provider/model is used for both LLM calls: Context Planner first and SQL Generator
second. Selecting Claude uses Claude for both calls; selecting GPT or Qwen behaves the same way.
Retrieval, grounding, validation, EXPLAIN, and execution remain deterministic.
`/api/models` reports each entry's real readiness instead of assuming it: the local model must be
installed in Ollama and fit in system RAM, the Claude entries need the Claude Code CLI, and every
AgentRouter entry needs `AGENTROUTER_API_KEY`. Unavailable models render greyed out with the reason
on hover, and the page preselects the first model that can actually serve a query. AgentRouter GPT
calls the gateway's OpenAI-compatible HTTP API directly, so it requires no local CLI; an installed
Codex CLI is used only as a fallback transport.

## Chat web application

Start the API and open <http://127.0.0.1:8000>. The page discovers configured databases and models,
keeps a browser session ID, supports follow-up questions and reset, and displays generated SQL,
validation/attempt metadata, column names, result rows, truncation, and end-to-end latency. API keys
remain server-side and are never sent to the browser.

Web requests run as cancellable server-side jobs. While a query runs, the page shows current job
status and elapsed time. **Cancel** stops the active async model task and preserves the last accepted
conversation state. The original synchronous `POST /api/chat` endpoint remains available for API
clients.

```bash
export TEXT2SQL_DATABASE_ROOT=../bird-bench/llm/mini_dev_data/minidev/MINIDEV/dev_databases
export TEXT2SQL_PROFILE_ROOT=profiles/bird_frozen50
export TEXT2SQL_GLOSSARY_ROOT=data/business_glossaries
../.local-tools/uv run uvicorn semantic_text2sql.api:app --host 127.0.0.1 --port 8000
```

The first database-wide chat benchmark used one session containing all 30 complete
`debit_card_specializing` questions. All 30 were correctly recognized as independent `NEW_QUERY`
turns, 29/30 produced accepted SQL, and BIRD result equality was 17/30 (56.7%). Historical-example
retrieval was disabled to prevent target leakage. This is lower than the non-chat v4.4 run
(21/30), so conversational correctness and SQL accuracy are reported separately.

Before Model 2, deterministic code supplies schema facts, keys, relationships, physical metadata,
missing-value facts, and approved glossary formulas only. It does not infer aggregation, grouped
grain, ranking, output operations, or filters from question keywords. Model 2 emits both a typed
semantic plan and the SQL compiled from it in one structured response. The controller verifies the
plan, then checks that SQL independently against the verified contract. Targeted repairs are retried
at most three total SQL attempts.

Sequential candidates are intentional: they avoid loading or driving three large local models at
once on a normal PC. Candidate prompts differ while temperature remains deterministic. The selector
prefers the largest result-equivalence cluster, then fewer inspection findings, fewer repair
attempts, and stable candidate order.

## Safety and evidence boundaries

- Only one `SELECT` or `WITH ... SELECT` is accepted.
- SQLGlot checks dialect, tables, columns, statement count, and row locks.
- SQLite uses immutable/query-only connections; PostgreSQL uses restricted read-only transactions.
- Every candidate must pass database `EXPLAIN` before execution.
- Execution rows, attempts, and candidate count are bounded.
- Gold SQL from protected evaluation IDs is never available to retrieval or generation.
- Historical SQL is labelled as a non-authoritative pattern and restricted to the same database.
- Empty results are not automatically treated as errors.
- Profiles are generated offline; request-time code does not scan arbitrary values.

## Source layout

```text
src/semantic_text2sql/
  querygpt.py      workspaces, intent, table proposal, question enhancement
  querygpt_metrics.py component-level post-generation evaluation
  historical.py   successful-query retrieval
  context.py      typed context plans, compact rendering, bounded reactive retrieval
  linker.py       table-first and column-second schema linking
  profiling.py    offline metadata and value profiles
  ensemble.py     candidates, inspection, clustering, selection
  agent.py        validation, bounded repair, explain, execution
  validator.py    deterministic SQLGlot safety
  semantic.py     conservative semantic planning and contract validation
  interpreter.py  compact schema memory, Claude Semantic IR, contract reconciliation
  conversation.py explicit session state, turn classification, standalone resolution
  llm.py          Ollama and AgentRouter provider adapters
  api.py          FastAPI endpoints
scripts/
  build_bird_history.py
  prepare_codex_queue.py
  run_codex_agent.py
  evaluate_codex_predictions.py
  evaluate_querygpt_signals.py
  profile_database.py
  evaluate_bird.py
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for inspected repositories, commits, licenses,
and adapted concepts. Reference repositories remain untouched outside this project.

## Install

```bash
cd /Users/meisam/Documents/text-to-sql/semantic_text2sql_v5
../.local-tools/uv sync --dev
../.local-tools/uv run python scripts/create_demo_db.py
```

Use any installed Ollama model through the generic adapter. The default is sized to fit in 16 GB of
unified memory; a larger model pages to disk and never returns a generation:

```bash
ollama pull qwen3.5:9b
```

Arctic-R1 and XiYanSQL model names are optional adapter targets, not bundled dependencies. Their
weights are large, and this setup does not download them automatically or claim they have been
verified locally.

## Build leakage-safe BIRD history

```bash
../.local-tools/uv run python scripts/build_bird_history.py \
  --dataset ../bird-bench/llm/mini_dev_data/minidev/MINIDEV/mini_dev_sqlite.json \
  --split-file ../bird-bench/local_agent/artifacts/semantic_ir_split_seed42.json \
  --output data/bird_history_seed42_400.json
```

The current frozen split contains 399 leakage-screened RAG IDs, 100 protected evaluation IDs, and
one SQL-duplicate record deliberately excluded from retrieval. The builder verifies those counts and
the absence of overlap before writing history.

Generate profiles separately with `scripts/profile_database.py`, including BIRD description CSVs
where available. Profiles and history should be refreshed when the database changes.

Model 1 receives only high-recall candidate table and column names. After selection, deterministic
grounding gives Model 2 physical types and one observed missing-data fact: `observed_nulls` states
whether the profiled column currently contains any SQL NULL value. Every selected text or
categorical column receives up to five safe observed `example_values` to ground spelling, case, and
storage form. Every selected date or datetime column must include its observed physical format; an
unknown format leaves context unresolved.

For example, regenerate the debit-card metadata with:

```bash
export TEXT2SQL_DATABASE_ROOT=../bird-bench/llm/mini_dev_data/minidev/MINIDEV/dev_databases
../.local-tools/uv run python scripts/profile_database.py \
  --dialect sqlite \
  --db-id debit_card_specializing \
  --output profiles/bird_frozen50 \
  --bird-description-dir "$TEXT2SQL_DATABASE_ROOT/debit_card_specializing/database_description"
```

## Run the API

```bash
set -a
source .env
set +a
export TEXT2SQL_DATABASE_ROOT="$PWD/data"
export TEXT2SQL_PROFILE_ROOT="$PWD/profiles"
export TEXT2SQL_HISTORY_PATH="$PWD/data/bird_history_seed42_400.json"
export TEXT2SQL_WORKSPACES_PATH="$PWD/data/workspaces.json"
../.local-tools/uv run uvicorn semantic_text2sql.api:app --host 127.0.0.1 --port 8000
```

The chat model selector also controls optimization. To use Claude, select a configured
`claude-opus-*` entry and ask `Optimize it` after an accepted query. Optimization is a fast-path:
it reuses the accepted semantic contract and approved tables, skips workspace/table retrieval and
the separate interpretation call, and sends the previous SQL to the selected model. The replacement
is accepted only when deterministic validation, EXPLAIN, and executed output-equivalence checks
pass. The API returns routing, planning, and generation/validation/execution timings in
`timings_ms`; the web UI shows total server latency and exposes the stage breakdown as a tooltip on
the assistant response.

Optimization now requires performance evidence, not only structural simplification. The original
`EXPLAIN` plan is included in the compact optimizer prompt. After semantic and result-equivalence
validation, both queries receive one warm-up and three alternating timed executions. A candidate is
selected only when its median is at least 5% and 0.1 ms faster. Otherwise the API returns the
previous SQL with `optimization.status=equivalent_not_faster`. The response records both EXPLAIN
plans, raw timing samples, medians, improvement percentage, equivalence status, and whether the
baseline or candidate was selected. These application-side measurements are a local acceptance
signal, not a replacement for production query telemetry under representative concurrency.

### Conversation and validation boundaries

- A follow-up is stored as a contract delta (`operation`, instruction, optional feedback category)
  against the last accepted semantic contract. It is not treated as an unrelated second question.
- Session memory is process-local and holds accepted conversation state. It is never written into
  the trusted historical-example corpus automatically; that corpus is loaded only from an explicit,
  reviewed file.
- Deterministic cleanup may extract one SQL code fence and normalize harmless formatting. It does
  **not** rewrite semantic SQL expressions. A physical mismatch such as `STRFTIME` over a profiled
  `YYYYMM` text column produces `PROFILE_DATE_FORMAT_MISMATCH` and targeted model feedback.
- Optimization equivalence compares identical output columns; unordered outputs use multiset/bag
  equality (duplicates retained), ordered outputs preserve row order, and floats are normalized to
  nine decimal places. Truncated results are never accepted as proven equivalent.

Model escalation based on explicit complexity signals (join count, nesting, temporal reasoning,
ambiguity, and prior repair failure) remains a separate planned increment. The current UI obeys the
model selected by the user and does not silently escalate to a paid provider.

### Latency fast path

SQLite schemas, cached profiles, and business glossaries are cached in memory and automatically
invalidated when their source file modification time changes. New table proposals are capped at
three tables before relationship-aware column selection. Local Qwen requests use one SQL-generation
call by default; the separate local semantic-interpreter call is opt-in with
`TEXT2SQL_LOCAL_INTERPRETER_ENABLED=true`. Claude retains its schema-grounded interpreter because it
is used for difficult questions where the added reasoning is intentional. Physical profile context
includes explicit safe operations for nonstandard formats such as `YYYYMM`, but application code
still validates rather than rewriting semantic SQL.

Generation context is question-aware. Selected schema, physical type, observed NULL presence,
table grain, keys, and relationships remain available. Date format is mandatory for every selected
date/datetime column. Formula, description, numeric min/max, or text pattern metadata is included
only when requested and verified.
Text and categorical examples are capped at five and are not treated as an exhaustive domain.
Median, quantiles, timezone, full distributions, and text-length statistics are not sent to Model 2.

### Structured table retrieval

Offline profiles now store a deterministic table summary, row grain, inferred metric ownership,
dimensions, exact date coverage, supported terms, warnings, and provenance. Retrieval combines
question-to-table/column lexical overlap with a strong metric-ownership bonus and an exact
date-coverage compatibility signal. Sampled ranges are never used to exclude a table. For example,
a 2013 annual-consumption request selects `customers` and `yearmonth`; `transactions_1k` is penalized
because its exact observed coverage ends in August 2012 and it owns transaction amount/price rather
than the `Consumption` measure.

Embeddings are intentionally not required for the current small BIRD schemas. They can improve
synonym recall across many tables, but should run after deterministic compatibility filtering and
be fused with lexical ranking. PostgreSQL/pgvector becomes useful for a large, shared catalog; it is
not necessary to retrieve five local tables and does not replace metric, grain, or date checks.

### Metric and grain dependency DAG

High-confidence nested questions carry a typed aggregation chain in the semantic contract:
`measures`, `derived_metrics`, `selectors`, and `output_operations`. For the annual least-customer
segment comparison, the contract fixes `AVG(yearmonth.Consumption)` at
`CustomerID + Segment` grain, selects `MIN(customer_annual_avg)` within each segment, and then emits
SME-LAM, LAM-KAM, and KAM-SME differences. SQLGlot reconstructs enough aggregation lineage to reject
the wrong input measure/function/grain or a selector applied to the wrong derived metric with
`SEMANTIC_METRIC_LINEAGE_MISMATCH`. Both `MIN(derived_metric)` and an ascending partitioned
`ROW_NUMBER`/`RANK` over that same metric are accepted compilation strategies.

Percentage-change questions can also carry `aggregation_stages`, `formula_metrics`, and named
outputs. For the EUR segment comparison from 2012 to 2013, the contract fixes
`SUM(Consumption)` at `Segment + Year` grain, computes `(2013 - 2012) / 2012 * 100` with a
NULL-safe baseline, then selects both `ARGMAX` and `ARGMIN` across segments. The validator requires
tie-preserving ascending and descending `RANK`/`DENSE_RANK`, rank-1 filtering, and final output of
both segment identity and percentage value. This prevents a plausible query that merely lists all
segments in descending order from satisfying the requested maximum/minimum result contract.

## Verification

```bash
../.local-tools/uv run pytest -q
../.local-tools/uv run ruff check .
../.local-tools/uv run mypy
```

## Inherited benchmark history (not rerun for v5)

| Pipeline | Model | Execution accuracy |
|---|---|---:|
| Original single-pass baseline | Claude Opus 5 | 31/50 (62%) |
| Table/column profile v2 | Claude Opus 5 | 30/50 (60%) |
| Codex-style investigation agent | Claude Opus 5 | **33/50 (66%)** |
| Dual planner + two-level schema memory v4.3 | Claude Opus 5 | **35/50 (70%)** |

These figures describe earlier pipelines and must not be presented as v5 accuracy. This v5
implementation task intentionally did not run the benchmark suite. The v4.3 run used the identical
frozen 50 IDs. A schema-grounded Claude interpreter and the
deterministic planner ran independently, their contracts were reconciled before deep profile and
historical-example retrieval, and all predictions were frozen before gold evaluation. It accepted
50/50 queries, fixed five v4.1 failures with no regressions, and reached 35/50 execution accuracy.
Mean end-to-end latency was 32,883 ms and median latency was 27,174 ms.

The Codex-style run used a gold-free 50-question queue, 399 leakage-screened same-database examples,
offline profiles, nine recorded actions per question, Semantic IR, a logical plan, full-schema
SQLGlot validation, and SQLite EXPLAIN. All 50 predictions were frozen before the separate evaluator
read target gold SQL. It produced 450 action events, accepted 50/50 queries, and had median generation
latency of 20,318 ms. See `CODEX_AGENT_ACTION_PLAN.md` for the frozen protocol and `artifacts/` for
predictions, actions, and evaluation results.

## QueryGPT-style evaluation

After predictions are frozen, `scripts/evaluate_querygpt_signals.py` reports execution accuracy,
table overlap, successful-run rate, non-empty-output rate, and deterministic structural similarity.
BIRD has no workspace-intent labels, so intent accuracy remains explicitly unmeasured. Gold-derived
signals are evaluation-only and never enter generation.
