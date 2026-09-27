# Semantic Text-to-SQL

A conversational Text-to-SQL application for SQLite and allowlisted PostgreSQL databases. The
normal path retrieves and grounds a compact schema, generates read-only SQL, and executes it.
Failures, empty or unexpected `NULL` results, and explicit correctness reviews can activate a
bounded evidence-gathering recovery agent. For ordinary successful results, the selected model
chooses a chart before writing a plain-language answer.

<p align="center">
  <img src="Arch-pic" alt="Semantic Text-to-SQL architecture" width="1000" />
</p>

_Visual overview of the query workflow. The architecture and runtime boundaries documented below
are authoritative._

## Demo

https://github.com/user-attachments/assets/ac593a9c-802f-44c7-875d-02c9b4e0ca1f

## Architecture

```text
Question -> conversation resolver
                 |
                 v
Hybrid schema retrieval (BM25 + dense + value matching -> RRF)
                 |  optional Model 1 context selection
                 v
Deterministic grounding (keys, join paths, grain, types, date formats, glossary)
                 |  optional, strongly matched historical SQL examples
                 v
Selected SQL model -> SQLGlot safety -> EXPLAIN -> read-only execution
                 |
       +---------+--------------------+
       |                              |
 ordinary result               failure / anomalous result /
       |                       explicit correctness review
       v                              |
 selected model chooses chart          v
       |                       bounded recovery agent
 validate chart against rows          | inspect_schema / inspect_values /
       |                       | query_database -> REPAIR / INFORM / ESCALATE
 selected model explains result        |       |
       |                               +-- focused SQL retry when justified
       v
 SQL + exact rows + answer + optional chart + context + attempt/usage history
```

The default context method is retrieval-only. Users may select optional Model 1 context selection
and choose the SQL model separately. Model 1 selects context; it does not write SQL. Historical SQL
examples are disabled by default and, when enabled, are admitted only after compatibility checks.
The same selected SQL model handles chart choice and result explanation; the chart is never used as
evidence that the generated SQL is business-correct.

### Stakeholder answers and charts

For an ordinary successful result, the selected model first proposes `bar`, `line`, `scatter`, or
`none` from the question, accepted SQL and up to 40 returned rows. Code verifies the proposed
columns, numeric values, date axis and row grain; invalid choices become `none`, not another
guessed chart. A second model call then explains the result using that validated choice. The UI
renders only actual returned rows and never runs model-generated chart code. SQL and the exact
result table remain available beside the answer.

Both calls share the request-wide model-call budget and have a 15-second timeout each. Their token
usage and outcomes appear separately in the call ledger. A model failure or exhausted budget leaves
the query result available with a factual fallback. Partial results are labelled, and explanations
are interpretations of returned data—not correctness proofs. Recovery and human-review messages
take precedence over this presentation path.

### Client workspace

- Browse and search database tables and columns before asking a question.
- View the result table, formatted SQL, selected context and validation issues separately.
- See a model-selected chart and plain-language answer when the returned data supports them.
- Follow live backend stages and a timestamped activity log, including recovery schema/value
  checks and read-only probes. Timings are observations, not predicted completion percentages.
- Cancel a running request; model/database selection stays locked until it finishes.

If the table selector is empty, check `TEXT2SQL_DATABASE_ROOT` in `.env`. It must contain
`<db_id>/<db_id>.sqlite`; profile JSON files do not replace the database files.
Frontend rendering regressions can be checked with `node --test tests/frontend_rendering.cjs`.

## Agent-first recovery

The agent is deliberately absent from successful routine queries. It activates only when additional
reasoning and evidence are useful:

```text
Generated SQL
    |
    v
Validate + execute
    |
    +-- ordinary result --------------------------> return
    |
    +-- SQL failure / zero rows / unexpected NULL
    |                                                |
    +-- explicit correctness review                  v
                                             Recovery agent
                                                  |
                                         choose one bounded tool
                                  /             |             \
                         inspect_schema  inspect_values  query_database
                                  \             |             /
                                           verified observation
                                                   |
                                            typed evidence claim
                                                   |
                                      deterministic evidence policy
                                                   |
                                              reason again
                                                   |
                                      REPAIR / INFORM / ESCALATE
```

`inspect_schema` returns only approved structural metadata: selected tables and columns, physical
and semantic types, keys, grain, relationships, NULL presence, date formats and temporal coverage.
It does not return sampled categorical values as proof. The deterministic `inspect_values` tool
checks identifier and category existence in `EXACT` mode; its separate `SEARCH` mode returns only
advisory candidates and cannot prove existence or absence. The more general `query_database` tool
is reserved for narrow analytical probes that value inspection cannot answer.

The model chooses what it needs to inspect, but it never controls authorization. SQLGlot parsing,
SELECT-only enforcement, database and table allowlists, timeouts, row limits and the three-tool-call
ceiling are mandatory code-level controls. The agent never receives a database connection and never
silently substitutes an explicit user value.

Every failure and tool observation receives a stable evidence ID. Before `REPAIR` or `INFORM` is
accepted, the agent must emit typed claims that cite IDs from that request's evidence ledger.
Deterministic policy rejects missing, malformed or invented citations; data-grounding and
correctness claims must cite live tool evidence rather than relying only on the failure message.
It also enforces evidence compatibility: schema claims require matching schema output, value claims
require a matching exact-value probe, filter mismatches require a zero-count probe, NULL causes
require an observed NULL, and join effects require differing before/after counts.

The three terminal decisions are:

| Decision | Meaning |
|---|---|
| `REPAIR` | Give Model 2 a focused, evidence-backed correction instruction |
| `INFORM` | Explain a verified empty, NULL or otherwise noteworthy result |
| `ESCALATE` | State what remains uncertain without guessing or changing the request |

## Supporting capabilities

- Conversational new queries, refinements, corrections, explanations and optimization requests
- SQLite discovery and allowlisted PostgreSQL connections
- BM25, local dense embeddings and profiled value matching fused with Reciprocal Rank Fusion
- Optional LightGBM column reranking with deterministic fallback to RRF
- PK/FK, formula-dependency and bridge-table restoration
- Compact verified context instead of full-schema prompting
- Mandatory observed format for every selected date/datetime column
- Advisory categorical examples and observed date formats in the compact model context
- SQL-only generation with at most three total attempts
- Syntax-coloured SQL, tabular results, progress, cancellation and token accounting
- Result-equivalence and performance gates for explicit optimization requests
- Evidence-based correctness review and human-in-the-loop escalation

## Recovery modes

The bounded LangGraph recovery agent operates in five modes:

| Mode | Trigger | Purpose |
|---|---|---|
| `FAILURE` | SQL/schema/data execution failure | Gather focused evidence for repair |
| `FILTER` | Unresolved explicit filter value | Verify storage without silent substitution |
| `ZERO_RESULT` | Successful execution with no rows | Distinguish a valid empty result from filter/join mismatch |
| `NULL_RESULT` | Result contains unexpected SQL `NULL` | Inspect filters, joins, aggregation and missing-value evidence |
| `CORRECTNESS` | User requests correctness review | Compare an independent evidence-based candidate |

Before SQL generation, categorical values and temporal metadata are advisory context rather than
hard validation rules. Sampled values never prove that another value is invalid. Data validity is
investigated only after an execution anomaly or an explicit correctness review.

Zero-row and unexpected-NULL diagnoses are displayed once as informational messages. Users may
provide a correction naturally in the normal chat, but the agent does not present a separate edit
form or silently modify the request.

## Validation boundary

Before execution, generated SQL must:

- Parse successfully with SQLGlot
- Contain exactly one statement
- Be `SELECT` or `WITH ... SELECT`
- Contain no writes, DDL, administrative commands or `SELECT INTO`

SQLite uses query-only connections. PostgreSQL uses read-only transactions, server-side database
allowlists and bounded timeouts. A query that executes is reported as executable—not automatically
business-correct. Returning rows is never treated as proof of correctness.

## Models

The API supports three project-scoped provider transports:

- AgentRouter: `gpt-5.6-sol`, `glm-5.3`, `deepseek-v4-flash`, `claude-opus-5`,
  `claude-opus-4-8`
- Groq: `qwen/qwen3.6-27b`
- JustDoWork: `gpt-5.6-sol`, `claude-opus-5`

Only configured models are available in the interface. Credentials remain server-side and are
never returned to the browser. JustDoWork has no hard-coded endpoint: set both
`JUSTDOWORK_API_KEY` and an endpoint you have verified in `JUSTDOWORK_BASE_URL`.

## Quick start

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/meisamgh/semantic_text2sql_ideal.git
cd semantic_text2sql_ideal
uv sync --dev --extra ml
cp .env.example .env
```

Add the required provider key to the ignored `.env`, then run:

```bash
uv run uvicorn semantic_text2sql.api:app \
  --host 127.0.0.1 \
  --port 8000 \
  --env-file .env
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Check the API with:

```bash
curl http://127.0.0.1:8000/api/health
```

## Database configuration

SQLite databases are discovered below `TEXT2SQL_DATABASE_ROOT`:

```text
data/
  books/
    books.sqlite
```

PostgreSQL databases are configured as a server-side JSON allowlist using read-only credentials:

```dotenv
TEXT2SQL_POSTGRES_DATABASES={"analytics":"postgresql://reader:password@localhost/analytics"}
```

Generate an offline profile with:

```bash
uv run python scripts/profile_database.py \
  --dialect sqlite \
  --db-id books \
  --output profiles
```

Important settings:

| Variable | Purpose | Default |
|---|---|---|
| `TEXT2SQL_DATABASE_ROOT` | SQLite database root | `data` |
| `TEXT2SQL_POSTGRES_DATABASES` | PostgreSQL database allowlist | unset |
| `TEXT2SQL_PROFILE_ROOT` | Offline profile root | `profiles` |
| `TEXT2SQL_GLOSSARY_ROOT` | Approved glossary root | `data/business_glossaries` |
| `TEXT2SQL_RETRIEVAL_TABLES` | High-recall table budget | `5` |
| `TEXT2SQL_RETRIEVAL_COLUMNS` | Approximate columns per table | `5` |
| `TEXT2SQL_SCHEMA_RERANKER_ENABLED` | Enable LightGBM column reranking | `true` |
| `TEXT2SQL_HISTORY_ENABLED` | Enable strongly matched historical examples | `false` |
| `TEXT2SQL_CONTEXT_MODEL` | Optional context-model override | selected model |
| `TEXT2SQL_SQL_MODEL` | Optional SQL-model override | selected model |
| `TEXT2SQL_REQUEST_TIMEOUT_SECONDS` | Shared end-to-end request deadline | `180` |
| `TEXT2SQL_REQUEST_MAX_MODEL_CALLS` | Maximum model-call budget per request | `6` |
| `TEXT2SQL_REQUEST_MAX_DATABASE_CALLS` | Maximum database-call budget per request | `24` |
| `TEXT2SQL_CONVERSATION_STORE` | Optional SQLite path for durable conversation state | in memory |
| `TEXT2SQL_CONVERSATION_TTL_SECONDS` | Durable/in-memory session lifetime | `86400` |
| `TEXT2SQL_JOB_TTL_SECONDS` | Completed asynchronous-job lifetime | `3600` |
| `TEXT2SQL_MAX_JOBS` | Maximum retained asynchronous jobs | `500` |

See [.env.example](.env.example) for provider-specific settings. Never commit `.env`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Health check |
| `GET /api/models` | Model discovery and configuration state |
| `GET /api/databases` | Configured database discovery |
| `POST /api/chat` | Synchronous conversational query |
| `POST /api/chat/jobs` | Start a cancellable query |
| `GET /api/chat/jobs/{job_id}` | Retrieve progress or result |
| `DELETE /api/chat/jobs/{job_id}` | Cancel an active query |
| `POST /api/check` | Parse supplied SQL; execution requires administrative enablement |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-1",
    "db_id": "books",
    "message": "Count the books by category.",
    "provider": "agentrouter",
    "model": "gpt-5.6-sol",
    "context_mode": "retrieval"
  }'
```

Conversation state is in memory by default. Set `TEXT2SQL_CONVERSATION_STORE` to retain versioned
state in SQLite across API restarts.

## Verification

```bash
uv run pytest -q
uv run ruff check src tests
uv run mypy src
git diff --check
```

The automated suite checks software behavior; it is not a claim of Text-to-SQL execution accuracy.
Model quality must be measured on a frozen dataset, database state, provider, prompt and
result-equivalence protocol. The optional browser-rendering smoke test requires Node.js:

```bash
node --test tests/frontend_rendering.cjs
```

## Project structure

```text
src/semantic_text2sql/
  api.py               FastAPI endpoints and web delivery
  conversation.py      turn classification and session state
  hybrid_retrieval.py  BM25, embeddings, values, RRF and ML reranking
  context.py           deterministic grounding and context assembly
  llm.py               provider adapters and generation prompts
  validator.py         SQLGlot syntax and read-only safety checks
  recovery.py          bounded LangGraph evidence recovery
  presentation.py      chart-first model calls, chart checks and result interpretation
  runtime.py           shared request deadline and model/database call budget
  agent.py             generation, repair and execution orchestration
web/                   conversational interface
tests/                 unit and integration tests
```

## Scope

This repository is suitable for controlled analytics pilots where database access, providers and
business definitions are governed. It is not presented as unrestricted autonomous production SQL,
and it does not claim semantic correctness solely from successful execution.
