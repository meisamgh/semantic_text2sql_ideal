# Semantic Text-to-SQL

A governed conversational Text-to-SQL application for SQLite and PostgreSQL. It combines compact
schema retrieval, optional model-based context selection, verified metadata grounding, SQL-only
generation, read-only execution, bounded recovery, and human review.

<p align="center">
  <img src="docs/assets/text-to-sql-architecture.png" alt="Semantic Text-to-SQL architecture" width="1000" />
</p>

_Visual overview of the query workflow. The architecture and runtime boundaries documented below
are authoritative._

## Architecture

```text
User question
    |
    v
Conversation resolver
    |
    v
Hybrid schema retrieval
BM25 + dense embeddings + value matching + RRF
    |
    +-- optional Model 1 context selector
    |
    v
Deterministic grounding
PK/FK, bridges, grain, cardinality, types, values and date formats
    |
    v
Model 2: SQL reasoning and generation
    |
    v
SQLGlot read-only safety validation
    |
    v
Read-only execution
    |
    +-- failure / zero rows / unexpected NULL --> bounded LangGraph recovery
    |                                               |
    |                                               +-- focused repair
    |                                               +-- human review when unresolved
    v
Formatted SQL + result + context + attempt history
```

The web application lets the user select the context method and SQL model independently. The
default retrieval-only path avoids the context-model call; Model 1 can be enabled for a controlled
A/B comparison.

## Key capabilities

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

## Recovery and human review

One bounded LangGraph recovery agent operates in five modes:

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

Recovery uses one reasoning agent with two capability-scoped tools:

- `inspect_schema`: bounded schema, grain, relationship, NULL, value and temporal metadata
- `query_database`: SQLGlot-checked, SELECT-only probes over approved tables

The agent may inspect evidence, reason again, and finish with `REPAIR`, `INFORM`, or `ESCALATE`.
Safety remains outside model control: one statement, read-only SQL, approved tables, short timeouts,
at most 20 returned rows and at most four recovery tool calls. It never silently changes an explicit
value or date. Zero-row and unexpected-NULL diagnoses are displayed once as informational messages;
users may provide a correction naturally in the normal chat.

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

The API model catalog currently supports:

- True SOTA Responses API: `gpt-5.5`
- JustDoWork: `gpt-5.6-sol`, `claude-opus-5`, `claude-opus-4-7`
- Groq: `qwen/qwen3.6-27b`

Only configured models are available in the interface. Credentials remain server-side and are
never returned to the browser.

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
| `POST /api/check` | Check and optionally run supplied read-only SQL |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-1",
    "db_id": "books",
    "message": "Count the books by category.",
    "provider": "sota",
    "model": "gpt-5.5",
    "context_mode": "retrieval"
  }'
```

Conversation state is process-local and resets when the API restarts.

## Verification

```bash
uv run pytest -q
uv run ruff check src tests
uv run mypy src
git diff --check
```

Current local verification: **115 passed, 1 skipped**. These are software tests, not a claim of
Text-to-SQL execution accuracy. Model quality must be measured on a frozen dataset, database state,
provider, prompt and result-equivalence protocol.

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
  agent.py             generation, repair and execution orchestration
web/                   conversational interface
tests/                 unit and integration tests
```

## Scope

This repository is suitable for controlled analytics pilots where database access, providers and
business definitions are governed. It is not presented as unrestricted autonomous production SQL,
and it does not claim semantic correctness solely from successful execution.
