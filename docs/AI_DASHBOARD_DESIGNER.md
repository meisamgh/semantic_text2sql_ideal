# AI dashboard designer (Phase 2)

The existing LangGraph assistant owns conversation and analytical changes. Its
`ai_dashboard_designer` tool is a bounded specialist, not another agent. The
Phase 1 deterministic dashboard remains the default UI option and fallback.

## Usage

After an accepted query, ask **“Design a professional dashboard from this data”**.
Alternatively choose **AI design** beside **Create dashboard**. Configure an
existing provider adapter with `TEXT2SQL_DASHBOARD_PROVIDER` and
`TEXT2SQL_DASHBOARD_MODEL` for that explicit UI action. Conversational design
defaults to the request's assistant provider/model when these are empty.
`TEXT2SQL_DASHBOARD_MAX_TOKENS` controls task-specific completion limits.
No credentials are added by this feature; the selected adapter must already be
configured. An unavailable provider produces the deterministic fallback.

Use the design-change input for requests such as “Make the main chart larger”
or “Remove the distribution chart”. These go through the same Top layer, not a
second natural-language router. **Undo design** restores the preceding revision.

## Evidence and validation

`ai_dashboard_designer.py` profiles at most 24 columns and sends three bounded
sample rows, role/cardinality/distribution summaries, approved definitions,
constraints, eligible KPI IDs and validated chart candidates. Full chart data is
not sent to the model. An oversized prompt falls back before calling a model.

The model chooses widgets, order, 12-column widths, titles, labels, read-only
filter metadata and green/neutral/contrast styling. The first selected chart is
primary. `DashboardProposal` forbids extra executable options. Materialization
checks IDs, axis references, duplicate selections, source identity, widget limits
and labels. Authoritative KPI values and chart data come from Phase 1, never
from the model. Numerical claims/markup in design annotations are rejected;
axis relabeling currently accepts only the source column's words. This is
deliberately conservative and can reject otherwise reasonable titles/aliases.

Sections remain filters → KPIs → charts → optional source table. Applied SQL
constraints cannot be removed by hiding optional filter metadata. Global filters
are informational, not editable cross-widget filtering.

One initial design consumes at most one additional model call, including bounded
provider retries through the shared request budget, with a 12-second specialist
deadline. Failed initial designs fall back to Phase 1; failed revisions retain
the previous validated widgets. No current-result design can execute SQL.
`REQUERY_REQUIRED` reports missing requirements to the Top layer. If it obtains
new accepted data through `ask_data`, the original dashboard goal resumes against
that new result. A dashboard is not reported ready without a materialized spec.

## Revisions and boundaries

Revisions are temporary RAM-only views, scoped by session/database/result, with
10 revisions per result and 100 result keys. Revision commits compare the prior
revision ID to reject concurrent overwrites. Undo and dashboard reads use existing
accepted-result scope checks. Reset/history deletion clears revisions. This does
not add user/tenant authentication or durable cross-worker dashboard storage.
Existing historical chart restoration remains unchanged.

`POST /api/dashboards` accepts the existing source request plus optional
`design_mode: "ai"` and `objective`. `POST /api/dashboards/undo` uses the same
source identifiers. Missing/incomplete/truncated sources are rejected before
model execution. `ChatResponse.dashboard` carries a validated serialized
`DashboardSpec`; rendering reuses existing Plotly/ECharts and table callbacks.

## Measured comparison

30 repeated builds of the same four-row fixture on the local machine:

| Path | p50 | p95 | Additional model calls | Additional SQL calls |
|---|---:|---:|---:|---:|
| Phase 1 deterministic | 1.266 ms | 1.662 ms | 0 | 0 |
| Phase 2 scripted designer | 2.868 ms | 4.173 ms | 1 scripted | 0 |

This measures local profiling/materialization plus an immediate scripted response,
not live inference, network, browser or production latency. It is additional
design flexibility, not a measured performance improvement. Provider tokens in
scripted tests are fixtures, not estimates of real usage. The Top-layer integration
test uses one Top decision plus one specialist call, with no final-answer call.

Tests cover eligibility differences, invalid references/labels, lineage, bounded
calls, fallback, patch preservation, undo/CAS, real SQLite result reuse with DB
and generation disabled, and missing-data → governed SQL → dashboard continuation.
Frontend tests cover rendering order, source integrity, AI mode, revisions and
automatic display without requery. Live providers, PostgreSQL and pixel-level
desktop/mobile browser inspection were not exercised for this phase.
