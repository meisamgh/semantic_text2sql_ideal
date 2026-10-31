# Phase 4: conversational dashboard builder

Ask **“Build a professional dashboard for revenue, customer count and product
performance.”** The existing LangGraph assistant owns intent and the workflow.
Ordinary analytical questions still return normal results, not dashboards.
There is no Create dashboard button or business-user design-mode selector.

## Execution path

User objective → Top decision → accepted evidence/business definitions → distinct
governed `ask_data` objectives for missing evidence → existing dashboard designer
→ deterministic validation → automatic conversation preview.

`ToolInput.dashboard_goal` preserves the original goal across tools; bounded
request-local results retain each accepted SQL/contract/result ID. Independent
queries use NEW_QUERY, not a forced refinement/join with another metric's grain.
`source_result_ids` explicitly select relevant evidence. Unknown, cross-database,
incomplete and truncated results are rejected. Required failed analyses prevent
the final assistant from claiming a completed dashboard.

The designer remains a specialist, not an autonomous agent. It independently
chooses eligible KPI/chart IDs, order, widths, titles/labels and approved theme.
Newly selected analyses must be represented by a KPI or chart; the Top layer must
not retrieve unrelated evidence. Initial failure uses a validated deterministic
fallback; failed edits retain the previous design. A repairable proposal error
permits **one** targeted correction, with both calls/failed usage in the ledger.
Provider/deadline failures do not trigger an unbounded loop or SQL refresh.

## Explicit per-widget lineage

`DashboardSpec.schema_version=2` adds a typed `sources` map. Each KPI and chart
references its own accepted result, contract, grain, constraints, SQL provenance
and timestamp. No cross-result sums, inferred joins or concatenated datasets are
created. Source IDs and unique widget IDs are checked independently of titles.
The legacy schema/version defaults remain supported for old saved dashboards.

Scope → KPIs → charts → optional data remains fixed. The first selected chart is
primary; the model controls widths and ordering. Existing Plotly/ECharts adapters
and server-calculated KPI values are reused. Source SQL and technical metadata
live in optional Details; ordinary SQL/Data views remain available inside the
dashboard's Analysis details rather than competing with its preview.

## Editing and saving

Describe changes or select **Edit** on a KPI/chart, then **Edit with AI**. The
server checks dashboard ID, revision/version and widget ID before using them.
Selected edits are constrained to that component; replacing a metric/chart uses
`replaces_widget_id` to retain the component's identity/order while rebinding to
an eligible source. Other widgets, filters and retained data stay unchanged.
Whole-dashboard editing clears the component selection.

Presentation changes reuse complete results and do not execute SQL. Explicit
typed analytical changes may add missing evidence through the same governed SQL
pipeline. Saved selected-widget queries start from that widget's own contract,
not an unrelated dashboard anchor. Existing snapshots cannot silently refresh or
change. Save appends a Draft revision; Save as New, Undo, Versions and explicit
owner approval keep the Phase 3 optimistic concurrency/permission rules.

Permanent documents now retain a bounded `sources` snapshot map in the existing
SQLite revision payload. The anchor remains for backward compatibility. No new
database/table migration is needed beyond Phase 3's migrations: versioned JSON
accepts legacy single-result records and new multi-result documents. Restart/open
validates retained evidence without executing SQL or calling an LLM. A preview
is not persisted until Save. Role/version changes during model work reject stale
or unauthorized previews. External sharing remains disabled; see
[Studio security](DASHBOARD_STUDIO.md).

## Bounds and privacy

- One database, up to **four complete results**, four KPI cards and four charts.
- Existing per-result limit: 500 complete scalar rows; saved document cap: 4 MB.
- Eight tool steps, default shared ceiling 12 model calls and 24 business DB calls;
  existing deadlines/repair/authorization still apply. Older `.env` files with
  an 8-call ceiling may need `TEXT2SQL_ASSISTANT_MAX_MODEL_CALLS=12` for larger goals.
- Designer calls: one normally, two maximum with targeted correction. The old
  Phase 2 API/default specialist still defaults to one call.
- Designer context contains bounded column/role/statistical profiles, category
  frequency counts, approved definitions and eligible references—not raw rows,
  category labels or identifier values. KPI numeric values are not needed in that
  prompt; rendering uses the exact server-calculated values. User-supplied text,
  schemas/definitions and aggregate statistics can still be sensitive: this is
  not universal anonymization or a substitute for provider/data-governance policy.

Unused retained source snapshots are kept for lineage; source-slot exhaustion
requires a narrower/new dashboard, not silently dropping earlier data. The
optional table is the anchor's result, not an invented union of unrelated results.
Temporary undo is per result identity; saved version history handles source-set
changes. RAM preview/bundle handles remain bounded and require sticky routing
across workers. There are no global interactive filters or automatic refresh.

## Verification evidence

`test_multi_dashboard.py` covers independent grains, widget/result mappings,
source limits/database isolation, legacy JSON, multi-source restart, profile
privacy, one correction/no third call, component replacement/preservation,
ordinary queries, and required-analysis failure. Real SQLite + scripted-provider
integration runs three independent governed queries, creates a multi-result
dashboard, saves and reopens it. Another integration adds new evidence to a saved
dashboard and verifies all old snapshots remain exactly unchanged.

Measured scripted call counts (not live-provider timing):

| Scenario | Top calls | SQL generation calls | Designer calls |
|---|---:|---:|---:|
| Three-query dashboard | 4 | 3 | 1 |
| Add missing metric to saved dashboard | 2 | 1 | 1 |
| Valid selected-widget design with retained data | 1 | 0 | 1 |
| Repair an invalid proposal | outside specialist | 0 | 2 maximum |
| Saved reopen | 0 | 0 | 0 |

Frontend regressions cover automatic rendering, absent creation controls,
ordinary responses, per-source KPI/chart attributes, selection metadata, Undo,
renderer cleanup, tables, fixed ordering, and responsive CSS. Existing SQL/chart,
ownership and voice regressions are retained. These are DOM/renderer tests, not
real-browser visual QA. No live-provider analytical accuracy/latency benchmark
or new PostgreSQL run was performed. Deterministic checks prove eligible choices
and retained lineage, not every natural-language business interpretation; existing
UNRESOLVED semantic states are not promoted to verified correctness.
