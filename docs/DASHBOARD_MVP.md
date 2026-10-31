# Phase 1: deterministic single-result dashboard

Historical Phase 1 UI description below. Phase 4 removes manual creation/mode
controls; use natural-language dashboard requests. The deterministic builder and
API remain supported as a validation/fallback foundation. See AGENTIC_DASHBOARDS.md.

Run a query normally, then click **Create dashboard** on its accepted response.
Select **Include source table** first if desired. The temporary dashboard opens
inside that response, in this fixed order:

1. Read-only applied SQL constraints and verified filter-field metadata.
2. Eligible backend-calculated KPIs.
3. Up to four complementary existing chart types.
4. Optional exact source table.

Closing/reopening the panel reuses its validated specification. This is not a
saved, multi-result dashboard, a conversational dashboard editor, or a global
filter engine. Checkbox/dropdown controls do not pretend to recalculate filters.

## Boundary and API

`POST /api/dashboards` accepts only `session_id`, `db_id`, `result_id` and optional
`include_data` (default false). It returns typed `DashboardSpec`, `DashboardWidget`
and `KPISpec` objects. It resolves the exact ID from the existing current-result
cache or retained chart snapshots, scoped to session and configured database.
There is no SQL input, SQL execution, schema inspection, LLM or agent capability
in `dashboard_service.py`. Missing/unauthorized IDs return 404; incomplete,
truncated, unexecuted, ambiguous-column or unsupported cell data returns 409.
Malformed request IDs return 422. Responses use `Cache-Control: no-store`.

The backend reuses `recommend_charts`, `validate_chart_proposal`, and
`materialize_recommendation`. It selects at most one candidate per chart family;
stack/composition charts additionally require an additive proof. Existing chart
types and Plotly-first/ECharts-fallback rendering are unchanged. The frontend
receives the existing rendering/table callbacks, not a second rendering engine.
All widgets reference the original result, and source SQL/rows/state are unchanged.

## KPI eligibility

- **Direct:** an exact numeric cell from a one-row accepted result.
- **Count:** the number of complete returned SQL groups, labeled as groups—not
  distinct customers, active users, orders, or other inferred entities.
- **Total:** only `SUM(physical_column)` across unique, selected SQL group keys
  from one relation with retained physical-grain metadata. No joins, CTEs,
  subqueries, DISTINCT, windows, HAVING, LIMIT or OFFSET are used for this proof.
- **Average:** arithmetic mean of those disjoint SUM group totals, explicitly
  labeled **Average group total**. It is not claimed to be the source-row mean.
- **Period growth:** two explicit, distinct contract comparison periods, a
  validated temporal field, the same non-temporal group population, an eligible
  additive measure, positive baseline and non-negative current total.

AVG outputs, ratios, distinct counts, joined/limited populations and unknown
source grain do not prove additive totals or growth. Single scalar results remain
eligible for direct display without recalculation. Unknown units are not guessed
from names. Profile/category examples are not promoted into applied filters or
verified value domains. Missing candidates are omitted with a clear empty state.
These conservative checks establish transformations of the accepted result; they
do not independently prove that the original SQL implements business intent.

## Local measurements and verification

30 builds per fixture; existing schema/rows already in RAM:

| Complete result | Build p50 | Build p95 | JSON bytes (with table) |
| --- | ---: | ---: | ---: |
| 4 rows | 1.139 ms | 1.398 ms | 3,823 |
| 500 rows | 18.384 ms | 19.012 ms | 101,027 |

Additional model calls: **0**. Additional business SQL queries: **0**. The API test
disables SQL generation, database execution and schema inspection after setup,
then verifies dashboard creation and unchanged cached result paging. Benchmarks
are reproducible in `tests/test_dashboard.py`; they are not a before/after speedup
claim or a measurement of network/browser latency.

`tests/dashboard_view.cjs` tests rendering callbacks, section order, source
identity, optional table, empty/error/loading states, no duplicate requests,
close/reopen disposal and responsive CSS rules. Existing tests cover asynchronous
Plotly failure to ECharts. Desktop/mobile pixel-level browser QA and live-provider
or PostgreSQL checks have not been performed for this feature.

## Limits

- Temporary RAM-backed result availability; no dashboard persistence or automatic
  historical query reruns. Cache eviction/server restart can make an ID unavailable.
- At most 500 complete retained rows. No extra pages, truncation, LIMIT rewriting,
  automatic SQL, global filters, advanced KPIs or multi-result comparisons.
- Reuses existing local session/database isolation. Session IDs are not a substitute
  for authenticated multi-user/tenant authorization in a public deployment.
- Complex SQL may produce charts/data but no derived KPIs when proof is insufficient.
