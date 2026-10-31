# V6.4 progress checkpoint

Local KPI comparison-bubble continuation: removed the visible Understood-as strip
and its response-render call. Each KPI now has a neutral value-side comparison
bubble. Existing PERIOD_CHANGE cards use exact retained baseline/current evidence,
formula validation, known units and chronological observations. Other summary KPIs
show No comparison rather than borrowing an unrelated period growth rate. No
unsupported min/max comparisons, inferred polarity, model or SQL calls. Per-card
container queries wrap the bubble at narrow card widths without changing the grid.
Tests cover comparison evidence and absent extrema baselines. Browser visual
verification remains unavailable.

Treemap/chart-change continuation: added native Plotly flat treemap tiles with
one unique ID/value per source row, and equivalent ECharts fallback; negative,
missing, or nonfinite values are rejected. Non-additive metrics are not newly
offered treemap. Forwarded KPI highlight methods through the shared chart handle.
Added reusable ChartChangeBubble with 36px neutral pill, verified percentage and
48x24 chronological microchart, plus exact values/period/formula tooltip. Existing
comparison eligibility remains strict; absent approved metric-polarity metadata
means neutral styling, never green/red based on sign alone. KPI icon bubbles are
44px with 20px semantic SVGs and matched shadows. Frontend/KPI/bubble tests: 104
passed. Synthetic 500-iteration mock-DOM build/render measured p50 0.037ms and p95
0.045ms, zero network/model/database calls; this is not real-browser render timing.
Full Python suite passed on rerun (one skip); an earlier run hit the pre-existing
timing-sensitive optimization assertion, which passed in isolation. Browser visual
verification remains unavailable. No SQL generation or backend calculations changed.

KPI/icon detail refinement: deterministic metric/calculation icon selection with
neutral fallback; green/amber/violet 40px outlined/shadowed bubbles; consistent
36px forest-green assistant leaf avatar. Chart title now receives a neutral,
result-ID-bound period-change badge only from existing valid comparison evidence.
Periods are formatted from known temporal granularity. Missing, zero, and unsupported
negative baselines remain suppressed; no business-performance polarity inferred.
No KPI calculations, SQL execution, or model calls were added for these UI changes.
KPI/frontend tests: 99 passed. Responsive CSS is preserved; connected-browser visual
validation remains unavailable.

Targeted frontend refinements: mint bordered user bubbles, consistent circular
leaf assistant/loading avatars, conditional chart tab (Data default without a
validated chart), metadata-aware compact number displays with raw values kept for
sorting/CSV, em-dash NULL display, comparison badges in KPI headings, and exact-value
hover text. Plotly range slider disabled, smaller bottom margin, white marker
outlines, and formatted hover text. Sidebar Clear styled with trash icon and
confirmation before any reset/deletion request. Regression verifies cancellation
causes zero API calls and absent charts cannot expose Visualization. Browser
desktop/mobile visual inspection remains unavailable; no backend logic changed.

KPI-only premium visual refinement: reusable icon bubbles and semantic calculation
accents (green primary, amber MIN, violet MAX), compact locale-aware K/M formatting,
right-aligned verified sparklines with extrema markers, and evidence-derived context
badges. Exact raw values/calculations remain in hover and accessible labels. Min/max
cards target exact Plotly source points via curve/point hover, without SQL/model
calls or source-row mutations. Existing calculation eligibility, placement, result
identities, and responsive grid are preserved. Added number/unit/locale, evidence
badge, DOM, and exact point-targeting tests. Connected-browser visual verification
is still unavailable; responsive CSS/DOM tests are not screenshot verification.

KPI completion continuation: cards consume explicit ChartSpec measure units and
accepted approved-metric/aggregation definitions. Additive totals require an exact
approved SUM input, matching SUM stage grouped by the selected temporal dimension,
unique periods, complete source rows, and no contract LIMIT. Other metrics keep
descriptive row statistics. Sparklines use unique, validated temporal values in
chronological order. Growth is emitted only for two explicitly named contract
periods with exact source-row matches and a positive baseline; unavailable periods
and zero/negative baselines suppress growth. Grouped cards retain exact series
names for local Plotly/ECharts highlighting. Currency/percentage units are never
guessed from metric names. Added DOM/unit tests without network or SQL access.
Browser visual verification was attempted but no browser is connected; not verified.

Local visualization styling/KPI continuation: enlarged filled white leaf; area
line width 3 with top-to-bottom fading fill in Plotly/ECharts; zoom-guide text
and reset control hidden while built-in zoom is preserved. Added typed JSDoc KPI
specifications and deterministic complete-result row summaries below charts.
Only numeric non-NULL complete accepted evidence with a stable result ID is used;
no inferred totals, units, growth or comparison baselines. KPI calculation runs
once per response, outside styling rerenders. Minimum/maximum and unweighted
single-series row averages are descriptive statistics, not business-total claims.
No additional SQL or LLM calls. KPI/front-end suites: 87 passed; browser visual
verification remains outstanding. Optional comparative KPIs/sparklines are not
implemented when the accepted data lacks explicit supporting semantics.

Local green frontend continuation: QueryFlow/leaf branding, hidden database selector,
removed Home/data-source navigation, icon-triggered local history search, and user
questions beside avatars. Restored exclusive Visualization/Data/SQL views with
response-local tab and vertical-scroll state. Added retained-row search/sorting,
column visibility, modifier-click cell/row selection and clipboard copying, SQL
download/expansion, and expandable actual backend validation/execution/context
metadata. Raw values remain the CSV source; numeric display adds grouping without
changing source values. Observed value types appear in header tooltips (not claimed
database types). Partial results explicitly scope local search/sort to cached rows.
Verification: 83 frontend tests, full Python suite (one skip), Ruff, mypy, and diff
check passed. Live-provider/browser end-to-end interaction remains unverified.

October 26 results-table update (local, uncommitted):

- Data now has responsive column widths, sticky numeric-aligned headers, drag-to-resize columns, compact small-result sizing, 20/50/100-row pages, a bounded 100-row virtual window, and a full-screen table. The source result and CSV values remain unchanged; truncated CSV export is explicitly labeled as partial.
- The old Visible chart data subview and verbose source-row note are removed. A Map tab is offered only for an explicit map request with a complete result containing valid numeric latitude/longitude columns; it plots exact returned coordinates with Plotly.
- A new `/api/results/page` endpoint accepts only session/database/result identity and an offset/size, never client SQL. Beyond the accepted 500-row snapshot, it re-runs only the saved read-only SQL with an outer ORDER BY, checks the saved prefix for drift, and caps access at 10,000 rows / five seconds. This is bounded offset paging, not a cross-request database snapshot; concurrent source updates can still require rerunning the question. No SQL LIMIT is added or changed.

October 26 single-tool response and view update (local, uncommitted):

- The initial Top-layer decision can now explicitly mark a simple governed SQL result, or SQL plus its requested validated chart, as complete. Those successful paths return the tool's grounded response without a separate final-answer model call. Multi-step refinement, investigation, explanation, and failure handling retain the existing continuation path. The optimization is conditional on the model setting `finish_after_tool`; it is not a guarantee for all provider outputs.
- The frontend no longer has an Answer tab. The response text, KPI, and requested chart remain visible above the Data/SQL tabs; switching technical tabs does not run SQL or regenerate a chart.
- Scripted SQLite regressions verify one Top-layer call for both simple SQL and SQL-plus-chart when marked complete. Frontend tab tests verify the persistent chart area and local switching. Live-provider latency impact is not yet measured.

October 25 same-turn consistency continuation (local, uncommitted):

- Delegated SQL calls now receive the most recent accepted state through a request-private, request-local handoff. The second same-turn SQL call preserves the turn's original optimistic commit version; no intermediate state or result is persisted. API JSON cannot supply this handoff.
- Same-turn result tools (`analyze_result`, `explain_existing_result`, `present_result`) read the current graph response before the persisted cache. A failed second SQL call retains the first accepted result but cannot be reported as a successful refinement.
- Added SQLite end-to-end regressions for NEW_QUERY → REFINE → chart, SQL → each result tool, failed second SQL, historical branch replay, cancellation, and rejection of client-supplied provisional state. Date-bound cleanup now removes only the exact old-year upper boundary, ranking extraction retains multiword metric names, and explicit `fanout_risk` produces an UNRESOLVED finding for one-to-many aggregate joins.
- Full semantic equivalence and the remaining V6.4 scenario matrix are still incomplete; these tests do not establish business correctness for every SQL rewrite.

October 25 provider continuation (uncommitted):

- Live, credential-backed Groq probes used only a synthetic decision prompt: text, JSON-object and tiny strict schema returned HTTP 200. The full Queryflow AgentDecision instructions returned HTTP 400 `json_validate_failed` in strict-schema mode; the same full prompt succeeded in JSON-object mode and passed local Pydantic validation. This establishes a prompt/schema-size-dependent Groq/Qwen strict-decoding failure, not a missing model or invalid key.
- The existing per-model/schema strict-mode failure cache and bounded JSON-object fallback remain active. Groq 429 retries now consume the shared model-call budget; a specific budget-exhausted error is no longer mislabeled as a deadline. Safe failure logs include response format and retry count. `/api/ui-config` now exposes effective assistant, context, SQL and plotting model identities without credentials, and the frontend Pipeline inspector displays them.
- Added a separate True SOTA Responses adapter and opt-in model catalog entry from the prior user request. This is not evidence of live True SOTA connectivity; no `TRUE_SOTA_API_KEY` was configured. The local, untracked `.env` now selects Groq/Qwen for Top-layer, Model 1, SQL and plotting; True SOTA remains selectable by configuration.
- A live end-to-end debit-card `/api/chat` test was blocked by the environment's data-egress review because it would send database schema and profile information to Groq. No such live debit-card request was made. A separate live end-to-end test used only a temporary synthetic books database: the initial strict AgentDecision call failed with `json_validate_failed`, JSON-object fallback succeeded, Model 1 and SQL generation succeeded, and read-only execution returned fiction=2/history=1. The final client answer was correct for those synthetic rows. This does not establish accuracy on BIRD or PostgreSQL.

Baseline: `c2564d4` on `main`. Untracked `screenshots/` belongs to the user and is not part of V6.4 work.

Implemented in this checkpoint:

- Added provable SQL AST checks for direct GROUP BY dimensions, physical aggregate inputs, and required predicate operand/operator/value. Complex lineage stays `UNRESOLVED`; many-to-many aggregate joins surface a fanout warning rather than an unsupported rejection.
- Reconciled several contract edits: metric changes clear stale metric-dependent fields; conflicting equality filters are rejected; period and ranking updates preserve unrelated constraints.
- Passed the request's verified schema/profile snapshot into generation, avoiding a duplicate live inspection in the governed service path.
- Preserved a stable chart handle when Plotly rejects asynchronously and replaced the failed chart with ECharts. A DOM-mocked test rejects the actual `newPlot` promise.

Checks run: Python pytest full suite passed with one skipped test; 71 Deno frontend tests passed; Ruff, mypy over `src`, and `git diff --check` passed. No live-provider or PostgreSQL integration run was performed. These checks do not prove the entire V6.4 brief complete.

October 24 continuation (uncommitted):

- Added frozen request-local `TurnContext`, immutable SQL-result/chart IDs, deferred Top-layer conversation commit, same-turn SQL-to-chart source binding, and historical result restore. The SQL-to-plot and restore regression tests pass; this does not close every concurrency or cancellation scenario.
- Reproduced a Groq Qwen strict-schema `json_validate_failed` on a minimal live decision call. Implemented a bounded JSON-object fallback with local `AgentDecision` validation and a per-model/schema native-failure cache, so subsequent decisions avoid repeating the rejected native mode. An unrelated HTTP 400 remains terminal. Two minimal live JSON-mode decision calls succeeded; no live SQL generation/execution or PostgreSQL test was run.
- Added task-specific Groq completion budgets, safe typed HTTP failures, truncation detection, and tests for 400/401/403/404/429 failures, structured fallback, exact call ledger, and no SQL execution on invalid provider response.
- Verification after these changes: full Python pytest passed with one skipped test; 71 Deno frontend tests passed; Ruff and mypy passed. Remaining V6.4 rows and partial statuses are recorded in `V6_4_MATRIX.md`.
- Added direct temporal-boundary checks and final ORDER BY/LIMIT checks to the conservative semantic validator, with positive, negative, and unresolved CTE/window tests. This does not prove full date-expression or ranked-metric equivalence.
- Applied the data-sufficiency gate after fresh SQL execution as well as on saved results. First-turn SQL plus plotting remains supported; an adversarial fresh result that returns currency instead of requested country now yields `REQUERY_REQUIRED` without invoking the plotting model.
- Wired the existing semantic catalog into conversation metric discovery; only human-approved catalog metrics are exposed. A test verifies approved formula visibility and inferred metric exclusion. Glossary/catalog unification is still incomplete.
- Bound the frontend chart view to `source_result_id` when present; mismatched result identity no longer renders the chart. A JavaScript regression covers matching, mismatching, and pre-identity saved responses.
- Extended approved-metric validation for a narrow, provable case: a single direct aggregate over the catalog's physical input. Wrong direct inputs fail; casts, competing aggregates and derived formulas remain `UNRESOLVED` instead of being treated as proven.
- End-of-continuation checks: the full Python suite passed with one skipped test on rerun; 72 Deno frontend tests, Ruff, mypy, and `git diff --check` passed. An earlier full-suite run had one non-reproducing failure in the legacy conversation test; the isolated test and immediate full rerun both passed. No claim of V6.4 completion follows from these tests.
- Verified that the Top-layer assistant exclusively handles `/api/chat` when both Top and legacy bridge flags are enabled. The legacy path remains available only as a compatibility route when Top mode is off.
- Reconciled explicit date equality filters with a four-digit active period. A conflicting year now requires a comparison-period declaration or a clarified change; positive and negative tests pass.
- Added a Top-layer regression proving that final-answer provider failure after accepted SQL preserves that result, marks the answer unavailable in provenance, and accounts for the failed model call and reported tokens.
- An end-to-end saved-aggregate → `REQUERY_REQUIRED` → governed monthly SQL → validated line-chart test exposed a false HTTP 409. Deferred state commits now use the version accepted at turn start rather than the new SQL state's version; the complete sequence passes and preserves chart source identity.
- Added a concurrent same-session ASGI test: both requests reach SQL generation, then exactly one commits and the other receives HTTP 409. This verifies stale-write rejection in the current process; parent-run restore still writes intermediate state and needs further work.
- Latest verification: full Python pytest passes with one skipped test; 72 Deno frontend tests pass; Ruff, mypy over 55 source files, and `git diff --check` pass. PostgreSQL and a live end-to-end Qwen SQL workflow were not run.

Still open: full semantic equivalence (derived formulas, temporal bounds, ranking, units, proven join fanout), complete delta/state semantics, all plotting-source identity/cancellation cases, conversational tool sequence consolidation, production identity/authorization, and the requested complete SQLite scenario matrix. See `V6_4_MATRIX.md` for individual statuses.

To resume: read this file and `V6_4_MATRIX.md`; inspect `git status`; complete the next high-severity NOT_STARTED or IN_PROGRESS row. Do not reimplement already VERIFIED behavior. Run `uv run pytest -q`, `uv run ruff check src tests`, `uv run mypy src`, `deno test --allow-read tests/frontend_rendering.cjs tests/voice_input_test.mjs`, and `git diff --check` after code changes. Record live-provider/PostgreSQL tests separately as NOT RUN when unavailable.

October 25 conversation-hardening checkpoint (uncommitted):

- The Top-layer prompt now includes the accepted analytical contract's approved metrics, formulas, mandatory filters, period/comparison boundaries, grouping, ranking, measure and correction context. A prompt regression confirms key fields are actually sent.
- A stale `CHANGE_PERIOD.previous_value` is rejected without mutating the previous contract. Simple final ranking validates the requested ranked metric as well as direction/limit. Complex ranking still remains `UNRESOLVED`.
- `parent_run_id` no longer writes a historical snapshot into accepted state before executing the new request. Its state and complete cached result are request-local; a successful branch commits against the current live version. Failed SQL no longer persists a failure-state copy as accepted state. An unresolved-branch regression and the existing branch/restart integration test pass.
- SQL safety success with unresolved semantic findings is now labeled `SQL_SEMANTICS_UNRESOLVED`, not `SQL_SAFETY_VALID`; execution still does not prove business correctness.
- Verification: full Python pytest passed with one skipped test, 72 Deno frontend tests passed, Ruff passed, mypy passed over 55 source files. Live PostgreSQL/provider tests were NOT RUN. Full derived-formula/date equivalence, cancellation matrix and all 32 requested SQLite scenarios remain open; do not mark V6.4 complete.

October 25 PR #3 chart-follow-up integration (local, uncommitted):

- Reviewed the open GitHub PR `#3` and integrated its chart-follow-up behavior into the local working tree without merging the PR or overwriting the earlier uncommitted conversation fixes.
- The plotting model receives the prior validated chart encodings and Top-layer typed changes. A chart-type-only request cannot silently switch x, y, metric, components, or series; invalid proposals fall back to deterministic chart validation preserving the prior encodings. A period change still returns `REQUERY_REQUIRED` before the plotting model is called.
- Replaced the chart proposal prompt with the requested expert business-analytics visualization-planner instructions, retaining the application's JSON schema and bounded result sample.
- Verification: focused visualization/V6.3 tests, full Python pytest (one skipped), 72 Deno frontend tests, Ruff, mypy over 55 source files, and `git diff --check` passed. No live-provider call was made; improved chart quality in live use remains unmeasured.

October 25 Top-layer routing hardening (local, uncommitted):

- Clarified DIRECT_ANSWER, NEW_QUERY, MODIFY_QUERY, VISUALIZE, EXPLAIN, INVESTIGATE and result-reuse instructions in the existing Top-layer prompt. No additional routing model or graph was added.
- Reconciled validated `AgentDecision` proposals against accepted `SemanticContract`, result/chart availability and typed changes. Presentation-only chart edits reuse accepted data; typed analytical changes delegate to governed SQL. Exact result recaps use `present_result` and preserve result identity. Unsupported direct answers to obvious database questions are grounded through `ask_data`.
- A safe `REQUERY_REQUIRED` follow-up now preserves the accepted question and original chart goal, runs one governed refinement and retries charting without a redundant decision call. Fresh mismatched SQL results and repeated insufficient chart results terminate without inventing a chart. This does not prove general semantic equivalence or every ambiguous natural-language change.
- Adversarial SQLite multi-turn tests cover model misroutes, chart-type edits, filter correction, same-turn SQL/chart, result reuse, failed SQL, and a premature answer after a requery signal. Full Python pytest passed with one skipped test; 72 Deno frontend tests, Ruff, mypy over 55 source files, and `git diff --check` passed. Live provider and PostgreSQL tests were NOT RUN.

October 26 same-turn orchestration and performance audit (local, uncommitted):

- Confirmed the request-local provisional result/contract path through SQL → refine → explain and SQL → chart. Added regressions for latest-result use and explanation-only timing isolation. Final answers now guard against unresolved required-tool failures even if a later benign tool observation intervenes.
- The call ledger now records actual Top-layer decision/final-answer durations, task names, retry counts and provider/model; context/SQL planner entries carry provider retry counts, and SQL validation/execution are timed separately. Removed a phantom legacy-interpreter model entry in Top-layer mode. The pipeline panel uses request-local stage timings and per-call metadata, rather than inherited SQL timings on result-only turns.
- Task-specific completion limits and schema-constrained output paths were checked and extended across configured AgentRouter, True SOTA, Groq, JustDoWork, CodeCraft and OpenRouter adapters where their HTTP interfaces support them. The CLI fallback cannot guarantee a per-call completion cap; live-provider behavior remains unverified.
- Corrected a local `.env` typo (`groqre` → `groq`) that made the assistant unavailable. The missing-provider error now names the setting and available configured providers. Restart the API process to reload `.env`.
- Scripted SQLite measurement for SQL → explain: graph workflow 194 ms, SQL generation 1 ms, SQL execution 1 ms; rounded decision/final-answer/validation/explanation stages reported 0 ms. These are mock-model timings, not live-provider latency; graph total includes unclassified orchestration and test-client overhead.
- Verification: full Python pytest passed with one skipped test, 68 Deno frontend tests passed, Ruff passed, mypy passed over 55 source files, and `git diff --check` passed. Live providers and PostgreSQL were NOT RUN.

October 26 local Gemma understanding integration (uncommitted):

- Verified the installed Ollama model tag `gemma4:12b`; local `.env` now selects it for the Top-layer assistant. SQL and plotting configuration is unchanged.
- Added an Ollama conversation completer with task token limits, token accounting, bounded HTTP timeout and native JSON/schema output. The installed Ollama build rejects the complete AgentDecision grammar; a specific capability fallback uses JSON mode and existing local decision validation, remembering the rejected capability per model.
- A live local greeting produced a valid AgentDecision via fallback in 6.98 seconds, reporting 123 tokens. This verifies local completion and parsing, not analytical routing accuracy or production latency. Provider regression tests, full pytest (one skipped), Ruff and mypy passed.
# Local premium chart styling — 2026-10-29

## Executable remote adapters — 2026-10-30

- Added a shared DatabaseDialect type and lazy RemoteDatabaseRegistry, extending
  the existing non-SQLite injection point rather than introducing another pipeline.
- MySQL, SQL Server, Snowflake and BigQuery are registered from explicit environment
  manifests and reach existing schema/retrieval/generation/validation/execution,
  recovery, paging and metadata paths. SQL remains validated and schema-qualified.
- Added parameterized native value lookup, explicit remote profiling CLI, driver
  extras and lockfile, bounded fetches/timeouts, sanitized errors and shutdown cleanup.
- BigQuery performs a cost dry run before the real job and attempts cancellation
  on timeout; the extra operation is charged to the active shared DB-call budget.
- Scripted integration tests cover all four engines via `/api/chat`, source scope,
  forbidden SQL, parameter binding, cost rejection, cancellation and unconfirmed roles.
- Full local verification: 477 passed, 1 skipped Python; 127 frontend passed;
  Ruff, mypy (59 source files) and diff checks passed. These are not live-server tests.
- Official driver import smoke tests passed for MySQL, Snowflake and BigQuery.
  Snowflake import failed under SQLAlchemy 2.1.6; pinning below 2.1 resolved it.
  SQL Server's Python driver is installed but its system ODBC library/driver is absent.
- Live DB credentials/IAM were not provided: live remote queries, live-provider SQL
  accuracy and production performance are NOT RUN. Role confirmation is an admin
  acknowledgement, not automatic server-permission verification. See EXTERNAL_DATABASES.md.
- Existing `.env` credentials and unrelated image additions/deletions were preserved.

## Historical inspection and categorical evidence hardening — local follow-up

- Added `RunHistoryStore.inspect_result` and POST `/api/results/inspect`, reusing
  history and accepted state. Both run/result IDs resolve; returned rows are bounded
  retained snapshots, never silently re-executed or installed as active results.
- Incomplete previews report `complete=false`, `calculation_allowed=false` and a
  warning. Tests cover 19, 20, 21 and 40 rows, memory/disk stores, database/session
  isolation, failed results and resets. Inspection reset cutoffs persist in the
  existing history SQLite database; audit records remain available separately.
- Existing bounded live category lookup now supports parameterized exact matching,
  complete-domain/ambiguity/provenance metadata, remaining cache lifetime, and
  profile-version invalidation. No automatic fuzzy substitution or new scans.
- Removed fallback bottom slider, retaining inside zoom. Explicit comparison-period
  KPI cards may show verified changes/sparklines; no inferred comparison bubble is
  displayed. Existing icons, shadows, raw rows and SQL are preserved.
- Isolated 1,000-call in-memory history microbenchmark: p50 0.0088 ms, p95
  0.0091 ms, zero model/business DB calls. No equivalent baseline or live-provider
  performance comparison was measured; these numbers are not production latency.
- Remaining: Top-layer historical tool exposure, broader fuzzy/alias category
  indexing, end-to-end semantic comparison enforcement, authenticated multi-user
  authorization, PostgreSQL/live-provider checks and browser visual regression.

Shared `chart_theme.js` and CSS palette tokens now align Plotly and ECharts.
Updated line/step markers, tooltip surfaces, legends, bar spacing, pie spacing,
treemap boundaries, signed heatmap scales and box styling without changing source rows.
Added renderer tests for cross-renderer series-color stability, reduced motion,
step geometry and missing-value preservation. Browser screenshot comparison is
NOT RUN: no browser executable or connected browser is available locally.
No backend generation, KPI calculations, SQL execution or provider calls changed.
# SQL-based historical memory refinement — 2026-10-30

Implemented indexed accepted-SQL memory, explicit historical-source delegation,
validated current-data refresh, snapshot-only historical figures, bounded chart
preview retention and transactional refresh commits. See `docs/SQL_MEMORY.md`
and `tests/test_sql_memory.py` for contracts, measurements and acceptance tests.
Live PostgreSQL/provider verification is NOT RUN. No commit or push requested.
Verification: Python 487 passed, 1 skipped (12.78s); frontend 127 passed;
Ruff passed; mypy passed across 59 source files; `git diff --check` passed.
Ten new SQL-memory acceptance cases pass. Existing dependency deprecation
warnings remain. Local metadata benchmark: p50 1.602 → 0.090 ms; p95
1.734 → 0.098 ms, without model or business-database calls.

## Conversational visualization integration — 2026-10-30

Retained one Top-layer orchestrator and existing LLMChartProposals planner.
Added accepted grain/completeness/result identity to chart context, validated
model-free chart-type edits, missing-provider support for those edits, and a
failed-historical-refresh stale-source guard. Historical refresh → chart now has
an actual SQLite integration test. See `VISUALIZATION_INTEGRATION.md` for
measured call-count changes and the explicitly simulated latency benchmark.
Verification: Python 489 passed/1 skipped (12.61s); frontend 127 passed (264ms);
Ruff, mypy (59 source files), and diff checks passed. Added failed historical
refresh → visualization regression: no stale chart, no replacement SQL query.
Live provider, PostgreSQL and browser screenshot verification remain NOT RUN.

## Premium chart presentation refinement — 2026-10-30

Preserved prior uncommitted orchestration work. Frontend-only changes:
- Shared metadata-aware number/period formatting across charts, KPIs and tables;
  explicit compact/exact modes, unchanged raw values, validated calendar dates.
- Default eight-color palette tested at >=3:1 contrast against white for graphical
  marks; collision-aware series assignment retains colors within a palette.
- 3px unsmoothed lines, 8px circular markers with 2px white borders; existing
  disabled bottom slider and zoom/pan/reset behavior preserved.
- Treemap tile separation, exact scatter hover labels, comparison-only visible
  KPI sparklines, focus states, dark surfaces and mobile wrapping.
- Cache-busted frontend assets and 22 new tests spanning all 19 chart types,
  raw-data integrity, formatting, contrast and responsive sizing constraints.
Verification: frontend 149 passed; Python 489 passed/1 skipped; Ruff/mypy/diff
checks passed. Existing tests cover renderer rejection fallback, comparison
evidence and no-network display changes. No live browser pixel-level or screen
reader verification performed. Color contrast does not prove color-vision
deficiency distinguishability. No SQL/LLM calls added; no runtime speedup claimed.

## Integrated first-layer visualization proposals — 2026-10-30

Added optional existing LLMChartProposal to AgentDecision and strict native JSON
schema/examples. Request-local proposal propagation validates after governed SQL;
valid proposals skip the plotting LLM, missing/rejected proposals use the same
existing recommender. Shared validation rejects silent measure/series changes.
Top-layer chart-only reconciliation preserves completion flags. Prior uncommitted
frontend and memory work remains intact. See VISUALIZATION_INTEGRATION.md for
measured calls and the scripted latency benchmark (not production claims).
Acceptance tests cover two-call SQL+chart, fallback, historical refresh, chart
reuse/lineage, malformed decision repair, incomplete results, incompatible units,
duplicate grain and exact multi-measure preservation. Live-provider/PostgreSQL
verification remains NOT RUN; no commit/push requested.
Verification: 507 Python tests passed, 1 skipped (13.93s); 149 frontend tests
passed (317ms); Ruff, mypy across 59 source files, and diff checks passed.

## Compact chart polish — 2026-10-30

Kept existing chart heights and KPI calculations. Added shared readable titles,
non-mutating category label thinning/ellipsis, width-aware tick budgets, legends
above Plotly charts, stable pie/donut source order, cleaner composition labels,
readable KPI titles and accessible renderer-failure messages. Exact source labels
remain in hover/data; no SQL, model, result identity or analytical behavior changes.
Asset versions updated so browsers fetch the current presentation. New frontend
cases assert 21-row integrity while thinning labels, unchanged 260px sample height,
original pie values/order, title semantics and dense-line behavior. No browser
screenshot/visual QA performed; these are renderer-option and DOM-mocked checks.
Verification: 153 frontend tests passed (304ms); Python 507 passed/1 skipped
(15.53s); Ruff, mypy across 59 source files, and diff checks passed.

## Stable chart selector and history restoration — 2026-10-30

- Chart choices preserve the original recommendation order while selected types
  change; compatibility derives from accepted encodings, not the current type.
- Browser display snapshots retain only complete chart rows/specifications with
  session/database/run/result/chart identity and SQL/column checks (100-view cap).
  New conversation resets analytical state without deleting displayed history;
  Clear history clears the browser cache.
- The history-turn endpoint hydrates exact complete charts from a bounded
  process-local index of existing RecentResult snapshots, keyed by session,
  database and run, validating result/chart IDs and SQL. It does not overwrite
  active state or execute SQL. This supports page reload while the server retains
  the snapshot. Clearing history removes its display index.
- Persisted history stays bounded to its configured preview. After cache eviction
  or server restart, incomplete previews remain incomplete; no hidden rerun or
  fabricated chart. UI metadata distinguishes complete retention from previews.
- Tests cover stable type switching, 19/20/21/100-row browser restoration,
  identity/session/database isolation, invalid snapshots, cache bounds and a
  real SQLite 21-row chart across new conversations/page reload. The API test
  forbids business DB execution while opening history and checks restart behavior.
Browser visual testing remains NOT RUN. Existing uncommitted work is preserved.
Verification: frontend 160 passed; Python 508 passed/1 skipped (16.38s);
Ruff, mypy and diff checks passed. Initial full run had an existing optimizer
timing-dependent assertion fail; isolated rerun and full rerun passed unchanged.
Already-discarded old snapshots cannot be reconstructed without an explicit rerun.

## Deterministic dashboard Phase 1 — 2026-10-31

Implemented focused typed dashboard service/models, scoped cache-only API, and
separate dashboard_view.js module (three-line app integration). Fixed sections
are filters → KPIs → charts → optional table. No agent/LLM, database or schema
query capability; existing chart validation/rendering is reused. KPI eligibility
uses accepted SQL AST, disjoint output groups and retained source-grain metadata,
not business-name heuristics. Distinct counts, averages of averages, joins and
insufficient comparison evidence are rejected for reaggregation. See
`docs/DASHBOARD_MVP.md` for contracts, benchmark evidence and limitations.
The unrelated working-tree deletion of improvement.md was not modified.
Verification: Python 528 passed/1 skipped; frontend 170 passed; Ruff, mypy
(60 source files) and diff checks passed. Local 30-build benchmarks: 4-row p50
1.139ms/p95 1.398ms, 3,823 JSON bytes; 500-row p50 18.384ms/p95 19.012ms,
101,027 bytes. These measure backend build cost, not browser/network latency.
Additional SQL/LLM calls are zero, tested with execution/inspection/generation
forbidden after initial setup. Browser pixel-level mobile/desktop, PostgreSQL
and live-provider verification remain NOT RUN. No commit or push requested.

## CTE authorization and PostgreSQL timeout hardening — 2026-10-31

Replaced global CTE-name exclusion with lexical scope resolution on a normalized
AST copy, preserving actual SQL identifiers. Remote preparation independently
checks each physical table against its allowlist in addition to existing schema
and catalog checks. Fanout analysis uses the same physical-source classification.
Added adversarial tests for qualified CTE shadowing, nested/forward/self scopes,
quoted alias case, qualified identities and permissive-validator defense in depth
across four scripted adapters; positive CTE cases remain supported.

Replaced bound SET LOCAL with parameterized transaction-local set_config and
tested clamping plus query binding. Real PostgreSQL 15.19/Psycopg 3 verification
passed against an isolated UTF-8 database and unprivileged role: timeout settings,
read-only mode, cancellation, recovery and no session-setting leakage. Server
stopped afterward. Full suite: 556 passed/1 skipped; final PostgreSQL focused
run: 8 passed/1 skipped; frontend: 170 passed; Ruff/mypy/diff checks passed.
See `docs/SECURITY_REGRESSIONS.md`. No production DB, commit or push was changed;
the unrelated improvement.md deletion remains untouched.

## Phase 2 AI dashboard designer — 2026-10-31

Added a one-call dashboard specialist under the existing Top-layer LangGraph
assistant. Typed proposals select eligible KPI IDs and validated chart candidates
from bounded result profiles, with deterministic materialization, lineage checks,
unchanged data, fixed section order and safe fallback. Revisions use patches to
preserve unaffected widgets; scoped RAM revision history supports undo and
compare-and-set commits. Missing analytical data returns REQUERY_REQUIRED to the
Top layer; an accepted governed requery resumes the original dashboard goal.

The explicit UI now offers deterministic/AI design, inline design-change requests,
automatic validated previews and undo. Existing Plotly/ECharts/table callbacks
remain the only renderers. Config is documented in .env.example; no local .env
was edited. Source modules: ai_dashboard_designer, dashboard_service, runtime,
conversational_agent, api, models and task budgets in llm; frontend dashboard_view,
small app integration, styles and cache-busted index. See AI_DASHBOARD_DESIGNER.md.

Acceptance coverage: 15 designer tests plus existing Phase 1 tests; frontend
dashboard coverage expanded to 12 tests. Full Python run: 570 passed/2 skipped;
frontend: 172 passed. Ruff, mypy (61 modules) and git diff --check passed.
30 identical four-row scripted builds measured Phase 1 p50/p95 1.266/1.662 ms
and Phase 2 2.868/4.173 ms. These exclude live inference/network/browser time.
Top design acceptance uses one Top call + one specialist call, no final-answer
call or SQL; initial invalid designs do not retry. API integration forbids DB
execution/inspection/generation after its original real SQLite query.

Limitations: live-provider/PostgreSQL checks and pixel-level desktop/mobile QA
NOT RUN for this phase; no new multi-user authentication, durable dashboard
storage or editable global filters. Conservative annotation validation can reject
legitimate numbered titles/aliases. Existing security fixes and unrelated
improvement.md deletion were preserved. No commit or push was requested.

## Phase 3 Permanent Dashboard Studio — 2026-10-31

Verified the clean local main checkpoint (32e3426) and the existing Phase 1/2
services/tests before extending them. Added dashboard_store.py with typed retained
snapshots/documents and two additive migrations in the existing history SQLite
file. Durable revisions retain source SQL, rows, semantic definitions, result/run
identity, provenance, owner and timestamps. Evidence is revalidated before saving,
reopening and approving, without executing source SQL. Snapshot tables cannot be
exposed as a registered business database. Private file mode and serialized
field selection protect credentials/configuration; this is not universal DLP or
encrypted storage.

dashboard_studio.py provides scoped Library/open/Save/Save as New/rename/delete,
Draft/explicit owner-approved Published, history and append-only Undo. Atomic
SQLite version checks reject stale writes and copies. Drafts are hidden from
viewers; editors cannot approve/delete. Validated preview handles bind owner,
dashboard and base version; consecutive unsaved edits preserve the previous
preview. A model-time version change rejects the stale preview. Approval validates
before committing, so corrupt/unsupported evidence cannot partially publish.

The existing LangGraph Top layer and Phase 2 designer handle saved design edits
through isolated request-local branch context. Saved editing forbids SQL/metadata
DB tools, even on an adversarial Top decision. It cannot refresh or overwrite the
active conversation. New analytical data requires a separate governed analysis.
Frontend dashboard_studio.js reuses the existing dashboard/Plotly/ECharts/table
renderers and adds a Library dialog, explicit save/approval/version controls and
unsaved previews. Only one-line integration hooks were added to app initialization
and dashboard mounting. No new agent/framework/chart types/global filters.

Checks: full Python 593 passed/2 skipped (18.02s); frontend/voice 180 passed;
Ruff, mypy (63 source modules), and git diff --check passed. New acceptance cases:
14 storage cases, 9 API/orchestration cases, 4 frontend cases. Valid scripted
saved editing uses one Top call plus one designer call; reopening uses zero model
and zero business-DB calls. Tests cover 0/1/21/100/500 complete rows, restart,
concurrency, ownership, ACLs, approval, invalid evidence, undo, history isolation,
configuration-secret exclusion and source consistency. These counts are not a
live-provider latency or accuracy benchmark.

External sharing remains DISABLED/BLOCKED: no legitimate deployment login/session
ownership mechanism exists. Local Studio is loopback-only, server-owned workspace
access, not production authentication. Trusted ASGI stable identities and explicit
authorized-session scope are required for remote creation. No fake login, header
identity, automatic session claiming, or public/grant API was added. Exact missing
deployment prerequisites are in DASHBOARD_STUDIO.md. Live providers, PostgreSQL and
real desktop/mobile browser visual QA NOT RUN. RAM previews need sticky routing
across workers and are not durable until Save; published snapshots remain permanent.
No commit, push, actual .env edit, new DB refresh or external infrastructure change.

### Phase 4 conversation-first multi-result dashboards (2026-10-31)

Implemented automatic dashboard composition through the existing Top-layer agent
and bounded designer. Independent governed SQL objectives retain separate typed
source identities, contracts, snapshots and provenance; no cross-result joins or
aggregations are introduced. Ordinary SQL questions expose no dashboard-creation
controls. Selected-component edits preserve unaffected widgets, and explicit
analytical additions append accepted evidence without replacing old snapshots.
Saved v1 dashboards remain readable; v2 documents reuse the existing store/CAS.

Designer inputs omit raw rows and category values. One targeted correction is
allowed for repairable invalid proposals under the shared budget. The example
model-call ceiling is 12; the actual local .env was not changed. Dashboard scope,
KPIs, charts and optional anchor table retain the fixed section ordering.

Verification: full Python 605 passed/2 skipped (17.92s); frontend/voice 180 passed
(537ms); Ruff, mypy (64 source modules), and git diff --check passed. New evidence
is in test_multi_dashboard.py (11 cases), extended test_dashboard_studio.py,
dashboard_view.cjs and dashboard_studio.cjs. Real SQLite with scripted providers
verifies three SQL objectives followed by composition: 4 Top decisions + 3 SQL
generation calls + 1 designer call, with no separate final-prose call. A saved
analytical addition uses 2 Top + 1 SQL generation + 1 designer call. These are
fixture measurements, not production latency/token or semantic-accuracy claims.

Limits: four sources in one authorized database, four KPIs/four charts, complete
bounded results only, optional data table for the anchor source. No interactive
global filters or automatic refresh. Temporary bundles require sticky routing;
saved snapshots survive restart. External sharing stays BLOCKED on legitimate
deployment identity/session authorization. Live providers, PostgreSQL and browser
visual QA NOT RUN. No commit or push; restart/reload is needed to load the UI.
See AGENTIC_DASHBOARDS.md for source files, behavior and remaining boundaries.

Publication review: rejected duplicate effective replacement targets before
materialization, preventing silent same-widget overwrite. Regression preserves
the original widgets on rejection. Final checks: 606 Python passed/2 skipped
(18.12s), 180 frontend/voice passed (515ms), Ruff/mypy/diff passed. Live-provider,
PostgreSQL and browser visual checks remain NOT RUN.
