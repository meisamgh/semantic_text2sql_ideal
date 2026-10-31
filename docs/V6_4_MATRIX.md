# Queryflow V6.4 requirement matrix

Statuses: NOT_STARTED, IN_PROGRESS, IMPLEMENTED_UNVERIFIED, VERIFIED, BLOCKED, NOT_APPLICABLE. VERIFIED requires code evidence and a relevant passing test. This matrix is a checkpoint, not a completion claim.

| ID | Requirement | Severity | Current status | Source files | Planned change | Test coverage | Acceptance condition | Evidence | Final status |
|---|---|---|---|---|---|---|---|---|---|
| V64-01 | Correct grouping expressions | P0 | IN_PROGRESS | `validator.py`, `agent.py` | Match actual group keys and aliases | `test_semantic_validation.py` | Wrong dimension rejected, equivalent form not falsely rejected | Direct wrong dimension, alias, ordinal and CTE tests pass; transformed lineage remains unresolved | IN_PROGRESS |
| V64-02 | Correct aggregation and measure lineage | P0 | NOT_STARTED | `validator.py`, `semantic_catalog.py` | Trace aggregate input to required measure | `test_semantic_validation.py` | Wrong measure rejected | Direct physical input and simple approved SUM formula tests reject wrong columns; casts, competing aggregates and derived formulas remain unresolved in passing tests | IN_PROGRESS |
| V64-03 | Predicate/filter correctness | P0 | IN_PROGRESS | `validator.py` | Match operand/operator/value in actual predicates | `test_semantic_validation.py` | Literal in unrelated expression not accepted | Wrong column/value and qualified-table tests pass; complex predicate equivalence remains | IN_PROGRESS |
| V64-04 | Temporal correctness | P0 | NOT_STARTED | `validator.py`, `conversation.py` | Reconcile active bounds and storage format | Date-range tests | Wrong period rejected when provable | Explicit temporal boundaries now enter predicate validation; direct wrong bound fails and transformed CTE remains unresolved in passing tests. Active-period equivalence remains open | IN_PROGRESS |
| V64-05 | Join fanout/cardinality | P0 | NOT_STARTED | `validator.py`, `context.py` | Use verified relationships and grain | `test_semantic_validation.py` | Provable fanout rejected | Many-to-many and explicit one-to-many `fanout_risk` aggregate joins yield UNRESOLVED; full grain proof remains | IN_PROGRESS |
| V64-06 | DISTINCT, ranking, NULL, units | P0 | NOT_STARTED | `validator.py`, `models.py` | Typed provable checks with UNRESOLVED fallback | Semantic negative tests | No false PASS | Final ORDER BY metric, direction and LIMIT are checked; wrong direct ranked metric fails, while complex lineage remains UNRESOLVED. DISTINCT, NULL and units remain open | IN_PROGRESS |
| V64-07 | Complete ContractDelta operations | P0 | NOT_STARTED | `conversation.py`, `models.py` | Reconcile dependent fields and clarify conflicts | `test_conversation.py` | Unaffected constraints retained | Existing preservation and conflict tests pass; stale CHANGE_PERIOD previous values now fail instead of silently changing the accepted period. Full delta set remains | IN_PROGRESS |
| V64-08 | Typed TurnContext and result identities | P1 | NOT_STARTED | `conversational_agent.py`, `models.py`, `result_cache.py` | Request-local state and stable IDs | First-turn SQL/plot identity tests | Chart source links exact result | Frozen TurnContext, result/chart IDs, same-turn NEW_QUERY → REFINE → chart, and result-tool read-your-writes tests pass; comprehensive identity matrix remains | IN_PROGRESS |
| V64-09 | Explicit state commit boundaries | P1 | NOT_STARTED | `api.py`, `conversation.py` | Remove intermediate persistence dependency | First-turn SQL+plot tests | No hidden state write | Request-private provisional state feeds subsequent SQL calls without intermediate persistence; original commit version is retained. Concurrent-commit, branch-replay, and cancelled-job preservation tests pass; full cancellation matrix remains | IN_PROGRESS |
| V64-10 | SQL/plot/requery/investigation tool sequences | P1 | NOT_STARTED | `conversational_agent.py`, `plotting_service.py` | Pending goals and bounded continuation | Integration scenarios 1–2, 12–15 | Goal completes or terminal status | Same-turn SQL then plot and saved aggregate → REQUERY_REQUIRED → governed SQL → validated chart pass end-to-end. New adversarial V6.3 tests cover wrong-model chart routing, recap reuse, typed filter correction, premature chart answer, and bounded requery; investigation and complete pending-goal matrix remain | IN_PROGRESS |
| V64-11 | History restore, reset, cancellation | P1 | NOT_STARTED | `api.py`, `run_history.py` | Restore exact source, enforce cancellation | History/concurrency tests | No stale result | Explicit restore still rebinds selected result; parent-run requests use a request-local snapshot/result and an unresolved branch leaves accepted state unchanged. Cancellation and full restore matrix remain | IN_PROGRESS |
| V64-12 | Remove competing routers | P1 | NOT_STARTED | `conversational_bridge.py`, `conversation.py`, `api.py` | Isolate compatibility routing | Routing tests | Top layer sole intent owner | `/api/chat` dispatches to the Top layer before consulting the legacy bridge flag; a both-flags integration test proves only the Top layer responds. Legacy bridge remains an explicit compatibility path | VERIFIED |
| V64-13 | VerifiedSchemaSnapshot reuse | P1 | NOT_STARTED | `service.py`, `agent.py`, `context.py` | Reuse request-scoped schema/profile | `test_agent.py` | Fewer redundant inspections, same authorization | Typed snapshot passed from service; direct agent test proves no second inspect | VERIFIED |
| V64-14 | Approved metric discovery | P1 | NOT_STARTED | `semantic_catalog.py`, `conversational_bridge.py` | One catalog-backed source | Discovery consistency test | Same approved definitions across tools | Conversation metric discovery now reads the configured semantic catalog and excludes inferred metrics; approval test passes. Glossary definitions still coexist, so complete unification remains | IN_PROGRESS |
| V64-15 | One PlottingService and sufficiency gate | P1 | NOT_STARTED | `plotting_service.py`, `api.py`, `visualization.py` | Consolidate chart ownership | Fresh-result and fallback tests | No insufficient chart accepted | PlottingService checks fresh and saved result sufficiency. PR #3 chart-type follow-up behavior is implemented locally: prior axes/metric/series are sent to the model, incompatible proposals are rejected, a validated fallback preserves encodings, and analytical period changes require SQL. Focused tests pass; full chart ownership consolidation remains | IN_PROGRESS |
| V64-16 | Task-specific provider limits and failures | P1 | NOT_STARTED | `llm.py`, `runtime.py`, `conversational_agent.py` | Separate output budgets and typed errors | Provider adapter and Qwen fallback tests | Truncation never silently accepted | Live synthetic Qwen full prompt reproduced strict `json_validate_failed`; JSON-object mode succeeded and locally validated. Groq budgets, safe typed errors, truncation and bounded fallback tests pass. Other adapters remain | IN_PROGRESS |
| V64-17 | Accurate budgets and telemetry | P1 | NOT_STARTED | `runtime.py`, `conversational_agent.py` | Count each actual model/DB call | Ledger tests | Counts match calls | Qwen native-failure + fallback test proves three calls and one failed ledger entry; Groq internal 429 retry now claims shared budget, and a test proves it cannot exceed the one-call cap. Full telemetry remains incomplete | IN_PROGRESS |
| V64-18 | Plotly async fallback | P2 | NOT_STARTED | `web/plotly_view.js`, `web/app.js` | Catch async render rejection | `frontend_rendering.cjs` | ECharts fallback after rejected promise | Rejected Plotly `newPlot` promise activates ECharts through the same stable chart handle in DOM-mocked test | VERIFIED |
| V64-19 | Frontend result synchronization | P1 | NOT_STARTED | `web/app.js`, `chart_explorer.js` | Bind SQL/Data/Answer/chart to result ID | Browser state tests | No stale chart | Chart rendering now rejects a mismatched source-result ID and records result/chart IDs on the response node; positive, mismatch and legacy cases pass. Full multi-turn browser identity matrix remains | IN_PROGRESS |
| V64-20 | Production session authorization | P1 security | BLOCKED | `api.py`, `run_history.py` | Integrate real identity when provided | Multi-user integration test | Cross-user isolation proven | No deployment identity mechanism configured | BLOCKED |
| V64-21 | Full SQLite end-to-end regression | P1 | NOT_STARTED | `tests/` | Real execution and multi-turn assertions | 32 required scenarios | State, SQL, chart and provenance verified | Existing tests plus accepted-contract, ranked-metric, stale-period and parent-branch regressions pass; the 32-scenario matrix is still incomplete | IN_PROGRESS |
| V64-22 | PostgreSQL/live-provider verification | P1 | BLOCKED | `tests/`, provider config | Run optional integration when reachable | Live integration test | Actual provider/DB result | Live synthetic SQLite books run succeeded through Qwen Top decision, JSON fallback, Model 1, SQL generation and read-only execution (fiction=2/history=1). Real debit-card egress was blocked by review; PostgreSQL was NOT RUN | BLOCKED |
# Historical SQL memory refinement

Acceptance coverage is in `tests/test_sql_memory.py`: indexed ordinal/ID lookup,
session/database/reset/branch isolation, real SQLite refresh, historical-source
modification, schema changes, unsafe SQL rejection, concurrent commits and
bounded snapshot retention. Verification results are recorded in the delivery
report; live provider and PostgreSQL checks remain NOT RUN.
SQL-memory acceptance cases: VERIFIED locally (10 cases); full Python suite
487 passed/1 skipped; frontend 127 passed; Ruff, mypy and diff checks passed.

Visualization refinement: existing SQL+plot+type-edit integration now asserts
one SQL generation and one plotting-model call across both turns (previously two
plotting calls). `test_historical_refresh_and_plot_share_new_result_without_regeneration`
verifies refreshed SQL/result/chart lineage. Broader V6.4 statuses above are unchanged.

Integrated proposal acceptance coverage: `test_integrated_chart_planning.py` and
the integrated/malformed/chart-only cases in `test_v63_conversation.py` validate
proposal schema, safe fallback, call counts and source identities. The historical
refresh integration runs with and without an embedded proposal. No claim that
all grain/unit/business semantics are provable; existing unresolved states remain.

### Phase 2 dashboard specialist (2026-10-31)

VERIFIED locally: typed eligible-catalog selection; exact result lineage; one-call
specialist budget; deterministic fallback; no-SQL design/undo; patch preservation
and revision CAS; Top → missing-data → governed SQL → resumed design; fixed
section ordering and existing renderer reuse. Evidence: 15 tests in
test_ai_dashboard_designer.py, existing test_dashboard.py, and 12 frontend cases
in dashboard_view.cjs. Full checks are recorded in V6_4_PROGRESS.md.
NOT RUN: live provider, PostgreSQL and pixel-level browser verification for this
feature. This does not change older V6.4 requirement statuses or claim production
tenant authentication/durable multi-worker dashboard state.

### Phase 3 permanent dashboards (2026-10-31)

VERIFIED locally: additive migration and restart recovery; typed complete source
snapshots; scoped library CRUD; Draft/explicit approval; append-only saved versions
and Undo; optimistic conflicts across independent SQLite connections; reference-only
Save and Copy; unchanged snapshot/source evidence; existing Top-layer/designer
integration; provisional preview continuity; blocked SQL refresh; renderer reuse.
Evidence: test_dashboard_store.py (14 cases), test_dashboard_studio.py (9 cases),
dashboard_studio.cjs (4 cases), full regression 593 Python passed/2 skipped,
180 frontend/voice passed, Ruff/mypy/diff checks passed.

BLOCKED for external sharing: real deployment identity and trusted conversation
session authorization, tenant/database sharing policy and production integration.
Local-only access is not counted as authenticated production ownership. ACL tests
use synthetic ASGI identities and do not verify a login system. No sharing/grant
endpoint is enabled. NOT RUN: live-provider/PostgreSQL and real browser visual QA.
Detailed boundaries are in DASHBOARD_STUDIO.md; older V6.4 statuses are unchanged.
