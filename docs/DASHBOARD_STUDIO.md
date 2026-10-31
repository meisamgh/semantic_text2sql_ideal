# Permanent Dashboard Studio (Phase 3)

## Client workflow

1. Run an analysis and create a Phase 1 or AI-designed Phase 2 dashboard.
2. Choose **Save dashboard**, name it, and explicitly retain a private **Draft**.
3. Open **Dashboard Library** in the sidebar. Saved dashboards reopen from their
   retained snapshot, even after a server restart, without a model or business
   database query.
4. Request design changes using **Update preview**. The existing LangGraph Top
   layer invokes the existing bounded AI dashboard designer. Consecutive previews
   preserve earlier unsaved design choices. No new agent or chart planner exists.
5. **Save** retains the preview as a new draft revision; **Save as New** creates
   an independent dashboard. **Approve & publish** requires explicit confirmation
   of the saved version. It does not publish a public link.
6. **Undo** appends a restoration revision rather than deleting history. **Versions**
   opens earlier immutable snapshots; historical versions can be copied with
   Save as New. Rename and Delete are version-checked; deletion requires confirmation.

Applied filters/scope → KPIs → analytical plots → optional data remains fixed.
Existing Plotly/ECharts and table components render all saved views. Library reads,
Save, Copy, Rename, Undo, Publish, Delete and history do not invoke an LLM or execute
the source SQL. Conversational edits use the Top layer plus at most one designer
call for a valid design; bounded decision-format repairs retain existing budgets.

Saved editing is intentionally snapshot-only: database/query/metadata tools are
blocked even if the Top layer requests them. Different analytical data requires
an explicit new governed analysis in Conversations, followed by a new dashboard.
No automatic refresh, editable global filters, new chart types or agents were added.

## Persistence and consistency

The store uses **the existing `TEXT2SQL_RUN_HISTORY_PATH` SQLite file**, not a new
database service. The default is `data/runtime/query_history.sqlite`. A `:memory:`
history installation returns a clear 503 for permanent Studio operations rather
than pretending to save permanently. Existing temporary dashboards still work.

Two additive, transactional migrations create `saved_dashboards`,
`saved_dashboard_revisions`, `saved_dashboard_access` and
`dashboard_studio_migrations`, then add revision names. They are idempotent and
leave existing query history intact. No destructive down-migration is run.

Every revision retains a typed DashboardSpec and full supporting snapshot: exact
rows/columns, SQL, question, result/run identity, execution timestamp, semantic
contract, source grain, filter metadata and provenance. Snapshot data is NOT a
20-row history preview. Models reject incomplete/truncated results; existing limits
remain 500 complete scalar rows and 4 MB per document. The persistent dashboard ID
is separate from its reused source-result/temporary-design IDs.

Every write uses SQLite `BEGIN IMMEDIATE`, an expected version, and access checks
inside the transaction. Stale saves/renames/approval/undo/delete/copies return 409.
Version numbers are monotonic. A save after publication creates a new Draft and
retains the previous published revision. Undo does not implicitly reapprove data.
The store revalidates KPI calculations, chart data/encodings, source IDs, constraints
and optional table against the retained source using Phase 1/2 validators, not
client-supplied chart code or numbers. A different source cannot overwrite a saved
analysis. Save as New has its own atomic source/version/permission checks.

Studio editing uses request-local branch context; it neither restores data into
the active conversation nor writes provisional conversation state. Validated
previews are bounded to 100 RAM entries and must match owner/dashboard/base version
when saved. Saved revisions survive restarts; unsaved previews do not. Multi-worker
editing currently requires sticky routing for preview handles; saved reads and
optimistic writes share the SQLite file. Snapshots are duplicated per revision;
large revision histories may require a future deduplication/retention policy.

## Security and external-sharing boundary

**External sharing is disabled. No sharing, grant-management or public-link API
is provided. Publishing is an approval state, not anonymous access.**

The repository has no production login/identity/session-ownership integration.
Local mode is a single OS-user workspace, NOT authenticated multi-user security.
It is limited to a loopback client and loopback Host, rejects cross-origin/site
requests and forwarded headers, and uses a server-selected `local-workspace` owner.
Bind the server to `127.0.0.1`; do not expose or reverse-proxy this mode.

Before any shared deployment set `TEXT2SQL_DASHBOARD_LOCAL_WORKSPACE=false`.
Studio accepts a remote principal only from trusted ASGI authentication middleware
with an implemented, stable `BaseUser.identity` (namespace by issuer/tenant).
Client headers, submitted owner IDs and session IDs are never authentication.
Saving conversation results additionally requires the middleware to provide the
trusted `scope['queryflow.authorized_sessions']` capability. Without it, remote
creation is denied. There is no automatic session claiming or fake JWT/login.

Owner/editor/viewer enforcement is implemented and tested at the storage boundary:
owners approve/delete, editors edit drafts, viewers see only explicitly published
versions, and unknown owners receive 404. Delegated access requires authenticated
principals. ACL provisioning is deliberately not exposed until deployment identity,
session/tenant/database authorization and sharing policy are implemented. Tests use
synthetic trusted ASGI identities; these are not verification of production login.

The SQLite file is created private (0600). API responses are private/no-store, and
application history storage is forbidden as a discoverable business SQL database.
Only deliberate snapshot fields are serialized; provider/connection configuration,
arbitrary model context and API keys are excluded. Credential-like result column
names are rejected. Business rows/SQL can still contain sensitive personal data:
this is NOT a universal DLP scanner or encryption-at-rest solution. Operators must
protect/encrypt the runtime volume, backups and access. Deletion removes live rows
and cascading revisions/ACLs, not copies in external backups. Removing a source DB
does not revoke an explicitly saved historical snapshot; delete/revoke its saved
access separately. Clearing conversation history does not delete saved dashboards.

## HTTP interface

- `GET/POST /api/studio`: scoped list; save a server-held validated design revision.
- `GET /api/studio/{id}[?version=N]`: authorized immutable snapshot.
- `GET /api/studio/{id}/versions`: scoped revision metadata.
- `POST .../edit`: conversational, validated unsaved preview.
- `POST .../save`: save preview, or copy with `save_as_new`/optional historical version.
- `POST .../rename`, `.../publish`, `.../undo`: version-checked explicit actions.
- `DELETE ...?expected_version=N`: owner-only permanent deletion.

Mutations reference server-held validated designs/previews; no endpoint accepts
client-authored SQL, rows, KPI values or authoritative DashboardSpec objects.
Lists/history currently return the latest 100 records; older known versions remain
inspectable by ID/version. There is no automatic query rerun during browsing.

## Verification

`test_dashboard_store.py` covers migrations/idempotence, restart equality,
0/1/21/100/500-row snapshots, owner/ACL isolation, explicit approval, append-only
undo, concurrent independent stores, conflicts, credential/config exclusions,
tampered evidence, and deletion cascades. `test_dashboard_studio.py` covers
API lifecycle, local/remote gates, synthetic authenticated identities, preview
save, in-flight version changes, and real Top-layer editing against a reopened
snapshot. Adversarial Top decisions requesting SQL are blocked. Scripted valid
edits use **one Top call + one designer call**; reopen uses **zero**. These are call
counts, not live-provider latency measurements.

`dashboard_studio.cjs` covers Library rendering, reference-only saving, preview →
Save → explicit approval, read-only controls and responsive CSS constraints.
Existing Phase 1/2, Text-to-SQL, chart and voice suites remain regression gates.
Production identity, external sharing, live-provider accuracy/latency and real
browser desktop/mobile pixel/accessibility verification remain unverified.
