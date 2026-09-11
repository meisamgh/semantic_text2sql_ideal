**Source-grounded V5 review and recommended V6 architecture**

Reviewed 10 September 2026, local checkout `semantic_text2sql_ideal`, HEAD `1e7e0cc`. This is an assessment of the implementation, not an implementation of the proposed fixes. The pre-existing edit to `improvement.md` was preserved.

**Verdict:** a promising, compact Text-to-SQL research application with useful retrieval and inspectable context, but insufficient controls for unattended business decisions or a shared production deployment. Its central weakness is that business meaning is primarily a generation instruction. The acceptance gate establishes much less than the context and recovery terminology can suggest.

Labels throughout: **CONFIRMED FROM CODE** means inspected implementation or a reproduced behavior; **INFERENCE** means an architectural judgment or a deployment-dependent risk; **RECOMMENDATION** means proposed work. Scores and expected benefits are judgments, not measured accuracy.

**1. Evidence, scope, and verification**

Read the core service, agent, API, model adapters and prompts, schema linkers, history, glossary/formulas, context planner, profiling, both database backends, recovery, conversation/explanation code, benchmark runners and saved results, training/evaluation code, relevant tests, UI code, configuration, SQL setup, and local commit history. The `autonomous_analytics` package was inspected as an adjacent foundation; it is not a complete running autonomous investigation system in this checkout.

Current verification:

| Check | Result and limit |
|---|---|
| `.venv/bin/python -m pytest` | **116 passed, 1 skipped**, 53 dependency deprecation warnings. This does not measure LLM SQL accuracy. |
| `.venv/bin/ruff check src tests` | Passed. |
| `.venv/bin/python -m mypy src/semantic_text2sql src/autonomous_analytics` | Passed across 37 source files. |
| Bare `.venv/bin/mypy` | Failed package discovery because `autonomous_analytics` lacks a `py.typed` marker in the installed-package path. Explicit source checking succeeds. Treat as tooling/package configuration drift, not a type-error finding. |
| [Offline reproductions][REPRO] | Passed. Synthetic SQLite data and fake model responses; no provider charges or external database changes. [Captured observations][RESULTS]. |
| GitHub issues and PRs | Attempted open and closed listings with `gh`; network connection failed. Browser retrieval also failed. No claim that the repository has no issues/PRs. |
| Live provider, PostgreSQL, UI rendering, load tests | Not verified in this review. No current accuracy percentage or production SLA established. |

`uv run pytest` initially could not access its sandboxed cache; the existing environment ran the suite successfully. Historical memory was used only for orientation and was checked against current code. Notably, current recovery differs substantially from older notes.

Local history matters: `e8ed331` added agentic anomaly recovery and removed `value_grounding.py` plus its tests; `3443974` strengthened the prompt requiring live value probes. The current runtime does not enforce that prompt requirement before accepting a diagnosis. Earlier `6877866`/`09ed727` commits concern recovery effort/accounting, but their intent is not evidence that current accounting and budgets work. [Recovery implementation][R]

**2. The actual architecture**

**CONFIRMED FROM CODE:** the main flow is:

```text
Browser POST /api/chat/jobs -> in-process asyncio task -> chat()
  -> session lookup and rule-based operation classification
  -> optional LLM interpretation of a stateful turn
  -> resolve question from root question + chronological modifications
  -> TextToSQLService.execute_question()
       inspect live schema + load offline profile/glossary
       retrieve historical schema features
       BM25 + dense + profile-value matching -> RRF
       optional LightGBM column reranker
       table/column selection + FK bridge closure
       optional LLM context selector (model1)
       deterministic identifier/dependency verification
       optional 0–2 historical SQL examples
       TextToSQLAgent.generate()
         second schema selection/adaptation
         compact context JSON
         SQL model call
         syntax/write-operation validation
         optional SQLGlot simplify + equivalence/performance experiment
         execute (when requested)
         on first failed SQL: deterministic recovery, then optional LLM tool loop
         repair, at most 3 total SQL-generation attempts
  -> accepted state is saved
  -> API-only zero-row / any-NULL diagnosis, or correctness comparison
  -> SQL, rows, context, attempts, partial usage -> browser polling/rendering
```

The normal generation path does **not** call the available database `explain()` methods. There is no routine semantic verifier between execution and acceptance. `execute=False` skips database execution and can nevertheless return `execution_status="EXECUTABLE"`. [Agent][A] [Service][S] [API][API]

| Component kind | Actual implementation |
|---|---|
| Deterministic | Operation rules, BM25, value matches, RRF, FK graph closure, context verification, profile inference, AST validation, SQL simplification, result comparisons, filter probes. |
| ML | FastEmbed dense encoder; optional LightGBM ranking model. No evidence of an online-trained SQL model in this repository. |
| LLM | Optional turn interpreter; optional context selector; SQL generation/repair; requested explanations; recovery reasoning/tool choice. |
| Agentic | The `ainvestigate()` loop selects metadata or SQL-probe tools and terminal actions. The compiled LangGraph itself has only `classify -> gather -> END`; it is the deterministic fallback, not the agentic loop. |
| Adjacent analytics | Typed evidence/KPI/investigation models, deterministic snapshot calculations, registry and `TextToSQLTool.ask()` over the service. No implemented autonomous Scout/Investigator orchestration should be inferred from these foundations. |

Evidence: [retrieval][H], [recovery][R], [LLM adapters][L], [analytics tool][AT], [snapshot builder][SN].

State is split between process-local `ConversationStore`, `chat_jobs`, `chat_tasks`, request/response Pydantic objects, per-attempt Python variables, profile/glossary files and caches, the loaded history corpus, embedding cache, and browser localStorage session ID. Profile timestamps refresh the file cache, not the relationship between a profile and current database contents. There is no durable, owner-bound conversation transaction or durable unified call ledger. The store lock protects individual dictionary operations, not an entire asynchronous turn. [ConversationStore][C] [API][API] [ProfileStore][P]

**INFERENCE:** the service extraction is useful, but orchestration is still split incorrectly: analytics callers use `execute_question()` and miss API-only anomaly diagnosis. Conversely, a chat call can acquire additional cost after the service has finished. The second linker overlaps with the hybrid retriever, although the usual service path supplies required tables/columns so it often acts as an adapter rather than a second independent pruning pass. Direct agent callers have different behavior. [S] [A] [AT]

**README disagreements confirmed from code:**

| Documented behavior | Actual behavior |
|---|---|
| Mandatory observed format for selected dates | Missing requirements are recorded, but neither coverage nor a required date-format gate blocks generation. |
| `REPAIR / INFORM / ESCALATE` controls anomaly outcomes | API zero/NULL paths take `diagnosis_summary`; they do not dispatch repair or surface escalation as a review request. |
| Live value evidence required before value diagnosis | Prompt says this; terminal-action parsing accepts a claim without any tool observations. |
| Recovery bounds and token accounting | LLM waits lack a shared deadline; fallback probes and agent phase have separate budgets; several calls disappear from totals. |
| Unexpected NULL triggers investigation | Any NULL in returned rows triggers it, including intentionally optional fields. |
| Three SQL attempts | Correct for `GenerateRequest`; unrelated recovery LLM calls and DB probes add further work. |

The README correctly describes safety-only SQL validation and explicitly avoids claiming benchmark accuracy. Its test count is simply stale by one passing test. [README][README] [A] [R] [API] [CTX]

**3. Text-to-SQL quality and plausible wrong answers**

| Area | CONFIRMED FROM CODE | INFERENCE: remaining failure |
|---|---|---|
| Schema/table linking | Joint table and column documents; capped table shortlist, relative-score threshold, column rescue, glossary additions, shortest FK paths. [H] | Missing a fact table before generation can prevent every later attempt from answering correctly. Dense/RRF ranks are not calibrated probabilities. |
| Column selection | Usually top five relevance columns plus keys and glossary requirements; verified identifiers are propagated through the contract. [H] [CP] | Ranking/denominator/date dependencies implied by meaning can be pruned despite low lexical overlap. |
| Values | Exact case-folded boundary matches against profiled top/allowed/example values; later live recovery probes. [H] [R] | Rare entities, abbreviations, code-to-label mappings and multilingual values are poorly covered. There is no general pre-generation live value resolver. |
| Historical queries | BM25 and hand-designed semantic signatures; at most two advisory examples; history features can feed reranking even when ICL is off. [HIST] [S] | Similar vocabulary does not imply the same denominator, population, grain, or period. `success=True` is admission metadata, not independent business validation. |
| Glossary | Versioned terms, synonyms, definitions, column dependencies, grain/caveats, formulas. Token-overlap selection. [G] | Single-token matches can activate irrelevant terms. `value_aliases` is declared but not consumed. Definitions are prose rather than executable semantic constraints. |
| Formulas | Approved operators and arguments are attached and the prompt asks for exact compilation. [F] [L] | No deterministic compilation or AST equivalence check guarantees the generated expression uses them. Ratio-of-sums versus average-of-ratios remains a major risk. |
| PK/FK and bridges | FK graph restoration and composite-key profiling exist. [H] [P] | Profile-to-context conversion loses composite key components; shortest path is not necessarily the intended relationship. Partial profiles can suppress live relationships. |
| Grain/cardinality | Table grain, unique keys, cardinality and fanout hints appear in context. Prompt suggests EXISTS/preaggregation. [CTX] [L] | No enforced measure-lineage or aggregate-grain check. Data-observed uniqueness can change, and declared uniqueness is not fully distinguished from sampled/observed facts. |
| Dates/time | Offline format/coverage inference; special YYYYMM prompt guidance. [P] [L] | Six-digit identifiers misclassify as dates; time zone excluded; no as-of time, fiscal calendar or business-period boundary contract. Native dates without profiles lose their type. |
| Generation | One SQL candidate per call, with configurable style; same style generally retained on repair. [A] [L] | Repairs tend to share the original interpretation and context omissions. |
| Validation | One parsed Query, forbidden writes and SELECT INTO blocked. Unknown tables/columns intentionally deferred to execution. [V] [VT] | Executable wrong joins, omitted filters, wrong population and ranking remain valid. Function/schema authorization is absent. |
| Execution feedback | SQL errors trigger bounded attempt repair; API inspects empty/NULL output. [A] [API] | COUNT returning zero, a wrong positive aggregate, duplicate multiplication or omitted NULL groups can evade anomaly handling. |
| Semantic correctness | Explicit review produces another candidate and compares bounded results. [API] | Agreement is corroboration only. Both candidates can share the same wrong schema, assumption, prompt, or model bias. |

Five concrete examples of answers that can pass the current gate:

1. **Revenue fanout:** join orders to line items and sum the order-level total; each order repeats per item. The output is a plausible positive total. Context warns against fanout but acceptance does not test it. [CTX] [V]
2. **Wrong denominator:** average customer conversion rates instead of total conversions divided by total eligible customers. Both return a reasonable percentage. A formula prompt cannot establish AST compliance. [F] [L]
3. **Wrong population:** “active customers” becomes anyone with an order rather than the approved activity window/status definition. Missing business semantics are not recoverable from successful execution. [G] [V]
4. **Wrong period:** compare partial current month with complete prior month, or interpret UTC timestamps as local business dates. The context explicitly excludes timezone and there is no reference clock contract. [CTX]
5. **Wrong extremum/ties:** select the customer with the smallest monthly record instead of the smallest annual total, or return one arbitrary tie when all are required. Neither syntax nor a nonempty result exposes the interpretation error. [A] [L]

**RECOMMENDATION:** enforce only semantics that are actually grounded: required metric lineage, denominator, grain, population filters, temporal boundary and tie policy. Use `UNRESOLVED` for absent authority. Do not replace a missing definition with a deterministic guess and then call it verified.

**4. Retrieval, profiling, and leakage**

**INFERENCE:** BM25, compact physical formats, glossary dependencies and relational closure offer the best complexity-to-value ratio. Dense retrieval is plausible for descriptions/synonyms, but no checked-in paired experiment isolates its SQL-accuracy benefit. The fallback defect currently prevents a clean BM25-only baseline. Column documents omit the table name despite accepting it as an argument, weakening disambiguation of columns such as `name`, `status` and `date`. [H]

The optional reranker has encouraging **historical artifact evidence**, not current end-to-end proof: saved metrics report column recall rising from **0.7484 to 0.8368**, with table exact recall **0.8551** unchanged, on **69 holdout questions**, with **290 training** and **40 skipped**. However, offline evaluation scores all identifiers with a learned model; runtime only reranks the top-30 mixed table/column pool, keeps RRF table selection, applies a relative cutoff, and restores dependencies. Training also omits runtime profiles/glossary. This is a material evaluation mismatch. [RM] [TRAIN] [H]

History is not dense semantic retrieval: `_signature_from_text()` and associated scoring use operations, lexical business concepts, grain and temporal tokens. Its hand-written concepts emphasize the debit-card domain. The service computes history schema evidence unconditionally, while only the optional reranker consumes those features. When reranking is disabled, that work brings no ranking gain. [HIST] [S] [H]

Profiling runs one table count, individual column scans up to 10,000 values, some DISTINCT scans, exact date min/max queries, and multiple uniqueness scans per relationship. Limits bound retrieved rows, not total server work. Samples are leading rows without randomization; formats inspect the first 20 non-NULL values. Observed exactness and database freshness are separate concerns. Neither profile load nor generation verifies a database/schema version against the profile. [P]

Leakage assessment:

- **CONFIRMED:** the history builder requires 399 RAG IDs, a one-entry excluded-duplicate list and no overlap with test IDs. Training excludes a record's own identifier from history features and uses training history for holdout features. These are useful protections. [BUILD] [TRAIN]
- **CONFIRMED:** the builder trusts the split's duplicate list; it does not recompute SQL equivalence or paraphrase/template separation. Runtime history has no evaluator-provided exclusion ID in the service call. [BUILD] [S]
- **INFERENCE:** random question splits within the same database can share SQL templates and domain semantics; they do not measure unseen-database generalization. Curating glossary entries against repeatedly inspected hard cases can also contaminate later evaluation. This review does not establish that protected labels actually leaked.
- **RECOMMENDATION:** freeze dataset, database, glossary/profile/history hashes and commit; separate by question family and database where possible; deduplicate structural templates and paraphrases; evaluate train-derived metadata separately from manually curated business definitions.

Table-first versus column-first is a false choice for this code: hybrid retrieval already uses both, then applies a table bottleneck. Keep joint retrieval but use separate table/column candidate pools and a dependency graph: identify likely metric/filter columns, collect parent tables, add exact composite joins, and preserve one alternative path when the margin is low. A mandatory metric operand should outrank the top-k budget. Model 1 should not be used to rescue information it never received. [H] [CP]

Improve retrieval without materially increasing tokens by using typed column identity (`table.column`, type, short description), clause-role-aware ranking, approved value aliases, targeted exact lookups for candidate values, compact rare-value hints, and schema-versioned cached documents. Replace redundant descriptive text with structured authoritative facts. Expand context only when a concrete dependency or ambiguity remains. Measure complete gold-dependency recall and final execution correctness, not only mean column recall.

**5. LLM calls, prompts, and actual cost drivers**

All context-size figures below are **INFERENCE**, approximate input-token ranges for short questions and small retrieved schemas, not measured provider usage or enforced ceilings. The current estimator is characters divided by four. Long conversations, schema metadata and recovery feedback can exceed these ranges. [CTX]

| Call site | Purpose, input and output | Typical input estimate | Necessity, deterministic alternative and failure modes |
|---|---|---:|---|
| `interpret_turn_detailed` [C] | Current message + prior root/resolved request + prior SQL -> operation/resolved instruction/confidence JSON. | 0.7–2.5k | Useful for ambiguous follow-ups; rules already handle explicit edits. Can misclassify or invent an interpretation; self-reported confidence is uncalibrated. History grows without a compact canonical state. |
| `plan_context_detailed` [CP] | Pruned identifier list, topology, relevant glossary, prior context, JSON schema -> selected tables/columns/metadata. | 1–4k | Optional. It explicitly must not define metrics, filters or grain. Saved A/B runs do not establish accuracy benefit. Cannot select pruned-out objects; malformed JSON falls back with lost call usage. |
| `_prompt` through provider `generate` [L] [A] | Question/evidence + compact context -> SQL only. | 1–4k initially | Necessary for general free-form questions; governed known metrics can use templates. Primary source of wrong aggregation, omitted conditions and invalid literals. “Trusted evidence” is caller text, not independently authenticated evidence. |
| SQL repair through the same generator [A] | Initial context + failed SQL, feedback, rejected fingerprints, recovery evidence -> SQL. | 2–8k | At most two additional generation attempts. Use deterministic dialect repairs only when provably semantics-preserving; avoid speculative rewrites. Rejected SHA hashes do not tell a model the rejected structure. |
| `ainvestigate` [R] | Question, SQL, failure, allowlist, growing observations -> tool action or final diagnosis/instruction. | 0.6–4k per call, up to four calls | Deterministic probes can answer explicit value/date questions. Model useful for choosing a targeted diagnostic for a complex error. It can stop without evidence, hallucinate a diagnosis, or spend four tool requests without a final action. |
| `explain_turn_detailed` [E] | Accepted SQL, extracted facts, schema context and result metadata -> prose. | 1–4k | Optional presentation call; deterministic fallback exists. Actual result values are not stored in this prompt, so causal explanations of a particular numeric answer are unsupported. Contradiction checks cover a narrow join-language case. |
| `_optimization_prompt` [L] [A] | Baseline SQL + referenced context + EXPLAIN placeholder -> replacement SQL. | 0.6–3k | Explicit optimization only; deterministic simplify may avoid it. EXPLAIN is generally unavailable because it is not populated. Snapshot equality is not a general algebraic proof. |

Correctness-review generation is an ordinary SQL generation call with previous SQL and diagnostic evidence included; it is not an independently isolated verifier. The routine final answer is deterministic formatting, not another LLM call. [API]

Provider behavior matters:

- Ollama uses deterministic sampling controls, SQL output cap 1,500 and generic completion cap 1,000; generic completion forces JSON even for prose explanations. [L]
- Groq sets **180 completion tokens** for both SQL and structured reasoning, risking truncation of complex SQL/JSON. It retries 429s up to three HTTP attempts, with bounded sleeps but no whole-request deadline. [L]
- JustDoWork caps output at 4,000 with no explicit sampling setting. True SOTA requests `xhigh` reasoning for every task and has no explicit output-token cap; its SSE response is buffered by `post()`. This weakens predictable cheap-path behavior. These are adapter facts, not verified gateway capabilities. [L]
- The API maps provider `agentrouter` to the JustDoWork object. Legacy AgentRouter HTTP/CLI classes remain in `llm.py`; the BIRD runner instead directly uses `AgentRouterClaudeModel`. Therefore the same provider label does not imply the same transport across paths. [API] [BIRD]

Cost is driven by full prompt size, hidden/reasoning output where billed, number of generation/diagnostic calls, provider retries, cold embedding/model loading, profile/schema inspection, and repeated DB work. One generation attempt is not one total LLM call.

For one ordinary new successful question the intended budget is **one SQL call**, zero context calls, zero recovery calls. A failed-then-repaired query can use **three SQL calls plus four recovery calls**. If the repaired result is empty/NULL, the API can run another four-call recovery. Context selection and conversation interpretation add independently; correctness review can run diagnosis before and after generation. There is no one shared budget across these stages. [A] [R] [API]

Database cost can be surprisingly large without extra tokens: a structurally changed SQLGlot simplification on the ordinary path can run **two equivalence executions + two warmups + six timed executions + final execution = eleven executions**. There is no preflight value-of-optimization gate. [A, `_validate_result_equivalence` and `_benchmark_optimization`][A]

**RECOMMENDATION: pre-generation estimator.** Compute features from the question, governed catalog, retrieval trace and cached database statistics before asking for SQL: candidate-table count, join-path length/ambiguity, multiple fact tables, requested aggregation stages, ranking/negation/ratios, temporal ambiguity, missing metric definitions, value resolution confidence, profile age, retrieval margin/entropy, estimated selected prompt length, provider/model health, table sizes and available indexes.

Initially use transparent rules with conservative intervals. After collecting representative labelled outcomes, fit a calibrated logistic model or small boosted model for repair/failure probability, and quantile regressors for tokens and latency. Train on pre-generation features only; do not leak generated SQL/errors into this estimator. Use leave-database-out validation and temporal holdouts. Maintain a separate post-generation EXPLAIN estimator: an exact physical plan cannot be known before SQL exists.

```text
expected_repair_probability = calibrated P(first candidate needs repair | features)

expected_llm_cost = sum over possible stages j:
  P(stage j runs | features) *
  [E(uncached_input_j)*price_input_j
   + E(cached_input_j)*price_cached_j
   + E(billed_output_j)*price_output_j] / 1,000,000

expected_latency = E(queue + retrieval + critical-path model time
                     + diagnostic DB time + final DB time | policy, features)

expected_database_cost = E(sum(CPU_seconds*CPU_rate
                              + bytes_scanned*scan_rate
                              + IO_operations*IO_rate) | features, policy)
```

Use the actual deployment billing model, not all DB cost terms indiscriminately. If server CPU/bytes are unavailable, return relative scan units and an interval; keep dollar cost `null`. Report median and p90 latency, model/version, features driving risk, and calibration freshness. Prices must come from a versioned provider tariff rather than assumed public-model pricing for third-party gateways.

An **illustrative**, non-measured policy with one 1,600-input/250-output SQL call, a 15% chance of a 2,000/300-token repair and a 10% chance of a 900/150 diagnostic call has expected input **1,990**, output **310**, and **1.25 calls**. Multiply these by configured tariffs. Add second-repair/candidate branches explicitly; do not hide them in a single average.

Highest-value cost reductions: skip history feature work when unused, cache stable retrieval documents and database schema versions, enforce context-token budgets, compact conversation state, use small completion budgets appropriate to each task rather than 180 universally or `xhigh` universally, avoid recovery for expected nullable outputs, carry deterministic observations into recovery, and move optimization experiments out of the normal execution path. [S] [H] [C] [L] [R] [A]

**6. Candidate generation and uncertainty gate**

**INFERENCE:** conditional diversity would help, but only after improving semantic authority and verification. Additional candidates sharing an incomplete schema cannot recover omitted facts; majority agreement does not resolve an ambiguous business definition.

| Approach | Fit for this project |
|---|---|
| One candidate | Default for a grounded simple query; preserves current philosophy. |
| Self-consistency / multiple samples of one prompt | Cheap orchestration but correlated errors; not the first escalation choice. Counts of samples are not independent evidence. |
| Different reasoning prompts | Useful second candidate: force a different decomposition while preserving the same semantic contract. |
| Divide-and-Conquer | Useful for multi-stage aggregation, cohorts, ratios and comparative periods. Prefer internal decomposition within one call before introducing separate calls per subproblem. |
| Skeleton/Plan | Useful when entity grain and aggregation order dominate difficulty. Output a compact auditable plan or SQL plus typed plan, not exposed chain-of-thought. |
| Retrieved-example/ICL | Useful only with a strong match on operation/grain and approved history provenance. Keep zero-to-two examples. |
| Multiple models | Add only when measured error complementarity beats a second prompt on one model after cost normalization. |
| Execution-result clustering | Useful for eliminating duplicate work and measuring agreement; use correct bag/order/type semantics and a stable snapshot. Never equate matching truncated prefixes with equivalence. |

**RECOMMENDATION: two separate gates.** First, an authority gate: unresolved metric definitions, ambiguous dates or unknown business population -> clarify before candidate generation. Second, a computational gate for resolvable SQL complexity.

Initial computational score, to calibrate rather than advertise as confidence:

| Observable feature | Initial points |
|---|---:|
| More than two necessary tables / nontrivial bridge | +1 |
| Two fact tables or aggregation across a fanout path | +2 |
| Two or more aggregation stages / cohort / nested ranking | +2 |
| Competing plausible schema paths or small retrieval margin | +2 |
| Complex negation, ratio, percent change, tie-sensitive extremum | +1 |
| Unresolved physical value/format after a cheap targeted lookup | +1; clarify if meaning itself remains unresolved |

Score 0–1: one candidate. Score 2–3: one candidate plus a specific deterministic semantic check. Score >=4: permit two diverse candidates when the calibrated expected accuracy improvement justifies added latency/cost and database budgets. Permit a third only after a material disagreement and an identifiable extra source of evidence; otherwise return a clarification or review state.

Use a utility condition such as `P(extra candidate resolves error) * error_impact > extra_compute_cost + latency_penalty`, with policy thresholds per deployment. The weights, repair probability, selection confidence and harm levels must be calibrated; a heuristic score is not a probability.

Generate at most two initially: direct SQL and skeleton or divide-and-conquer selected by failure type. Validate authorization, references and the semantic contract before executing either. Deduplicate ASTs, preflight cost, execute within a shared snapshot/budget, compare typed results, then select using contract compliance, dependency coverage and genuinely independent support. A disagreement on business meaning requires clarification. Use an LLM adjudicator only if the evidence identifies a decidable technical difference; no judge call for equal supported candidates.

Track candidate oracle accuracy, chosen-candidate accuracy, disagreement rate, escalation frequency, repair success, added cost and database work separately. A larger oracle-to-selected gap calls for a better selector, not more generation.

**7. Database and security assessment**

**CONFIRMED FROM CODE:** SQLite validates database IDs and resolved paths; opens with `mode=ro&immutable=1`, sets `query_only`, uses a progress handler for execution, and fetches `max_rows+1`. PostgreSQL resolves server-configured database IDs to DSNs, uses read-only transactions and statement/lock/idle timeouts. Demo SQL provisions a reader role. These are worthwhile defenses. [DB] [PG] [SEC]

They are not complete production authorization. The API has no authentication/owner check, and `/api/check` accepts caller SQL against any configured database. The main validator does not enforce allowed tables/columns, schemas or functions. Read-only transactions are not a general side-effect/resource sandbox; PostgreSQL documents the scope of its high-level read-only guarantee. [V] [API] [PostgreSQL transaction documentation](https://www.postgresql.org/docs/current/sql-set-transaction.html)

Recovery has a different validator, with bare table-name matching rather than fully qualified relations and scoped CTE resolution. The existence of a SELECT node is insufficient as a universal policy. Qualified schemas or CTE shadowing can defeat a name-only allowlist where the database role can see additional objects. Function capabilities remain uncontrolled. This is a **confirmed control gap**; a particular privileged exploit depends on installed functions and DB permissions and was not executed. [R]

SQLite `immutable=1` is appropriate for frozen benchmark files; it tells SQLite the file cannot change and bypasses normal locking/change detection. It is unsuitable as the default contract for an actively updated application database. [DB] [SQLite URI documentation](https://www.sqlite.org/uri.html)

PostgreSQL `fetchmany()` bounds the Python result slice but a normal client cursor executes before fetching; it is not a server scan/row/memory limit. Statement timeout starts with statements, not arbitrary connection establishment, and the supplied per-probe timeout is ignored. A DSN may set connection timeout, but the application does not enforce one. New connections for comparisons also mean separate snapshots. [PG] [PostgreSQL connection documentation](https://www.postgresql.org/docs/current/libpq-connect.html)

**RECOMMENDATION:** authenticated principal -> database/schema/table/column policy -> fully resolved SQL AST and approved functions -> cost preflight -> least-privilege read-only worker. Enforce RLS where needed and propagate principal identity to the DB role/session securely. Use trusted search paths, connection deadlines, DB-specific cancellation, byte/row/time/concurrency limits, and sandbox benchmark databases separately. Bind literal values through parameters where supported after interpreting them; parameterization does not replace authorizing the generated SQL structure. Recovery must use the same executor and policy as final execution.

Prompt content and database values are untrusted text. Calling them “trusted evidence” does not make them authority. Keep source/provenance separate from instructions, and enforce access before retrieving examples or transmitting context to a provider. Heuristic profile redaction is useful but incomplete; min/max and live results can expose data that sample redaction hides. No authenticated tenant boundary or data-classification egress gate is present. [L] [P] [R] [API]

UI inspection found text/SQL/results predominantly rendered with `textContent`/text nodes; the inspected `innerHTML` occurrence is a constant progress template. I did not find a confirmed dynamic HTML injection defect. The browser busy flag prevents one-tab overlap only; sessions are shared through localStorage and the server permits concurrent calls. [UI]

**8. Findings register**

Severity: P0 immediate catastrophic exposure demonstrated; P1 serious correctness/security/reliability failure; P2 material defect or misleading behavior; P3 maintenance/configuration issue. **No P0 was established.** P1 security severities assume deployment beyond a trusted single-user local environment. Every item below is confirmed from code unless explicitly marked risk.

| ID / severity | File and function/class | Problem and why it matters | Recommended fix |
|---|---|---|---|
| F01 **P1** | `validator.py:validate_sql`; `agent.py:generate` [V] [A] | Approved tables restrict context, not SQL execution. Reproduction selects a synthetic secret from an unapproved table and is accepted. Query function/schema policy is also absent. | Resolve scoped identifiers against authorization, including schema/catalog and functions; keep database privileges as an independent boundary. |
| F02 **P1** | `api.py:create_app`, chat/job/check endpoints [API] | No authentication or ownership checks. Caller-supplied sessions identify state; job retrieval/cancellation has no owner binding. **Deployment risk:** access to any reachable configured DB via `/api/check`. | Authenticate, bind sessions/jobs to principals, authorize each DB/query, limit request rate. |
| F03 **P1** | `api.py:chat`, accepted branch [API] | Correctness candidate is saved before equivalence is checked. Reproduction replaces SUM with MAX while asking the user which interpretation to trust. | Keep a pending review candidate; commit it only after policy/user resolution. |
| F04 **P1** | `recovery.py:ainvestigate`; `postgres.py:execute` [R] [PG] | Recovery deadline is restarted after deterministic work; LLM awaits are not bounded by it; PostgreSQL deletes the supplied timeout. Reproduction completes past a 100ms budget without exhausted status. | One immutable end-to-end deadline shared across probes/LLM/repair; honor remaining timeout; server cancellation. |
| F05 **P1** | `recovery.py:ainvestigate` [R] | Terminal diagnosis accepted with no tool evidence; deterministic fallback evidence is not passed into the initial agent prompt. Reproduction accepts “value does not exist” without a probe. Existing test accepts an unsupported year-range diagnosis. | Require typed evidence IDs and proposition-specific proof before factual final actions; reuse fallback observations. |
| F06 **P1** | `context.py:_relationships` [CTX] | Composite profile has arrays of key columns, but context uses only singular first columns and uniqueness of the whole key. Reproduced: `(a,b)->(x,y)` becomes `a->x`. Any profile relationship also prevents fallback adding other live FK edges. | Preserve grouped composite edges atomically; merge current live topology with profile annotations instead of returning early. |
| F07 **P1** | `api.py:chat`; `service.py:execute_question`; `database.py:execute` [API] [S] [DB] | Synchronous DB work, profile IO and embedding inference occur inside async paths. Event loop cannot handle cancellation/health/jobs while these calls block. No global DB-work admission control. | Bounded worker execution or async DB driver; semaphores/queue limits and explicit cancellation, not just `task.cancel()`. |
| F08 **P2** | `hybrid_retrieval.py:retrieve`, `_ranks` [H] | Dense failure creates all-zero scores but ranks them with `positive_only=False`; alphabetical order still votes in RRF. Reproduced. | Omit the failed channel entirely and report degraded retrieval; test invariance to identifier renaming. |
| F09 **P2** | `context.py:_execution_column` [CTX] | Missing profile converts known live types to UNKNOWN. Reproduced REAL -> UNKNOWN. | Carry live ColumnInfo through the plan/context; profiles enrich rather than replace physical schema. |
| F10 **P2** | `profiling.py:_profile_column`, `_observed_format` [P] | Any first 20 six-digit strings imply YYYYMM and override identifier semantics, including impossible months. Reproduced IDs 123456/123457 -> date. | Preserve explicit key/type semantics; validate month/calendar ranges and heterogeneous-format confidence. |
| F11 **P2** | `agent.py:generate` [A] | `execute=False` can accept `SELECT missing` as EXECUTABLE; no EXPLAIN happens. | Return `SAFETY_VALIDATED` or preflight before claiming executability. |
| F12 **P1** | `api.py:chat` anomaly branches [API] | `zero_trace`/`null_trace` are not returned/stored in generation; REPAIR and ESCALATE actions are not acted on. Analytics service callers skip these branches entirely. | Centralize result review in service, dispatch typed actions and retain full provenance. |
| F13 **P2** | `recovery.py:ainvestigate`; `agent.py:_sum_usage`; `api.py:chat` [R] [A] [API] | Recovery tokens absent from chat totals; failed/four-tool-only recovery returns zero-call fallback; fallback probes omitted from agent totals; positive model usage retains estimated USD 0. Reproduced. | Append every attempt to a call ledger in finally blocks; aggregate once; unknown cost remains null. |
| F14 **P2** | `recovery.py:probe_filter_counts` [R] | “Independent” filter probes preserve original joins. A real parent value with no child match yields NO_MATCH. Only root top-level WHERE conjunctions are probed; HAVING/nested/ON semantics are incomplete. Reproduced. | Separate base-domain existence, join survival and joint predicate checks; scope each claim to its actual probe. |
| F15 **P2** | `context.py:build_context_plan`; `agent.py:generate` [CTX] [A] | `coverage_complete`, missing HARD requirements and token budget are telemetry only. Expansion codes are not emitted by the active validator, making the advertised semantic expansion path effectively dormant. Appending prose would also invalidate JSON and invoke a lossy fallback if activated. | Make missing dependencies actionable; use typed context updates, then serialize; enforce budget/coverage policy. |
| F16 **P2** | `database.py:connect/inspect/execute`; `api.py:chat_jobs/chat_tasks`; `ConversationStore` [DB] [API] [C] | SQLite `with connection` does not close the connection; reproduction confirms it remains usable. Jobs/tasks/sessions lack retention limits. **Risk:** resource accumulation under sustained use, with SQLite cleanup timing depending on GC. | Explicit closing; bounded TTL stores; remove completed task handles; durable session storage with ownership. |
| F17 **P2** | `agent.py:generate` optimization branch [A] | Explicit model optimization equivalence/benchmark calls can raise uncaught DB errors before normal execution handling. Optimization can run DB comparisons even with execute=False. Normal generation can incur eleven executions. | Apply execute policy consistently; catch comparison failures; reserve benchmarks for explicit offline/low-risk optimization with a total DB budget. |
| F18 **P2** | `benchmark.py:compare_sql`; `benchmarks/evaluate_bird.py:main` [B] [BIRD] | Set equality ignores duplicates/order; reproduced DISTINCT and duplicate-preserving query counted equal. BIRD runner bypasses service/retrieval/planner and generates with execute=False; DB execution errors therefore do not reach repair. | Keep benchmark-compatible EX separately; add strict semantic comparison; evaluate the production service path with frozen inputs. |
| F19 **P2** | `benchmarks/compare_model1_spider.py:run`; debit runner `run` [SPIDER] [DEBIT] | Spider reads nonexistent serialized `token_usage.total_tokens`, yielding zero; also reads generation rather than all-call usage. Debit stores evidence but omits it from API input. Both need explicit truncation handling. | Sum actual token fields/ledger; pass evidence or label no-evidence mode; reject truncated comparisons and randomize arm order. |
| F20 **P2** | `scripts/train_schema_reranker.py:evaluate`; runtime `retrieve` [TRAIN] [H] | Offline all-candidate ranking differs from runtime top-30 reranking and table/bridge policies; reported gains do not measure deployment behavior. | Run validation through the exact runtime retrieval function with frozen assets and per-stage recall. |
| F21 **P2** | `llm.py:GroqSQLModel`; `api.py:create_app`; `service.py:execute_question` [L] [API] [S] | 180-token cap for all Groq tasks; agentrouter label points at JustDoWork; SQL-model env override beats web-selected model despite `.env.example` comment. Flags used to disable catalog options do not enforce request policy. | Typed provider/model capability registry with explicit aliases, task-specific budgets, validated override precedence and admission checks. |
| F22 **P2** | `api.py:chat` explanation path [API] | PostgreSQL explanation calls hard-code `dialect="sqlite"`, including deterministic fallback. | Resolve and store dialect in conversation state and pass it through. |
| F23 **P2** | `ConversationStore.get/put`; `api.py:chat` [C] [API] | Two concurrent turns can read the same state, then overwrite each other; a reset can be followed by an older pending task saving stale state. Dictionary locking does not make the turn atomic. | Session version/CAS or per-session queue; cancel or invalidate obsolete generations. |
| F24 **P3** | `glossary.py:GlossaryTerm`; config/model helpers; BIRD CLI [G] [L] [BIRD] | `value_aliases` unused; local-model availability helper not wired into current catalog; stale planner-enabled env knob; evaluator CLI permits 4 attempts while request schema permits 3. | Remove dead compatibility surfaces or test and document supported behavior; unify configuration. |

Reproductions intentionally assert current defects; they are review evidence, not desired regression-test assertions. Convert them to expected-safe tests when fixing the source. Python documents the distinction between a connection transaction context and closing it. [Python sqlite3 documentation](https://docs.python.org/3/library/sqlite3.html#how-to-use-the-connection-context-manager)

**Additional risks, not demonstrated production incidents:** stale/malformed profile files can break or mislead requests; PostgreSQL comparisons use different snapshots; unknown unique constraints are overclassified as many-to-many; embedding cache eviction can remove entries needed by a single batch larger than its 20,000-entry cap; sensitive min/max statistics are not fully redacted; role grants/search paths may be broader than intended. Tests do not establish worst-case behavior under these conditions. [P] [PG] [CTX] [H] [SEC]

**9. What the benchmark evidence supports**

The saved Spider A/B run reports **17/20 correct in each arm**, with average latency **6,897ms retrieval** versus **11,505ms model1**. Each arm uniquely solved one case. The saved debit hard-20 run reports **6/20 correct in each arm**, while all 20 were accepted; average latency **18,989ms** versus **23,402ms**. These are historical artifacts, not measurements rerun against HEAD. [SPRESULT] [DERESULT]

**INFERENCE:** this supports keeping the context-model call optional. It does not establish that context selection never helps, that 85% generalizes, or that current hard-query accuracy is 30%. The sample is tiny; debit evidence was omitted; provider labels differ across paths; token accounting is broken in Spider; artifact signatures do not fully pin code/database/profile/provider configuration. Some gold business interpretations themselves warrant review. Report benchmark mismatch separately from adjudicated business correctness.

**RECOMMENDATION:** one shared evaluation harness over the service with: frozen inputs/assets; schema dependency recall; execution EX and strict bag/order comparison; known-result metric tests; ambiguity/abstention quality; dialect/security negatives; repair success and regression rates; all-call tokens and p50/p95 latency; DB statements/CPU/bytes; nullable/empty expected outcomes; cancellation and concurrent sessions. Add small adversarial datasets where a wrong join or denominator still returns plausible results. Repeat tuning on development only, then evaluate once on a fresh holdout.

**10. Comparison with strong public Text-to-SQL systems**

Public primary sources were checked during this review. These are architecture comparisons, not apples-to-apples latency, security or accuracy rankings. Paper benchmark success does not establish enterprise authorization or audited business definitions.

| System | Published architecture | What to borrow | What not to copy by default |
|---|---|---|---|
| DeepEye-SQL | Semantic value retrieval, complementary schema linking, skeleton/ICL/divide-and-conquer generators, deterministic checking and confidence-aware selection. | Question-conditioned value hints, structured failure evidence, diverse second candidate. | Multiple linking/generation samples for every query; its experiment's broad sampling budget is incompatible with the one-call target. [Paper](https://arxiv.org/html/2510.17586v3) |
| XiYan-SQL | Schema filtering, multiple generators with distinct learned SQL styles, selection and candidate reorganization. | Measure candidate complementarity and use a compact schema representation. | Fine-tune multiple generators/selectors before fixing local semantic/evaluation gaps; duplicate samples do not create independent support. [Paper](https://arxiv.org/abs/2507.04701) |
| SIRIUS-SQL | Specialist/generalist candidate sources, typed execution outcomes and repair, re-execution before readmission, confidence-gated selection and structural support. | Typed candidate lifecycle; evidence from distinct sources; deterministic resolution of close selection cases. | Treat every repair as beneficial or count correlated candidate votes as confidence. Its conclusions about complementary training distributions require validation here. [Paper](https://arxiv.org/html/2606.01246v1) |
| Agentar-Scale-SQL | Internal reasoning training, diverse reasoning/ICL synthesis, iterative refinement, execution grouping and pairwise tournament selection; default experiment uses 9+ ICL plus 8 reasoning candidates. | Separate generation coverage from selection quality; refine representatives rather than every duplicate. | Large default pools, round-robin selection costs and RL training investment for a small cost-constrained application. [Paper](https://arxiv.org/html/2509.24403v1) |

Relative assessment by dimension:

- **Semantic accuracy and verification:** the public architectures explicitly invest in diversity/selection or structured checks. V5 mainly invests in compact context and syntax/error recovery; it has no equivalent routine semantic gate. Neither strategy supplies an absent business definition.
- **Schema linking:** V5's deterministic FK/glossary closure is useful, but its early shortlist and composite-key loss constrain recall. Complementary linking is worth borrowing conditionally, especially when a missing dependency can be named.
- **Business semantics:** V5 already has curated glossary/formula structures that can become executable contracts. They are not currently enforced. A benchmark-specialized generator cannot replace metric governance.
- **Cost and latency:** V5 has a simpler intended fast path. This is a structural advantage, not measured superiority: its fixed high reasoning setting, diagnostics and optimization executions can dominate difficult requests. Large paper candidate pools add cost and serial selection latency; parallel generation trades latency for resource use.
- **Database load:** execution-based ensembles and tournaments require multiple queries. V5 can also execute many probes/optimization trials; a shared DB budget matters more than counting SQL candidates alone.
- **Security and production readiness:** do not rank research papers as secure products. V5 exposes concrete least-privilege helpers but lacks auth, fully resolved authorization, durable owned state and robust cancellation.
- **Explainability:** V5 exposes context/attempts and extracts SQL facts, a useful foundation. Confidence voting should add candidate/probe provenance, not opaque scores. Missing recovery traces currently undermine that advantage.

These are **INFERENCES** from the cited methods and the inspected V5 implementation, not live comparative measurements. No paper's headline leaderboard percentage is used to assign V5 a numerical accuracy estimate.

**11. Scores and priorities**

Scores assess current design and implementation for a general business analytics application. They do not mean percentage correctness; uncertainty is roughly ±1 point.

| Requested score | /10 | Rationale |
|---|---:|---|
| A. Architecture | **6** | Good service, typed context and optional escalation; split orchestration and unenforced control boundaries. [S] [API] [R] |
| B. SQL accuracy potential | **7** | Useful grounding and adapters provide room to improve; current semantic verification is weak. Potential, not measured accuracy. [H] [G] [V] |
| C. Retrieval quality | **6** | Hybrid channels, glossary/key closure and encouraging reranker artifact; fallback/composite flaws and evaluation mismatch. [H] [RM] [TRAIN] |
| D. Cost efficiency | **5** | One-call path possible, but global accounting/budgets and DB-cost gating are absent. [A] [R] [L] |
| E. Production readiness | **3** | Suitable for controlled local experimentation; ownership, durability, cancellation, concurrency and current holdout evaluation remain. [API] [C] [BIRD] |
| F. Code quality | **6** | Strict models, modular boundaries and passing checks; duplicated validators/adapters, stale semantics and incomplete wiring. [V] [R] [L] |
| G. Security | **3** | Real read-only defenses, but no application identity or complete query authorization. [DB] [PG] [V] [API] |
| H. Novelty | **4** | Thoughtful integration of established retrieval/grounding/recovery ideas; no demonstrated new method or controlled accuracy/cost advance. [H] [R] [RM] |

**Five biggest reasons it will fail on difficult real-world questions:**

1. The business metric, denominator, population or period is ambiguous, yet generation proceeds without resolving it. [G] [CP] [V]
2. Early pruning or incomplete relationship representation removes a necessary dependency. [H] [CTX]
3. Plausible wrong aggregations and joins pass the acceptance gate. [A] [V]
4. Recovery reuses the same assumptions and can accept unsupported explanations instead of gathering decisive evidence. [R]
5. Runtime budgets, state updates and benchmark/accounting gaps conceal operational failures and the true accuracy/cost tradeoff. [API] [BIRD] [SPIDER]

**Five strongest aspects:**

1. A viable one-SQL-call default with optional context planning. [S]
2. Compact structured schema/profile context with explicit provenance concepts. [CTX]
3. Complementary lexical/dense/value retrieval and deterministic dependency restoration. [H]
4. Mockable providers, typed contracts, bounded SQL attempts and inspectable attempts/context. [L] [A]
5. A reusable natural-language service boundary and a testable basis for targeted DB evidence, without requiring a general-purpose agent for every query. [S] [AT] [R]

**Ten highest-impact improvements, ranked.** Effort assumes one experienced engineer with access to representative databases. Accuracy gains are qualitative hypotheses; numerical percentage-point promises would be unsupported. “None” means no necessary token increase, not zero engineering/runtime cost.

| Rank | Improvement | Impact | Effort | Token/cost change | Expected accuracy gain | Production value |
|---|---|---|---|---|---|---|
| 1 | Unified authorized executor: identity, scoped AST/catalog/function policy, least privilege, deadlines/cancellation | Critical | Large | None in tokens; small preflight overhead | Mainly safety/reliability | Critical; F01/F02/F04/F07 |
| 2 | Governed semantic contract + deterministic metric/grain/filter/time checks, clarify absent authority | Very high | Large | Small compact-context increase; occasional clarification | High on semantic failures | Critical |
| 3 | Repair context fidelity: complete composite FKs, merge live/profile topology, preserve live types, valid date inference | Very high | Medium | Negligible; necessary keys only | High on affected schemas | High; F06/F09/F10 |
| 4 | One budgeted recovery lifecycle with typed evidence, action dispatch and pending-state commit | Very high | Medium–large | Usually decreases wasted calls | Medium–high on repairable failures | Critical; F03/F05/F12/F14 |
| 5 | Production-path evaluation, full call ledger and versioned artifacts | Very high | Medium | Offline evaluation expense; no required online tokens | Indirect but essential | Critical; F13/F18/F19/F20 |
| 6 | Correct dense fallback; joint role-aware retrieval and targeted value lookup | High | Medium | Similar prompt size; bounded extra lookups | Medium–high on retrieval/value errors | High; F08 |
| 7 | Calibrated conditional second candidate with deterministic selection | High after 2–6 | Medium–large | About +1 generation only on gated fraction; bounded DB work | Medium on complex resolvable questions | High |
| 8 | Disable routine optimization benchmarking; use cached estimates and explicit performance workflow | High | Small–medium | Large DB/latency reduction where simplification fires | Neutral if behavior preserved | High; F17 |
| 9 | Durable owner-bound/versioned sessions, retained audit records and bounded job queues | High | Medium | No necessary model tokens | Prevents conversational regressions | Critical for multi-user; F16/F23 |
| 10 | Typed provider registry, task budgets, exact routing and config cleanup | Medium–high | Small–medium | Usually reduces output/latency waste | Medium where truncation/routing occurs | High; F21/F22/F24 |

Do not launch model fine-tuning until the evaluation can distinguish missing authority, retrieval omission, generation error and selection failure. Otherwise training will optimize around measurement defects or teach benchmark-specific interpretations.

**12. Recommended V6 architecture**

**RECOMMENDATION:** extend existing modules in place. Keep `TextToSQLService` as the sole entry point for chat and analytical tools. Keep SQL generation/execution inside this boundary. The browser and analytics orchestrator submit natural-language requests; they do not acquire arbitrary SQL access.

```text
Authenticated question + owned/versioned conversation state
    |
Deterministic request normalization + approved metric/period resolution
    | unresolved authority -> clarification
    v
Versioned joint schema/value/history retrieval
    -> restore complete key/metric dependencies
    -> verify coverage, provenance and token budget
    v
Pre-generation risk/cost policy + shared RequestBudget
    | low risk: 1 candidate
    | justified complexity: 2 different reasoning routes
    v
One provider interface -> SQL + compact optional semantic plan
    v
Authorized AST resolution + contract checks + bounded EXPLAIN
    v
Read-only execution in isolated worker / consistent snapshot
    v
Typed result outcome + targeted invariants
    | supported: accept
    | repairable: bounded evidence/probe -> one focused repair
    | disagreement/unknown meaning: clarification or pending review
    v
Atomic accepted-state commit + auditable result envelope
    -> SQL, result, interpretation, evidence, all-call cost/latency
```

Concrete module evolution:

- `service.py` owns the whole request lifecycle, including result review and final acceptance. `api.py` handles authentication, transport, jobs and presentation only.
- A `SemanticContract` revision separates approved business facts, explicit user constraints, inferred physical facts and unresolved items. Include metric ID/version, measure lineage, entity grain, population, filters, denominator, calendar/timezone/reference date, currency/unit and tie policy. Do not fill these fields speculatively.
- `context.py` serializes current live types, grouped composite relationships, selected value evidence and approved formulas. Coverage checks operate over all required dependencies. Profiles include database/schema version, observation time, exactness and validity period.
- `validator.py` evolves into separate safety/authorization, reference/type and grounded-semantic passes with typed failure codes. Recovery calls the same policy; remove the second weaker validator.
- `recovery.py` consumes existing observations and a shared `RequestBudget`; the controller requires evidence for terminal claims. A simple state machine is sufficient. Keep LangGraph only if checkpointing/state transitions justify it; no additional agents are necessary.
- `llm.py` becomes a provider registry with validated model/task capabilities and reusable async clients. Preserve selected transport identity in every record; distinguish requested model from effective model.
- The executor enforces total deadline, per-statement timeout, maximum rows and bytes, concurrency and cancellation. Deterministic parameterized diagnostic templates cover common value/date checks; model-chosen SQL probes require the same resolved authorization as final SQL.
- An append-only call ledger records successful, failed, cancelled and malformed calls with trace ID, model/transport, token usage or unknown, DB statements, time and evidence links. Immutable candidates remain separate from accepted state.
- Evaluation invokes the exact service and a frozen catalog/database snapshot. Include business ambiguity and safe abstention, not just executable-query agreement.

Suggested initial limits, **to tune from measurements**: one candidate by default, two only when gated, three maximum across diversity and repairs, at most four diagnostic DB probes, one shared interactive deadline and cost ceiling, and no automatic online performance tournament. Reserve enough time for cancellation and a truthful partial/clarification response. Long jobs should use an explicit separate policy rather than silently expanding interactive budgets.

Deliver V6 in this order: (1) authorization/state/budget defects; (2) context fidelity and evaluation; (3) grounded semantic checks and targeted recovery; (4) measured conditional diversity. The success criterion is fewer silently wrong accepted answers at a known total cost, with the existing one-call path preserved for simple questions.

**13. How to work toward 10/10 in every category**

This section is **RECOMMENDATION**, including all thresholds and milestones. It does not change the current scores or imply that the proposed capabilities exist. A 10/10 means an exceptional, independently evidenced implementation **for a declared workload and deployment scope**. It cannot mean perfect answers to every possible natural-language question or a guarantee of zero security defects. The score must be reassessed when the workload, database, model or deployment changes.

Start by declaring the supported scope: databases/dialects, languages, analytical question families, approved metrics, data sensitivity, number of concurrent users, freshness requirements and maximum acceptable latency/cost. Initially choose one real domain and one primary production dialect. Add another only after its behavior is verified. A narrow system with an honest unsupported-question boundary is easier to make excellent than a universal system with silent gaps.

Distinguish three milestones: **8/10 = strong and demonstrably reliable within scope; 9/10 = robust under representative edge cases and operational stress; 10/10 = sustained evidence, independent review and no material unresolved gaps within that scope.** These are review criteria, not an external industry certification.

**A. Architecture: 6 -> 10**

The architectural problem is fragmented ownership of execution, review, state and budgets. Fix the boundaries before adding new capabilities. [S] [A] [API] [R]

1. Make `TextToSQLService.execute_question()` own retrieval, generation, validation, execution, result review and final acceptance. Chat and analytical tools must use the same lifecycle.
2. Introduce a typed request context with principal, session version, catalog version, semantic contract, shared deadline, cost ceiling and trace ID. Pass it explicitly rather than rediscovering settings from environment variables during execution.
3. Separate candidate state from accepted state. Define transitions such as `CONTEXT_READY -> CANDIDATE -> POLICY_VALID -> EXECUTED -> VERIFIED -> ACCEPTED`, with `CLARIFICATION`, `REJECTED` and `CANCELLED` outcomes. Only the acceptance transition commits conversation state.
4. Keep retrieval, provider and database interfaces replaceable, but require identical authorization, accounting and error semantics across implementations. Eliminate the weaker recovery validator and the benchmark-only orchestration fork.
5. Use a simple controller unless durable checkpointing or resumable transitions justify LangGraph. More agents do not increase this score.

**8/10 gate:** one end-to-end execution owner, common executor, common budget and atomic acceptance; chat/service integration tests show equivalent policy behavior. **9/10 gate:** provider failure, cancellation, stale metadata and concurrent turns produce explicit, consistent outcomes. **10/10 evidence:** architecture invariants survive representative load/fault injection, adapters pass shared conformance tests, and an independent reviewer can reconstruct any accepted answer from its audit record. No hidden bypass may remain through check, recovery, optimization or benchmark endpoints.

**B. SQL accuracy potential: 7 -> 10**

Change the goal from “generate valid SQL” to “answer a specified business question, or explain which required fact is unresolved.” Do not try to obtain the last points solely by selecting a larger model. [V] [G] [F] [CTX]

1. Build an approved metric catalog. For every supported metric, define entity grain, numerator, denominator, eligible population, date column/calendar, currency/unit, NULL policy and ownership/version.
2. Compile common approved metrics deterministically to SQL expression/plan fragments. Let the LLM compose supported operations around these fragments, with AST checks that confirm the approved lineage and constraints remain present.
3. Detect semantic hazards before acceptance: fanout across multiple facts, aggregation order, ratio-of-sums versus mean-of-ratios, missing population filters, tie policy, anti-join NULL behavior and incomplete period comparisons. Each check must have an explicit proof scope; an uncertain check must not silently claim success.
4. Use targeted counterexamples in tests: two child records per parent, zero denominator, duplicate dimension keys, tied extrema, missing periods and nullable filters. A query that happened to match one snapshot should fail tests when its incorrect semantics become visible.
5. Add conditional candidate diversity only after the reference semantics and selector are reliable. Measure how much extra coverage translates into correctly selected answers.

**8/10 gate:** the main governed metrics have independently reviewed known-result tests, and unresolved definitions produce clarification. **9/10 gate:** complex question families and temporal/schema drift tests pass without a material rise in silent errors. **10/10 evidence:** high answer correctness at useful coverage on fresh, representative holdouts, independently adjudicated semantic disagreements, and stable performance after deployment/model updates.

As a **proposed initial product target**, aim for at least **99% semantic precision among accepted answers and at least 80% coverage** on the declared eligible workload. These numbers are neither achieved nor sufficient by themselves for 10/10; choose stricter thresholds for higher-impact use. Report confidence intervals, difficulty slices and the number of queries. Never improve precision by silently excluding difficult eligible questions. Separately report initial-answer correctness, final correctness after clarification and unresolved-question rate. A small perfect test set cannot justify a perfect rating.

**C. Retrieval quality: 6 -> 10**

Retrieval should preserve every dependency needed for a correct answer while minimizing irrelevant context. Mean column recall alone is insufficient. [H] [CTX] [P] [TRAIN]

1. Fix the dense fallback, live-type loss, false date inference and composite relationship loss first. These are defects with a clearer payoff than adding another retriever.
2. Use role-aware candidates for measures, dimensions, filters, time and join keys. Include fully qualified identity and concise descriptions. Restore complete composite paths and formula dependencies after pruning.
3. Add a bounded value resolver: approved aliases first, then exact lookup on plausible columns, then fuzzy/semantic candidate retrieval where needed. Keep alternatives when a value maps ambiguously; do not automatically choose a similar spelling.
4. Version schema documents, profiles, glossary and history together. Declare stale/degraded states; verify critical physical metadata against the live catalog.
5. Evaluate all learned ranking through the runtime retriever. Remove a channel or reranker if paired accuracy/cost ablations do not justify it.

**8/10 gate:** no known dependency-loss defects; actual-runtime retrieval evaluation and visible degradation. **9/10 gate:** strong recall on rare values, composite keys, competing paths and unfamiliar terminology under the same context budget. **10/10 evidence:** a measured recall-versus-token frontier, reliable out-of-scope/ambiguity detection, and stable behavior across fresh databases and metadata updates.

A useful **proposed target** is **>=98% complete required-dependency recall** on a labelled supported-workload holdout, counting a case as successful only if every necessary table, column, key component and metric operand survives. Also report per-component recall and final SQL accuracy. Retrieval can score highly without carrying most of the schema; it cannot score highly by retaining everything and hiding the token cost.

**D. Cost efficiency: 5 -> 10**

The right objective is cost per correctly answered eligible question at an agreed accuracy, coverage and latency level. Cheap wrong answers are not efficient. [A] [R] [L] [SPIDER]

1. Capture all provider calls and DB executions, including failed JSON, cancellations, recovery, retries, context selection and optimization. Keep unknown usage separate from zero and reconcile records against provider billing when available.
2. Remove routine online optimization benchmarks. Cache schema/retrieval artifacts and compact conversation state; skip unused history features and expected-NULL investigations.
3. Use task-specific model settings: modest budgets for classification, sufficient SQL output limits, and expensive reasoning only when measured uncertainty justifies it. Test provider truncation/finish states rather than accepting incomplete text.
4. Fit the pre-generation cost/risk estimator described above, then measure its calibration. Compare one-call, deterministic-verification and two-candidate policies at equal answer-quality requirements.
5. Track database CPU/bytes or defensible proxy units alongside token cost. A policy that saves tokens by repeatedly scanning a large warehouse may be more expensive overall.

**8/10 gate:** complete call accounting and enforced request ceilings; simple governed questions normally need one generation call. **9/10 gate:** a calibrated escalation policy reduces wasted work without degrading semantic precision/coverage. **10/10 evidence:** the selected policy lies on the measured quality/cost/latency frontier of credible alternatives for the deployment, with sustained billing/telemetry reconciliation and no hidden work outside budgets.

Suggested initial targets: **>=90% of correctly answered simple questions use one SQL-generation call**, no unexplained model-call omissions, and no execution beyond the enforced request ceiling except documented cancellation grace. Do not impose an arbitrary universal dollar target before measuring the workload and actual gateway tariffs.

**E. Production readiness: 3 -> 10**

Production quality requires operational evidence, not only additional unit tests. [API] [C] [PG] [DB]

1. Add authentication/authorization, durable owned sessions, bounded job retention and admission control. Use versioned state updates so a late response cannot overwrite a newer turn or undo a reset.
2. Move blocking work into bounded execution workers or async drivers. Propagate cancellation to database/model work and terminate abandoned requests predictably.
3. Define operational SLOs separately for simple, complex and asynchronous workloads: availability, p95 latency, cancellation completion, queue wait, error/clarification rates and freshness.
4. Add readiness checks for required dependencies, graceful shutdown, resource limits, structured metrics, protected audit retention, rollback and recovery procedures.
5. Run load tests at the declared concurrency and a controlled overload level; exercise provider outages, slow databases, corrupt/stale metadata and worker restarts. Verify backup restore and migration rollback where persistent state is introduced.

**8/10 gate:** a controlled pilot with owned state, bounded execution and operational visibility. **9/10 gate:** documented SLOs are met under representative load and fault tests, with rollback/recovery proven. **10/10 evidence:** sustained monitored operation across an agreed observation window, no unresolved critical operational findings, tested incident procedures and a repeatable release process. A 30-day pilot may be a useful initial checkpoint, not automatic proof of 10/10.

**F. Code quality: 6 -> 10**

Focus on reducing ways for interfaces to disagree. The existing typed models and mockable boundaries provide a good foundation. [L] [V] [R] [S]

1. Replace broad `Any`/runtime signature fallback at core boundaries with typed protocols for generation, completion, execution and usage. Distinguish provider errors, policy failures, syntax errors, execution errors and semantic uncertainty.
2. Consolidate provider routing and configuration once at startup. Remove stale flags and unused compatibility code or give each supported path a contract test.
3. Convert the review reproductions into expected-safe regression tests as fixes land. Add property/metamorphic tests where useful: renaming irrelevant columns should not alter fallback rankings; composite keys remain atomic; unapproved relations cannot be executed.
4. Make the benchmark harness use runtime code, share result-equivalence utilities with explicit order/bag/type semantics, and ensure tests assert desired user-visible behavior rather than perpetuate broken behavior.
5. Document invariants next to the enforcing code. Make CI reproduce lint, typing, unit/integration checks and provider/DB adapter conformance in a clean environment.

**8/10 gate:** identified defects repaired with relevant regression tests, common interfaces and reproducible CI. **9/10 gate:** changes have localized effects, error handling and ownership are explicit, and adversarial invariant tests cover critical boundaries. **10/10 evidence:** independent review, demonstrated maintainability during real changes, and no material untested critical path. A high test count or coverage percentage alone does not justify the score.

**G. Security: 3 -> 10**

Treat security as an enforced capability model rather than a prompt instruction. [V] [R] [API] [SEC]

1. Derive every permitted database, relation, column, function and data-egress capability from an authenticated principal. Bind session/job ownership server-side.
2. Resolve SQL scope correctly, including schemas, CTE shadowing, aliases, subqueries and composite joins. Reuse this authorization for final SQL, diagnostics and optimization; retire public arbitrary SQL checking or restrict it to an explicitly authorized administrative role.
3. Enforce least-privilege DB roles, row/column controls where required, trusted search paths, disabled unnecessary capabilities and bounded isolated execution. Do not rely on AST checks as the sole barrier.
4. Apply data classification before context/history/value retrieval and provider transmission. Protect logs/audit records, manage secrets separately, validate retention and prevent cross-tenant cache/result reuse.
5. Test malicious prompts, poisoned metadata/history, schema-qualified bypasses, role misuse, expensive SELECTs, stale sessions and cross-owner job access. Obtain an independent security review of the actual deployed configuration.

**8/10 gate:** shared authorization/execution policy, no known bypass in tested interfaces, and realistic least-privilege deployment. **9/10 gate:** adversarial tests and independent review find no unresolved critical/high-severity issues; operational secret/retention controls are verified. **10/10 evidence:** defense in depth holds under independent assessment and ongoing monitored operation, with a maintained threat model and remediation process. This remains a scoped assurance judgment, never a claim that compromise is impossible.

**H. Novelty: 4 -> 10**

Novelty has a different path. Correctness fixes, enterprise deployment and more integrations can make an excellent product without making a new research contribution. Raising novelty is optional for the business goal. The public systems in section 10 already explore diverse generation and execution-guided selection.

My most promising **research recommendation** is a **budgeted, dependency-aware verifier**: estimate which unresolved semantic dependency is most likely to make an answer wrong, then select the cheapest decisive action—catalog lookup, value probe, semantic check, clarification or second candidate. This is a research hypothesis, not a claim that the idea is unprecedented.

1. State a falsifiable hypothesis, such as: “Selecting diagnostic actions by expected error reduction per unit cost improves semantic precision at fixed coverage and total compute compared with fixed recovery and unconditional two-candidate generation.”
2. Build simple baselines first: single call, single call plus deterministic checks, fixed two candidates, heuristic gate and learned gate. Equalize model access, context, token/DB budgets and evaluation data.
3. Measure which action resolved each error and whether the selected answer is correct, not only whether a correct candidate existed. Include unseen databases and uncertainty/abstention cases.
4. Release reproducible method details, benchmark contracts, code/configuration and ablations where data permissions allow. Commission a fresh prior-art review before asserting novelty.
5. Seek independent reproduction and comparison with strong current methods. Negative results should simplify the product rather than lead to increasingly elaborate unsupported claims.

**8/10 gate:** a clearly differentiated method with credible controlled gains. **9/10 gate:** robust gains across workloads, fair baselines and substantial ablations. **10/10 evidence:** a significant original contribution, independently reproduced and useful beyond this repository. No implementation checklist or schedule can guarantee that result. I would prioritize production/accuracy scores over novelty unless publication is a primary objective.

**14. Recommended delivery sequence and acceptance checklist**

Implement a few reviewable increments, not a full rewrite. Estimates below are **rough planning ranges for one experienced engineer**, assume access to representative data and reviewers, and do not include an unknown amount of domain-definition work. They are not commitments or predicted dates for reaching 10/10. Each phase ends with evidence before the next expands scope.

| Phase | Approximate effort | Concrete deliverable | Acceptance evidence |
|---|---|---|---|
| 1. Correct the demonstrated defects | 1–2 weeks | F03, F06, F08–F11, F13–F15, F19 and routing/dialect fixes; preserve existing APIs where practical | Review probes converted to expected-safe regression tests; no unsupported EXECUTABLE or recovery-cost claims |
| 2. Enforce shared policy and budgets | 2–4 weeks | Authorized executor, owner-bound state/jobs, common deadlines, cancellation, action dispatch and ledger | Cross-owner/table/schema/function negative tests; cancellation/load tests; every call accounted for |
| 3. Govern the first business domain | 2–4 weeks plus stakeholder review | Versioned metric catalog, compact semantic contracts and deterministic checks/templates for key metrics | Independently reviewed known-result and adversarial semantic cases; explicit clarification for absent authority |
| 4. Establish a trustworthy evaluation baseline | 1–2 weeks, begun during earlier phases | One runtime-path harness with frozen assets, fresh holdout and strict result semantics | Precision/coverage, dependency recall, latency, all-call cost and DB work reported by question family |
| 5. Optimize retrieval and selective compute | 2–4 weeks | Correct runtime reranker evaluation, targeted values, calibrated gate and optional diverse second candidate | Paired ablations show benefit at fixed precision/coverage and bounded total cost; otherwise leave feature disabled |
| 6. Validate deployment and independent review | 2–4 weeks plus observation period | Production runbooks, rollback/restore, security assessment and monitored pilot | SLO evidence, fault/overload results, no unresolved critical/high findings, approved domain scope |

Security controls needed for external exposure must precede exposing a pilot; the table is not permission to deploy phase 1 publicly. Evaluation instrumentation should start early so each later phase can be compared against an honest baseline.

For the **next concrete implementation increment**, I recommend fixing five things together: preserve disputed candidates separately from accepted state; preserve complete composite relationships/live types; remove the failed dense channel from RRF; repair recovery accounting and final-action evidence checks; and fix benchmark evidence/token inputs. These produce immediate correctness gains and make subsequent improvements measurable. In parallel planning, specify the shared authorization/executor contract before connecting real multi-user data.

Use this release scorecard rather than manually increasing ratings after features are merged:

| Question | Required evidence |
|---|---|
| Does it answer the intended business question? | Independently adjudicated semantic precision, coverage and fresh holdout results |
| Does retrieval retain all necessary facts? | Complete-dependency recall, key/metric closure and rare-value tests |
| Does it know when information is missing? | Clarification/abstention accuracy with no hidden exclusion of difficult eligible queries |
| Are extra calls justified? | Paired marginal accuracy gain versus all-call cost, latency and DB work |
| Can every result be reconstructed? | Versioned contract/context/SQL/result evidence and complete call ledger |
| Can it fail safely under load or attack? | Authorization, isolation, cancellation, concurrency and fault-injection evidence |
| Can another engineer operate and change it? | Reproducible CI, adapter contracts, runbooks, recovery and independent review |

The strongest practical destination is an **excellent scoped analytics product** with calibrated uncertainty and auditable cost. Aim for 9–10 in accuracy assurance, architecture, security and operational quality first. Treat a novelty score of 10 as a separate research ambition, and keep every score provisional until the corresponding evidence exists.

[REPRO]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/docs/review_v5_reproductions.py
[RESULTS]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/docs/review_v5_reproduction_results.json
[A]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/agent.py:95
[S]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/service.py:81
[API]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/api.py:69
[R]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/recovery.py:442
[H]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/hybrid_retrieval.py:122
[L]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/llm.py:1070
[C]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/conversation.py:94
[CTX]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/context.py:38
[CP]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/context_planner.py:31
[P]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/profiling.py:67
[G]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/glossary.py:38
[F]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/formulas.py:8
[HIST]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/historical.py:78
[V]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/validator.py:51
[VT]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/tests/test_validator.py:18
[DB]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/database.py:25
[PG]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/postgres.py:41
[SEC]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/sql/postgres_security.sql:1
[E]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/explanation.py:92
[B]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/semantic_text2sql/benchmark.py:22
[BIRD]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/benchmarks/evaluate_bird.py:61
[SPIDER]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/benchmarks/compare_model1_spider.py:40
[DEBIT]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/benchmarks/compare_model1_debit20.py:71
[BUILD]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/benchmarks/build_bird_history.py:11
[TRAIN]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/scripts/train_schema_reranker.py:238
[RM]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/models/schema_reranker/v1/metrics.json:1
[SPRESULT]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/benchmarks/results/spider_model1_ab_20.json:1
[DERESULT]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/benchmarks/results/debit_card_model1_ab_20.json:1
[AT]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/autonomous_analytics/tools/text_to_sql.py:47
[SN]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/src/autonomous_analytics/metrics/snapshots.py:18
[UI]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/web/app.js:188
[README]: /Users/meisam/Documents/text-to-sql/semantic_text2sql_ideal/README.md:1

|   Rank | Change                                                 | Impact          | Effort  | Token increase           | Expected hard-query gain     | Production value |
| -----: | ------------------------------------------------------ | --------------- | ------- | ------------------------ | ---------------------------- | ---------------- |
|  **1** | Semantic AST contract validator                        | Very high       | Medium  | **0**                    | potentially +3–10 pp         | Very high        |
|  **2** | Fix recovery action/control/budget bugs                | Very high       | Medium  | lower or neutral         | reliability more than raw EX | Very high        |
|  **3** | Fix AgentRouter/provider wiring + benchmark provenance | Very high       | Low     | 0                        | measurement integrity        | Very high        |
|  **4** | Conditional 2–3 candidate generation                   | Very high       | Medium  | only hard queries        | potentially +5–15 pp         | High             |
|  **5** | Result clustering + confidence score                   | High            | Medium  | near 0 beyond candidates | enables #4 safely            | High             |
|  **6** | Fix dense fallback + calibrated retrieval confidence   | High            | Low/Med | 0                        | ~1–4 pp possible             | High             |
|  **7** | Unified SQL safety boundary + PostgreSQL timeout       | High            | Low     | 0                        | none directly                | **Very high**    |
|  **8** | Unified trustworthy benchmark evaluator                | High            | Low/Med | 0                        | none directly                | Very high        |
|  **9** | Full cost/recovery telemetry + cost predictor          | Medium/High     | Medium  | negligible               | indirect                     | Very high        |
| **10** | Durable state/auth/resource controls/CI                | High production | High    | 0                        | none directly                | **Critical**     |

| Sev.      | Location                       | Problem                                                                        | Why it matters                                                          | Fix                                                   |
| --------- | ------------------------------ | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------- | ----------------------------------------------------- |
| **P0**    | `api.py:create_app()`          | `agentrouter` is wired to `JustDoWorkSQLModel`                                 | Requests/benchmarks labelled AgentRouter don't actually use AgentRouter | Instantiate `AgentRouterModel` and wire it explicitly |
| **P0**    | `recovery.py:ainvestigate()`   | complete deterministic fallback runs **before** agent                          | duplicate DB work, false latency/budget telemetry                       | Agent first; fallback only after agent failure        |
| **P0**    | `agent.py:generate()`          | `INFORM/ESCALATE/REPAIR` don't fully control state transitions                 | agent's decision is partly cosmetic                                     | explicit action-based transitions                     |
| **P0**    | `postgres.py:execute()`        | `del timeout_seconds`                                                          | 2s recovery probes may run for fixed 5s                                 | honor bounded caller timeout                          |
| **P0/P1** | `recovery.py:query_database()` | duplicates weaker SQL safety implementation                                    | `MERGE`, Lock, Into etc. policy can drift                               | reuse `validate_sql()`                                |
| **P1**    | `recovery.py:ainvestigate()`   | 4 iterations may allow a fourth tool call with no final reasoning turn         | recovery can end without interpreting last evidence                     | 3 tool calls + 4 model calls                          |
| **P1**    | `benchmark.py:compare_sql()`   | `set()` comparison loses duplicate multiplicity                                | fanout error can be scored correct                                      | use `Counter`                                         |
| **P1**    | `hybrid_retrieval.py`          | failed dense retrieval still receives RRF ranks                                | random noise alters retrieval                                           | remove dense channel on failure                       |
| **P1**    | `evaluate_bird.py`             | CLI allows `max-attempts=4`; `GenerateRequest` allows max 3                    | valid CLI argument crashes validation                                   | make both 1–3                                         |
| **P1**    | `evaluate_bird.py`             | AgentRouter route hardcodes `AgentRouterClaudeModel`                           | GPT AgentRouter evaluation may use wrong adapter                        | use `AgentRouterModel`                                |
| **P1**    | `api.py` correctness review    | result comparison uses exact row-list equality regardless of `ORDER BY`        | semantically same unordered results can trigger false disagreement      | shared ordered/Counter comparator                     |
| **P2**    | `api.py`                       | jobs/tasks/conversation state process-local and unbounded                      | memory growth; multi-worker inconsistency                               | TTL/LRU, then shared durable state                    |
| **P2**    | `web/app.js`                   | database dropdown filters to SQLite                                            | backend Postgres unavailable through UI                                 | show configured dialects                              |
| **P2**    | README vs `api.py`             | advertised JustDoWork models differ from API model list                        | configuration/documentation drift                                       | generate catalog from one source                      |
| **P2**    | service config                 | `TEXT2SQL_HISTORY_ENABLED=false` doesn't disable history-based schema evidence | confusing experiment semantics                                          | split schema-history/example-history switches         |
| **P2**    | repository                     | no CI checks                                                                   | reported test state is not enforced per commit                          | add GitHub Actions                                    |
