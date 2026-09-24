const $ = (selector) => document.querySelector(selector);
const sessionId = localStorage.getItem("queryRoomSession") || crypto.randomUUID().replaceAll("-", "");
localStorage.setItem("queryRoomSession", sessionId);
$("#sessionId").textContent = sessionId;

const state = { busy: false, jobId: null, cancelled: false, currentQuestion: "", ready: false };
const recentStorageKey = "queryRoomRecentQueries";
const postSafetyOptimizationCodes = new Set([
  "OPTIMIZATION_NOT_FASTER",
  "OPTIMIZATION_CHANGED_RESULT",
  "OPTIMIZATION_EQUIVALENCE_UNPROVEN",
]);

function attemptOutcome(attempt) {
  const validation = attempt.validation || {};
  if (validation.valid === true) return "passed";
  if (postSafetyOptimizationCodes.has(validation.code)) return "not_selected";
  return "failed";
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

async function initialize() {
  try {
    await api("/api/health");
    $("#healthStatus").classList.add("online");
    $("#healthStatus").lastChild.textContent = " Online";
    $("#connectionStatus").textContent = "API online";
    const databases = await api("/api/databases");
    fillSelect(
      "#databaseSelect",
      databases.filter((item) => item.configured),
      "db_id",
      (item) => `${item.db_id} (${item.dialect})`,
      null,
    );
    await loadDatabaseSchema();
    const models = await api(`/api/models?refresh=${Date.now()}`, { cache: "no-store" });
    const preferredModel = models.find((item) => item.configured);
    const preferred = preferredModel && `${preferredModel.provider}|${preferredModel.model}`;
    fillContextSelect(models);
    fillSelect("#sqlModelSelect", models, (item) => `${item.provider}|${item.model}`, modelLabel, preferred);
    state.ready = !!$("#databaseSelect").value && !!preferredModel;
    $("#sendButton").disabled = !state.ready;
    if (!preferredModel) showError("No model can serve queries right now. Hover an entry in the model list to see why.");
    renderRecentQueries();
  } catch (error) {
    $("#healthStatus").lastChild.textContent = " Offline";
    $("#connectionStatus").textContent = "Connection issue";
    showError(error.message);
  }
}

let schemaRequest = 0;
let browserTables = [];
let browserRelationships = [];
let selectedExploreTable = null;

function setView(view, tableName = null) {
  const explore = view === "explore";
  $("#exploreView").hidden = !explore;
  document.querySelectorAll(".workspace-only").forEach(element => { element.hidden = explore; });
  document.querySelectorAll(".side-nav a[data-view]").forEach(link => {
    link.classList.toggle("active", link.dataset.view === view);
  });
  if (tableName) {
    selectedExploreTable = tableName;
    $("#exploreSearch").value = "";
  }
  if (explore) renderExplore();
  window.scrollTo({ top: 0, behavior: "instant" });
}

function renderExplore() {
  const container = $("#exploreTables");
  const relations = $("#exploreRelationships");
  container.replaceChildren();
  relations.replaceChildren();
  const dbId = $("#databaseSelect").value;
  const term = $("#exploreSearch").value.trim().toLowerCase();
  const tables = browserTables.filter(table => table.name.toLowerCase().includes(term)
    || table.columns.some(column => column.name.toLowerCase().includes(term)));
  $("#exploreTitle").textContent = dbId ? `Explore ${dbId}` : "Explore database";
  $("#exploreSummary").textContent = dbId
    ? `${browserTables.length} tables · ${browserRelationships.length} relationships${term ? ` · ${tables.length} matching tables` : ""}`
    : "Select a configured database in Connections to inspect its schema.";
  if (!tables.length) {
    const empty = document.createElement("p");
    empty.className = "explore-empty";
    empty.textContent = term ? "No tables or columns match this search." : "No tables are available.";
    container.append(empty);
  }
  tables.forEach(table => {
    const card = document.createElement("section");
    card.className = "explore-table-card";
    if (table.name === selectedExploreTable) card.classList.add("selected");
    const header = document.createElement("header");
    const name = document.createElement("h3");
    name.textContent = table.name;
    const count = document.createElement("span");
    count.textContent = `${table.columns.length} columns`;
    header.append(name, count);
    const columns = document.createElement("ul");
    table.columns.forEach(column => {
      const row = document.createElement("li");
      const label = document.createElement("strong");
      label.textContent = column.name;
      const meta = document.createElement("span");
      meta.textContent = `${column.type}${column.primary_key ? " · Primary key" : ""}`;
      row.append(label, meta);
      columns.append(row);
    });
    card.append(header, columns);
    container.append(card);
  });
  const visibleTables = new Set(tables.map(table => table.name));
  const matchingRelations = browserRelationships.filter(item =>
    !term || visibleTables.has(item.from_table) || visibleTables.has(item.to_table));
  if (!matchingRelations.length) {
    const empty = document.createElement("p");
    empty.className = "explore-empty";
    empty.textContent = term
      ? "No relationships connect the matching tables. Clear the search to see all relationships."
      : "No declared or profiled relationships are available for this database.";
    relations.append(empty);
  }
  matchingRelations.forEach(item => {
    const row = document.createElement("div");
    row.className = "explore-relationship";
    const path = document.createElement("strong");
    path.textContent = `${item.from_table}.${item.from_column} → ${item.to_table}.${item.to_column}`;
    const source = document.createElement("span");
    source.textContent = item.source === "declared_foreign_key" ? "Declared foreign key"
      : `Profile inference${item.cardinality ? ` · ${item.cardinality.replaceAll("_", " ").toLowerCase()}` : ""}`;
    row.append(path, source);
    relations.append(row);
  });
}

async function loadDatabaseSchema() {
  const request = ++schemaRequest;
  const dbId = $("#databaseSelect").value;
  browserTables = [];
  browserRelationships = [];
  selectedExploreTable = null;
  $("#exploreSearch").value = "";
  if ($("#databaseTables")) $("#databaseTables").replaceChildren();
  if ($("#schemaCount")) $("#schemaCount").textContent = dbId || "No database selected";
  if ($("#schemaStatus")) $("#schemaStatus").textContent = dbId ? "Loading table and column names…"
    : "No database files were found. Check TEXT2SQL_DATABASE_ROOT in the project's .env file, then restart the server. Profiles alone do not contain queryable data.";
  if (!dbId) { renderExplore(); return; }
  try {
    const schema = await api(`/api/databases/${encodeURIComponent(dbId)}/schema`);
    if (request !== schemaRequest) return;
    browserTables = schema.tables || [];
    browserRelationships = schema.relationships || [];
    if ($("#schemaCount")) $("#schemaCount").textContent = `${dbId} · ${browserTables.length} tables`;
    renderDatabaseTables();
    renderExplore();
  } catch (error) {
    if (request === schemaRequest && $("#schemaStatus")) $("#schemaStatus").textContent = `Could not load tables: ${error.message}. Use Refresh to retry.`;
    renderExplore();
  }
}

function renderDatabaseTables() {
  if (!$("#schemaSearch") || !$("#databaseTables")) return;
  const term = $("#schemaSearch").value.trim().toLowerCase();
  const container = $("#databaseTables");
  container.replaceChildren();
  const matches = browserTables.filter(table => table.name.toLowerCase().includes(term)
    || table.columns.some(column => column.name.toLowerCase().includes(term)));
  $("#schemaStatus").textContent = matches.length
    ? "Select a table to inspect its columns and relationships in Explore."
    : "No tables match your search.";
  matches.forEach(table => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "database-table-link";
    button.textContent = `${table.name} · ${table.columns.length} columns`;
    button.addEventListener("click", () => {
      setView("explore", table.name);
    });
    container.append(button);
  });
}

function fillSelect(selector, items, valueKey, labelKey, preferred) {
  const select = $(selector);
  select.replaceChildren();
  for (const item of items) {
    const option = document.createElement("option");
    option.value = typeof valueKey === "function" ? valueKey(item) : item[valueKey];
    option.textContent = typeof labelKey === "function" ? labelKey(item) : item[labelKey];
    option.disabled = item.configured === false;
    if (item.unavailable_reason) option.title = item.unavailable_reason;
    option.selected = option.value === preferred;
    select.append(option);
  }
  if (!select.value) {
    const first = [...select.options].find(option => !option.disabled);
    if (first) select.value = first.value;
  }
  if (!items.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No configured databases";
    option.disabled = true;
    select.append(option);
  }
}

function modelLabel(item) {
  const label = item.model;
  return item.configured ? label : `${label} — unavailable`;
}

function fillContextSelect(models) {
  const select = $("#contextModelSelect");
  select.replaceChildren();
  const retrieval = document.createElement("option");
  retrieval.value = "retrieval";
  retrieval.textContent = "Hybrid retrieval only";
  retrieval.selected = true;
  select.append(retrieval);
  for (const item of models) {
    const option = document.createElement("option");
    option.value = `${item.provider}|${item.model}`;
    option.textContent = modelLabel(item);
    option.disabled = item.configured === false;
    if (item.unavailable_reason) option.title = item.unavailable_reason;
    select.append(option);
  }
}

function formatSqlForDisplay(sql) {
  if (!sql || sql === "No SQL was accepted.") return sql;

  // Protect quoted values and identifiers before formatting SQL keywords.
  const protectedParts = [];
  const masked = sql.replace(/'(?:''|[^'])*'|"(?:""|[^"])*"|`(?:``|[^`])*`|\[[^\]]*\]/g, (part) => {
    const marker = `__SQL_PART_${protectedParts.length}__`;
    protectedParts.push(part);
    return marker;
  });

  const clauses = [
    "UNION ALL", "GROUP BY", "ORDER BY", "LEFT OUTER JOIN", "RIGHT OUTER JOIN",
    "FULL OUTER JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN",
    "CROSS JOIN", "UNION", "WITH", "SELECT", "FROM", "JOIN", "WHERE", "HAVING",
    "LIMIT", "OFFSET", "RETURNING",
  ];
  const clausePattern = new RegExp(`\\s*\\b(${clauses.join("|").replaceAll(" ", "\\s+")})\\b\\s*`, "gi");
  let formatted = masked
    .replace(/\s+/g, " ")
    .trim()
    .replace(clausePattern, (_, clause) => `\n${clause.toUpperCase().replace(/\s+/g, " ")} `)
    .replace(/^\n/, "")
    .replace(/\s*;\s*$/, ";");

  // Put top-level output expressions on separate lines without splitting function arguments.
  let depth = 0;
  formatted = [...formatted].map((character) => {
    if (character === "(") depth += 1;
    if (character === ")") depth = Math.max(0, depth - 1);
    return character === "," && depth === 0 ? ",\n  " : character;
  }).join("");

  formatted = formatted
    .split("\n")
    .map((line) => line.trimEnd())
    .filter(Boolean)
    .join("\n");

  protectedParts.forEach((part, index) => {
    formatted = formatted.replaceAll(`__SQL_PART_${index}__`, part);
  });
  return formatted;
}

const sqlKeywords = new Set([
  "ALL", "AND", "AS", "ASC", "BETWEEN", "BY", "CASE", "CAST", "CROSS", "DESC",
  "DISTINCT", "ELSE", "END", "EXCEPT", "EXISTS", "FROM", "FULL", "GROUP", "HAVING",
  "IN", "INNER", "INTERSECT", "IS", "JOIN", "LEFT", "LIKE", "LIMIT", "NOT", "NULL",
  "OFFSET", "ON", "OR", "ORDER", "OUTER", "OVER", "PARTITION", "RIGHT", "SELECT", "THEN",
  "UNION", "WHEN", "WHERE", "WINDOW", "WITH",
]);

const sqlFunctions = new Set([
  "AVG", "COALESCE", "COUNT", "DATE", "DATETIME", "IIF", "MAX", "MIN", "NULLIF",
  "ROUND", "ROW_NUMBER", "STRFTIME", "SUBSTR", "SUM", "TOTAL",
]);

function renderHighlightedSql(element, sql) {
  element.replaceChildren();
  String(sql || "").split("\n").forEach((line, index) => {
    const row = document.createElement("span");
    row.className = "sql-line";
    const number = document.createElement("span");
    number.className = "sql-line-number";
    number.textContent = String(index + 1);
    const content = document.createElement("span");
    content.className = "sql-line-content";
    renderSqlTokens(content, line || " ");
    row.append(number, content);
    element.append(row);
  });
}

function renderSqlTokens(element, sql) {
  const tokens = sql.match(/--[^\n]*|\/\*.*?\*\/|'(?:''|[^'])*'|"(?:""|[^"])*"|`(?:``|[^`])*`|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][A-Za-z0-9_$]*\b|[^A-Za-z0-9_]+/g) || [sql];
  tokens.forEach((token) => {
    let className = "";
    const upper = token.toUpperCase();
    if (token.startsWith("--") || token.startsWith("/*")) className = "sql-comment";
    else if (token.startsWith("'") || token.startsWith('"') || token.startsWith("`")) className = "sql-string";
    else if (/^\d/.test(token)) className = "sql-number";
    else if (sqlKeywords.has(upper)) className = "sql-keyword";
    else if (sqlFunctions.has(upper)) className = "sql-function";
    if (!className) {
      element.append(document.createTextNode(token));
      return;
    }
    const span = document.createElement("span");
    span.className = className;
    span.textContent = token;
    element.append(span);
  });
}

function appendUser(message) {
  $("#emptyState").hidden = true;
  state.currentQuestion = message;
  saveRecentQuery(message);
  const article = document.createElement("article");
  article.className = "message user-message";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message;
  article.append(bubble);
  $("#messages").append(article);
}

function appendAssistant(body, elapsed) {
  const fragment = $("#assistantTemplate").content.cloneNode(true);
  const article = fragment.querySelector("article");
  const generation = body.generation || {};
  const accepted = generation.accepted === true;
  const explanatory = body.operation === "EXPLAIN" || body.operation?.startsWith("EXPLAIN_");
  const attempts = generation.attempts || [];
  const usage = body.token_usage || generation.token_usage || {};
  const tokenTotal = usage.input_tokens == null && usage.output_tokens == null
    ? null
    : (usage.input_tokens || 0) + (usage.output_tokens || 0);
  const timings = body.timings_ms || {};
  const optimization = generation.optimization || null;
  const serverElapsed = timings.total == null ? `${elapsed.toFixed(1)}s` : `${(timings.total / 1000).toFixed(1)}s server`;
  const modelUnavailable = !accepted && generation.termination_reason === "model_error";
  const status = explanatory ? "NO NEW QUERY" : accepted
    ? (generation.execution_status === "ACCEPTED" ? "EXECUTED" : "SAFE SQL")
    : modelUnavailable ? "MODEL UNAVAILABLE" : "FAILED";
  const tokenBadge = tokenTotal == null ? "tokens unavailable" : `${tokenTotal.toLocaleString()} tokens`;
  const badges = [body.operation, status, serverElapsed, `${attempts.length} attempt${attempts.length === 1 ? "" : "s"}`, tokenBadge];
  if (optimization) badges.splice(2, 0, optimization.status.toUpperCase().replaceAll("_", " "));
  for (const value of badges) {
    const badge = document.createElement("span");
    badge.className = `badge${value === "FAILED" || value === "MODEL UNAVAILABLE" ? " error" : ""}`;
    badge.textContent = value;
    fragment.querySelector(".response-meta").append(badge);
  }
  const rejectedAttempts = attempts.filter((attempt) => attemptOutcome(attempt) !== "passed");
  const failureSummary = rejectedAttempts.length
    ? `${rejectedAttempts.length} candidate${rejectedAttempts.length === 1 ? " was" : "s were"} rejected; details are available under Issues.`
    : "";
  const responseMessage = body.explanation || body.presentation?.summary || body.message || "";
  fragment.querySelector(".response-note").textContent = [responseMessage, failureSummary].filter(Boolean).join(" ");
  renderResultChart(fragment.querySelector(".answer-pane"), body.presentation, generation);
  fragment.querySelector(".response-heading h3").textContent = modelUnavailable
    ? "Model unavailable" : explanatory ? "Explanation" : accepted ? "Answer" : "Query issue";
  fragment.querySelector(".response-status-icon").textContent = accepted ? "✓" : "!";
  fragment.querySelector(".response-status-icon").classList.toggle("error", !accepted);
  renderTokenAccounting(fragment, body, generation);
  renderPipelineDetails(fragment, timings, body, generation);
  if (Object.keys(timings).length) {
    fragment.querySelector(".response-note").title = `Routing ${timings.routing || 0} ms · Planning ${timings.planning || 0} ms · Generation/validation/execution ${timings.generation_validation_execution || 0} ms`;
  }
  if (explanatory || modelUnavailable || body.clarification_required) {
    fragment.querySelector(".sql-panel").hidden = true;
    fragment.querySelector(".result-panel").hidden = true;
    fragment.querySelector('[data-tab="sql"]').hidden = true;
    fragment.querySelector('[data-tab="result"]').hidden = true;
  }
  if (body.clarification_required) {
    fragment.querySelector(".feedback-panel").hidden = true;
  }
  renderAttempts(fragment, attempts);
  const sql = generation.sql || "No SQL was accepted.";
  const displaySql = generation.formatted_sql || formatSqlForDisplay(sql);
  renderHighlightedSql(fragment.querySelector(".sql-panel code"), displaySql);
  fragment.querySelector(".copy-button").addEventListener("click", (event) => {
    navigator.clipboard.writeText(sql);
    event.currentTarget.textContent = "Copied";
  });
  renderTable(fragment.querySelector(".table-wrap"), generation.columns || [], generation.rows || []);
  fragment.querySelector(".row-count").textContent = `${generation.row_count || 0} rows${generation.truncated ? " · truncated" : ""}`;
  fragment.querySelector(".csv-button").addEventListener("click", () => {
    downloadCsv(generation.columns || [], generation.rows || []);
  });
  renderModelContext(fragment, body, generation);
  renderHumanReview(fragment, body.human_review);
  setupTechnicalTabs(fragment, rejectedAttempts.length, body.presentation ? "answer" : accepted && !explanatory && generation.columns?.length ? "result" : "answer");
  $("#messages").append(fragment);
  const correctionButton = article?.querySelector(".correction-button");
  correctionButton?.addEventListener("click", () => {
    const category = article.querySelector(".feedback-category").value;
    const correction = article.querySelector(".correction-input").value.trim();
    if (!correction) return;
    send(correction, category);
  });
  article?.querySelector(".correctness-button")?.addEventListener("click", () => {
    send("Check whether the previous SQL and result are correct.");
  });
  article?.querySelector(".optimize-button")?.addEventListener("click", () => {
    send("Optimize the previous SQL without changing its result.");
  });
  article?.scrollIntoView({ behavior: "smooth", block: "end" });
}

function renderHumanReview(fragment, review) {
  const panel = fragment.querySelector(".human-review-panel");
  if (!panel || !review) return;
  panel.hidden = false;
  panel.querySelector(".human-review-reason").textContent = review.reason || "Automated review is uncertain.";
  panel.querySelector(".human-review-question").textContent = review.question || "Please clarify.";
  const options = panel.querySelector(".human-review-options");
  const editableActions = {
    "Enter another ID": "Enter the replacement ID in the chat…",
    "Edit filters": "Describe the corrected filters in the chat…",
    "Edit the question": "Enter the complete revised question…",
    "Choose another period": "Enter the replacement period in the chat…",
    "Check another period": "Enter the period to check in the chat…",
  };
  (review.options || []).forEach((label) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => {
      if (editableActions[label]) {
        const input = $("#messageInput");
        input.placeholder = editableActions[label];
        input.focus();
        return;
      }
      if (review.replacement_target) {
        send(
          `Replace filter value ${JSON.stringify(review.replacement_target)} with ${JSON.stringify(label)}.`,
          "missing_filter",
        );
        return;
      }
      send(label, "other");
    });
    options.append(button);
  });
}

function usageTotal(usage) {
  if (!usage || (usage.input_tokens == null && usage.output_tokens == null)) return null;
  return (usage.input_tokens || 0) + (usage.output_tokens || 0);
}

function sumKnown(values) {
  if (values.some((value) => value == null)) return null;
  return values.reduce((total, value) => total + value, 0);
}

function renderTokenAccounting(fragment, body, generation) {
  const panel = fragment.querySelector(".token-card");
  const grid = fragment.querySelector(".token-grid");
  const note = fragment.querySelector(".token-note");
  if (!panel || !grid || !note) return;

  const attempts = generation.attempts || [];
  const telemetry = generation.telemetry || {};
  const usage = body.token_usage || generation.token_usage || {};
  const total = usageTotal(usage);
  const context = telemetry.planner_call_used ? usageTotal(telemetry.planner_call) : 0;
  const chartCall = (body.call_ledger || []).find((entry) => entry.component === "chart_selection");
  const interpretationCall = (body.call_ledger || []).find((entry) => entry.component === "result_interpretation");
  const chartTokens = chartCall ? usageTotal(chartCall) : 0;
  const interpretationTokens = interpretationCall ? usageTotal(interpretationCall) : 0;
  const attemptTotals = attempts.map((attempt) => usageTotal(attempt.token_usage));
  const sql = attempts.length ? sumKnown(attemptTotals) : 0;
  const discardedAttempts = generation.accepted ? attemptTotals.slice(0, -1) : attemptTotals;
  const wasted = discardedAttempts.length ? sumKnown(discardedAttempts) : 0;
  const conversation = total == null || context == null || sql == null
    || chartTokens == null || interpretationTokens == null
    ? null
    : Math.max(0, total - context - sql - chartTokens - interpretationTokens);
  const cacheRead = body.token_usage?.cache_read_tokens ?? generation.token_usage?.cache_read_tokens ?? null;
  const cacheCreation = body.token_usage?.cache_creation_tokens ?? generation.token_usage?.cache_creation_tokens ?? null;
  const sentTokens = telemetry.selected_model_context_tokens ?? null;
  const avoidedTokens = telemetry.pruned_tokens ?? null;
  const fullTokens = sentTokens == null || avoidedTokens == null ? null : sentTokens + avoidedTokens;
  const reduction = fullTokens ? Math.round((avoidedTokens / fullTokens) * 1000) / 10 : null;
  const reductionBox = fragment.querySelector(".token-reduction");
  if (reductionBox) {
    reductionBox.innerHTML = `<div class="token-reduction-grid"><div><strong>${fullTokens == null ? "—" : fullTokens.toLocaleString()}</strong><small>Full schema estimate</small></div><b>→</b><div><strong>${sentTokens == null ? "—" : sentTokens.toLocaleString()}</strong><small>Sent to LLM</small></div></div><div class="reduction-pill">${reduction == null ? "Reduction unavailable" : `${reduction}% fewer context tokens`}</div>`;
  }

  const rows = [
    ["Total provider tokens", total],
    ["Other calls (resolver / recovery)", conversation],
    ["Context model", context],
    ["SQL attempts", sql],
    ["Chart selection", chartTokens],
    ["Result interpretation", interpretationTokens],
    ["Discarded-attempt tokens", wasted],
    ["Input tokens", usage.input_tokens ?? null],
    ["Output tokens", usage.output_tokens ?? null],
    ["Cache read", cacheRead],
    ["Cache creation", cacheCreation],
  ];
  rows.forEach(([label, value]) => {
    const item = document.createElement("div");
    if (label === "Discarded-attempt tokens") item.className = "token-waste";
    const heading = document.createElement("strong");
    heading.textContent = label;
    const amount = document.createElement("span");
    amount.textContent = value == null ? "Unavailable" : Number(value).toLocaleString();
    item.append(heading, amount);
    grid.append(item);
  });

  attempts.forEach((attempt, index) => {
    const item = document.createElement("div");
    const discarded = !generation.accepted || index < attempts.length - 1;
    if (discarded) item.className = "token-waste";
    const heading = document.createElement("strong");
    heading.textContent = `Attempt ${attempt.number || index + 1}${discarded ? " · discarded" : " · selected"}`;
    const amount = document.createElement("span");
    const value = attemptTotals[index];
    amount.textContent = value == null ? "Unavailable" : Number(value).toLocaleString();
    item.append(heading, amount);
    grid.append(item);
  });

  note.textContent = total == null
    ? "This provider did not return token usage; unavailable values are not treated as zero."
    : "Discarded-attempt tokens are the measurable retry waste. Context and conversation tokens are shown separately because they may be necessary rather than wasted.";
  panel.open = (wasted || 0) > 0;
}

function renderAttempts(fragment, attempts) {
  const panel = fragment.querySelector(".attempts-panel");
  const list = fragment.querySelector(".attempts-list");
  const summary = fragment.querySelector(".attempts-summary");
  const visibleAttempts = attempts.filter((attempt) => attemptOutcome(attempt) !== "passed");
  if (!panel || !list || !summary) return;
  if (!visibleAttempts.length) {
    summary.textContent = attempts.length
      ? `${attempts.length} attempt${attempts.length === 1 ? "" : "s"} passed validation.`
      : "No SQL validation attempt was required for this response.";
    return;
  }

  const failed = visibleAttempts.filter((attempt) => attemptOutcome(attempt) === "failed").length;
  const notSelected = visibleAttempts.filter((attempt) => attemptOutcome(attempt) === "not_selected").length;
  const summaryParts = [`${visibleAttempts.length} rejected candidate${visibleAttempts.length === 1 ? "" : "s"}`];
  if (failed) summaryParts.push(`${failed} failed`);
  if (notSelected) summaryParts.push(`${notSelected} not selected`);
  summary.textContent = summaryParts.join(" · ");
  visibleAttempts.forEach((attempt, index) => {
    const validation = attempt.validation || {};
    const outcome = attemptOutcome(attempt);
    const passed = outcome === "passed";
    const notSelected = outcome === "not_selected";
    const item = document.createElement("section");
    item.className = `attempt-item ${passed ? "attempt-pass" : notSelected ? "attempt-warn" : "attempt-fail"}`;

    const header = document.createElement("div");
    header.className = "attempt-header";
    const title = document.createElement("strong");
    const outcomeLabel = passed ? "PASSED" : notSelected ? "NOT SELECTED" : "FAILED";
    title.textContent = `Attempt ${attempt.number || index + 1} · ${outcomeLabel} · ${validation.code || "UNKNOWN_VALIDATION_RESULT"}`;
    header.append(title);

    const reason = document.createElement("p");
    reason.className = "attempt-reason";
    reason.textContent = validation.message || "No validator explanation was returned.";
    item.append(header, reason);

    if (!passed && attempt.sql) {
      const sql = document.createElement("pre");
      const sqlCode = document.createElement("code");
      sqlCode.className = "language-sql";
      renderHighlightedSql(sqlCode, formatSqlForDisplay(attempt.sql));
      sql.append(sqlCode);
      item.append(sql);
    }
    list.append(item);
  });
}

function setupTechnicalTabs(fragment, issueCount, initialTab = "answer") {
  const article = fragment.querySelector(".assistant-message");
  if (!article) return;
  const validationTab = article.querySelector('[data-tab="validation"]');
  validationTab.textContent = issueCount ? `Validation · ${issueCount}` : "Validation";
  validationTab.classList.toggle("has-issues", issueCount > 0);

  const activate = (name) => {
    article.querySelectorAll(".response-tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.tab === name);
    });
    article.querySelectorAll(".response-pane").forEach((pane) => {
      pane.classList.toggle("active", pane.dataset.tabPanel === name);
    });
  };
  article.querySelectorAll(".response-tab").forEach((tab) => {
    tab.addEventListener("click", () => activate(tab.dataset.tab));
  });
  activate(initialTab);
}

function renderModelContext(fragment, body, generation) {
  const panel = fragment.querySelector(".context-panel");
  if (!panel) return;
  const modelContext = generation.model_context || null;
  if (!modelContext) {
    panel.hidden = true;
    fragment.querySelector(".context-card").hidden = true;
    fragment.querySelector('[data-tab="context"]').hidden = true;
    return;
  }
  renderSchemaVisual(fragment.querySelector(".schema-visual"), modelContext);
  fragment.querySelector(".context-json").textContent = JSON.stringify(modelContext, null, 2);
  const tables = modelContext.tables || modelContext.execution_context?.tables || {};
  const relationships = modelContext.relationships || [];
  const columnCount = Object.values(tables).reduce((count, table) => {
    const columns = table.columns || {};
    return count + (Array.isArray(columns) ? columns.length : Object.keys(columns).length);
  }, 0);
  fragment.querySelector(".context-summary").textContent = `${Object.keys(tables).length} tables · ${columnCount} columns · ${relationships.length} relationships`;
}

function renderSchemaVisual(container, modelContext) {
  if (!container) return;
  const tables = modelContext.tables || modelContext.execution_context?.tables || {};
  const tableGrid = document.createElement("div");
  tableGrid.className = "schema-table-grid";
  Object.entries(tables).forEach(([tableName, tableData]) => {
    const card = document.createElement("section");
    card.className = "schema-table-card";
    const heading = document.createElement("h4");
    heading.textContent = tableName;
    const grain = document.createElement("p");
    grain.textContent = tableData.grain || "grain not profiled";
    const list = document.createElement("ul");
    const columns = tableData.columns || {};
    const primaryKeys = new Set(tableData.primary_key || []);
    const relationshipKeys = new Set(tableData.relationship_keys || []);
    const entries = Array.isArray(columns)
      ? columns.map((name) => [name, {}])
      : Object.entries(columns);
    entries.forEach(([columnName, metadata]) => {
      const item = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = columnName;
      const tags = document.createElement("span");
      tags.className = "column-tags";
      const roles = new Set(metadata.key_roles || []);
      if (primaryKeys.has(columnName) || roles.has("PRIMARY_KEY")) tags.append(makeTag("PK"));
      if (relationshipKeys.has(columnName) || roles.has("FOREIGN_KEY")) tags.append(makeTag("FK"));
      if (metadata.type) tags.append(makeTag(metadata.type));
      if (metadata.format) tags.append(makeTag(metadata.format));
      item.append(name, tags);
      list.append(item);
    });
    card.append(heading, grain, list);
    tableGrid.append(card);
  });
  container.append(tableGrid);

  const relationships = modelContext.relationships || [];
  if (relationships.length) {
    const relationshipList = document.createElement("div");
    relationshipList.className = "relationship-list";
    relationships.forEach((relationship) => {
      const item = document.createElement("div");
      item.className = "relationship-item";
      item.textContent = `${relationship.left}  →  ${relationship.right}`;
      const details = [relationship.cardinality, relationship.fanout_risk ? "fanout risk" : null]
        .filter(Boolean)
        .join(" · ");
      if (details) item.append(makeTag(details));
      relationshipList.append(item);
    });
    container.append(relationshipList);
  }
}

function makeTag(value) {
  const tag = document.createElement("small");
  tag.className = "schema-tag";
  tag.textContent = value;
  return tag;
}

function renderTable(container, columns, rows) {
  if (!columns.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No rows returned, or execution was not completed.";
    container.append(empty);
    return;
  }
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  columns.forEach((column) => { const cell = document.createElement("th"); cell.textContent = column; head.append(cell); });
  const body = table.createTBody();
  rows.forEach((row) => {
    const tr = body.insertRow();
    row.forEach((value) => {
      const td = tr.insertCell();
      if (value === null) {
        const badge = document.createElement("span");
        badge.className = "null-value";
        badge.textContent = "NULL";
        td.append(badge);
      } else {
        td.textContent = String(value);
        if (typeof value === "number") td.classList.add("numeric");
      }
    });
  });
  container.append(table);
}

function downloadCsv(columns, rows) {
  if (!columns.length) return;
  const escape = (value) => {
    if (value === null || value === undefined) return "";
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  const csv = [columns, ...rows].map((row) => row.map(escape).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `query-result-${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function renderPipelineDetails(fragment, timings, body, generation) {
  const container = fragment.querySelector(".pipeline-details");
  if (!container) return;
  const stages = [
    ["Conversation routing", timings.routing],
    ["Context planning", timings.planning],
    ["Generation / validation / execution", timings.generation_validation_execution],
    ["Total request", timings.total],
  ];
  stages.forEach(([label, milliseconds]) => {
    const row = document.createElement("div");
    row.className = `pipeline-row${milliseconds == null ? " unmeasured" : ""}`;
    const status = document.createElement("span");
    status.textContent = label;
    const duration = document.createElement("span");
    duration.textContent = milliseconds == null ? "Not reported" : `${Number(milliseconds).toLocaleString()} ms`;
    row.append(status, duration);
    container.append(row);
  });
  if (body.progress_events?.length) {
    const history = document.createElement("details");
    history.className = "progress-history";
    const summary = document.createElement("summary");
    summary.textContent = "Request activity";
    const list = document.createElement("ol");
    body.progress_events.forEach(event => {
      const item = document.createElement("li");
      item.textContent = `${(event.elapsed_ms / 1000).toFixed(1)}s · ${progressLabels[event.stage] || event.stage}`;
      list.append(item);
    });
    history.append(summary, list);
    container.append(history);
  }
}

function getRecentQueries() {
  try { return JSON.parse(localStorage.getItem(recentStorageKey) || "[]"); }
  catch { return []; }
}

function saveRecentQuery(question) {
  const recent = getRecentQueries().filter((item) => item !== question);
  recent.unshift(question);
  try { localStorage.setItem(recentStorageKey, JSON.stringify(recent.slice(0, 7))); }
  catch { /* Browser storage restrictions must not prevent submitting a question. */ }
  renderRecentQueries();
}

function renderRecentQueries() {
  const container = $("#recentQueries");
  if (!container) return;
  container.replaceChildren();
  const recent = getRecentQueries();
  if (!recent.length) {
    const empty = document.createElement("p");
    empty.className = "recent-empty";
    empty.textContent = "Your recent questions will appear here.";
    container.append(empty);
    return;
  }
  recent.forEach((question) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "recent-item";
    button.textContent = `◯  ${question}`;
    button.title = question;
    button.addEventListener("click", () => {
      $("#messageInput").value = question;
      $("#messageInput").focus();
    });
    container.append(button);
  });
}

function showError(message) {
  appendAssistant({ operation: "ERROR", message, generation: { accepted: false, attempts: [] } }, 0);
}

const pipelineStages = [
  ["conversation", "Question"],
  ["retrieval", "Retrieval"],
  ["context_selection", "Context"],
  ["grounding", "Grounding"],
  ["generation", "Generate SQL"],
  ["validation", "Validate"],
  ["execution", "Run query"],
  ["result_review", "Review result"],
  ["recovery", "Recovery"],
  ["presentation", "Choose chart & explain"],
];

const progressLabels = {
  queued: "Waiting to start…",
  presentation: "Choosing the chart, then preparing a plain-language answer",
  conversation: "Understanding your request",
  retrieval: "Finding relevant tables and columns",
  context_selection: "Selecting context with your context model",
  grounding: "Verifying schema and metadata",
  generation: "Generating SQL — waiting for the model",
  validation: "Checking SQL safety",
  optimization: "Checking a possible SQL optimization",
  execution: "Running the read-only query",
  result_review: "Reviewing an empty or NULL result",
  recovery: "Investigating the query issue",
  recovery_reasoning: "Recovery agent: deciding what evidence to check",
  recovery_schema: "Recovery agent: inspecting schema",
  recovery_values: "Recovery agent: checking actual database values",
  recovery_probe: "Recovery agent: running a bounded read-only check",
  recovery_checks: "Running bounded diagnostic checks",
  recovery_decision: "Recovery agent: evaluating evidence",
  completed: "Finished",
};

function appendProgress(provider, model, contextMode) {
  const article = document.createElement("article");
  article.className = "message assistant-message progress-message";
  article.setAttribute("role", "status");
  article.innerHTML = '<div class="message-content"><div class="progress-heading"><span class="progress-spinner" aria-hidden="true"></span><p class="progress-stage">Queued…</p></div><p class="progress-detail"></p><div class="pipeline-tracker"></div><details class="progress-history"><summary>Activity log</summary><ol></ol></details></div>';
  const tracker = article.querySelector(".pipeline-tracker");
  pipelineStages.forEach(([stage, label]) => {
    if (stage === "context_selection" && contextMode === "retrieval") return;
    const step = document.createElement("span");
    step.dataset.stage = stage;
    step.textContent = label;
    tracker.append(step);
  });
  article.querySelector(".progress-detail").textContent = `${provider} · ${model}`;
  $("#messages").append(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  return article;
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function waitForJob(jobId, progress) {
  while (true) {
    const job = await api(`/api/chat/jobs/${jobId}`);
    const seconds = ((job.elapsed_ms || 0) / 1000).toFixed(1);
    updatePipeline(progress, job.stage, job.status, job.events || []);
    const label = progressLabels[job.stage] || "Processing your request";
    progress.querySelector(".progress-stage").textContent = label;
    progress.querySelector(".progress-detail").textContent = `${seconds}s elapsed · ${Number(seconds) > 20 ? "Still working. You can cancel without changing your previous result." : "You can cancel at any time."}`;
    const log = progress.querySelector(".progress-history ol");
    log.replaceChildren();
    (job.events || []).forEach(event => {
      const item = document.createElement("li");
      item.textContent = `${(event.elapsed_ms / 1000).toFixed(1)}s · ${progressLabels[event.stage] || event.stage}`;
      log.append(item);
    });
    if (job.status === "completed") return { ...job.response, progress_events: job.events || [] };
    if (job.status === "cancelled") throw new Error("Request cancelled.");
    if (job.status === "failed") throw new Error(job.error || "Chat job failed.");
    await wait(500);
  }
}

function visiblePipelineStage(stage) {
  if (stage.startsWith("recovery_")) return "recovery";
  if (stage === "optimization") return "validation";
  return stage;
}

function updatePipeline(progress, currentStage, jobStatus, events = []) {
  const visibleStage = visiblePipelineStage(currentStage);
  const seen = new Set(JSON.parse(progress.dataset.seenStages || "[]"));
  events.forEach((event) => seen.add(visiblePipelineStage(event.stage)));
  if (progress.dataset.activeStage && progress.dataset.activeStage !== visibleStage) seen.add(progress.dataset.activeStage);
  seen.add(visibleStage);
  progress.dataset.activeStage = visibleStage;
  progress.dataset.seenStages = JSON.stringify([...seen]);
  progress.querySelectorAll(".pipeline-tracker span").forEach((step) => {
    const completed = seen.has(step.dataset.stage)
      && (jobStatus === "completed" || step.dataset.stage !== visibleStage);
    step.classList.toggle("complete", completed);
    step.classList.toggle("active", jobStatus !== "completed" && step.dataset.stage === visibleStage);
  });
}

async function send(message, feedbackCategory = null) {
  if (state.busy || !message.trim()) return;
  setView("workspace");
  if (!state.ready) {
    showError("A configured database and SQL model are required before asking a question.");
    return;
  }
  state.busy = true;
  $("#sendButton").disabled = true;
  ["#databaseSelect", "#sqlModelSelect", "#contextModelSelect", "#resetButton"].forEach(id => { $(id).disabled = true; });
  appendUser(message.trim());
  const [provider, model] = $("#sqlModelSelect").value.split("|");
  const contextSelection = $("#contextModelSelect").value;
  const contextMode = contextSelection === "retrieval" ? "retrieval" : "model1";
  const [contextProvider, contextModel] = contextMode === "model1"
    ? contextSelection.split("|")
    : [null, null];
  const started = performance.now();
  const progress = appendProgress(provider, model, contextMode);
  state.cancelled = false;
  $("#cancelButton").hidden = false;
  try {
    const created = await api("/api/chat/jobs", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        db_id: $("#databaseSelect").value,
        message: message.trim(),
        provider,
        model,
        context_provider: contextMode === "model1" ? contextProvider : null,
        context_model: contextMode === "model1" ? contextModel : null,
        context_mode: contextMode,
        execute: true,
        max_rows: 100,
        feedback_category: feedbackCategory,
      }),
    });
    state.jobId = created.job_id;
    if (state.cancelled) await api(`/api/chat/jobs/${created.job_id}`, { method: "DELETE" });
    const body = await waitForJob(created.job_id, progress);
    progress.remove();
    appendAssistant(body, (performance.now() - started) / 1000);
  } catch (error) {
    progress.remove();
    showError(error.message);
  } finally {
    state.busy = false;
    state.jobId = null;
    $("#sendButton").disabled = !state.ready;
    ["#databaseSelect", "#sqlModelSelect", "#contextModelSelect", "#resetButton"].forEach(id => { $(id).disabled = false; });
    $("#cancelButton").hidden = true;
    $("#messageInput").focus();
  }
}

$("#cancelButton").addEventListener("click", async () => {
  if (state.cancelled) return;
  state.cancelled = true;
  $("#cancelButton").disabled = true;
  try {
    if (state.jobId) await api(`/api/chat/jobs/${state.jobId}`, { method: "DELETE" });
  } catch (error) {
    state.cancelled = false;
    showError(`Could not cancel: ${error.message}`);
  } finally {
    $("#cancelButton").disabled = false;
  }
});

$("#chatForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $("#messageInput");
  const message = input.value;
  input.value = "";
  send(message);
});

$("#messageInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chatForm").requestSubmit();
  }
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    $("#messageInput").value = button.dataset.example;
    $("#messageInput").focus();
  });
});

$("#clearRecent").addEventListener("click", () => {
  localStorage.removeItem(recentStorageKey);
  renderRecentQueries();
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    $("#resetButton").click();
  }
});

$("#resetButton").addEventListener("click", async () => {
  try {
    await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, db_id: $("#databaseSelect").value, message: "Reset context" }),
    });
    $("#messages").replaceChildren();
    $("#emptyState").hidden = false;
    $("#messageInput").focus();
  } catch (error) { showError(error.message); }
});

initialize();

$("#databaseSelect").addEventListener("change", loadDatabaseSchema);
if ($("#refreshSchema")) $("#refreshSchema").addEventListener("click", loadDatabaseSchema);
if ($("#schemaSearch")) $("#schemaSearch").addEventListener("input", renderDatabaseTables);
$("#exploreSearch").addEventListener("input", renderExplore);
document.querySelectorAll(".side-nav a[data-view]").forEach(link => {
  link.addEventListener("click", event => {
    event.preventDefault();
    if (link.dataset.view === "connections") {
      setView("workspace");
      $("#connectionCard").scrollIntoView({ behavior: "smooth", block: "center" });
      document.querySelectorAll(".side-nav a[data-view]").forEach(item => {
        item.classList.toggle("active", item === link);
      });
    } else {
      setView(link.dataset.view);
    }
  });
});
