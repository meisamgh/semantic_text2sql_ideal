const $ = (selector) => document.querySelector(selector);
const sessionId = localStorage.getItem("queryRoomSession") || crypto.randomUUID().replaceAll("-", "");
localStorage.setItem("queryRoomSession", sessionId);
$("#sessionId").textContent = sessionId;

const state = { busy: false, jobId: null, cancelled: false };
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
    const [databases, models] = await Promise.all([
      api("/api/databases"),
      api(`/api/models?refresh=${Date.now()}`, { cache: "no-store" }),
    ]);
    const preferredModel = models.find((item) => item.configured);
    fillSelect("#databaseSelect", databases.filter((item) => item.configured && item.dialect === "sqlite"), "db_id", "db_id", null);
    const preferred = preferredModel && `${preferredModel.provider}|${preferredModel.model}`;
    fillContextSelect(models);
    fillSelect("#sqlModelSelect", models, (item) => `${item.provider}|${item.model}`, modelLabel, preferred);
    if (!preferredModel) showError("No model can serve queries right now. Hover an entry in the model list to see why.");
  } catch (error) {
    $("#healthStatus").lastChild.textContent = " Offline";
    showError(error.message);
  }
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
  const responseMessage = body.explanation || body.message || "";
  fragment.querySelector(".response-note").textContent = [responseMessage, failureSummary].filter(Boolean).join(" ");
  renderTokenAccounting(fragment, body, generation);
  if (Object.keys(timings).length) {
    fragment.querySelector(".response-note").title = `Routing ${timings.routing || 0} ms · Planning ${timings.planning || 0} ms · Generation/validation/execution ${timings.generation_validation_execution || 0} ms`;
  }
  if (explanatory || modelUnavailable || body.clarification_required) {
    fragment.querySelector(".sql-panel").hidden = true;
    fragment.querySelector(".result-panel").hidden = true;
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
  renderModelContext(fragment, body, generation);
  renderHumanReview(fragment, body.human_review);
  setupTechnicalTabs(fragment, rejectedAttempts.length);
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
  const panel = fragment.querySelector(".token-panel");
  const grid = fragment.querySelector(".token-grid");
  const note = fragment.querySelector(".token-note");
  if (!panel || !grid || !note) return;

  const attempts = generation.attempts || [];
  const telemetry = generation.telemetry || {};
  const total = usageTotal(body.token_usage || generation.token_usage);
  const context = telemetry.planner_call_used ? usageTotal(telemetry.planner_call) : 0;
  const attemptTotals = attempts.map((attempt) => usageTotal(attempt.token_usage));
  const sql = attempts.length ? sumKnown(attemptTotals) : 0;
  const discardedAttempts = generation.accepted ? attemptTotals.slice(0, -1) : attemptTotals;
  const wasted = discardedAttempts.length ? sumKnown(discardedAttempts) : 0;
  const conversation = total == null || context == null || sql == null
    ? null
    : Math.max(0, total - context - sql);
  const cacheRead = body.token_usage?.cache_read_tokens ?? generation.token_usage?.cache_read_tokens ?? null;
  const cacheCreation = body.token_usage?.cache_creation_tokens ?? generation.token_usage?.cache_creation_tokens ?? null;

  const rows = [
    ["Total provider tokens", total],
    ["Conversation resolver", conversation],
    ["Context model", context],
    ["SQL attempts", sql],
    ["Discarded-attempt tokens", wasted],
    ["Model context estimate", telemetry.selected_model_context_tokens ?? null],
    ["Tokens avoided by pruning", telemetry.pruned_tokens ?? null],
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
  if (!panel || !list || !summary || !visibleAttempts.length) {
    if (panel) panel.hidden = true;
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

function setupTechnicalTabs(fragment, issueCount) {
  const panel = fragment.querySelector(".technical-panel");
  if (!panel) return;
  const issueTab = panel.querySelector('[data-tab="issues"]');
  if (!issueCount) issueTab.hidden = true;
  else {
    issueTab.classList.add("has-issues");
    issueTab.textContent = `Issues · ${issueCount}`;
  }

  const activate = (name) => {
    panel.querySelectorAll(".technical-tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.tab === name);
    });
    panel.querySelectorAll(".tab-pane").forEach((pane) => {
      pane.classList.toggle("active", pane.dataset.tabPanel === name);
    });
  };
  panel.querySelectorAll(".technical-tab").forEach((tab) => {
    tab.addEventListener("click", () => activate(tab.dataset.tab));
  });
  const initial = panel.querySelector('[data-tab="context"]')?.hidden ? "tokens" : "context";
  activate(initial);
}

function renderModelContext(fragment, body, generation) {
  const panel = fragment.querySelector(".context-panel");
  if (!panel) return;
  const modelContext = generation.model_context || null;
  if (!modelContext) {
    panel.hidden = true;
    return;
  }
  renderSchemaVisual(fragment.querySelector(".schema-visual"), modelContext);
  fragment.querySelector(".context-json").textContent = JSON.stringify(modelContext, null, 2);
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
  ["recovery", "Recovery"],
];

function appendProgress(provider, model, contextMode) {
  const article = document.createElement("article");
  article.className = "message assistant-message progress-message";
  article.innerHTML = '<div class="avatar">Q</div><div class="message-content"><p class="progress-stage">Queued…</p><div class="pipeline-tracker"></div><p class="progress-detail"></p></div>';
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
    updatePipeline(progress, job.stage, job.status);
    const label = job.status === "queued" ? "Queued…" : "Building and executing your query…";
    progress.querySelector(".progress-stage").textContent = label;
    progress.querySelector(".progress-detail").textContent = `${seconds}s elapsed`;
    if (job.status === "completed") return job.response;
    if (job.status === "cancelled") throw new Error("Request cancelled.");
    if (job.status === "failed") throw new Error(job.error || "Chat job failed.");
    await wait(500);
  }
}

function updatePipeline(progress, currentStage, jobStatus) {
  // Optimization and execution still run internally, but the compact user-facing
  // path intentionally ends at validation.
  const visibleStage = ["optimization", "execution"].includes(currentStage)
    ? "validation"
    : currentStage;
  const activeIndex = pipelineStages.findIndex(([stage]) => stage === visibleStage);
  progress.querySelectorAll(".pipeline-tracker span").forEach((step) => {
    const index = pipelineStages.findIndex(([stage]) => stage === step.dataset.stage);
    step.classList.toggle("complete", jobStatus === "completed" || (activeIndex >= 0 && index < activeIndex));
    step.classList.toggle("active", jobStatus !== "completed" && index === activeIndex);
  });
}

async function send(message, feedbackCategory = null) {
  if (state.busy || !message.trim()) return;
  state.busy = true;
  $("#sendButton").disabled = true;
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
    const body = await waitForJob(created.job_id, progress);
    progress.remove();
    appendAssistant(body, (performance.now() - started) / 1000);
  } catch (error) {
    progress.remove();
    showError(error.message);
  } finally {
    state.busy = false;
    state.jobId = null;
    $("#sendButton").disabled = false;
    $("#cancelButton").hidden = true;
    $("#messageInput").focus();
  }
}

$("#cancelButton").addEventListener("click", async () => {
  if (!state.jobId || state.cancelled) return;
  state.cancelled = true;
  $("#cancelButton").disabled = true;
  try {
    await api(`/api/chat/jobs/${state.jobId}`, { method: "DELETE" });
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

$("#resetButton").addEventListener("click", async () => {
  try {
    await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, db_id: $("#databaseSelect").value, message: "Reset context" }),
    });
    $("#messages").replaceChildren();
    $("#emptyState").hidden = false;
  } catch (error) { showError(error.message); }
});

initialize();
