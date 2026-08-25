const $ = (selector) => document.querySelector(selector);
const sessionId = localStorage.getItem("queryRoomSession") || crypto.randomUUID().replaceAll("-", "");
localStorage.setItem("queryRoomSession", sessionId);
$("#sessionId").textContent = sessionId;

const state = { busy: false, jobId: null, cancelled: false };

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
    const [databases, models] = await Promise.all([api("/api/databases"), api("/api/models")]);
    const preferredModel = models.find((item) => item.configured);
    fillSelect("#databaseSelect", databases.filter((item) => item.configured && item.dialect === "sqlite"), "db_id", "db_id", null);
    const preferred = preferredModel && `${preferredModel.provider}|${preferredModel.model}`;
    fillSelect("#contextModelSelect", models, (item) => `${item.provider}|${item.model}`, modelLabel, preferred);
    fillSelect("#sqlModelSelect", models, (item) => `${item.provider}|${item.model}`, modelLabel, preferred);
    updateContextModelState();
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
  const label = `${item.provider} · ${item.model}`;
  return item.configured ? label : `${label} — unavailable`;
}

function updateContextModelState() {
  const retrievalOnly = $("#contextModeSelect").value === "retrieval";
  $("#contextModelSelect").disabled = retrievalOnly;
}

$("#contextModeSelect").addEventListener("change", updateContextModelState);

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
  const explanatory = body.operation === "EXPLAIN";
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
  const failedAttempts = attempts.filter((attempt) => attempt.validation?.valid !== true);
  const failureSummary = failedAttempts.length
    ? failedAttempts.map((attempt) => {
      const validation = attempt.validation || {};
      return `Attempt ${attempt.number} failed: ${validation.code || "UNKNOWN"} — ${validation.message || "No reason returned."}`;
    }).join(" ")
    : "";
  const responseMessage = body.explanation || body.message || "";
  fragment.querySelector(".response-note").textContent = [responseMessage, failureSummary].filter(Boolean).join(" ");
  renderSemanticStatus(fragment, generation, explanatory || modelUnavailable);
  renderTokenAccounting(fragment, body, generation);
  if (Object.keys(timings).length) {
    fragment.querySelector(".response-note").title = `Routing ${timings.routing || 0} ms · Planning ${timings.planning || 0} ms · Generation/validation/execution ${timings.generation_validation_execution || 0} ms`;
  }
  if (explanatory || modelUnavailable) {
    fragment.querySelector(".sql-panel").hidden = true;
    fragment.querySelector(".result-panel").hidden = true;
  }
  renderAttempts(fragment, attempts);
  const sql = generation.sql || "No SQL was accepted.";
  fragment.querySelector("code").textContent = formatSqlForDisplay(sql);
  fragment.querySelector(".copy-button").addEventListener("click", (event) => {
    navigator.clipboard.writeText(sql);
    event.currentTarget.textContent = "Copied";
  });
  renderTable(fragment.querySelector(".table-wrap"), generation.columns || [], generation.rows || []);
  fragment.querySelector(".row-count").textContent = `${generation.row_count || 0} rows${generation.truncated ? " · truncated" : ""}`;
  renderModelContext(fragment, body, generation);
  $("#messages").append(fragment);
  const correctionButton = article?.querySelector(".correction-button");
  correctionButton?.addEventListener("click", () => {
    const category = article.querySelector(".feedback-category").value;
    const correction = article.querySelector(".correction-input").value.trim();
    if (!correction) return;
    send(correction, category);
  });
  article?.scrollIntoView({ behavior: "smooth", block: "end" });
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

function renderSemanticStatus(fragment, generation, hidden) {
  const container = fragment.querySelector(".semantic-status");
  if (!container || hidden) {
    if (container) container.hidden = true;
    return;
  }
  const attempts = generation.attempts || [];
  const safetyPassed = attempts.some((attempt) => attempt.validation?.valid === true);
  const executionPassed = generation.accepted === true && generation.execution_status === "ACCEPTED";
  const items = [
    ["Safety", safetyPassed ? "Passed" : "Failed", safetyPassed ? "pass" : "fail"],
    ["Execution", executionPassed ? "Succeeded" : "Not completed", executionPassed ? "pass" : "fail"],
    ["Correctness", "Not measured", "unknown"],
  ];
  items.forEach(([label, value, stateName]) => {
    const item = document.createElement("div");
    item.className = `status-card status-${stateName}`;
    const heading = document.createElement("strong");
    heading.textContent = label;
    const result = document.createElement("span");
    result.textContent = value;
    item.append(heading, result);
    container.append(item);
  });
}

function renderAttempts(fragment, attempts) {
  const panel = fragment.querySelector(".attempts-panel");
  const list = fragment.querySelector(".attempts-list");
  const summary = fragment.querySelector(".attempts-summary");
  if (!panel || !list || !summary || !attempts.length) {
    if (panel) panel.hidden = true;
    return;
  }

  const failed = attempts.filter((attempt) => attempt.validation?.valid !== true).length;
  summary.textContent = `Validation attempts (${attempts.length}) · ${failed} failed`;
  panel.open = failed > 0;
  attempts.forEach((attempt, index) => {
    const validation = attempt.validation || {};
    const passed = validation.valid === true;
    const item = document.createElement("section");
    item.className = `attempt-item ${passed ? "attempt-pass" : "attempt-fail"}`;

    const header = document.createElement("div");
    header.className = "attempt-header";
    const title = document.createElement("strong");
    title.textContent = `Attempt ${attempt.number || index + 1} · ${passed ? "PASSED" : "FAILED"} · ${validation.code || "UNKNOWN_VALIDATION_RESULT"}`;
    header.append(title);

    const reason = document.createElement("p");
    reason.className = "attempt-reason";
    reason.textContent = validation.message || "No validator explanation was returned.";
    item.append(header, reason);

    if (!passed && attempt.sql) {
      const sql = document.createElement("pre");
      const sqlCode = document.createElement("code");
      sqlCode.className = "language-sql";
      sqlCode.textContent = formatSqlForDisplay(attempt.sql);
      sql.append(sqlCode);
      item.append(sql);
    }
    list.append(item);
  });
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
    row.forEach((value) => { const td = tr.insertCell(); td.textContent = value === null ? "NULL" : String(value); });
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
  ["generation_validation_execution", "SQL + validation + execution"],
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
  const activeIndex = pipelineStages.findIndex(([stage]) => stage === currentStage);
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
  const [contextProvider, contextModel] = $("#contextModelSelect").value.split("|");
  const started = performance.now();
  const contextMode = $("#contextModeSelect").value;
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
        evidence: $("#evidenceInput").value.trim() || null,
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

$("#suggestions").addEventListener("click", (event) => {
  if (event.target.matches("button")) send(event.target.textContent);
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
