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
    fillSelect("#modelSelect", models, (item) => `${item.provider}|${item.model}`, modelLabel, preferredModel && `${preferredModel.provider}|${preferredModel.model}`);
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
  return item.configured ? item.model : `${item.model} — unavailable`;
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
    ? (generation.execution_status || "ACCEPTED")
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
  fragment.querySelector(".context-summary")?.remove();
  fragment.querySelector(".context-json").textContent = JSON.stringify(modelContext, null, 2);
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

function appendProgress(provider, model) {
  const article = document.createElement("article");
  article.className = "message assistant-message progress-message";
  article.innerHTML = '<div class="avatar">Q</div><div class="message-content"><p class="progress-stage">Queued…</p><p class="progress-detail"></p></div>';
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
    const stage = job.status === "queued" ? "Queued…" : "Generating and validating SQL…";
    progress.querySelector(".progress-stage").textContent = stage;
    progress.querySelector(".progress-detail").textContent = `${seconds}s elapsed · ${job.stage}`;
    if (job.status === "completed") return job.response;
    if (job.status === "cancelled") throw new Error("Request cancelled.");
    if (job.status === "failed") throw new Error(job.error || "Chat job failed.");
    await wait(500);
  }
}

async function send(message, feedbackCategory = null) {
  if (state.busy || !message.trim()) return;
  state.busy = true;
  $("#sendButton").disabled = true;
  appendUser(message.trim());
  const [provider, model] = $("#modelSelect").value.split("|");
  const started = performance.now();
  const progress = appendProgress(provider, model);
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
