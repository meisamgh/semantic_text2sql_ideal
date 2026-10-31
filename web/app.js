const $ = (selector) => document.querySelector(selector);
const sessionId = localStorage.getItem("queryRoomSession") || crypto.randomUUID().replaceAll("-", "");
localStorage.setItem("queryRoomSession", sessionId);

const state = {
  busy: false, jobId: null, cancelled: false, currentQuestion: "", ready: false,
  lastBody: null, history: [], saved: [], glossaryByDb: {}, schemaRequest: 0,
  activeRunId: null, branchFromRunId: null, diffCache: new Map(), modelConfig: null,
  restorePending: false,
};
let voiceInput = null;
import("/static/voice.js?v=voice-1").then(async ({VoiceInput}) => {
  voiceInput = new VoiceInput({
    button: $("#voiceButton"), cancelButton: $("#voiceCancelButton"),
    input: $("#messageInput"), status: $("#voiceStatus"),
    onBusy: busy => { $("#sendButton").disabled = busy || state.busy || !state.ready; },
  });
  await voiceInput.initialize();
}).catch(() => { $("#voiceStatus").textContent = "Voice input could not load. You can still type a question."; });

const recentStorageKey = "queryRoomRecentQueries";
const postSafetyOptimizationCodes = new Set([
  "OPTIMIZATION_NOT_FASTER",
  "OPTIMIZATION_CHANGED_RESULT",
  "OPTIMIZATION_EQUIVALENCE_UNPROVEN",
]);
const EXPLANATION_OPERATIONS_UI = new Set(["EXPLAIN", "EXPLAIN_SQL", "EXPLAIN_RESULT", "EXPLAIN_INTERPRETATION", "EXPLAIN_CONTEXT", "EXPLAIN_FAILURE"]);

function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((slot) => {
    const name = slot.dataset.icon;
    if (!slot.innerHTML.trim()) slot.innerHTML = icon(name);
  });
}

function attemptOutcome(attempt) {
  const validation = attempt.validation || {};
  if (validation.valid === true) return "passed";
  if (postSafetyOptimizationCodes.has(validation.code)) return "not_selected";
  return "failed";
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const raw = await response.text();
  let body = null;
  if (raw.trim()) {
    try { body = JSON.parse(raw); }
    catch { if (response.ok) throw new Error(`Invalid JSON response from ${path}.`); }
  }
  if (!response.ok) {
    let detail = body?.detail;
    if (Array.isArray(detail)) {
      detail = detail.map((item) => {
        const path = Array.isArray(item.loc) ? item.loc.join(".") : item.loc;
        return [path, item.msg].filter(Boolean).join(": ");
      }).join("; ");
    }
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return body;
}

async function initialize() {
  window.QueryflowDashboardStudio?.install({api,renderChart:renderChartInto,renderTable});
  hydrateIcons();
  if (window.matchMedia("(max-width: 760px)").matches) {
    document.body.classList.add("sidebar-collapsed");
    $("#sidebarToggle").setAttribute("aria-expanded", "false");
    $(".sidebar").inert = true;
  }
  try {
    await api("/api/health");
    setHealth(true);
    const [databases, models, uiConfig] = await Promise.all([
      api("/api/databases"),
      api(`/api/models?refresh=${Date.now()}`, { cache: "no-store" }),
      api("/api/ui-config", { cache: "no-store" }),
    ]);
    fillSelect(
      "#databaseSelect",
      databases.filter((item) => item.configured && item.db_id === uiConfig.default_db_id),
      "db_id",
      (item) => `${item.db_id} (${item.dialect})`,
      uiConfig.default_db_id,
    );
    [...$("#databaseSelect").options].forEach((option) => {
      const match = databases.find((item) => item.db_id === option.value);
      if (match) option.dataset.dialect = match.dialect;
    });
    await Promise.all([loadDatabaseSchema(), refreshHistory(), loadGlossary($("#databaseSelect").value)]);
    const configuredSql = models.find(item => item.provider === uiConfig.sql_provider
      && item.model === uiConfig.sql_model && item.configured);
    const configuredContext = uiConfig.context_mode !== "model1" || models.some(item =>
      item.provider === uiConfig.context_provider && item.model === uiConfig.context_model && item.configured);
    state.modelConfig = uiConfig;
    state.ready = !!$("#databaseSelect").value && !!configuredSql && configuredContext;
    $("#sendButton").disabled = !state.ready || !!voiceInput?.busy;
    if (!$("#databaseSelect").value) showError(`Database ${uiConfig.default_db_id} is not configured. Check TEXT2SQL_DEFAULT_DB_ID and TEXT2SQL_DATABASE_ROOT in .env.`);
    else if (!configuredSql) showError("The SQL model configured in .env is unavailable. Check TEXT2SQL_SQL_PROVIDER and TEXT2SQL_SQL_MODEL.");
    else if (!configuredContext) showError("The context model configured in .env is unavailable. Check TEXT2SQL_CONTEXT_PROVIDER and TEXT2SQL_CONTEXT_MODEL.");
  } catch (error) {
    setHealth(false);
    showError(error.message);
  }
}

function setHealth(online) {
  const pill = $("#healthStatus");
  pill.classList.toggle("online", online === true);
  pill.classList.toggle("offline", online === false);
  pill.lastChild.textContent = online === false ? " Offline" : " Online";
}

let schemaRequest = 0;
let browserTables = [];
let browserRelationships = [];

async function loadDatabaseSchema() {
  const request = ++schemaRequest;
  const dbId = $("#databaseSelect").value;
  browserTables = [];
  browserRelationships = [];
  if (!dbId) { renderInspectorSchema(); return; }
  try {
    const schema = await api(`/api/databases/${encodeURIComponent(dbId)}/schema`);
    if (request !== schemaRequest) return;
    browserTables = schema.tables || [];
    browserRelationships = schema.relationships || [];
    renderInspectorSchema();
  } catch (error) {
    if (request !== schemaRequest) return;
    renderInspectorSchema();
    console.warn(`Could not load database schema: ${error.message}`);
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

function formatSqlForDisplay(sql) {
  if (!sql || sql === "No SQL was accepted.") return sql;

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

function schemaIdentifiers() {
  const tables = new Set();
  const columns = new Set();
  browserTables.forEach((table) => {
    tables.add(table.name.toLowerCase());
    (table.columns || []).forEach((column) => columns.add(column.name.toLowerCase()));
  });
  return { tables, columns };
}

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
  const ident = schemaIdentifiers();
  const tokens = sql.match(/--[^\n]*|\/\*.*?\*\/|'(?:''|[^'])*'|"(?:""|[^"])*"|`(?:``|[^`])*`|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][A-Za-z0-9_$]*\b|[^A-Za-z0-9_]+/g) || [sql];
  tokens.forEach((token, index) => {
    let className = "";
    const upper = token.toUpperCase();
    const lower = token.toLowerCase();
    if (token.startsWith("--") || token.startsWith("/*")) className = "sql-comment";
    else if (token.startsWith("'") || token.startsWith('"') || token.startsWith("`")) className = "sql-string";
    else if (/^\d/.test(token)) className = "sql-number";
    else if (sqlKeywords.has(upper)) className = "sql-keyword";
    else if (sqlFunctions.has(upper)) className = "sql-function";
    if (!className && /[A-Za-z_]/.test(token) && (ident.tables.has(lower) || ident.columns.has(lower))) {
      const span = document.createElement("span");
      span.className = "sql-ident";
      span.textContent = token;
      if (ident.tables.has(lower)) span.dataset.table = token;
      else span.dataset.column = token;
      element.append(span);
      return;
    }
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

function shortLabel(text, max = 46) {
  return String(text).length > max ? `${String(text).slice(0, max - 3)}…` : String(text);
}

function appendUser(message) {
  $("#emptyState").hidden = true;
  const intro = $("#threadIntro");
  if (intro) {
    intro.hidden = false;
    if (!state.activeRunId) {
      const title = $("#threadIntroTitle");
      if (title) title.textContent = shortLabel(message, 52);
    }
  }
  const crumb = $("#threadBreadcrumb");
  if (crumb) crumb.textContent = shortLabel(message);
  state.currentQuestion = message;
  const article = document.createElement("article");
  article.className = "message user-message";
  const speaker = document.createElement("div");
  speaker.className = "speaker";
  const avatar = document.createElement("span");
  avatar.className = "avatar";
  avatar.innerHTML = icon("user", 13);
  const name = document.createElement("span");
  name.textContent = "You";
  const time = document.createElement("time");
  time.textContent = "just now";
  speaker.append(avatar, name, time);
  const question = document.createElement("p");
  question.className = "question-text";
  question.textContent = message;
  speaker.append(question);
  article.append(speaker);
  $("#messages").append(article);
}

const pipelineStages = [
  ["conversation", "Understanding"],
  ["retrieval", "Retrieving"],
  ["context_selection", "Context"],
  ["grounding", "Grounding"],
  ["generation", "Generating"],
  ["validation", "Validating"],
  ["execution", "Executing"],
  ["result_review", "Review"],
  ["recovery", "Recovery"],
  ["presentation", "Explaining"],
];

const progressLabels = {
  queued: "Waiting to start…",
  presentation: "Preparing a plain-language answer",
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

function appendProgress(contextMode) {
  $("#emptyState").hidden = true;
  const fragment = $("#progressTemplate").content.cloneNode(true);
  const article = fragment.querySelector("article");
  const tracker = article.querySelector(".pipeline-tracker");
  tracker.replaceChildren();
  pipelineStages.forEach(([stage, label]) => {
    if (stage === "context_selection" && contextMode === "retrieval") return;
    const step = document.createElement("span");
    step.dataset.stage = stage;
    step.textContent = label;
    tracker.append(step);
  });
  article.querySelector(".progress-detail").textContent = "Using the configured query pipeline";
  hydrateIcons(article);
  $("#messages").append(article);
  return article;
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function waitForJob(jobId, progress) {
  while (true) {
    const job = await api(`/api/chat/jobs/${jobId}`);
    const seconds = ((job.elapsed_ms || 0) / 1000).toFixed(1);
    updatePipeline(progress, job.stage, job.status, job.events || [], job.response);
    const label = progressLabels[job.stage] || "Processing your request";
    progress.querySelector(".progress-stage").textContent = label;
    progress.querySelector(".progress-detail").textContent = `${seconds}s elapsed · ${Number(seconds) > 20 ? "Still working. You can cancel without changing your previous result." : "You can cancel at any time."}`;
    if (job.understanding) {
      const understanding = $("#understandingContent");
      understanding.replaceChildren();
      const title = document.createElement("strong");
      title.textContent = `Interpreting as ${job.understanding.operation.replaceAll("_", " ").toLowerCase()}`;
      const detail = document.createElement("p");
      detail.textContent = job.understanding.resolved_instruction;
      understanding.append(title, detail);
    }
    const log = progress.querySelector(".progress-history ul");
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

function updatePipeline(progress, currentStage, jobStatus, events = [], response = null) {
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
    const recoverySkipped = jobStatus === "completed"
      && ["recovery", "result_review"].includes(step.dataset.stage)
      && !seen.has(step.dataset.stage);
    step.classList.toggle("skipped", recoverySkipped);
    if (recoverySkipped) step.title = "Not called during this request";
  });
}

async function send(message, feedbackCategory = null) {
  if (state.busy || voiceInput?.busy || !message.trim()) return;
  if (!state.ready) {
    showError("A configured database and SQL model are required before asking a question.");
    return;
  }
  state.busy = true;
  voiceInput?.setQueryBusy(true);
  $("#sendButton").disabled = true;
  ["#databaseSelect", "#resetButton"].forEach(id => { $(id).disabled = true; });
  appendUser(message.trim());
  const contextMode = state.modelConfig?.context_mode || "retrieval";
  const started = performance.now();
  const progress = appendProgress(contextMode);
  state.cancelled = false;
  state.pendingVizRequest = vizRequested(message);
  $("#cancelButton").hidden = false;
  try {
    if (state.restorePending && state.activeRunId) await continueRun(state.activeRunId, false);
    const created = await api("/api/chat/jobs", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        db_id: $("#databaseSelect").value,
        message: message.trim(),
        // An ordinary follow-up uses the accepted in-memory result. A parent
        // run is sent only after an explicit historical Continue action.
        parent_run_id: state.branchFromRunId,
        execute: true,
        // Keep the bounded full result available in server RAM for a later "Plot it".
        max_rows: 500,
        feedback_category: feedbackCategory,
      }),
    });
    state.jobId = created.job_id;
    if (state.cancelled) await api(`/api/chat/jobs/${created.job_id}`, { method: "DELETE" });
    const body = await waitForJob(created.job_id, progress);
    progress.remove();
    state.activeRunId = body.history_saved ? body.run_id : null;
    state.branchFromRunId = null;
    await appendAssistant(body, (performance.now() - started) / 1000,
      {question: message.trim()});
    void refreshHistory();
    return body;
  } catch (error) {
    progress.remove();
    showError(error.message);
    return null;
  } finally {
    state.busy = false;
    voiceInput?.setQueryBusy(false);
    state.jobId = null;
    $("#sendButton").disabled = !state.ready;
    ["#databaseSelect", "#resetButton"].forEach(id => { $(id).disabled = false; });
    $("#cancelButton").hidden = true;
    $("#messageInput").focus();
  }
}

function responseStatus(body, attempts) {
  const accepted = body.generation?.accepted === true;
  const explanatory = EXPLANATION_OPERATIONS_UI.has(body.operation || "");
  const chartOnly = body.operation === "VISUALIZE" || Boolean(body.plot_status);
  const modelUnavailable = body.generation && !accepted && body.generation.termination_reason === "model_error";
  const status = chartOnly ? (body.presentation?.chart_spec ? "CHART FROM STORED DATA" : "CHART UNAVAILABLE")
    : explanatory ? "NO NEW QUERY"
    : accepted ? (body.generation.execution_status === "ACCEPTED" ? "EXECUTED" : body.generation.execution_status || "SAFE SQL")
    : modelUnavailable ? "MODEL UNAVAILABLE" : "FAILED";
  return { accepted, explanatory, modelUnavailable, status };
}

function renderCallout(root, kind, iconName, text) {
  if (!text) return;
  const callout = root.querySelector(".callout");
  callout.hidden = false;
  callout.classList.remove("error", "warn", "info");
  callout.classList.add(kind);
  callout.querySelector(".callout-icon").innerHTML = icon(iconName);
  callout.querySelector(".callout-text").textContent = text;
}

function sameDisplayText(left, right) {
  const normalize = value => String(value || "").trim().replace(/\s+/g, " ").replace(/[.!?]+$/, "").toLowerCase();
  return Boolean(normalize(left)) && normalize(left) === normalize(right);
}

function updateStatusPill(fragment, body, accepted, status) {
  const pill = fragment.querySelector(".status-pill");
  if (!pill) return;
  if (accepted && status !== "CHART UNAVAILABLE") { pill.hidden = true; return; }
  const label = pill.querySelector(".status-pill-label");
  const slot = pill.querySelector("[data-icon]");
  let iconName = "check";
  let text = accepted ? "Ready" : "Query issue";
  let kind = "ok";
  if (status === "MODEL UNAVAILABLE") { kind = "error"; iconName = "alert"; text = "Model unavailable"; }
  else if (status === "FAILED") { kind = "error"; iconName = "alert"; text = "Failed"; }
  else if (status === "NO NEW QUERY") { kind = "warn"; iconName = "info"; text = "Answered without a new query"; }
  else if (status === "CHART UNAVAILABLE") { kind = "warn"; iconName = "alert"; text = "Chart unavailable"; }
  else if (status === "CHART FROM STORED DATA") { text = "Chart ready"; }
  else if (accepted && String(status).startsWith("EXEC")) { text = "Executed"; }
  pill.classList.remove("warn", "error");
  if (kind === "warn") pill.classList.add("warn");
  if (kind === "error") pill.classList.add("error");
  if (label) label.textContent = text;
  if (slot) slot.innerHTML = icon(iconName);
}

/* ---------- Intent-adaptive response modes (Queryflow v3) ---------- */

const VIZ_REQUEST_PATTERN = /\b(?:plot(?:s|ted|ting)?|chart(?:s|ed|ing)?|graph(?:s|ed|ing)?|visuali[sz]|histogram|heatmap|treemap|box ?plot|donut|scatter plot|line chart|bar chart|area chart|pie chart|stacked (?:bar|area))\b/i;

function vizRequested(...messages) {
  return messages.filter(Boolean).some((message) => VIZ_REQUEST_PATTERN.test(String(message)));
}

function responseMode(body, opts = {}) {
  if (EXPLANATION_OPERATIONS_UI.has(body.operation || "") || body.clarification_required) return "explanation";
  const generation = body.generation || {};
  if (generation.accepted !== true) return "sql";
  const presentation = body.presentation;
  if (presentation?.chart_spec) return "chart";
  if (opts.viz || body.operation === "VISUALIZE" || body.plot_status) return "chart_unavailable";
  return "sql";
}

/* The backend's versioned PresentationState is authoritative when present:
   analytical corrections received a "chart" mode stay chart-first, and a new
   independent question comes back as "sql". Client heuristics only serve
   responses produced before this contract existed. */
function presentationMode(body, opts = {}) {
  if (body.plot_status && body.plot_status !== "CHART_READY"
      && !body.presentation?.chart_spec && body.generation?.accepted) return "chart_unavailable";
  const stateMode = body.presentation_state?.mode;
  if (stateMode === "chart") {
    return body.presentation?.chart_spec ? "chart" : "chart_unavailable";
  }
  if (stateMode === "sql") {
    const fallback = responseMode(body, opts);
    return fallback === "explanation" ? "explanation" : "sql";
  }
  return responseMode(body, opts);
}

function isChartMode(screen) {
  return screen === "chart";
}

/* Queryflow v4: contextual investigations are explicit user actions, never SQL executed
   from a table cell or a chart event. All follow-ups go through the existing API. */
async function sendFromAnswer(runId, prompt) {
  if (runId && runId !== state.activeRunId) await continueRun(runId, false);
  await send(prompt);
}
let activeInsight = null;
function dismissInsight() {
  if (activeInsight) { activeInsight._cleanup?.(); activeInsight.remove(); }
  activeInsight = null;
}
function placeInsight(anchor, panel) {
  dismissInsight();
  panel.className = "query-insight-popover";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Investigate result");
  panel.addEventListener("click", event => event.stopPropagation());
  document.body.append(panel);
  activeInsight = panel;
  const bounds = anchor.getBoundingClientRect();
  const x = Math.max(12, Math.min(bounds.left, innerWidth - panel.offsetWidth - 12));
  const y = bounds.bottom + panel.offsetHeight + 12 > innerHeight
    ? Math.max(12, bounds.top - panel.offsetHeight - 8) : bounds.bottom + 8;
  panel.style.left = x + "px";
  panel.style.top = y + "px";
  const escape = event => {
    if (event.key === "Escape") { dismissInsight(); document.removeEventListener("keydown", escape); }
  };
  document.addEventListener("keydown", escape);
  // Dispose keyboard listener when the next popover is opened.
  panel._cleanup = () => document.removeEventListener("keydown", escape);
}
function insightMenu(anchor, heading, description, items, runId = null) {
  const panel = document.createElement("div");
  const title = document.createElement("strong");
  title.className = "query-insight-title";
  title.textContent = heading;
  const caption = document.createElement("p");
  caption.textContent = description;
  panel.append(title, caption);
  items.forEach(({label, prompt}) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "query-insight-action";
    button.textContent = label;
    button.addEventListener("click", async () => {
      dismissInsight();
      if (state.busy || !state.ready) return;
      // Deliberate click on an action, routed through existing interpretation and SQL safety.
      try { await sendFromAnswer(runId, prompt); }
      catch (error) { showError(error.message); }
    });
    panel.append(button);
  });
  placeInsight(anchor, panel);
}
function editSemanticChip(anchor, dimension, current, runId, body, filterItem = null) {
  const panel = document.createElement("form");
  const title = document.createElement("label");
  title.className = "query-insight-title";
  title.textContent = "Change " + dimension.toLowerCase();
  const contract = body.state?.semantic_contract || body.generation?.semantic_contract || {};
  const verified = body.generation?.model_context?.tables || {};
  let suggestions = [];
  if (dimension === "Metric") suggestions = contract.measures || [];
  if (dimension === "Grain") suggestions = contract.grain || [];
  if (dimension === "Aggregation") suggestions = ["count", "sum", "average", "minimum", "maximum"];
  if (dimension === "Filter" && filterItem) {
    const [tableName, columnName] = String(filterItem.operand || filterItem.column || "").split(".");
    const profile = verified[tableName]?.columns?.[columnName] || {};
    suggestions = profile.example_values || profile.top_values || [];
  }
  const input = suggestions.length > 1 && dimension !== "Filter"
    ? document.createElement("select") : document.createElement("input");
  input.className = "query-insight-input";
  if (input.tagName === "SELECT") {
    [...new Set([current, ...suggestions])].forEach(value => {
      const option = document.createElement("option"); option.value = value;
      option.textContent = readableResultLabel(value); input.append(option);
    });
  }
  if (dimension === "Period" && /^\d{4}-\d{2}$/.test(current)) input.type = "month";
  else if (dimension === "Period" && /^\d{4}-\d{2}-\d{2}$/.test(current)) input.type = "date";
  else if (dimension === "Period" && /^\d{4}$/.test(current)) {
    input.type = "number"; input.min = "1900"; input.max = "2200";
  }
  input.value = current;
  input.maxLength = 220;
  input.required = true;
  input.setAttribute("aria-label", "New " + dimension.toLowerCase());
  const note = document.createElement("p");
  note.textContent = "Applies a conversational correction through validated SQL.";
  const apply = document.createElement("button");
  apply.className = "query-insight-primary";
  apply.type = "submit";
  apply.textContent = "Apply change";
  panel.append(title, input, note, apply);
  if (dimension === "Filter" && filterItem) {
    const operator = document.createElement("select");
    operator.className = "query-insight-input";
    for (const value of ["=", "!=", ">", ">=", "<", "<=", "IN"]) {
      const option = document.createElement("option"); option.value = value; option.textContent = value;
      operator.append(option);
    }
    operator.value = filterItem.operator || "=";
    panel.insertBefore(operator, input);
    panel._operator = operator;
    if (suggestions.length) {
      const list = document.createElement("datalist");
      list.id = `verified-values-${Math.random().toString(36).slice(2)}`;
      suggestions.slice(0, 20).forEach(value => {
        const option = document.createElement("option"); option.value = String(value); list.append(option);
      });
      input.setAttribute("list", list.id); panel.append(list);
    }
    input.value = Array.isArray(filterItem.value) ? filterItem.value.join(", ") : String(filterItem.value ?? "");
  }
  panel.addEventListener("submit", async event => {
    event.preventDefault();
    if (state.busy || !state.ready || !input.value.trim()) return;
    const revised = input.value.trim();
    dismissInsight();
    const update = filterItem && dimension === "Filter"
      ? `${filterItem.operand || filterItem.column} ${panel._operator.value} ${JSON.stringify(revised)}`
      : filterItem && dimension === "Period"
        ? `${filterItem.operand || filterItem.column} ${filterItem.operator || "="} ${JSON.stringify(revised)}`
        : JSON.stringify(revised);
    const prompt = "Change the " + dimension.toLowerCase() + " in the previous query from " +
      JSON.stringify(current) + " to " + update +
      ". Preserve all other confirmed metrics, filters, and grouping requirements.";
    try { await sendFromAnswer(runId, prompt); }
    catch (error) { showError(error.message); }
  });
  placeInsight(anchor, panel);
  input.focus();
  input.select?.();
}
function semanticFilterText(item, includeColumn = true) {
  const column = String(item.operand || item.column || "").split(".").at(-1);
  const value = Array.isArray(item.value) ? item.value.join(", ") : String(item.value ?? "");
  const operator = item.operator || "=";
  if (!includeColumn && operator === "=") return value;
  const label = includeColumn ? readableResultLabel(column) : "";
  return operator === "=" && includeColumn ? `${label}: ${value}`
    : [label, operator, value].filter(Boolean).join(" ");
}

function renderSemanticChips(fragment, body) {
  const strip = fragment.querySelector(".query-semantic-strip");
  if (!strip) return;
  const contract = body.state?.semantic_contract;
  if (!contract || body.clarification_required || body.generation?.accepted === false) return;
  const items = [];
  if (contract.measures?.length) items.push([
    "Metric", contract.measures.map(readableResultLabel).join(", "), contract.measures.join(", "),
  ]);
  const temporal = [], otherFilters = [];
  if (contract.filters?.length) {
    for (const item of contract.filters) {
      const operand = String(item.operand || item.column || "");
      if (/(?:^|[._])(?:date|year|month|period|timestamp|time)$/i.test(operand)) temporal.push(item);
      else otherFilters.push(item);
    }
  }
  if (temporal.length) items.push([
    "Period", temporal.map(item => semanticFilterText(item, false)).join("; "),
    temporal.length === 1 ? String(temporal[0].value ?? "")
      : temporal.map(item => semanticFilterText(item)).join("; "),
    temporal.length === 1 ? temporal[0] : null,
  ]);
  if (contract.grain?.length) items.push([
    "Grain", contract.grain.map(item => readableResultLabel(String(item).split(".").at(-1).replace(/(?:_id|ID)$/, ""))).join(", "),
    contract.grain.join(", "),
  ]);
  if (contract.aggregation) items.push(["Aggregation", readableResultLabel(contract.aggregation), contract.aggregation]);
  for (const item of otherFilters) items.push(["Filter", semanticFilterText(item),
    [item.operand || item.column, item.operator || "=", JSON.stringify(item.value)].filter(Boolean).join(" "), item]);
  if (!items.length) return;
  strip.hidden = false;
  const eyebrow = document.createElement("span");
  eyebrow.className = "query-semantic-eyebrow";
  eyebrow.textContent = "UNDERSTOOD AS";
  strip.append(eyebrow);
  const appendChip = ([kind, display, raw, filterItem]) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "query-semantic-chip";
    chip.title = `Change ${kind.toLowerCase()}: ${display}`;
    chip.setAttribute("aria-label", chip.title);
    const k = document.createElement("small");
    k.textContent = kind;
    const v = document.createElement("span");
    v.className = "chip-value";
    v.textContent = display;
    const arrow = document.createElement("span");
    arrow.className = "chip-chevron";
    arrow.setAttribute("data-icon", "chevron-right");
    chip.append(k, v, arrow);
    chip.addEventListener("click", () => editSemanticChip(chip, kind, raw, body.run_id, body, filterItem));
    strip.append(chip);
  };
  items.slice(0, 6).forEach(appendChip);
  if (items.length > 6) {
    const more = document.createElement("button");
    more.type = "button"; more.className = "query-semantic-chip query-semantic-more";
    more.textContent = `+${items.length - 6} filters`;
    more.setAttribute("aria-expanded", "false");
    more.addEventListener("click", () => {
      items.slice(6).forEach(appendChip); more.remove();
    });
    strip.append(more);
  }
}
function attachResultInvestigations(table, body, displayedRows = body.generation?.rows || []) {
  const cols = body.generation?.columns || [];
  const actual = table?.tBodies?.[0];
  if (!actual || !cols.length) return;
  Array.from(actual.rows).forEach((tr, r) => {
    if (tr.dataset.spacer) return;
    Array.from(tr.cells).forEach((cell, c) => {
      const value = displayedRows[Number(tr.dataset.rowIndex ?? r)]?.[c], column = cols[c];
      if (value == null || !column) return;
      cell.classList.add("investigable-cell");
      cell.tabIndex = 0;
      cell.title = "Select to investigate this value";
      const action = () => {
        const key = JSON.stringify(String(column)), val = JSON.stringify(value);
        insightMenu(cell, String(column) + ": " + String(value), "Use this observed value as the starting point for a follow-up.", [
          {label:"Investigate this value",prompt:"Investigate the observed value " + val +
            " in result column " + key + " from the previous query. Keep the existing analytical context and explain what can be verified."},
          {label:"Filter to this value",prompt:"Refine the previous query to focus on " + key + " = " + val +
            ". Verify that this result column maps to an appropriate source dimension; ask for clarification if it is an aggregate or ambiguous alias."},
          {label:"Compare with other values",prompt:"Compare " + key + " = " + val +
            " to other values of the same dimension, preserving the previous metric definition and relevant filters."}
        ], body.run_id);
      };
      cell.addEventListener("click", action);
      cell.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); action(); }
      });
    });
  });
}
function attachChartInvestigations(article, presentation, runId) {
  const spec = presentation?.chart_spec;
  if (!article._echart?.onSelect || !spec?.data?.length) return;
  article._echart.onSelect(point => {
    const observedX = point.rawX ?? (point.name !== "" && point.name != null
      ? point.name : (Array.isArray(point.value) ? point.value[0] : undefined));
    const matches = spec.data.filter(row => String(row[spec.x]) === String(observedX)
      && (!spec.series || !point.seriesName || String(row[spec.series]) === String(point.seriesName)));
    if (!matches.length) return;
    const value = matches[0][spec.x];
    const anchor = article.querySelector(".chart-slot");
    const condition = Array.isArray(spec.x_components) && spec.x_components.length === 2
      ? spec.x_components.map(name => `${JSON.stringify(name)} = ${JSON.stringify(matches[0][name])}`).join(" and ")
      : `${JSON.stringify(spec.x)} = ${JSON.stringify(value)}`;
    const y = JSON.stringify(spec.y);
    insightMenu(anchor, String(spec.x) + ": " + String(value), "Explore this chart category using the original query.", [
      {label:"Break down this point", prompt:"For the previous chart, break down the " + y +
        " measure where " + condition +
        " by a meaningful verified dimension. Preserve existing filters."},
      {label:"Compare with other periods or groups", prompt:"Compare the previous chart result at " +
        condition + " with relevant neighboring periods or groups, maintaining its metric definition."},
      {label:"Show contributing records", prompt:"Show the underlying records contributing to the previous chart at " +
        condition + ". Verify the source grain and joins; do not infer individual records from aggregates."}
    ], runId);
  });
}

function renderChartInto(slot, presentation, generation) {
  slot._echart?.dispose?.();
  slot.replaceChildren();
  // Plotly measures the element during newPlot, so size it before rendering.
  slot.classList.add("chart-slot-plotly");
  let plotly = null;
  let active = null;
  let disposed = false;
  const selectionCallbacks = [];
  const zoomCallbacks = [];
  const handle = {
    get renderer() { return active?.renderer; },
    onSelect(callback) { selectionCallbacks.push(callback); active?.onSelect?.(callback); },
    onZoom(callback) { zoomCallbacks.push(callback); active?.onZoom?.(callback); },
    highlightSeries(name) { active?.highlightSeries?.(name); },
    highlightPoint(point) { active?.highlightPoint?.(point); },
    resetZoom() { active?.resetZoom?.(); },
    resize() { active?.resize?.(); },
    getDataURL(options) { return active?.getDataURL?.(options); },
    dispose() { disposed = true; active?.dispose?.(); },
  };
  const useFallback = () => {
    if (disposed || active !== plotly) return;
    active?.dispose?.();
    slot.replaceChildren();
    slot.classList.remove("chart-slot-plotly");
    slot.classList.add("chart-slot-ec");
    active = window.QueryflowECharts?.render(slot, presentation, generation) || null;
    if (active) {
      selectionCallbacks.forEach(callback => active.onSelect?.(callback));
      zoomCallbacks.forEach(callback => active.onZoom?.(callback));
    } else {
      const note = document.createElement("p");
      note.className = "chart-note";
      note.textContent = "This chart could not be displayed. Open Data to inspect the results, or choose another chart.";
      note.setAttribute?.("role", "status");
      slot.append(note);
    }
  };
  try {
    plotly = window.QueryflowPlotly?.render(slot, presentation, generation, useFallback) || null;
  }
  catch (error) { console.warn(`Plotly rendering failed: ${error.message}`); }
  const chart = plotly || window.QueryflowECharts?.render(slot, presentation, generation) || null;
  active = chart;
  slot.classList.toggle("chart-slot-plotly", Boolean(plotly));
  slot.classList.toggle("chart-slot-ec", Boolean(chart && !plotly));
  slot._echart = chart ? handle : null;
  if (chart) {
    return { echarts: handle };
  }
  const note = document.createElement("p");
  note.className = "chart-note";
  note.textContent = "This chart could not be displayed. Your results and SQL are still available in their tabs.";
  note.setAttribute?.("role", "status");
  slot.append(note);
  return { echarts: null };
}

/* Response views are local to one message. They never call the query API. */
let responseTabSequence = 0;
function setResponseTab(article, view) {
  const tabs = article.querySelector(".response-tabs");
  const answerPane = article.querySelector(".answer-tab-panel");
  const split = article.querySelector(".sql-result-split");
  const sqlPane = article.querySelector(".sql-panel-pane");
  const dataPane = article.querySelector(".result-panel-pane");
  const mapPane = article.querySelector(".map-panel-pane");
  if (!tabs || tabs.hidden || !answerPane || !split || !sqlPane || !dataPane) return;
  const button = tabs.querySelector(`[data-response-tab="${view}"]`);
  if (!button || button.disabled) return;
  const oldView = article.dataset.activeResponseTab;
  article._tabScroll ||= {};
  if (oldView) article._tabScroll[oldView] = article.querySelector(oldView === "sql"
    ? ".sql-editor" : oldView === "data" ? ".table-wrap" : ".answer-tab-panel")?.scrollTop || 0;
  article.dataset.activeResponseTab = view;
  answerPane.hidden = view !== "answer";
  split.hidden = view === "answer";
  sqlPane.hidden = view !== "sql";
  dataPane.hidden = view !== "data";
  if (mapPane) mapPane.hidden = view !== "map";
  const versions = article.querySelector(".versions-panel");
  if (versions) versions.hidden = view !== "sql" || versions.dataset.available !== "true";
  tabs.querySelectorAll("[data-response-tab]").forEach((tab) => {
    const active = tab === button;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  if (view === "map") article._renderMap?.();
  const scrollPane = article.querySelector(view === "sql" ? ".sql-editor"
    : view === "data" ? ".table-wrap" : ".answer-tab-panel");
  if (scrollPane) scrollPane.scrollTop = article._tabScroll[view] || 0;
  window.requestAnimationFrame?.(() => article._echart?.resize?.());
}

function wireResponseTabs(article, available = {answer: true, sql: true, data: true, map: false}) {
  const tabs = article.querySelector(".response-tabs");
  if (!tabs) return;
  tabs.hidden = false;
  const group = `response-tabs-${++responseTabSequence}`;
  tabs.querySelectorAll("[data-response-tab]").forEach((tab) => {
    const view = tab.dataset.responseTab;
    tab.hidden = !available[view];
    tab.id = `${group}-${view}`;
    tab.disabled = !available[view];
    const panel = article.querySelector(view === "sql" ? ".sql-panel-pane"
      : view === "map" ? ".map-panel-pane" : view === "answer"
      ? ".answer-tab-panel" : ".result-panel-pane");
    if (panel) {
      panel.id = `${group}-${view}-panel`;
      panel.setAttribute("aria-labelledby", tab.id);
      tab.setAttribute("aria-controls", panel.id);
    }
    tab.addEventListener("click", () => setResponseTab(article, view));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const enabled = [...tabs.querySelectorAll("[data-response-tab]:not(:disabled)")]
        .filter(item => !item.hidden);
      const next = enabled[(enabled.indexOf(tab) + (event.key === "ArrowRight" ? 1 : enabled.length - 1)) % enabled.length];
      setResponseTab(article, next.dataset.responseTab);
      next.focus();
    });
  });
  setResponseTab(article, available.answer ? "answer" : available.data ? "data" : "sql");
}

/* The Data tab always shows exact executed rows; chart display filters never
   rewrite this table or the accepted SQL. */
function wireDataScope(article, generation, response) {
  const pane = article.querySelector(".result-panel-pane");
  if (!pane) return;
  const wrap = pane.querySelector(".table-wrap");
  const sourceRows = generation.rows || [];
  article._visibleChartSpec = null;
  article._refreshDataScope = () => {};
  article._setClientFilter = filter => article._setChartViewRowFilters?.([filter]);
  article._setClientFilters = filters => article._setChartViewRowFilters?.(filters);
  article._clearClientFilter = () => article._setChartViewRowFilters?.([]);
  article._hasClientFilter = () => false;
  renderTable(wrap, generation.columns || [], sourceRows, {
    units: response.presentation?.chart_spec?.measure_units || {},
    temporal: response.presentation?.chart_spec?.time_format
      && response.presentation.chart_spec.time_format !== "raw"
      && response.presentation.chart_spec.x
      ? {
          column: response.presentation.chart_spec.x,
          time_format: response.presentation.chart_spec.time_format,
        }
      : null,
    truncated: Boolean(generation.truncated || generation.row_count > sourceRows.length),
    onPageRendered: (table, pageRows) => attachResultInvestigations(
      table, {generation, run_id: article.dataset.runId}, pageRows),
    onPageError: error => showError(error.message),
    loadPage: generation.truncated && response.result_id ? async (offset, pageSize) => {
      const page = await api("/api/results/page", {
        method: "POST",
        body: JSON.stringify({
          session_id: response.session_id, db_id: generation.db_id,
          result_id: response.result_id, offset, page_size: pageSize,
        }),
      });
      if (page.result_id !== response.result_id ||
          JSON.stringify(page.columns) !== JSON.stringify(generation.columns)) {
        throw new Error("The result changed. Run the question again.");
      }
      return {rows: page.rows, has_more: page.has_more,
        limit_reached: page.limit_reached};
    } : null,
  });
  const expand = pane.querySelector(".table-expand");
  if (expand) {
    expand.addEventListener("click", () => {
      pane.classList.toggle("table-fullscreen");
      expand.setAttribute("aria-label", pane.classList.contains("table-fullscreen")
        ? "Close expanded table" : "Expand table");
    });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && pane.classList.contains("table-fullscreen")) {
        pane.classList.remove("table-fullscreen");
        expand.setAttribute("aria-label", "Expand table");
      }
    });
  }
}

function mapCoordinates(columns, rows, requested, truncated) {
  if (!requested || truncated) return null;
  const latitude = columns.findIndex(name => /^(?:lat|latitude)$/i.test(name));
  const longitude = columns.findIndex(name => /^(?:lon|lng|long|longitude)$/i.test(name));
  if (latitude < 0 || longitude < 0) return null;
  const points = rows.filter(row => typeof row[latitude] === "number"
    && typeof row[longitude] === "number"
    && Number.isFinite(row[latitude]) && Number.isFinite(row[longitude])
    && Math.abs(row[latitude]) <= 90 && Math.abs(row[longitude]) <= 180);
  return points.length === rows.length && points.length ? {
    lat: points.map(row => row[latitude]), lon: points.map(row => row[longitude]),
  } : null;
}

function wireChartSection(article, presentation, generation, runId, recommendations = null) {
  const section = article.querySelector(".chart-section");
  const slot = article.querySelector(".chart-slot");
  if (!section || !slot) return;
  const kinds = section.querySelector(".chart-kinds");
  section.querySelector(".chart-tools")?.prepend(kinds);
  const more = section.querySelector(".chart-more");
  const menu = section.querySelector(".chart-menu");
  const explorer = window.QueryflowChartExplorer;
  const original = presentation.chart_spec;
  const stableTypeOrder = [...new Set([
    recommendations?.primary_chart?.chart_type,
    ...(recommendations?.alternative_charts || []).map(item=>item.chart_type),
    ...(explorer?.choices(original) || []),
  ].filter(Boolean))];
  const settingKey = runId ? `queryflow-chart-${runId}` : null;
  let stored = null;
  try { stored = settingKey ? JSON.parse(sessionStorage.getItem(settingKey) || "null") : null; }
  catch { stored = null; }
  // Do not revive filters from older sessions after removing their controls.
  const settings = stored && typeof stored === "object"
    ? {chartType: stored.chartType, zoom: stored.zoom} : {};
  const chartPres = { ...presentation, chart_spec: original };
  function saveSettings() {
    if (settingKey) {
      try { sessionStorage.setItem(settingKey, JSON.stringify(settings)); } catch { /* private browsing */ }
    }
  }
  function renderCurrent() {
    const next = explorer?.apply(original, settings);
    if (!next) {
      article._echart?.dispose?.();
      article._echart = null;
      slot.replaceChildren();
      const message = document.createElement("p");
      message.className = "chart-note";
      message.textContent = "This chart change would combine separate result rows. Ask a follow-up question to calculate it at the new grouping level.";
      slot.append(message);
      article._visibleChartSpec = null;
      article._refreshDataScope?.();
      return;
    }
    if (!next.data.length) {
      article._echart?.dispose?.();
      article._echart = null;
      slot.replaceChildren();
      const message = document.createElement("p");
      message.className = "chart-note";
      message.textContent = "No rows in the executed result match these display filters. The source query is unchanged.";
      slot.append(message);
      article._visibleChartSpec = next;
      article._refreshDataScope?.();
      return;
    }
    if (settings.zoom && new Set(next.data.map(row => String(row[next.x]))).size > 2) {
      next.zoom_start = settings.zoom.start;
      next.zoom_end = settings.zoom.end;
    }
    chartPres.chart_spec = next;
    article._visibleChartSpec = next;
    article._refreshDataScope?.();
    const subtitle = section.querySelector(".chart-sub");
    subtitle.textContent = (next.warnings || []).join(" · ");
    subtitle.hidden = !subtitle.textContent;
    article._echart = renderChartInto(slot, chartPres, generation).echarts;
    article._echart?.onZoom?.(range => {
      settings.zoom = range;
      saveSettings();
    });
    attachChartInvestigations(article, chartPres, runId);
    const zoomActions = section.querySelector(".chart-zoom-actions");
    zoomActions.hidden = true;
    const zoomGuide = section.querySelector(".chart-zoom-guide");
    zoomGuide.hidden = true;
    kinds.replaceChildren();
    if (explorer) {
      const recommended = [recommendations?.primary_chart, ...(recommendations?.alternative_charts || [])]
        .filter(item => item && explorer.choices(next).includes(item.chart_type));
      const types = explorer.orderedChoices(next,stableTypeOrder,original.chart_type);
      const picker = document.createElement("details");
      picker.className = "chart-choice";
      picker.addEventListener("keydown", event => {
        if (event.key === "Escape") { picker.open = false; summary.focus(); }
      });
      const summary = document.createElement("summary");
      const chartSymbol = type => ({heatmap:"chart-heatmap",treemap:"chart-treemap",
        scatter:"chart-scatter",bubble:"chart-bubble",area:"chart-area",stacked_area:"chart-area",
        step_line:"chart-step",box:"chart-box",donut:"chart-donut",pie:"chart-pie",
        horizontal_bar:"chart-horizontal",stacked_bar:"chart-stacked",percent_stacked_bar:"chart-stacked",
        calendar_heatmap:"chart-calendar",bar:"chart-bar",grouped_bar:"chart-bar",histogram:"chart-bar"})[type] || "chart-line";
      summary.innerHTML = icon(chartSymbol(next.chart_type), 19);
      const selectedLabel = document.createElement("span");
      selectedLabel.textContent = readableResultLabel(next.chart_type);
      summary.append(selectedLabel);
      summary.insertAdjacentHTML("beforeend", icon("chevron-down", 12));
      summary.setAttribute("aria-label", "Choose chart type");
      const menu = document.createElement("div"); menu.className = "chart-choice-menu";
      picker.append(summary, menu); kinds.append(picker);
      types.forEach(type => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "chart-kind";
        button.innerHTML = icon(chartSymbol(type), 19);
        const label = document.createElement("span"); label.textContent = readableResultLabel(type);
        button.append(label);
        button.title = `Show ${button.textContent} using these returned rows`;
        button.classList.toggle("active", type === next.chart_type);
        button.setAttribute("aria-pressed", String(type === next.chart_type));
        button.addEventListener("click", () => {
          const proposal = recommended.find(item => item.chart_type === type)
            || {...next, chart_type: type};
          // Every displayed choice is a local view of the validated result.
          // A failed local switch must not silently call the model or SQL.
          if (article._chartViewState?.tryChart(proposal)) return;
          showError("This chart cannot display the current result without changing its data.");
        });
        menu.append(button);
      });
    }
  }
  article._echart = null;
  article._chartType = presentation.chart_type;
  kinds.hidden = !explorer;
  /* Chart controls change display only. Analytical changes still go through
     the conversational orchestrator when requested in the chat. */
  article._chartViewState = {
    sourceRunId: runId,
    settings,
    accepted: original,
    tryChart(chart) {
      if (!chart || !chart.x || !chart.y) return false;
      const specColumns = new Set([
        ...Object.keys(original.data?.[0] || {}),
        ...(original.source_columns || []),
      ]);
      if (!specColumns.has(chart.x)) return false;
      const measures = [...new Set([...(chart.y_measures || []), chart.y].filter(Boolean))];
      if (!measures.every(name => specColumns.has(name))) return false;
      const merged = {
        ...settings,
        x: chart.x,
        metrics: measures,
        series: chart.series || null,
        chartType: chart.chart_type,
        rowFilters: {},
        categories: [],
        topN: null,
      };
      const applied = explorer?.apply(original, merged);
      if (!applied || !applied.data.length) return false;
      if (applied.chart_type !== chart.chart_type) return false;
      Object.keys(settings).forEach(key => delete settings[key]);
      Object.assign(settings, merged);
      saveSettings();
      renderCurrent();
      return true;
    },
    setRowFilters(rows) {
      settings.rowFilters = (Array.isArray(rows) ? rows : [rows]).reduce((map, filter) => {
        if (!filter?.column) return map;
        const values = filter.values && filter.values.length
          ? filter.values.map(String) : [String(filter.value ?? "")];
        return { ...map, [filter.column]: values };
      }, {});
      saveSettings();
      renderCurrent();
    },
  };
  article._setChartViewRowFilters = filters => article._chartViewState.setRowFilters(filters);
  section.querySelector(".chart-reset-zoom").addEventListener("click", () => {
    delete settings.zoom;
    saveSettings();
    article._echart?.resetZoom?.();
  });
  renderCurrent();
  /* Theme switches recolor the shared token palette, so live charts repaint. */
  article._chartRerender = renderCurrent;
  const closeMenu = () => { menu.hidden = true; more.setAttribute("aria-expanded", "false"); };
  more.addEventListener("click", (event) => {
    event.stopPropagation();
    menu.hidden = !menu.hidden;
    more.setAttribute("aria-expanded", String(!menu.hidden));
  });
  menu.addEventListener("click", (event) => event.stopPropagation());
  document.addEventListener("click", closeMenu, { once: false });
  menu.querySelectorAll("[data-chart-action]").forEach((item) => {
    item.addEventListener("click", () => {
      closeMenu();
      const action = item.dataset.chartAction;
      if (action === "view-sql") {
        setResponseTab(article, "sql");
      } else if (action === "view-data") {
        setResponseTab(article, "data");
      } else if (action === "download") {
        exportChartPng(article);
      } else if (action === "expand") {
        openChartFullscreen(article, chartPres, generation);
      } else if (action === "inspect") {
        revealInspector("pipeline");
        const pipeline = $("#pipelineSection");
        if (pipeline) { pipeline.open = true; pipeline.scrollIntoView({ behavior: "smooth", block: "start" }); }
      }
    });
  });
}

function toast(message) {
  document.querySelector(".toasty")?.remove();
  const node = document.createElement("div");
  node.className = "toasty";
  node.textContent = message;
  document.body.append(node);
  window.setTimeout(() => node.remove(), 2600);
}

function exportedChartSource(article) {
  if (article._echart?.getDataURL) {
    return article._echart.getDataURL({ pixelRatio: 2, backgroundColor: themeBackgroundColor() });
  }
  return null;
}

function themeBackgroundColor() {
  return getComputedStyle(document.body).backgroundColor || "#ffffff";
}

async function exportChartPng(article) {
  let dataUrl;
  try { dataUrl = await exportedChartSource(article); }
  catch { dataUrl = null; }
  if (!dataUrl) { toast("Chart export is unavailable while the chart library is offline."); return; }
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = `queryflow-chart-${new Date().toISOString().slice(0, 10)}.png`;
  link.click();
}

function openChartFullscreen(article, presentation, generation) {
  closeChartFullscreen();
  const overlay = document.createElement("div");
  overlay.className = "chart-fullscreen";
  const inner = document.createElement("div");
  inner.className = "chart-fullscreen-inner";
  inner.setAttribute("role", "dialog");
  inner.setAttribute("aria-modal", "true");
  inner.setAttribute("aria-label", "Expanded chart");
  const header = document.createElement("header");
  const title = document.createElement("strong");
  title.textContent = window.QueryflowChartTheme?.title(presentation.chart_spec || presentation)
    || presentation.title || `${presentation.y} by ${presentation.x}`;
  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.className = "icon-button";
  closeButton.id = "chartFullscreenClose";
  closeButton.title = "Close expanded chart";
  closeButton.innerHTML = icon("x", 16);
  const slotHolder = document.createElement("div");
  slotHolder.className = "chart-fullscreen-slot";
  header.append(title, closeButton);
  inner.append(header, slotHolder);
  overlay.append(inner);
  overlay.addEventListener("click", (event) => { if (event.target === overlay) closeChartFullscreen(); });
  document.body.append(overlay);
  closeButton.addEventListener("click", closeChartFullscreen);
  const slot = slotHolder;
  const { echarts } = renderChartInto(slot, presentation, generation);
  overlay._echart = echarts;
  overlay._chartRerender = () => {
    overlay._echart = renderChartInto(slot, presentation, generation).echarts;
  };
}

function closeChartFullscreen() {
  document.querySelectorAll(".chart-fullscreen").forEach((overlay) => {
    overlay._echart?.dispose?.();
    overlay.remove();
  });
}

function revealInspector(tab = "understanding") {
  selectInspectorTab(tab);
  setInspectorOpen(true);
}

function mergeRunChanges(changes) {
  const merged = [];
  for (const change of changes) {
    const last = merged[merged.length - 1];
    if (last && last.kind === change.kind) last.token = `${last.token} ${change.token}`.trim();
    else merged.push({ kind: change.kind, token: change.token });
  }
  return merged;
}

function recoveryTimelineSteps(recovery, events = []) {
  if (!recovery || recovery.activated === false) return [];
  if (events.length) {
    const titles = {
      SQL_FAILED: "SQL attempt failed", SCHEMA_INSPECTED: "Inspected live schema",
      VALUES_INSPECTED: "Checked database values", DATABASE_PROBED: "Ran a bounded database check",
      TOOL_OBSERVED: "Collected evidence", CLAIM_REPORTED: "Agent interpretation",
      RECOVERY_DECISION: "Recovery decision", SQL_EXECUTED: "Selected SQL executed",
    };
    return events.map(event => ({
      icon: event.status === "FAILED" ? "alert" : event.status === "EXECUTED" ? "check" : "search",
      tone: event.status === "FAILED" ? "warn" : event.status === "EXECUTED" ? "ok" : "accent",
      title: `${titles[event.event_type] || event.event_type}${event.attempt ? ` · Attempt ${event.attempt}` : ""}`,
      detail: event.summary, evidenceId: event.evidence_ids.join(", "),
      observation: event.detail, sql: event.sql,
    }));
  }
  const steps = [];
  const mode = recovery.mode || "FAILURE";
  steps.push({
    icon: "alert", tone: "warn", title: `Initial ${mode === "FAILURE" ? "failure" : mode.toLowerCase()}`,
    detail: [recovery.failure_code, recovery.diagnosis_summary].filter(Boolean).join(" — ") || "The first candidate hit an anomaly.",
  });
  (recovery.tool_calls || []).forEach((call) => {
    const toolIcons = { inspect_schema: "table", inspect_values: "search", query_database: "database" };
    steps.push({
      icon: toolIcons[call.tool] || "search",
      tone: "accent",
      title: `Recovery tool · ${call.tool}`,
      detail: [call.purpose, call.result_summary].filter(Boolean).join(" — "),
      evidenceId: call.evidence_id || null,
    });
  });
  (recovery.claims || []).forEach((claim) => {
    steps.push({
      icon: "check", tone: "ok", title: `Claim · ${claim.claim_type.replaceAll("_", " ").toLowerCase()}`,
      detail: `${claim.statement} — evidence: ${(claim.evidence_ids || []).join(", ")}`,
    });
  });
  (recovery.evidence || []).forEach((item, index) => {
    steps.push({
      icon: "shield", tone: "info", title: "Recorded evidence",
      detail: typeof item === "string" ? item : JSON.stringify(item),
    });
  });
  if (recovery.agent_diagnosis || recovery.agent_action) {
    steps.push({
      icon: recovery.agent_action === "REPAIR" ? "refresh" : recovery.agent_action === "INFORM" ? "info" : "alert",
      tone: recovery.agent_action === "REPAIR" ? "accent" : recovery.agent_action === "INFORM" ? "ok" : "warn",
      title: `Agent decision · ${recovery.agent_action || "REVIEW"}`,
      detail: [recovery.agent_diagnosis, recovery.repair_instruction ? `Repair instruction: ${recovery.repair_instruction}` : null]
        .filter(Boolean).join(" "),
    });
  }
  return steps;
}

function readableResultLabel(column) {
  const words = String(column).replace(/_/g, " ").replace(/([a-z0-9])([A-Z])/g, "$1 $2").trim();
  return words ? words[0].toUpperCase() + words.slice(1).replace(/\bId\b/g, "ID") : "Result";
}

function formatKpiValue(value, unit) {
  if (value == null) return "—";
  if (typeof value === "number" && Number.isFinite(value)) {
    if (typeof window !== "undefined" && window.QueryflowKpis) return window.QueryflowKpis.format(value,unit);
    return new Intl.NumberFormat(undefined,{maximumFractionDigits:2,
      notation:Math.abs(value)>=1000 ? "compact" : "standard"}).format(value);
  }
  const raw = String(value);
  // Group ordinary decimal numbers without rounding or altering identifier-like strings.
  if (!/^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(raw)) return raw;
  const [integer, fraction] = raw.split(".");
  const grouped = integer.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return fraction === undefined ? grouped : `${grouped}.${fraction}`;
}

function renderAnswerPreview(fragment, body, mode) {
  const preview = fragment.querySelector(".answer-preview");
  const generation = body.generation || {};
  const columns = generation.columns || [];
  const rows = generation.rows || [];
  if (!preview || mode === "explanation" || body.clarification_required
      || !generation.accepted || generation.truncated || rows.length !== 1) return;
  const metricColumns = columns.length === 1 ? columns : columns.filter((column, index) =>
    !/(?:^|_)(?:id|key|code|year|month|date)(?:$|_)/i.test(column)
    && !/[a-z](?:ID|Id)$/.test(column)
    && (rows[0][index] == null || typeof rows[0][index] === "number"
      && Number.isFinite(rows[0][index])));
  if (!metricColumns.length || metricColumns.length > 8) return;
  const context = columns.filter(column => !metricColumns.includes(column))
    .map(column => `${readableResultLabel(column)}: ${formatKpiValue(rows[0][columns.indexOf(column)])}`)
    .join(" · ");
  preview.hidden = false;
  preview.classList.add(metricColumns.length > 1 ? "answer-kpi-grid" : "answer-kpi");
  for (const column of metricColumns) {
    const card = metricColumns.length > 1 ? document.createElement("div") : preview;
    if (card !== preview) card.className = "kpi-tile";
    const label = document.createElement("span");
    label.textContent = readableResultLabel(column);
    const value = document.createElement("strong");
    value.textContent = formatKpiValue(rows[0][columns.indexOf(column)]);
    value.title = String(rows[0][columns.indexOf(column)] ?? "SQL NULL");
    card.append(label, value);
    if (context) {
      const detail = document.createElement("small");
      detail.className = "kpi-context";
      detail.textContent = context;
      card.append(detail);
    }
    if (card !== preview) preview.append(card);
  }
}


function chartMatchesResultIdentity(body) {
  const sourceId = body.presentation_state?.source_result_id;
  return !sourceId || !body.result_id || sourceId === body.result_id;
}

function isPlotOnlyTurn(body) {
  if (body.operation !== "VISUALIZE") return false;
  // A visualization may follow new SQL in the same turn. Only hide the
  // technical views when this turn reused an already accepted result.
  return !(body.call_ledger || []).some(entry =>
    ["sql_generation", "sql_repair", "sql_execution"].includes(entry.component));
}

async function appendAssistant(body, elapsed, options = {}) {
  if (!options.restored) rememberChartSnapshot(body);
  const fragment = $("#assistantTemplate").content.cloneNode(true);
  const article = fragment.querySelector("article");
  const generation = body.generation || {};
  const plotOnly = isPlotOnlyTurn(body);
  const accepted = generation.accepted === true;
  const attempts = generation.attempts || [];
  const { explanatory, modelUnavailable, status } = responseStatus(body, attempts);
  updateStatusPill(fragment, body, accepted, status);

  const responseMessage = body.explanation || body.presentation?.summary || body.message || "";
  fragment.querySelector(".response-note").textContent = responseMessage;

  // Distinct empty / anomaly states, grounded in the real response.
  if (accepted && generation.row_count === 0) {
    const diagnosis = generation.recovery?.diagnosis_summary
      || "The query executed but returned no rows. Adjust the question or check the filters.";
    if (!sameDisplayText(responseMessage, diagnosis)
        && !sameDisplayText(body.human_review?.reason, diagnosis)) {
      renderCallout(fragment, "warn", "search", diagnosis);
    }
  } else if (!accepted && !explanatory && !modelUnavailable) {
    const failure = body.human_review?.reason || body.message || "The query could not be completed.";
    if (!body.human_review && !sameDisplayText(responseMessage, failure)) {
      renderCallout(fragment, "error", "alert", failure);
    }
  }

  // The Answer tab adapts to intent; SQL and Data use the same executed response.
  const viz = options.viz ?? state.pendingVizRequest === true;
  state.pendingVizRequest = null;
  const mode = presentationMode(body, { viz });
  const chartSection = fragment.querySelector(".chart-section");
  const presentation = body.presentation || null;
  const chartDataComplete = chartMatchesResultIdentity(body)
    && body.presentation_state?.data_complete !== false
    && !generation.truncated
    && (!(presentation?.chart_spec) || !presentation.chart_spec.source_row_count
      || presentation.chart_spec.source_row_count === generation.rows?.length);
  if (mode === "chart" && chartDataComplete) {
    chartSection.hidden = false;
    const spec = presentation.chart_spec;
    fragment.querySelector(".chart-title").textContent = window.QueryflowChartTheme?.title(spec)
      || presentation.title || `${presentation.y} by ${presentation.x}`;
    const subtitle = fragment.querySelector(".chart-sub");
    subtitle.textContent = (spec.warnings || []).join(" · ");
    subtitle.hidden = !subtitle.textContent;
  } else if (mode === "chart") {
    // A versioned "chart" state with incomplete data must never render a
    // partial chart; demand a fresh validated execution instead.
    chartSection.remove();
    renderCallout(fragment, "warn", "alert",
      "The stored chart snapshot is incomplete. Rerun the query to restore this visualization from a fresh, complete result.");
    state.lastPresentationMode = "chart_incomplete";
  } else if (mode === "chart_unavailable") {
    chartSection.remove();
    const reason = presentation?.chart_reason || body.message
      || "This result cannot support the requested chart. SQL and data are available in their tabs.";
    if (!sameDisplayText(responseMessage, reason)) renderCallout(fragment, "warn", "alert", reason);
  } else {
    chartSection.remove();
  }
  renderAnswerPreview(fragment, body, mode === "chart" && !chartDataComplete ? "sql" : mode);

  const sqlNotAvailable = explanatory || modelUnavailable || body.clarification_required;
  const sqlText = generation.sql || "";
  const displaySql = sqlNotAvailable ? "" : (generation.formatted_sql || formatSqlForDisplay(sqlText));
  fragment.querySelector(".sql-panel-pane").hidden = sqlNotAvailable || plotOnly;
  if (!sqlNotAvailable) {
    renderHighlightedSql(fragment.querySelector(".language-sql"), displaySql);
    const validity = fragment.querySelector(".sql-validity");
    if (validity) {
      const parts = [generation.dialect, generation.execution_status].filter(Boolean);
      if (body.sql_version) parts.push(`SQL v${body.sql_version}`);
      if (generation.truncated) parts.push("preview");
      validity.textContent = parts.join(" · ");
    }
  }
  fragment.querySelector(".result-panel-pane").hidden = sqlNotAvailable || explanatory || plotOnly;

  const actions = fragment.querySelector(".answer-actions");
  if (accepted && !explanatory && !sqlNotAvailable && !plotOnly) {
    const sqlActions = fragment.querySelector(".sql-actions");
    const runId = body.run_id || options.run?.run_id || null;
    sqlActions.append(
      actionButton("Check correctness", "shield", () =>
        sendFromAnswer(runId, "Check whether the previous SQL and result are correct.")
          .catch(error => showError(error.message))),
      actionButton("Optimize", "zap", () =>
        sendFromAnswer(runId, "Optimize the previous SQL without changing its result.")
          .catch(error => showError(error.message))),
    );
  }
  if (options.restored && options.run) {
    actions.append(actionButton("Continue from here", "message", () => continueRun(options.run.run_id).catch(error => showError(error.message))));
    if (options.run.accepted && options.run.sql) {
      actions.append(actionButton("Rerun saved SQL", "rerun", () => rerunSaved(options.run)));
    }
    const snapshot = document.createElement("p");
    snapshot.className = "snapshot-note";
    const retained = generation.rows?.length || 0;
    const complete = !generation.truncated && retained === generation.row_count;
    snapshot.textContent = `Retained snapshot · ${retained} of ${generation.row_count || 0} rows${complete ? " · complete" : " · preview only"}. Opening history did not run the query.`;
    actions.before(snapshot);
  }
  actions.hidden = !actions.children.length;

  renderHumanReview(fragment, body.human_review, responseMessage);
  if (!sqlNotAvailable) {
    const metadata = fragment.querySelector(".sql-metadata");
    if (metadata) {
      const status = document.createElement("div");
      status.className = "sql-execution-summary";
      const heading = document.createElement("strong");
      heading.textContent = ["ACCEPTED", "EXECUTED", "SUCCEEDED"].includes(generation.execution_status)
        ? "✓ Query executed successfully" : generation.execution_status || "Execution status unavailable";
      const detail = document.createElement("small");
      const duration = body.timings_ms?.sql_execution;
      detail.textContent = [
        `${generation.rows?.length || 0} retained rows${generation.truncated ? " · partial result" : ""}`,
        typeof duration === "number" ? `${(duration / 1000).toFixed(2)}s execution` : null,
      ].filter(Boolean).join(" · ");
      status.append(heading, detail); metadata.append(status);
    }
    for (const [label, value] of [
      ["Validation Details", attempts.map(attempt => attempt.validation).filter(Boolean)],
      ["Execution Plan", attempts.map(attempt => attempt.validation?.explain_plan)
        .filter(plan => Array.isArray(plan) && plan.length).at(-1) || "No execution plan was recorded."],
      ["Execution Details", {status: generation.execution_status,
        timings_ms: body.timings_ms, returned_rows: generation.row_count,
        retained_rows: generation.rows?.length, truncated: generation.truncated}],
      ["Query Context", {original_question: body.state?.root_question,
        resolved_question: body.resolved_question, result_id: body.result_id,
        contract: generation.semantic_contract || body.state?.semantic_contract}],
    ]) {
      if (!metadata) break;
      const details = document.createElement("details");
      const summary = document.createElement("summary"); summary.textContent = label;
      const pre = document.createElement("pre"); pre.textContent = JSON.stringify(value, null, 2);
      details.append(summary, pre); metadata.append(details);
    }
    fragment.querySelector(".copy-button").addEventListener("click", (event) => {
      navigator.clipboard.writeText(sqlText);
      event.currentTarget.classList.add("active");
      window.setTimeout(() => event.currentTarget.classList.remove("active"), 1200);
    });
    fragment.querySelector(".row-count").textContent = generation.truncated
      ? `${generation.rows?.length || 0} returned · partial result`
      : `${generation.row_count || 0} rows · ${generation.columns?.length || 0} columns`;
    wireSqlIdents(fragment, article);
    const versionPanel = fragment.querySelector(".versions-panel");
    const previous = latestAcceptedPreviousSql(body, options.restored ? options.run : null);
    if (sqlText && previous) {
      versionPanel.hidden = false;
      versionPanel.dataset.available = "true";
      versionPanel.querySelector("summary").onclick = () => { void loadVersionPanel(versionPanel, previous, sqlText); };
    }
  }
  if (generation.recovery && generation.recovery.activated) {
    renderRecoveryTimeline(fragment.querySelector(".recovery-panel"), generation.recovery, body.recovery_events || []);
  }
  hydrateIcons(article);
  $("#messages").append(fragment);
  if (!sqlNotAvailable && !explanatory) {
    article.dataset.runId = body.run_id || "";
    article.dataset.resultId = body.result_id || "";
    article.dataset.chartId = body.chart_id || "";
    wireDataScope(article, generation, body);
  }
  if (mode === "chart") wireChartSection(article, presentation, generation, body.run_id, body.chart_recommendations);
  const kpiContainer = article.querySelector(".chart-kpi-row");
  if (mode === "chart" && chartDataComplete && kpiContainer && window.QueryflowKpis) {
    // Built once from the accepted source, not during cosmetic chart rerenders.
    const kpis = window.QueryflowKpis.build(body, presentation.chart_spec)
      .filter(spec => spec.calculation !== "PERIOD_CHANGE"
        || generation.semantic_contract?.comparison_periods?.length === 2);
    article._kpiSpecs = kpis;
    const titleBadge=article.querySelector(".chart-change-badge");
    titleBadge?.remove();
    window.QueryflowKpis.render(kpiContainer, kpis, spec => {
      if (spec.point && article._echart?.highlightPoint) article._echart.highlightPoint(spec.point);
      else article._echart?.highlightSeries?.(spec.series_name);
    });
  }
  if (accepted && !explanatory && !body.clarification_required) {
    const mapRequested = /\bmap\b/i.test([
      options.question, body.resolved_question,
      body.state?.semantic_contract?.original_question,
    ].filter(Boolean).join(" "));
    const coordinates = mapCoordinates(generation.columns || [], generation.rows || [],
      mapRequested, generation.truncated);
    if (coordinates && window.Plotly) {
      article._renderMap = () => {
        if (article._mapRendered) return;
        article._mapRendered = true;
        const slot = article.querySelector(".map-slot");
        Promise.resolve(window.Plotly.newPlot(slot, [{type: "scattergeo", mode: "markers",
          lat: coordinates.lat, lon: coordinates.lon,
          marker: {size: 9, color: "#6e60e6"}}],
        {margin: {t: 10, r: 10, b: 10, l: 10}, geo: {showland: true}},
        {responsive: true, displaylogo: false})).catch(() => {
          article._mapRendered = false;
          slot.textContent = "The map renderer is unavailable. The source data remains in Data.";
        });
      };
    }
    wireResponseTabs(article, {answer:mode === "chart" && chartDataComplete,
      sql: Boolean(sqlText), data: Boolean(generation.columns?.length),
      map: Boolean(article._renderMap)});
  }
  window.QueryflowDashboard?.mountAction(article,body,{
    api,sessionId,renderChart:renderChartInto,renderTable,
    reviseDashboard:message=>sendFromAnswer(body.run_id,message).catch(error=>showError(error.message)),
  });
  if (!options.restored) updateInspector(body);
  return article;
}

function actionButton(label, iconName, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ghost-button";
  button.innerHTML = `${icon(iconName)}<span>${label}</span>`;
  button.addEventListener("click", onClick);
  return button;
}

function latestAcceptedPreviousSql(body, restoredRun) {
  return body.previous_sql || state.history.find(run => run.run_id === body.parent_run_id)?.sql || null;
}

async function loadVersionPanel(panel, previousSql, currentSql) {
  const list = panel.querySelector(".version-list");
  const diffBox = panel.querySelector(".diff-changes");
  if (list.dataset.loaded === "1") return;
  list.replaceChildren();
  list.dataset.loaded = "1";
  const difference = await fetchDiff(previousSql, currentSql);
  const changes = mergeRunChanges(difference.changes || []);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "version-item";
  const num = document.createElement("span");
  num.className = "version-num";
  num.textContent = "diff";
  const text = document.createElement("span");
  text.textContent = `vs previous accepted SQL · ${changes.length} change${changes.length === 1 ? "" : "s"}`;
  button.append(num, text);
  button.addEventListener("click", () => renderDiff(diffBox, changes));
  list.append(button);
  renderDiff(diffBox, changes);
  if (Object.keys(difference.invariants || {}).length) {
    const comparisons = document.createElement("p");
    comparisons.className = "snapshot-note";
    comparisons.textContent = "SQL structure: " + Object.entries(difference.invariants).map(([name, status]) => `${name} ${status}`).join(" · ");
    diffBox.append(comparisons);
  }
  for (const change of difference.filter_changes || []) {
    const update = document.createElement("p");
    update.className = "snapshot-note";
    update.textContent = `${change.operation.replaceAll("_", " ")}: ${change.previous_value || "—"} → ${change.new_value || "—"}`;
    diffBox.append(update);
  }
}

async function fetchDiff(previousSql, currentSql) {
  const key = `${previousSql}\u0000${currentSql}`;
  if (state.diffCache.has(key)) return state.diffCache.get(key);
  const observedDialect = $("#databaseSelect")?.selectedOptions?.[0]?.dataset.dialect || "sqlite";
  const body = await api("/api/sql/diff", {
    method: "POST",
    body: JSON.stringify({ previous_sql: previousSql, updated_sql: currentSql, dialect: observedDialect }),
  });
  state.diffCache.set(key, body);
  return body;
}

function renderDiff(diffBox, changes) {
  diffBox.hidden = false;
  diffBox.querySelectorAll(".diff-line").forEach(node => node.remove());
  const title = diffBox.querySelector(".diff-changes-title");
  if (!changes.length) {
    title.textContent = "No structural changes detected. This comparison does not prove business correctness.";
    return;
  }
  title.textContent = `${changes.filter(c => c.kind === "insert").length} added · ${changes.filter(c => c.kind === "remove").length} removed expressions (SQLGlot-aware)`;
  const existing = diffBox.querySelector(".diff-lines");
  if (existing) existing.remove();
  const container = document.createElement("div");
  container.className = "diff-lines";
  mergeRunChanges(changes).forEach((change) => {
    const line = document.createElement("div");
    line.className = `diff-line ${change.kind}`;
    line.textContent = change.token;
    container.append(line);
  });
  diffBox.append(container);
}

function renderRecoveryTimeline(panel, recovery, events = []) {
  panel.hidden = false;
  const modeBadge = panel.querySelector(".recovery-mode-badge");
  if (recovery.mode) {
    modeBadge.hidden = false;
    modeBadge.textContent = recovery.mode;
  }
  const decisionBadges = panel.querySelector(".recovery-decision-badges");
  decisionBadges.replaceChildren();
  const actions = [["REPAIR", "refresh", "accent"], ["INFORM", "info", ""], ["ESCALATE", "alert", ""]];
  actions.forEach(([action, iconName, tone]) => {
    if (recovery.agent_action !== action) return;
    const badge = document.createElement("span");
    badge.className = `badge ${tone === "accent" ? "accent" : "warn"}`;
    badge.innerHTML = `${icon(iconName)}${action}`;
    decisionBadges.append(badge);
  });
  const agentBadges = panel.querySelector(".recovery-agent-badges");
  agentBadges.replaceChildren();
  const usage = recovery.usage || {};
  [
    `${(usage.llm_calls || 0)} LLM calls`,
    `${(usage.database_probe_count || 0)} database probes`,
    `${Math.round((usage.latency_ms || 0) / 100) / 10}s`,
    usage.budget_exhausted ? "budget exhausted" : null,
  ].filter(Boolean).forEach((text) => {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = text;
    agentBadges.append(badge);
  });
  const steps = panel.querySelector(".recovery-steps");
  steps.replaceChildren();
  recoveryTimelineSteps(recovery, events).forEach((step) => {
    const node = document.createElement("div");
    node.className = "recovery-step";
    const iconWrap = document.createElement("span");
    iconWrap.className = `recovery-step-icon ${step.tone === "info" ? "" : step.tone}`;
    iconWrap.innerHTML = icon(step.icon, 13);
    const title = document.createElement("div");
    title.className = "recovery-step-title";
    title.textContent = step.title;
    const detail = document.createElement("div");
    detail.className = "recovery-step-detail";
    detail.textContent = step.detail;
    node.append(iconWrap, title, detail);
    if (step.evidenceId) {
      const evidence = document.createElement("span");
      evidence.className = "evidence-id";
      evidence.textContent = step.evidenceId;
      detail.append(" ", evidence);
    }
    if (step.observation || step.sql) {
      const observation = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = step.sql ? "View SQL" : "View observation";
      const pre = document.createElement("pre");
      if (step.sql) renderHighlightedSql(pre, step.sql);
      else pre.textContent = step.observation;
      observation.append(summary, pre);
      detail.append(observation);
    }
    steps.append(node);
  });
}

function underRows(interpretation, stateLike) {
  const rows = [];
  if (!interpretation && !stateLike) return rows;
  const source = (interpretation?.source || "rules");
  if (interpretation) {
    rows.push(["Operation", `${interpretation.operation} · ${source === "model" ? "model interpretation" : "rules"}`]);
    rows.push(["Resolved request", interpretation.resolved_instruction]);
    if (interpretation.confidence != null) rows.push(["Confidence", String(interpretation.confidence)]);
  }
  const contract = stateLike?.semantic_contract;
  if (contract) {
    if (contract.measures?.length) rows.push(["Metric", contract.measures.join(", ")]);
    if (contract.aggregation) rows.push(["Aggregation", contract.aggregation]);
    if (contract.filters?.length) rows.push(["Filters", contract.filters.map(item => `${item.operand || item.column || ""} ${item.operator || "="} ${JSON.stringify(item.value)}`).join("; ")]);
    if (contract.grain?.length) rows.push(["Grouping / grain", contract.grain.join(", ")]);
    if (contract.limit != null) rows.push(["Limit", String(contract.limit)]);
    if (contract.outputs?.length) rows.push(["Metric columns", contract.outputs.join(", ")]);
    if (contract.proposed_tables?.length) rows.push(["Candidate tables", contract.proposed_tables.join(", ")]);
    if (contract.ambiguities?.length) rows.push(["Ambiguities", contract.ambiguities.join("; ")]);
  }
  for (const change of interpretation?.changes || []) {
    rows.push(["Requested change", `${change.operation}: ${change.target} · ${change.previous_value ?? "—"} → ${change.new_value ?? "—"}`]);
  }
  return rows;
}

function updateInspector(body) {
  state.lastBody = body;
  const understanding = $("#understandingContent");
  understanding.replaceChildren();
  const rows = underRows(body.conversation_interpretation, body.state);
  const chartSpec = body.presentation?.chart_spec;
  if (chartSpec) {
    // A chart answer: provenance of what is on screen comes first.
    const provenance = [
      ["Chart type", String(chartSpec.chart_type || "").replaceAll("_", " ")],
      ["X axis", chartSpec.x_label || chartSpec.x || ""],
      ["Y axis", chartSpec.y_label || chartSpec.y || ""],
      chartSpec.series ? ["Series", String(chartSpec.series)] : null,
      ["Result rows", String(chartSpec.source_row_count ?? chartSpec.data?.length ?? "")],
      ...(chartSpec.warnings || []).map((warning) => ["Chart note", warning]),
    ].filter((row) => row && row[1]);
    rows.splice(0, 0, ...provenance);
  }
  if (!rows.length) {
    understanding.append(emptyNote("Run a query to see how it was understood."));
  } else {
    const kicker = document.createElement("div");
    kicker.className = "inspect-kicker";
    kicker.textContent = "QUERY UNDERSTANDING · CURRENT RESPONSE";
    understanding.append(kicker);
    const contract = body.state?.semantic_contract || {};
    const dateFilter = (contract.filters || []).find(item =>
      /(?:date|time|year|month|period)/i.test(String(item.operand || item.column || "")));
    const formatFilter = item => `${item.operand || item.column || "Value"} ${item.operator || "="} ${item.value ?? "?"}`;
    const chartPeriods = chartSpec?.time_format !== "raw" && chartSpec?.x
      ? [...new Set((chartSpec.data || []).map(row => String(row[chartSpec.x])))].sort()
      : [];
    const period = dateFilter ? formatFilter(dateFilter)
      : chartPeriods.length ? `Observed ${chartPeriods[0]}${chartPeriods.length > 1 ? `–${chartPeriods.at(-1)}` : ""}`
        : "Not specified";
    const tiles = [
      ["Measure", contract.measures?.join(", ") || chartSpec?.y_label || contract.outputs?.join(", ") || "Not specified"],
      ["Period", period],
      ["Dimension", contract.grain?.join(", ") || chartSpec?.x_label || "Not specified"],
      ["Filter", contract.filters?.length ? contract.filters.map(formatFilter).join("; ") : "None specified"],
    ];
    const grid = document.createElement("div");
    grid.className = "inspect-grid";
    tiles.forEach(([key, value]) => {
      const card = document.createElement("div");
      card.className = "inspect-tile";
      const label = document.createElement("span");
      label.className = "inspect-tile-label";
      label.textContent = key;
      const content = document.createElement("span");
      content.className = "inspect-tile-value";
      content.textContent = value;
      card.append(label, content);
      grid.append(card);
    });
    understanding.append(grid);
    const modelContext = body.generation?.model_context || {};
    const contextTables = Object.keys(modelContext.tables || modelContext.execution_context?.tables || {});
    if (contextTables.length) {
      const label = document.createElement("div");
      label.className = "inspect-label";
      label.textContent = "Selected SQL reference";
      understanding.append(label);
      const node = document.createElement("div");
      node.className = "inspect-reference";
      const title = document.createElement("strong");
      title.textContent = contextTables.join(" · ");
      const detail = document.createElement("code");
      detail.textContent = contextTables.map(name => {
        const selectedColumns = (modelContext.tables || modelContext.execution_context?.tables || {})[name]?.columns || {};
        const columns = Array.isArray(selectedColumns) ? selectedColumns : Object.keys(selectedColumns);
        return `${name}: ${columns.length ? columns.join(" · ") : "selected table"}`;
      }).join("\n");
      node.append(title, detail);
      understanding.append(node);
    }
    const presentation = document.createElement("div");
    presentation.className = "inspect-presentation";
    const presentationLabel = document.createElement("div");
    presentationLabel.className = "inspect-label";
    presentationLabel.textContent = "Presentation";
    const presentationValue = document.createElement("div");
    presentationValue.className = "inspect-presentation-value";
    presentationValue.textContent = chartSpec ? "Chart requested by the user"
      : body.generation?.accepted ? "SQL and executed data" : "No accepted result";
    presentation.append(presentationLabel, presentationValue);
    understanding.append(presentation);
    const extra = rows.filter(([key]) => !["Metric", "Grouping / grain", "Filters", "Metric columns"].includes(key));
    if (extra.length) {
      const more = document.createElement("details");
      more.className = "inspect-more";
      const summary = document.createElement("summary");
      summary.textContent = "More context";
      more.append(summary);
      extra.forEach(([key, value]) => {
        const row = document.createElement("div");
        row.className = "uq-row";
        const label = document.createElement("span");
        label.className = "k";
        label.textContent = key;
        const content = document.createElement("span");
        content.className = "v";
        content.textContent = value;
        row.append(label, content);
        more.append(row);
      });
      understanding.append(more);
    }
  }
  renderInspectorSchema();
  renderInspectorRecovery(body);
  renderInspectorTokenSummary(body);
  renderInspectorRunDetails(body);
  prioritizeInspector(body);
}

/* Keep measured pipeline timing separate from token accounting and attempts. */
function renderInspectorRunDetails(body) {
  const target = $("#tokensContent");
  const pipelineTarget = $("#pipelineContent");
  const schemaContent = $("#schemaContent");
  if (!target) return;
  const generation = body.generation || {};
  const tech = document.createElement("div");
  tech.innerHTML = [
    '<details class="token-card inspector-block">',
    '  <summary><span data-icon="hash"></span> Token usage</summary>',
    '  <div class="token-reduction"></div>',
    '  <div class="token-grid"></div>',
    '  <p class="token-note"></p>',
    '</details>',
    '<details class="pipeline-details inspector-block">',
    '  <summary><span data-icon="clock"></span> Pipeline timings</summary>',
    '  <div class="pipeline-rows"></div>',
    '</details>',
  ].join("\n");
  renderTokenAccounting(tech, body, generation);
  const tokenCard = tech.querySelector(".token-card");
  if (tokenCard) { tokenCard.open = true; target.append(tokenCard); }
  const pipelineCard = tech.querySelector(".pipeline-details");
  if (pipelineCard && pipelineTarget) {
    pipelineTarget.replaceChildren();
    const kicker = document.createElement("div");
    kicker.className = "inspect-kicker";
    kicker.textContent = "PIPELINE · MEASURED TIME";
    pipelineTarget.append(kicker);
    pipelineTarget.append(pipelineCard);
    pipelineCard.open = true;
    renderPipelineDetails(pipelineTarget, body.timings_ms || {}, body, generation);
  }
  const context = JSON.stringify(generation.model_context || {}, null, 2);
  if (context !== "{}" && schemaContent) {
    schemaContent.querySelectorAll("[data-context-panel]").forEach((node) => node.remove());
    const details = document.createElement("details");
    details.className = "inspector-block";
    details.dataset.contextPanel = "true";
    details.innerHTML = '<summary><span data-icon="network"></span> Verified context sent to the model</summary>';
    const pre = document.createElement("pre");
    pre.className = "context-json";
    pre.append(Object.assign(document.createElement("code"), { textContent: context }));
    details.append(pre);
    schemaContent.append(details);
  }
}

function prioritizeInspector(body) {
  const recoveryActive = body.generation?.recovery?.activated === true;
  selectInspectorTab(recoveryActive ? "recovery" : "understanding");
}

function emptyNote(text) {
  const p = document.createElement("p");
  p.className = "inspector-empty";
  p.textContent = text;
  return p;
}

function renderInspectorSchema() {
  const graphBox = $("#schemaGraph");
  const detail = $("#schemaDetail");
  detail.replaceChildren();
  const generation = state.lastBody?.generation;
  const context = generation?.model_context || {};
  const rawTables = context.tables || context.execution_context?.tables || {};
  const names = Object.keys(rawTables);
  graphBox.replaceChildren();
  if (!names.length && !browserTables.length) {
    graphBox.append(emptyNote("The verified tables used by the current query appear here."));
    return;
  }
  if (names.length) {
    const shownRelationships = new Set();
    names.forEach((name) => {
      const table = rawTables[name] || {};
      const card = document.createElement("button");
      card.type = "button";
      card.className = "inspect-schema-node";
      const title = document.createElement("strong");
      title.textContent = name;
      const columns = document.createElement("code");
      const selected = Array.isArray(table.columns)
        ? table.columns.map(column => [column, ""])
        : Object.entries(table.columns || {});
      columns.textContent = selected.map(([column, metadata]) => {
        const kind = typeof metadata === "object" ? metadata.type || metadata.database_type || "" : "";
        const role = (table.primary_key || []).includes(column) ? " · PRIMARY KEY" : "";
        return `${column}${kind ? ` · ${kind}` : ""}${role}`;
      }).join("\n") || "Selected table";
      card.append(title, columns);
      card.addEventListener("click", () => showSchemaDetail(detail, { table: name }));
      graphBox.append(card);
      (context.relationships || []).forEach((relationship, index) => {
        if (shownRelationships.has(index)) return;
        const left = String(relationship.left || "");
        const right = String(relationship.right || "");
        if (left.split(".")[0] !== name && right.split(".")[0] !== name) return;
        const join = document.createElement("div");
        join.className = "inspect-schema-join";
        const cardinality = relationship.cardinality
          ? ` · ${relationship.cardinality.replaceAll("_", " ").toLowerCase()}` : "";
        const source = relationship.state === "VERIFIED_FK" ? " · declared FK"
          : relationship.state ? " · inferred relationship" : "";
        join.textContent = `${left} → ${right}${cardinality}${source}`;
        graphBox.append(join);
        shownRelationships.add(index);
      });
    });
  } else {
    graphBox.append(emptyNote("Run a query to see the verified tables it used."));
  }
}

async function loadGlossary(dbId) {
  if (!dbId || state.glossaryByDb[dbId]) return state.glossaryByDb[dbId] || {};
  try {
    const terms = await api(`/api/databases/${encodeURIComponent(dbId)}/glossary`);
    state.glossaryByDb[dbId] = Object.fromEntries(terms.map((term) => [term.term.toLowerCase(), term]));
  } catch {
    state.glossaryByDb[dbId] = {};
  }
  return state.glossaryByDb[dbId];
}

function showSchemaDetail(container, selection) {
  const { table, column } = selection;
  container.replaceChildren();
  loadGlossary($("#databaseSelect").value).then((glossary) => {
    const tableInfo = browserTables.find(item => item.name.toLowerCase() === String(table || "").toLowerCase());
    const glossaryMatch = glossary[String(table || "").toLowerCase()] || glossary[String(column || "").toLowerCase()];
    const box = document.createElement("section");
    box.className = "table-meta";
    if (column) {
      const candidates = browserTables.filter(t => !table || t.name.toLowerCase() === String(table).toLowerCase())
        .flatMap(t => (t.columns || []).filter(c => c.name.toLowerCase() === String(column).toLowerCase())
          .map(c => ({...c, tableWithColumn: t.name})));
      const columnInfo = candidates.length === 1 ? candidates[0] : null;
      if (candidates.length > 1) {
        box.append(metaRow("Column", `${column} appears in multiple tables`));
        for (const candidate of candidates) {
          box.append(actionButton(candidate.tableWithColumn, "table", () => showSchemaDetail(container, {
            table: candidate.tableWithColumn, column,
          })));
        }
      }
      if (columnInfo) {
        box.append(metaRow("Column", `${columnInfo.tableWithColumn}.${columnInfo.name}`));
        box.append(metaRow("Physical type", columnInfo.type || "unknown"));
        if (columnInfo.primary_key) box.append(metaRow("Key", "Primary key"));
        const tags = document.createElement("div");
        tags.className = "tagrow";
        if (columnInfo.primary_key) tags.innerHTML += `<span class="schema-tag pk">PK</span>`;
        box.append(tags);
      } else {
        box.append(metaRow("Column", String(column)));
      }
    } else if (tableInfo) {
      box.append(metaRow("Table", tableInfo.name));
      box.append(metaRow("Columns", String(tableInfo.columns?.length || 0)));
      const primaryKeys = (tableInfo.columns || []).filter(c => c.primary_key).map(c => c.name);
      if (primaryKeys.length) box.append(metaRow("Primary key", primaryKeys.join(", ")));
      const relations = browserRelationships.filter(item => item.from_table === tableInfo.name || item.to_table === tableInfo.name);
      relations.forEach(relationship => {
        if (relationship.source === "declared_foreign_key") {
          box.append(metaRow("Declared FK", `${relationship.from_table}.${relationship.from_column} → ${relationship.to_table}.${relationship.to_column}`));
        }
      });
    }
    if (glossaryMatch) {
      box.append(metaRow("Glossary", glossaryMatch.definition));
      if (glossaryMatch.grain) box.append(metaRow("Grain", glossaryMatch.grain));
    }
    container.append(box);
  });
}

function metaRow(k, v) {
  const row = document.createElement("div");
  row.className = "meta-row";
  const kEl = document.createElement("span");
  kEl.className = "k";
  kEl.textContent = k;
  const vEl = document.createElement("span");
  vEl.className = "v";
  vEl.textContent = v;
  row.append(kEl, vEl);
  return row;
}

function inspectorRecoverySteps(recovery, events = []) {
  if (!recovery?.activated) return [];
  const signal = events.find(event => event.event_type === "SQL_FAILED");
  const modes = {
    ZERO_RESULT: ["Unmatched result detected", "The query ran but returned no rows."],
    NULL_RESULT: ["NULL result detected", "The query returned a NULL value."],
    FAILURE: ["Query failure detected", signal?.summary || "The SQL attempt failed."],
    CORRECTNESS: ["Correctness review requested", "The answer was flagged for review."],
    FILTER: ["Filter issue detected", "A filter produced an unexpected result."],
  };
  const [title, detail] = modes[recovery.mode] || ["Recovery started", recovery.failure_category || "An anomaly was recorded."];
  const steps = [{title, detail, meta: recovery.failure_code || "", icon: "alert", tone: "warn"}];
  const toolTitles = {
    inspect_schema: "Schema inspection", inspect_values: "Value check",
    query_database: "Database probe", run_probe: "Database probe",
  };
  for (const call of recovery.tool_calls || []) {
    const observations = [call.purpose, call.result_summary].filter(Boolean);
    steps.push({
      title: toolTitles[call.tool] || "Evidence check",
      detail: observations.filter((item, index) => !index || !sameDisplayText(item, observations[0])).join(" — "),
      meta: [call.tool, call.evidence_id ? `Observation ${call.evidence_id}` : null].filter(Boolean).join(" · "),
      icon: call.tool === "inspect_schema" ? "table" : "database", tone: "accent",
    });
  }
  if (recovery.agent_action || recovery.diagnosis_summary || recovery.agent_diagnosis || recovery.claims?.length) {
    const evidenceIds = [...new Set((recovery.claims || []).flatMap(claim => claim.evidence_ids || []))];
    steps.push({
      title: "Evidence evaluated",
      detail: recovery.diagnosis_summary || recovery.agent_diagnosis || recovery.claims[0]?.statement || "A recovery decision was recorded.",
      meta: [recovery.agent_action ? `Outcome: ${recovery.agent_action}` : "Agent interpretation",
        evidenceIds.length ? `Evidence ${evidenceIds.join(", ")}` : null].filter(Boolean).join(" · "),
      icon: recovery.agent_action === "ESCALATE" ? "alert" : "check",
      tone: recovery.agent_action === "ESCALATE" ? "warn" : "ok",
    });
  }
  return steps;
}

function renderInspectorRecovery(body) {
  const box = $("#recoveryMini");
  const recovery = body.generation?.recovery;
  box.replaceChildren();
  const kicker = document.createElement("div");
  kicker.className = "inspect-kicker";
  kicker.textContent = "INVESTIGATION · RECORDED EVIDENCE";
  box.append(kicker);
  if (!recovery || !recovery.activated) {
    box.append(emptyNote("The recovery agent only activates on anomalies."));
    return;
  }
  const info = document.createElement("div");
  info.className = "inspect-recovery-timeline";
  for (const step of inspectorRecoverySteps(recovery, body.recovery_events || [])) {
    const item = document.createElement("div");
    item.className = "inspect-recovery-step";
    const marker = document.createElement("span");
    marker.className = `inspect-recovery-marker ${step.tone}`;
    marker.innerHTML = icon(step.icon, 17);
    const content = document.createElement("div");
    content.className = "inspect-recovery-content";
    const title = document.createElement("strong");
    title.textContent = step.title;
    const detail = document.createElement("p");
    detail.textContent = step.detail;
    content.append(title, detail);
    if (step.meta) {
      const meta = document.createElement("small");
      meta.textContent = step.meta;
      content.append(meta);
    }
    item.append(marker, content);
    info.append(item);
  }
  box.append(info);
}

function renderInspectorTokenSummary(body) {
  const box = $("#tokensContent");
  box.replaceChildren();
  const kicker = document.createElement("div");
  kicker.className = "inspect-kicker";
  kicker.textContent = "MODEL USAGE · TOKENS";
  box.append(kicker);
  const generation = body.generation || {};
  const usage = body.token_usage || generation.token_usage || {};
  const total = usage.input_tokens == null && usage.output_tokens == null ? null : (usage.input_tokens || 0) + (usage.output_tokens || 0);
  box.append(metaRow("Model calls in ledger", String(
    (body.call_ledger || []).filter((entry) => entry.kind === "MODEL").length
  )));
  box.append(metaRow("Total tokens", total == null ? "unavailable" : total.toLocaleString()));
}

function wireSqlIdents(fragment, article) {
  fragment.querySelectorAll(".sql-ident").forEach((span) => {
    span.addEventListener("click", () => {
      const detail = $("#schemaDetail");
      showSchemaDetail(detail, span.dataset.table ? { table: span.dataset.table } : { column: span.dataset.column });
      revealInspector("schema");
      $("#schemaContent").scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  });
}

function renderHumanReview(fragment, review, responseMessage = "") {
  const panel = fragment.querySelector(".human-review");
  if (!panel || !review) return;
  panel.hidden = false;
  const reason = panel.querySelector(".human-review-reason");
  reason.textContent = review.reason || "Automated review is uncertain.";
  reason.hidden = sameDisplayText(responseMessage, reason.textContent);
  const question = panel.querySelector(".human-review-question");
  question.textContent = review.question || "Please clarify.";
  question.hidden = sameDisplayText(responseMessage, question.textContent)
    || (!reason.hidden && sameDisplayText(reason.textContent, question.textContent));
  const options = panel.querySelector(".review-actions");
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

function filterAndSortResultRows(rows, query, sortColumn = -1, direction = 1) {
  const needle = query.trim().toLocaleLowerCase();
  const filtered = rows.filter(row => !needle || row.some(value =>
    String(value ?? "NULL").toLocaleLowerCase().includes(needle)));
  if (sortColumn < 0) return filtered;
  return filtered.slice().sort((a, b) => {
    const left = a[sortColumn], right = b[sortColumn];
    if (left == null) return right == null ? 0 : 1;
    if (right == null) return -1;
    return direction * (typeof left === "number" && typeof right === "number"
      ? left - right : String(left).localeCompare(String(right), undefined, {numeric:true}));
  });
}

function renderTable(container, columns, rows, options = {}) {
  container._tableResizeObserver?.disconnect();
  container.replaceChildren();
  if (!columns.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No rows returned, or execution was not completed.";
    container.append(empty);
    return;
  }
  const table = document.createElement("table");
  table.className = "result-grid";
  const canvas = document.createElement("canvas");
  const measureContext = canvas.getContext?.("2d");
  const manualWidths = new Set();
  const hiddenColumns = new Set();
  const visibleColumns = columns.map(() => true);
  const widths = columns.map(() => 84);
  let textFont = "12.3px sans-serif";
  const sampled = (data, maximum = 120) => {
    if (data.length <= maximum) return data;
    return Array.from({length: maximum}, (_, index) =>
      data[Math.round(index * (data.length - 1) / (maximum - 1))]);
  };
  const measure = (value, numeric = false) => {
    const label = String(value ?? "NULL");
    if (measureContext) measureContext.font = numeric ? "12.3px monospace" : textFont;
    return measureContext ? measureContext.measureText(label).width : label.length * 7;
  };
  const applyWidths = () => {
    // Hidden columns leave no reserved gap inside the table's own borders.
    widths.forEach((width, index) => {
      colgroup.children[index].style.width = `${hiddenColumns.has(index) ? 0 : width}px`;
    });
    const contentWidth = widths.reduce(
      (sum, width, index) => sum + (hiddenColumns.has(index) ? 0 : width), 0);
    table.style.width = `${contentWidth}px`;
    const pane = container.parentElement;
    if (pane?.classList?.contains?.("result-panel-pane")) {
      // The response card stays full-width and responsive; the table itself is
      // content-sized and scrolls inside the card instead of stretching its
      // columns, so two-column results never fill the whole chat width.
      pane.style.width = "100%";
    }
  };
  const colgroup = document.createElement("colgroup");
  widths.forEach(width => {
    const col = document.createElement("col");
    col.style.width = `${width}px`;
    colgroup.append(col);
  });
  table.append(colgroup);
  const recalculateWidths = (data = rows) => {
    if (typeof getComputedStyle === "function") textFont = getComputedStyle(table).font || textFont;
    const examples = sampled(data);
    columns.forEach((name, index) => {
      if (manualWidths.has(index)) return;
      const content = examples.reduce((largest, row) =>
        Math.max(largest, measure(row[index], typeof row[index] === "number")), measure(name));
      // 12px cell padding on each side plus the resize affordance; a sensible
      // minimum keeps headers readable and a maximum reserves room for the rest.
      widths[index] = Math.round(Math.max(84, Math.min(420, content + 40)));
    });
    applyWidths();
  };
  recalculateWidths();
  const head = table.createTHead().insertRow();
  columns.forEach((column, index) => {
    const cell = document.createElement("th");
    cell.textContent = column;
    const observed = [...new Set(rows.map(row => row[index]).filter(value => value != null)
      .map(value => typeof value))];
    cell.title = `${column} · Observed values: ${observed.join(" / ") || "NULL only"}`;
    cell.classList.toggle("numeric", rows.some(row => typeof row[index] === "number"));
    const handle = document.createElement("span");
    handle.className = "column-resize";
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-label", `Resize ${column} column`);
    handle.addEventListener("pointerdown", event => {
      event.preventDefault();
      const initialX = event.clientX;
      const initialWidth = widths[index];
      handle.setPointerCapture?.(event.pointerId);
      const move = moved => {
        widths[index] = Math.max(72, Math.min(600, initialWidth + moved.clientX - initialX));
        manualWidths.add(index);
        applyWidths();
      };
      const done = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", done);
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", done);
    });
    cell.append(handle);
    head.append(cell);
  });
  const body = table.createTBody();
  container.append(table);
  const controls = container.parentElement?.querySelector(".table-controls");
  if (controls) controls.replaceChildren();
  let pageSize = 5;
  let page = 0;
  let currentRows = rows.slice(0, pageSize);
  let hasMore = Boolean(options.truncated && options.loadPage) || rows.length > pageSize;
  let limitReached = false;
  let loading = false;
  const pages = new Map();
  let viewRows = rows;
  let localView = false;
  let sortColumn = -1;
  let sortDirection = 1;
  /* Table tools live in the Data section header (top right); page controls
     stay underneath the table. The search sorts over every retained row of
     the authorized result scope, never just the visible page. */
  const paneForTools = container.parentElement;
  const headerEl = paneForTools && typeof paneForTools.querySelector === "function"
    ? paneForTools.querySelector(".data-header") : null;
  const tools = headerEl && typeof headerEl.querySelector === "function"
    ? headerEl.querySelector(".data-header-tools") : null;
  const applyView = () => {
    localView = Boolean(tools?.querySelector?.(".data-search")?.value?.trim()) || sortColumn >= 0;
    viewRows = filterAndSortResultRows(rows, tools?.querySelector?.(".data-search")?.value ?? "", sortColumn, sortDirection);
    pages.clear(); page = 0; currentRows = viewRows.slice(0, pageSize);
    hasMore = viewRows.length > pageSize || (!localView && Boolean(options.truncated && options.loadPage));
    renderRows(); updateControls();
  };
  if (tools) {
    tools.replaceChildren();
    const search = document.createElement("input");
    search.type = "search"; search.className = "data-search";
    search.placeholder = "Search rows…";
    search.setAttribute("aria-label", "Search retained result rows");
    search.addEventListener("input", applyView);
    const filters = document.createElement("details");
    filters.className = "data-column-controls";
    const summary = document.createElement("summary");
    summary.textContent = "Filters";
    filters.append(summary);
    const hint = document.createElement("small");
    hint.textContent = "Search filters retained rows. Choose visible columns:";
    filters.append(hint);
    columns.forEach((column, index) => {
      const label = document.createElement("label");
      const check = document.createElement("input");
      check.type = "checkbox"; check.checked = true;
      check.addEventListener("change", () => {
        visibleColumns[index] = check.checked;
        head.children[index].hidden = !check.checked;
        colgroup.children[index].style.display = check.checked ? "" : "none";
        [...body.children].forEach(row => { if (row.cells?.[index]) row.cells[index].hidden = !check.checked; });
        if (!check.checked) hiddenColumns.add(index); else hiddenColumns.delete(index);
        applyWidths();
      });
      label.append(check, column); filters.append(label);
    });
    tools.append(search, filters);
  }
  const exportButton = container.parentElement?.querySelector?.(".data-header .csv-button") || null;
  if (exportButton && !exportButton._exportWired) {
    exportButton._exportWired = true;
    exportButton.addEventListener("click", () => {
      // Export the current retained view: raw values, applied search/sort,
      // and only the visible columns. Never reformatted display strings.
      const exportColumns = columns.filter((_, index) => visibleColumns[index]);
      const exportRows = viewRows.map(row => columns.map((_, index) => row[index])
        .filter((_, index) => visibleColumns[index]));
      downloadCsv(exportColumns, exportRows);
      appendExportBubble(exportColumns, exportRows, {
        truncated: Boolean(options.truncated),
        source: `Data tab retained result · ${localView ? "search/sort applied · " : ""}raw values`,
      });
    });
  }
  columns.forEach((column, index) => {
    const th = head.children[index];
    const button = document.createElement("button"); button.type = "button";
    button.textContent = `${column} ↕`; button.title = `Sort retained rows by ${column}`;
    th.firstChild?.remove(); th.prepend(button);
    button.addEventListener("click", () => {
      sortDirection = sortColumn === index ? -sortDirection : 1; sortColumn = index;
      [...head.children].forEach(cell => cell.removeAttribute("aria-sort"));
      th.setAttribute("aria-sort", sortDirection > 0 ? "ascending" : "descending");
      button.textContent = `${column} ${sortDirection > 0 ? "↑" : "↓"}`; applyView();
    });
  });
  const renderRows = () => {
    body.replaceChildren();
    const virtual = currentRows.length >= 80;
    const start = virtual ? Math.max(0, Math.min(currentRows.length - 36,
      Math.floor((container.scrollTop || 0) / 38) - 6)) : 0;
    const visibleRows = virtual ? currentRows.slice(start, start + 36) : currentRows;
    const spacer = height => {
      if (!height) return;
      const tr = body.insertRow();
      tr.dataset.spacer = "true";
      const td = tr.insertCell();
      td.colSpan = columns.length;
      td.style.height = `${height}px`;
      td.className = "virtual-spacer";
    };
    spacer(start * 46);
    visibleRows.forEach((row, index) => {
      const tr = body.insertRow();
      tr.dataset.rowIndex = String(start + index);
      row.forEach((value, columnIndex) => {
        const column = columns[columnIndex];
        const td = tr.insertCell();
        if (value === null) {
          const badge = document.createElement("span");
          badge.className = "null-value";
          badge.textContent = "—"; badge.title = "SQL NULL (missing value)";
          td.append(badge);
        } else {
          // Display formatting is metadata-aware; the exact raw value stays on
          // hover and drives sorting, filtering, and CSV export unchanged.
          const formatDisplay = globalThis.QueryflowFormat?.formatCellValue;
          td.textContent = typeof formatDisplay === "function"
            ? formatDisplay(value, {
                unit: options.units?.[column],
                timeFormat: options.temporal?.time_format ?? null,
                timeColumn: options.temporal?.column ?? null,
                column,
              })
            : typeof value === "number"
            && !/(?:id|key|code)$/i.test(column) && typeof formatKpiValue === "function"
            ? formatKpiValue(value, options.units?.[column])
            : String(value);
          td.title = String(value);
          if (typeof value === "number") td.classList.add("numeric");
        }
      });
    });
    spacer((currentRows.length - start - visibleRows.length) * 46);
    options.onPageRendered?.(table, currentRows);
    columns.forEach((_, index) => {
      if (visibleColumns[index]) return;
      [...body.children].forEach(row => { if (row.cells?.[index]) row.cells[index].hidden = true; });
    });
  };
  const updateControls = () => {
    if (!controls) return;
    controls.replaceChildren();
    const status = document.createElement("span");
    status.className = "table-page-status";
    const offset = page * pageSize;
    status.textContent = currentRows.length
      ? `Showing ${offset + 1}–${offset + currentRows.length} of ${localView ? viewRows.length : rows.length} rows`
      : `No rows match · ${options.truncated ? `${rows.length} cached · partial result` : `${rows.length} rows · complete result`}`;
    if (localView) status.textContent += ` · matching cached rows only · ${options.truncated ? "searching partial result" : "complete retained result"}`;
    if (limitReached) status.textContent += " · 10,000-row paging limit reached";
    controls.append(status);
    const meta = container.parentElement?.querySelector?.(".row-count");
    if (meta) meta.textContent = currentRows.length
      ? `Showing ${offset + 1}–${offset + currentRows.length} of ${localView ? viewRows.length : rows.length} rows · ${columns.length} columns${options.truncated ? " · partial" : ""}`
      : `No rows match · ${columns.length} columns${options.truncated ? " · partial result" : ""}`;
    if (viewRows.length <= 5 && !options.truncated) return;
    const pagesHost = document.createElement("div");
    pagesHost.className = "table-pages";
    const count = Math.ceil(viewRows.length / pageSize);
    for (let target = Math.max(0, page - 2); target < Math.min(count, page + 3); target++) {
      const numbered = document.createElement("button"); numbered.type = "button";
      numbered.textContent = String(target + 1);
      numbered.className = target === page ? "table-page active" : "table-page";
      numbered.setAttribute("aria-label", `Page ${target + 1}`);
      if (target === page) numbered.setAttribute("aria-current", "page");
      numbered.disabled = loading;
      numbered.addEventListener("click", () => { void goToPage(target); }); pagesHost.append(numbered);
    }
    for (const [label, target, disabled] of [
      ["Previous", page - 1, page === 0],
      ["Next", page + 1, !hasMore],
    ]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.disabled = disabled || loading;
      button.addEventListener("click", () => { void goToPage(target); });
      pagesHost.append(button);
    }
    controls.append(pagesHost);
  };
  async function goToPage(target) {
    if (target < 0 || loading) return;
    const offset = target * pageSize;
    const key = `${offset}:${pageSize}`;
    let next = pages.get(key);
    if (!next) {
      if (offset < viewRows.length || localView) {
        next = {
          rows: viewRows.slice(offset, offset + pageSize),
          has_more: (!localView && Boolean(options.truncated && options.loadPage))
            || offset + pageSize < viewRows.length,
        };
      } else if (options.loadPage) {
        loading = true;
        updateControls();
        try { next = await options.loadPage(offset, pageSize); }
        catch (error) { options.onPageError?.(error); return; }
        finally { loading = false; }
      } else return;
      pages.set(key, next);
    }
    page = target;
    currentRows = next.rows;
    recalculateWidths([...sampled(rows, 80), ...sampled(currentRows, 40)]);
    hasMore = next.has_more;
    limitReached = Boolean(next.limit_reached);
    container.scrollTop = 0;
    renderRows();
    updateControls();
  }
  container.addEventListener("scroll", () => {
    if (currentRows.length >= 80) renderRows();
  });
  container.classList.toggle("table-compact", rows.length <= 8 && columns.length <= 5);
  if (typeof ResizeObserver !== "undefined") {
    let previousWidth = container.clientWidth;
    container._tableResizeObserver = new ResizeObserver(() => {
      if (container.clientWidth === previousWidth) return;
      previousWidth = container.clientWidth;
      recalculateWidths([...sampled(rows, 80), ...sampled(currentRows, 40)]);
    });
    container._tableResizeObserver.observe(container);
  }
  document.fonts?.ready?.then(() => recalculateWidths([...sampled(rows, 80), ...sampled(currentRows, 40)]));
  renderRows();
  updateControls();
  return {goToPage, get pageSize() { return pageSize; }};
}

/* Exporting data stays a client-side action: it never re-runs SQL or calls a
   model. The bubble records what was exported (raw values) so retained-result
   and truncated previews stay clearly distinguishable in the conversation. */
function appendExportBubble(columns, rows, { source = null, truncated = false } = {}) {
  const fragment = $("#exportTemplate").content.cloneNode(true);
  const article = fragment.querySelector("article");
  const count = rows.length;
  fragment.querySelector(".export-note").textContent = truncated
    ? `Exported the first ${count} retained rows as CSV — this is a partial result preview; more rows can be paged from the Data tab.`
    : `Exported ${count} rows × ${columns.length} columns as CSV from the retained result.`;
  fragment.querySelector(".export-meta").textContent = source
    || `${columns.join(" · ")} · raw values, no formatting applied`;
  const again = fragment.querySelector(".export-again");
  again.addEventListener("click", () => downloadCsv(columns, rows));
  hydrateIcons(article);
  const messages = $("#messages");
  messages.append(article);
  article.scrollIntoView?.({ behavior: "smooth", block: "end" });
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
    ...(chartTokens > 0 ? [["Legacy chart-selection call", chartTokens]] : []),
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
  const summary = fragment.querySelector(".attempt-summary");
  const count = fragment.querySelector(".attempts-count");
  if (!panel || !list || !summary) return;
  const visibleAttempts = attempts.filter((attempt) => attemptOutcome(attempt) !== "passed");
  if (count) count.textContent = attempts.length ? `· ${attempts.length}` : "";
  if (!visibleAttempts.length) {
    summary.textContent = attempts.length
      ? `${attempts.length} attempt${attempts.length === 1 ? "" : "s"} passed validation.`
      : "No SQL validation attempt was required for this response.";
    panel.open = false;
    return;
  }

  const failed = visibleAttempts.filter((attempt) => attemptOutcome(attempt) === "failed").length;
  const notSelected = visibleAttempts.filter((attempt) => attemptOutcome(attempt) === "not_selected").length;
  const summaryParts = [`${visibleAttempts.length} rejected candidate${visibleAttempts.length === 1 ? "" : "s"}`];
  if (failed) summaryParts.push(`${failed} failed`);
  if (notSelected) summaryParts.push(`${notSelected} not selected`);
  summary.textContent = summaryParts.join(" · ");
  panel.open = true;
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
    title.textContent = `Attempt ${attempt.number || index + 1} · ${outcomeLabel}`;
    header.append(title);
    const code = document.createElement("code");
    code.textContent = validation.code || "UNKNOWN_VALIDATION_RESULT";
    header.append(code);

    const reason = document.createElement("p");
    reason.className = "attempt-reason";
    reason.textContent = validation.message || "No validator explanation was returned.";
    item.append(header, reason);

    if (!passed && attempt.sql) {
      const sqlPre = document.createElement("pre");
      const sqlCode = document.createElement("code");
      sqlCode.className = "language-sql";
      renderHighlightedSql(sqlCode, formatSqlForDisplay(attempt.sql));
      sqlPre.append(sqlCode);
      item.append(sqlPre);
    }
    list.append(item);
  });
}

function renderPipelineDetails(fragment, timings, body, generation) {
  const container = fragment.querySelector(".pipeline-rows");
  if (!container) return;
  container.replaceChildren();
  const total = Number(timings.total);
  const knownTotal = Number.isFinite(total) && total > 0 ? total : null;
  const headline = document.createElement("div");
  headline.className = "pipeline-total";
  const headlineLabel = document.createElement("span");
  headlineLabel.textContent = "Total request";
  const headlineValue = document.createElement("strong");
  headlineValue.textContent = knownTotal == null ? "Not reported" : `${(knownTotal / 1000).toFixed(1)} s`;
  headline.append(headlineLabel, headlineValue);
  container.append(headline);
  const sectionHeading = title => {
    const heading = document.createElement("div");
    heading.className = "pipeline-section-heading";
    heading.textContent = title;
    container.append(heading);
  };
  sectionHeading("Stage timing");
  const topLayerMeasured = Object.hasOwn(timings, "top_decisions");
  const stages = topLayerMeasured ? [
    ["Top-layer decisions", timings.top_decisions],
    ["Analytics planning model", timings.analytics_planning],
    ["Context planning", timings.context_planning],
    ["SQL generation and repair", timings.sql_generation],
    ["SQL validation", timings.sql_validation],
    ["Database execution", timings.sql_execution],
    ["Result analysis", timings.result_analysis],
    ["Result explanation", timings.result_explanation],
    ["Charting model", timings.charting_model],
    ["Top-layer final answer", timings.top_final_answer],
  ] : [
    ["Conversation interpretation", timings.conversation_interpretation],
    ["Analytics planning", timings.analytics_planning],
    ["Routing", timings.routing],
    ["Context planning", timings.planning],
    ["Generation / validation / execution", timings.generation_validation_execution],
  ];
  stages.forEach(([label, milliseconds]) => {
    if (topLayerMeasured && milliseconds == null) return;
    const row = document.createElement("div");
    row.className = `pipeline-run${milliseconds == null ? " unmeasured" : ""}`;
    const status = document.createElement("span");
    status.textContent = label;
    const duration = document.createElement("b");
    const measured = milliseconds == null ? null : Number(milliseconds);
    duration.textContent = measured == null || !Number.isFinite(measured)
      ? "Not reported" : `${measured.toLocaleString()} ms`;
    row.append(status, duration);
    const track = document.createElement("div");
    track.className = "pipeline-bar-track";
    track.setAttribute("aria-hidden", "true");
    const bar = document.createElement("span");
    bar.className = "pipeline-bar-fill";
    bar.style.width = `${knownTotal == null || measured == null || !Number.isFinite(measured)
      ? 0 : Math.min(100, Math.max(0, measured / knownTotal * 100))}%`;
    track.append(bar);
    row.append(track);
    container.append(row);
  });
  const modelCalls = (body.call_ledger || []).filter(entry => entry.kind === "MODEL");
  if (modelCalls.length) sectionHeading("Model calls");
  for (const entry of modelCalls) {
    const row = document.createElement("div");
    row.className = "pipeline-run pipeline-model-call";
    const label = document.createElement("span");
    label.textContent = entry.task ? `${entry.component} · ${entry.task}` : entry.component;
    const detail = document.createElement("b");
    const tokens = entry.input_tokens == null && entry.output_tokens == null
      ? "tokens unavailable" : `${(entry.input_tokens || 0) + (entry.output_tokens || 0)} tokens`;
    detail.textContent = [
      [entry.provider, entry.effective_model || entry.requested_model].filter(Boolean).join("/"),
      entry.latency_ms == null ? "time unavailable" : `${entry.latency_ms} ms`,
      tokens,
      entry.retry_count > 0 ? `retry ${entry.retry_count}` : null,
      entry.status === "FAILED" ? "failed" : null,
    ].filter(Boolean).join(" · ");
    row.append(label, detail);
    container.append(row);
  }
  if (body.progress_events?.length) {
    const history = container.parentElement.querySelector(".progress-history");
    const list = history ? history.querySelector("ul") : null;
    if (list) {
      list.replaceChildren();
      body.progress_events.forEach(event => {
        const item = document.createElement("li");
        item.textContent = `${(event.elapsed_ms / 1000).toFixed(1)}s · ${progressLabels[event.stage] || event.stage}`;
        list.append(item);
      });
    }
  }
}

function showError(message) {
  appendAssistant({ operation: "ERROR", message, generation: { accepted: false, attempts: [] } }, 0);
}

/* ---------- History ---------- */

function summarizeRun(run) {
  const created = run.created_at ? new Date(run.created_at * 1000) : null;
  const time = created && !Number.isNaN(created.getTime())
    ? created.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : "";
  return {
    title: run.original_message || run.resolved_question || "Query",
    subtitle: [run.operation, `${run.row_count || 0} rows`, run.outcome?.toLowerCase().replaceAll("_", " ") || (run.accepted ? "executed" : "failed"), time]
      .filter(Boolean).join(" · "),
    failed: run.outcome ? run.outcome === "FAILED" : run.accepted === false,
  };
}

async function refreshHistory() {
  try {
    state.history = (await api(`/api/history?session_id=${encodeURIComponent(sessionId)}&limit=50`))
      .filter(run => run.db_id === $("#databaseSelect").value);
    renderHistory();
  } catch {
    state.history = [];
    renderHistory();
  }
}

$("#historySearch")?.addEventListener("input", () => renderHistory());
$("#historySearchToggle")?.addEventListener("click", event => {
  const input = $("#historySearch"); input.hidden = !input.hidden;
  event.currentTarget.setAttribute("aria-expanded", String(!input.hidden));
  if (!input.hidden) input.focus();
  else { input.value = ""; renderHistory(); }
});
$("#sidebarToggle")?.addEventListener("click", () => {
  const collapsed = document.body.classList.toggle("sidebar-collapsed");
  $("#sidebarToggle").setAttribute("aria-expanded", String(!collapsed));
  $(".sidebar").inert = collapsed;
  window.dispatchEvent(new Event("resize"));
});
$("#dataNavigation")?.addEventListener("click", event => {
  event.preventDefault();
  $("#databaseSelect").focus();
});
$("#settingsNavigation")?.addEventListener("click", event => {
  event.preventDefault();
  $("#themeToggle").click();
});
$("#homeNavigation")?.addEventListener("click", event => {
  event.preventDefault();
  $("#messageInput").focus();
});

function renderHistory() {
  const container = $("#conversationHistory");
  container.replaceChildren();
  const groups = new Map();
  for (const run of state.history) {
    const id = run.conversation_id || run.run_id;
    if (!groups.has(id)) groups.set(id, run);
  }
  const search = $("#historySearch")?.value?.trim().toLocaleLowerCase() || "";
  const visible = [...groups.values()].filter(run => !search ||
    state.history.some(turn => (turn.conversation_id || turn.run_id) ===
      (run.conversation_id || run.run_id) &&
      `${turn.original_message || ""} ${turn.resolved_question || ""}`.toLocaleLowerCase().includes(search)));
  if (!visible.length) {
    container.append(emptyNote(search ? "No matching conversations." : "Completed queries appear here."));
    return;
  }
  for (const run of visible) {
    const item = summarizeRun(run);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "run-item";
    button.dataset.runId = run.run_id;
    if (run.run_id === state.activeRunId) button.classList.add("active");
    const iconWrap = document.createElement("span");
    iconWrap.innerHTML = icon(item.failed ? "alert" : "message", 14);
    const main = document.createElement("div");
    main.className = "run-main";
    const title = document.createElement("span");
    title.className = "run-title";
    const turns = state.history.filter(turn => (turn.conversation_id || turn.run_id) === (run.conversation_id || run.run_id));
    const first = turns[turns.length - 1];
    title.textContent = first?.original_message || item.title;
    const sub = document.createElement("span");
    sub.className = "run-sub";
    sub.textContent = `${run.db_id} · ${turns.length} turn${turns.length === 1 ? "" : "s"} · ${item.subtitle}`;
    main.append(title, sub);
    const dot = document.createElement("span");
    dot.className = `run-status${item.failed ? " failed" : ""}`;
    button.append(iconWrap, main, dot);
    button.addEventListener("click", () => openRun(run));
    container.append(button);
  }
}

async function openRun(run) {
  if (state.busy || voiceInput?.busy) return;
  $("#emptyState").hidden = true;
  state.activeRunId = run.run_id;
  state.branchFromRunId = null;
  renderHistory();
  try {
    const records = await api(`/api/history/${encodeURIComponent(run.run_id)}/turns?session_id=${encodeURIComponent(sessionId)}`);
    const record = records[records.length - 1];
    if (!record) return;
    if (record.db_id !== $("#databaseSelect").value) throw new Error("This conversation belongs to another database.");
    await loadGlossary(record.db_id);
    await loadDatabaseSchema();
    $("#messages").replaceChildren();
    for (const turn of records) {
      appendUser(turn.original_message);
      await appendAssistant(restoredRunToResponse(turn), 0, {restored: true, run: turn});
    }
    state.restorePending = true;
    updateInspector(restoredRunToResponse(record));
    $("#mainColumn").scrollTo({top: 0});
  } catch (error) { showError(`Could not open the saved conversation: ${error.message}`); }
}

async function continueRun(runId, focus = true) {
  const savedState = await api(`/api/history/${encodeURIComponent(runId)}/continue?session_id=${encodeURIComponent(sessionId)}`, {method: "POST"});
  if (savedState.db_id !== $("#databaseSelect").value) throw new Error("This conversation belongs to another database.");
  state.activeRunId = runId;
  state.branchFromRunId = runId;
  state.restorePending = false;
  renderHistory();
  if (focus) {
    $("#messageInput").placeholder = "Continue this conversation or correct the previous query…";
    $("#messageInput").focus();
  }
}

async function rerunSaved(record) {
  if (state.busy || voiceInput?.busy) return;
  state.busy = true;
  voiceInput?.setQueryBusy(true);
  $("#sendButton").disabled = true;
  try {
    const response = await api(`/api/history/${encodeURIComponent(record.run_id)}/rerun?session_id=${encodeURIComponent(sessionId)}`, {method: "POST"});
    state.activeRunId = response.run_id;
    state.branchFromRunId = null;
    state.restorePending = false;
    appendUser("Rerun the saved SQL against current data.");
    await appendAssistant(response, 0);
    await refreshHistory();
  } catch (error) { showError(error.message); }
  finally {
    state.busy = false;
    voiceInput?.setQueryBusy(false);
    $("#sendButton").disabled = !state.ready;
  }
}

/* Bounded, session-local display snapshots. Never persisted as analytical memory. */
function chartSnapshotKey(session, db, run, result, chart) {
  return JSON.stringify([session,db,run,result,chart || null]);
}

function rememberChartSnapshot(body) {
  state.chartSnapshots ||= new Map();
  if (body.operation === "RESET_CONTEXT") return;
  const generation=body.generation, chart=body.presentation?.chart_spec;
  if (!body.run_id || !body.result_id || !generation?.accepted || !chart
      || generation.truncated || generation.row_count !== generation.rows?.length
      || !chartMatchesResultIdentity(body)) return;
  const key=chartSnapshotKey(sessionId,generation.db_id,body.run_id,body.result_id,body.chart_id);
  state.chartSnapshots.delete(key);
  state.chartSnapshots.set(key,{
    sql:generation.sql, columns:generation.columns, rows:generation.rows,
    row_count:generation.row_count, chart_spec:chart,
  });
  if (state.chartSnapshots.size>100) state.chartSnapshots.delete(state.chartSnapshots.keys().next().value);
}

function restoredRunToResponse(record) {
  if (record.response_payload && Object.keys(record.response_payload).length) {
    const payload = {...record.response_payload};
    const key=chartSnapshotKey(sessionId,record.db_id,record.run_id,payload.result_id,payload.chart_id);
    const snapshot=state.chartSnapshots?.get(key);
    if (snapshot && payload.generation?.accepted && snapshot.sql === payload.generation.sql
        && snapshot.row_count === payload.generation.row_count
        && JSON.stringify(snapshot.columns) === JSON.stringify(payload.generation.columns)) {
      payload.generation={...payload.generation,rows:snapshot.rows,truncated:false};
      payload.presentation={...payload.presentation,chart_spec:snapshot.chart_spec};
    }
    if (payload.presentation_state == null) {
      // History saved before the typed presentation state existed: synthesize
      // it from the validated chart spec so restored threads stay chart-first.
      const spec = payload.presentation?.chart_spec;
      payload.presentation_state = spec
        ? {
            mode: "chart",
            chart_type: spec.chart_type,
            x: spec.x,
            y: spec.y,
            series: spec.series,
            size: spec.size,
            title: spec.title,
          }
        : payload.generation?.accepted
          ? { mode: "sql" }
          : null;
    }
    return payload;
  }
  return {
    session_id: record.session_id,
    operation: record.operation,
    resolved_question: record.resolved_question,
    message: record.presentation_summary || record.resolved_question,
    state: { turn_count: record.turn_count, semantic_contract: null },
    token_usage: record.token_usage || {},
    timings_ms: record.timings_ms || {},
    presentation: record.presentation_summary ? { summary: record.presentation_summary, note: "" } : null,
    generation: {
      accepted: record.accepted,
      execution_status: record.execution_status,
      question: record.resolved_question,
      sql: record.sql,
      columns: record.columns || [],
      rows: record.rows_preview || [],
      row_count: record.row_count || 0,
      truncated: record.truncated,
      attempts: [],
      token_usage: record.token_usage || {},
      recovery: record.recovery || null,
    },
  };
}

/* ---------- Theme, inspector, keyboard ---------- */

function applyThemeButtons() {
  const dark = document.documentElement.dataset.theme === "dark";
  [["#themeToggle", "Toggle appearance"], ["#themeToggleTop", ""]].forEach(([selector, label]) => {
    const button = $(selector);
    if (!button) return;
    button.innerHTML = `<span data-icon="${dark ? "sun" : "moon"}"></span>${label ? ` ${label}` : ""}`;
    hydrateIcons(button);
  });
}

$("#themeToggle")?.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("queryStudioTheme", next);
  applyThemeButtons();
  document.querySelectorAll("article").forEach((article) => article._chartRerender?.());
  document.querySelectorAll(".chart-fullscreen").forEach((overlay) => overlay._chartRerender?.());
});
$("#themeToggleTop")?.addEventListener("click", () => $("#themeToggle").click());
applyThemeButtons();

const INSPECTOR_SECTIONS = {
  understanding: "#understandingSection",
  schema: "#schemaSection",
  recovery: "#recoverySection",
  pipeline: "#pipelineSection",
  tokens: "#tokensSection",
};

function selectInspectorTab(tab) {
  const selected = INSPECTOR_SECTIONS[tab] ? tab : "understanding";
  Object.entries(INSPECTOR_SECTIONS).forEach(([name, selector]) => {
    const section = $(selector);
    if (section) {
      section.hidden = name !== selected;
      if (name === selected) section.open = true;
    }
  });
  document.querySelectorAll("[data-inspector-tab]").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.inspectorTab === selected));
  });
}

function setInspectorOpen(open) {
  document.body.classList.toggle("inspector-open", open);
  $("#inspectorBackdrop").hidden = !open;
  $("#inspectorPane").setAttribute("aria-hidden", String(!open));
  $("#inspectorToggle").setAttribute("aria-expanded", String(open));
  if (open) $("#inspectorClose").focus();
  else $("#inspectorToggle").focus();
}

$("#inspectorToggle")?.addEventListener("click", () => {
  setInspectorOpen(!document.body.classList.contains("inspector-open"));
});
$("#inspectorClose")?.addEventListener("click", () => setInspectorOpen(false));
$("#inspectorBackdrop")?.addEventListener("click", () => setInspectorOpen(false));
document.querySelectorAll("[data-inspector-tab]").forEach((button) => {
  button.addEventListener("click", () => selectInspectorTab(button.dataset.inspectorTab));
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.body.classList.contains("inspector-open")) {
    setInspectorOpen(false);
  }
});
selectInspectorTab("understanding");

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
  if (state.busy || voiceInput?.busy || !state.ready) return;
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

document.addEventListener("pointerdown", event => {
  if (activeInsight && !activeInsight.contains(event.target)) dismissInsight();
});
$("#clearHistory").addEventListener("click", async () => {
  if (state.busy) return;
  if (!window.confirm("Clear all saved conversations in this session? This removes your conversation history.")) return;
  try {
    await api(`/api/conversations/${encodeURIComponent(sessionId)}/reset`, { method: "POST" });
    await api(`/api/history?session_id=${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  } catch (error) { showError(`Could not clear history: ${error.message}`); return; }
  state.history = [];
  state.chartSnapshots?.clear();
  state.activeRunId = null;
  state.branchFromRunId = null;
  state.restorePending = false;
  $("#messages").replaceChildren();
  $("#emptyState").hidden = false;
  $("#threadIntro").hidden = true;
  $("#threadBreadcrumb").textContent = "Live database";
  renderHistory();
});

$("#resetButton").addEventListener("click", async () => {
  if (state.busy) return;
  try {
    await api(`/api/conversations/${encodeURIComponent(sessionId)}/reset`, { method: "POST" });
    $("#messages").replaceChildren();
    state.lastBody = null;
    state.activeRunId = null;
    state.branchFromRunId = null;
    state.restorePending = false;
    $("#emptyState").hidden = false;
    $("#threadIntro").hidden = true;
    $("#threadBreadcrumb").textContent = "Live database";
    $("#mainColumn").scrollTop = 0;
    $("#messageInput").focus();
  } catch (error) { showError(error.message); }
});

initialize();

document.querySelectorAll(".side-nav a[data-view]").forEach(link => {
  link.addEventListener("click", event => {
    event.preventDefault();
    $("#messageInput").focus();
  });
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    $("#resetButton").click();
  }
  if (event.key === "Escape") closeChartFullscreen();
});
