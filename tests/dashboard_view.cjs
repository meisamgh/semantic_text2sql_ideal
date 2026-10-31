const { test } = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), vm = require("node:vm");
const source = fs.readFileSync("web/dashboard_view.js", "utf8");
class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.hidden = false;
    this.style = {
      setProperty: (key, value) => {
        this.style[key] = value;
      },
    };
    this.classList = { add: () => {} };
    this.listeners = {};
    this.checked = false;
  }
  append(...items) {
    this.children.push(...items);
  }
  replaceChildren(...items) {
    this.children = [...items];
  }
  setAttribute(name, value) {
    this.attributes[name] = value;
  }
  addEventListener(name, callback) {
    this.listeners[name] = callback;
  }
  click() {
    return this.listeners.click?.();
  }
}
function setup() {
  const context = vm.createContext({
    window: {},
    document: {
      createElement: (tag) => new Element(tag),
      createTextNode: (text) => ({ textContent: text }),
    },
  });
  vm.runInContext(source, context);
  return context.window.QueryflowDashboard;
}
function spec(includeData = true) {
  return {
    result_id: "result",
    complete: true,
    row_count: 2,
    title: "Monthly revenue",
    applied_constraints: ["country = 'DE'"],
    kpis: [{
      result_id: "result",
      metric: "revenue",
      title: "Total",
      value: 300,
      definition: "Sum of disjoint returned groups",
      calculation: "TOTAL",
    }],
    widgets: [{
      widget_id: "chart-0",
      result_id: "result",
      column_span: 6,
      chart_spec: {
        chart_type: "line",
        x: "month",
        y: "revenue",
        title: "Revenue by month",
        source_row_count: 2,
        data: [{ month: "202601", revenue: 100 }, {
          month: "202602",
          revenue: 200,
        }],
      },
    }],
    data: includeData
      ? {
        columns: ["month", "revenue"],
        rows: [["202601", 100], ["202602", 200]],
      }
      : null,
    notices: ["Temporary view of one retained result. Filters are read-only."],
  };
}

function walk(e) {
  return [e, ...e.children.filter((x) => x.children).flatMap(walk)];
}
function find(e, text) {
  const button = walk(e).find((x) => x.textContent === text);
  assert.ok(button, text);
  return button;
}
function body(data = spec()) {
  return {
    result_id: "result",
    dashboard: data,
    generation: {
      accepted: true,
      execution_status: "ACCEPTED",
      db_id: "sales",
      rows: [[1], [2]],
      row_count: 2,
    },
  };
}
test("sections preserve authoritative KPI values, chart data and table rows", () => {
  const view = setup(),
    container = new Element("div"),
    data = spec(),
    before = JSON.stringify(data);
  let charts = 0, tables = 0, disposed = 0;
  const handle = view.render(container, data, {
    renderChart(slot, presentation) {
      charts++;
      assert.equal(presentation.chart_spec, data.widgets[0].chart_spec);
      return {
        echarts: {
          dispose() {
            disposed++;
          },
        },
      };
    },
    renderTable(slot, columns, rows) {
      tables++;
      assert.equal(rows, data.data.rows);
    },
  });
  assert.deepEqual(
    container.children.filter((n) => n.dataset.dashboardSection).map((n) =>
      n.dataset.dashboardSection
    ),
    ["filters", "kpis", "charts", "data"],
  );
  assert.ok(walk(container).some((e) => e.textContent === "300"));
  assert.equal(charts, 1);
  assert.equal(tables, 1);
  assert.equal(JSON.stringify(data), before);
  handle.dispose();
  assert.equal(disposed, 1);
});
test("ordinary analytical answers have no dashboard creation controls or calls", () => {
  const article = new Element("article");
  let calls = 0;
  const response = body();
  delete response.dashboard;
  setup().mountAction(article, response, {
    api() {
      calls++;
    },
  });
  assert.equal(article.children.length, 0);
  assert.equal(calls, 0);
  assert.equal(source.includes('"Create dashboard"'), false);
  assert.equal(source.includes('"Verified layout"'), false);
});
test("validated dashboards appear automatically with zero API or LLM calls", () => {
  const article = new Element("article");
  let charts = 0;
  setup().mountAction(article, body(), {
    api() {
      throw Error("No request");
    },
    renderChart() {
      charts++;
    },
    renderTable() {},
  });
  assert.equal(charts, 1);
  assert.equal(article.children[0].hidden, false);
  assert.ok(find(article, "Edit with AI"));
});
test("component selection passes dashboard, revision and widget identity to the existing agent", () => {
  const article = new Element("article"), data = spec(false), messages = [];
  data.dashboard_id = "dashboard";
  data.revision_id = "revision";
  data.widgets[0].widget_id = "widget";
  setup().mountAction(article, body(data), {
    renderChart() {},
    renderTable() {},
    reviseDashboard: (message, target) => messages.push([message, target]),
  });
  const input = walk(article).find((e) =>
    e.attributes?.["aria-label"] === "Dashboard design change"
  );
  const select = walk(article).find((e) =>
    e.attributes?.["aria-label"] === "Edit Revenue by month"
  );
  assert.ok(select);
  select.click();
  input.value = "Make this larger";
  find(article, "Edit with AI").click();
  assert.equal(messages[0][0], "Make this larger");
  assert.deepEqual({ ...messages[0][1] }, {
    dashboard_id: "dashboard",
    revision_id: "revision",
    widget_id: "widget",
  });
  find(article, "Whole dashboard").click();
  find(article, "Edit with AI").click();
  assert.equal(messages[1][1].widget_id, null);
  assert.equal(messages[1][1].dashboard_id, "dashboard");
});
test("undo is scoped to the response and never generates SQL", async () => {
  const article = new Element("article"), calls = [];
  setup().mountAction(article, body(spec(false)), {
    sessionId: "owner",
    renderChart() {},
    renderTable() {},
    api: async (path, request) => {
      calls.push([path, JSON.parse(request.body)]);
      return spec(false);
    },
  });
  await find(article, "Undo").click();
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], "/api/dashboards/undo");
  assert.deepEqual(calls[0][1], {
    session_id: "owner",
    db_id: "sales",
    result_id: "result",
  });
});
test("failed undo retains the existing dashboard and shows an actionable message", async () => {
  const article = new Element("article");
  let charts = 0;
  setup().mountAction(article, body(spec(false)), {
    renderChart() {
      charts++;
    },
    renderTable() {},
    api: async () => {
      throw Error("Reopen the latest dashboard");
    },
  });
  await find(article, "Undo").click();
  assert.equal(charts, 1);
  assert.ok(
    walk(article).some((e) => e.textContent === "Reopen the latest dashboard"),
  );
});
test("multi-result charts and KPIs keep separate source identities and row counts", () => {
  const data = spec(false);
  data.schema_version = 2;
  data.db_id = "sales";
  data.sources = {
    result: {
      result_id: "result",
      db_id: "sales",
      row_count: 2,
      question: "Revenue",
    },
    customers: {
      result_id: "customers",
      db_id: "sales",
      row_count: 1,
      question: "Customers",
    },
  };
  data.kpis[0].result_id = "customers";
  const container = new Element("div");
  setup().render(container, data, { renderChart() {}, renderTable() {} });
  const metric = walk(container).find((e) =>
    e.className === "chart-kpi-card dashboard-kpi"
  );
  assert.equal(metric.dataset.resultId, "customers");
  assert.ok(walk(container).some((e) => e.textContent === "Customers"));
});
test("unknown widget sources and mismatched counts are rejected", () => {
  const data = spec(false);
  data.widgets[0].result_id = "wrong";
  assert.throws(
    () => setup().render(new Element("div"), data, {}),
    /complete result/,
  );
  const other = spec(false);
  other.widgets[0].chart_spec.source_row_count = 99;
  assert.throws(
    () => setup().render(new Element("div"), other, {}),
    /complete result/,
  );
});
test("empty results remain empty without invented KPIs or charts", () => {
  const data = spec(false);
  data.kpis = [];
  data.widgets = [];
  data.row_count = 0;
  const container = new Element("div");
  setup().render(container, data, {
    renderChart() {
      throw Error("No chart");
    },
    renderTable() {},
  });
  assert.ok(walk(container).some((e) => e.className === "dashboard-empty"));
});
test("partial renderer failure disposes earlier renderer resources", () => {
  const data = spec(false);
  data.widgets.push({ ...data.widgets[0] });
  let count = 0, disposed = 0;
  assert.throws(() =>
    setup().render(new Element("div"), data, {
      renderChart() {
        if (++count === 2) throw Error("Renderer unavailable");
        return {
          echarts: {
            dispose() {
              disposed++;
            },
          },
        };
      },
    })
  );
  assert.equal(disposed, 1);
});
test("dashboard rendering stays inside its own response card", () => {
  const article = new Element("article"), card = new Element("div");
  article.querySelector = () => card;
  setup().mountAction(article, body(spec(false)), {
    renderChart() {},
    renderTable() {},
  });
  assert.equal(article.children.length, 0);
  assert.equal(card.children.length, 2);
});
test("responsive layout reuses existing rendering engines without new frameworks", () => {
  const css = fs.readFileSync("web/styles.css", "utf8");
  assert.ok(css.includes("grid-template-columns:repeat(12,minmax(0,1fr))"));
  assert.ok(css.includes(".dashboard-chart-card { grid-column:span 12;"));
  assert.equal(source.includes("Plotly.newPlot"), false);
  assert.equal(source.includes("echarts.init"), false);
});
