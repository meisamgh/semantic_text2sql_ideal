const { test } = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), vm = require("node:vm");
class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.listeners = {};
    this.dataset = {};
    this.style = { setProperty: (key, value) => this.style[key] = value };
    this.classList = { add() {} };
    this.value = "";
    this.open = false;
    this.disabled = false;
  }
  append(...items) {
    this.children.push(...items);
  }
  replaceChildren(...items) {
    this.children = [...items];
  }
  setAttribute(key, value) {
    this[key] = value;
  }
  addEventListener(name, callback) {
    this.listeners[name] = callback;
  }
  click() {
    if (!this.disabled) return this.listeners.click?.();
  }
  showModal() {
    this.open = true;
  }
  close() {
    this.open = false;
    this.listeners.close?.();
  }
}
function all(element) {
  return [element, ...element.children.flatMap(all)];
}
function find(root, text) {
  const result = all(root).find((e) =>
    e.tagName === "button" && e.textContent === text
  );
  assert.ok(result, `Missing ${text}`);
  return result;
}
function saved() {
  return {
    dashboard_id: "saved",
    owner: "local-workspace",
    name: "Revenue",
    version: 1,
    status: "Draft",
    permission: "owner",
    document: {
      source: {
        db_id: "sales",
        result_id: "result",
        sql: "SELECT month,revenue FROM result",
        row_count: 2,
      },
      spec: {
        result_id: "result",
        complete: true,
        row_count: 2,
        title: "Revenue",
        revision_id: "rev-1",
        theme: "green",
        kpis: [],
        widgets: [{
          result_id: "result",
          column_span: 12,
          chart_spec: {
            title: "Revenue",
            source_row_count: 2,
            chart_type: "line",
            x: "month",
            y: "revenue",
            data: [{ month: "Jan", revenue: 1 }, { month: "Feb", revenue: 2 }],
          },
        }],
        applied_constraints: [],
        notices: [],
      },
    },
  };
}
function setup(api) {
  const body = new Element("body"), library = new Element("button");
  let renders = 0;
  const window = { confirm: () => true, prompt: () => "My dashboard" };
  const context = vm.createContext({
    window,
    document: {
      body,
      createElement: (tag) => new Element(tag),
      getElementById: () => library,
    },
  });
  vm.runInContext(fs.readFileSync("web/dashboard_view.js", "utf8"), context);
  vm.runInContext(fs.readFileSync("web/dashboard_studio.js", "utf8"), context);
  const dependencies = {
    api,
    renderChart() {
      renders++;
      return { echarts: { dispose() {} } };
    },
    renderTable() {},
  };
  window.QueryflowDashboardStudio.install(dependencies);
  return {
    body,
    library,
    window,
    renders: () => renders,
    studio: window.QueryflowDashboardStudio,
  };
}
test("library reopening renders saved evidence with only GET requests", async () => {
  const calls = [], record = saved();
  const env = setup(async (path, options) => {
    calls.push([path, options]);
    assert.equal(options, undefined);
    return path === "/api/studio"
      ? [{
        id: "saved",
        name: "Revenue",
        status: "Draft",
        version: 1,
        updated_at: 1,
      }]
      : record;
  });
  await env.library.click();
  await find(env.body, "Revenue").click();
  assert.deepEqual(calls.map((c) => c[0]), [
    "/api/studio",
    "/api/studio/saved",
  ]);
  assert.equal(env.renders(), 1);
  const sections = all(env.body).filter((e) => e.dataset.dashboardSection).map(
    (e) => e.dataset.dashboardSection,
  );
  assert.deepEqual(sections, ["filters", "kpis", "charts"]);
  assert.ok(
    all(env.body).some((e) => e.textContent === record.document.source.sql),
  );
  await find(env.body, "Close").click();
  assert.equal(env.body.children[0].open, false);
});
test("saving sends a validated revision reference, not SQL rows or a client spec", async () => {
  const env = setup(async () => saved()), host = new Element("div"), calls = [];
  env.studio.mountSave(host, () => saved().document.spec, {
    sessionId: "owner",
    body: { result_id: "result", generation: { db_id: "sales" } },
    api: async (path, options) => {
      calls.push([path, JSON.parse(options.body)]);
      return saved();
    },
  });
  await find(host, "Save dashboard").click();
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], "/api/studio");
  assert.deepEqual(Object.keys(calls[0][1]).sort(), [
    "db_id",
    "design_revision_id",
    "name",
    "result_id",
    "session_id",
  ]);
  assert.equal(calls[0][1].design_revision_id, "rev-1");
  assert.equal(calls[0][1].owner, undefined);
});
test("editing is an explicit unsaved preview; save and publish require separate actions", async () => {
  let record = saved();
  const posts = [];
  const env = setup(async (path, options) => {
    if (!options) {
      return path === "/api/studio"
        ? [{
          id: "saved",
          name: "Revenue",
          status: "Draft",
          version: 1,
          updated_at: 1,
        }]
        : record;
    }
    const payload = JSON.parse(options.body);
    posts.push([path, payload]);
    if (path.endsWith("/edit")) {
      return {
        preview_id: "preview",
        spec: { ...record.document.spec, theme: "contrast" },
      };
    }
    if (path.endsWith("/save")) {
      record = {
        ...record,
        version: 2,
        document: {
          ...record.document,
          spec: { ...record.document.spec, theme: "contrast" },
        },
      };
      return record;
    }
    if (path.endsWith("/publish")) {
      record = { ...record, version: 3, status: "Published" };
      return record;
    }
    throw Error("Unexpected request");
  });
  await env.library.click();
  await find(env.body, "Revenue").click();
  const input = all(env.body).find((e) =>
    e["aria-label"] === "Saved dashboard design change"
  );
  input.value = "Higher contrast";
  await find(env.body, "Update preview").click();
  assert.equal(record.version, 1);
  assert.equal(find(env.body, "Approve & publish").disabled, true);
  assert.equal(posts.length, 1);
  await find(env.body, "Save").click();
  assert.equal(posts[1][1].preview_id, "preview");
  assert.equal(posts[1][1].expected_version, 1);
  env.window.confirm = () => false;
  await find(env.body, "Approve & publish").click();
  assert.equal(posts.length, 2);
  env.window.confirm = () => true;
  await find(env.body, "Approve & publish").click();
  // The click handler starts an asynchronous action; yield its immediate mock awaits.
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(posts[2][1].approve, true);
  assert.equal(posts[2][1].expected_version, 2);
});
test("viewer controls are read-only and the studio retains responsive layout", async () => {
  const record = { ...saved(), permission: "viewer" };
  const env = setup(async (path) =>
    path === "/api/studio"
      ? [{
        id: "saved",
        name: "Revenue",
        status: "Draft",
        version: 1,
        updated_at: 1,
      }]
      : record
  );
  await env.library.click();
  await find(env.body, "Revenue").click();
  for (
    const label of [
      "Rename",
      "Save",
      "Save as New",
      "Undo",
      "Delete",
      "Approve & publish",
      "Update preview",
    ]
  ) assert.equal(find(env.body, label).disabled, true, label);
  const css = fs.readFileSync("web/styles.css", "utf8");
  assert.ok(css.includes("width:min(1120px,94vw)"));
  assert.ok(css.includes(".dashboard-studio { width:96vw;"));
  assert.ok(css.includes("grid-template-columns:repeat(12,minmax(0,1fr))"));
});
