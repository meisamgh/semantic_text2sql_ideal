/* Permanent snapshots. Pure rendering never executes SQL or calls a model.
   Only an explicit conversational design request invokes the existing Top layer. */
window.QueryflowDashboardStudio = (() => {
  let dependencies = null,
    dialog = null,
    content = null,
    status = null,
    handle = null,
    busy = false;
  const node = (tag, cls, text) => {
    const e = document.createElement(tag);
    e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const button = (label, action) => {
    const e = node("button", "secondary-button", label);
    e.type = "button";
    e.addEventListener("click", action);
    return e;
  };
  async function action(operation) {
    if (busy) return;
    busy = true;
    status.textContent = "Working…";
    try {
      await operation();
      status.textContent = "";
    } catch (error) {
      status.textContent = error.message || "Dashboard unavailable.";
    } finally {
      busy = false;
    }
  }
  const request = (path, body, method = "POST") =>
    dependencies.api(path, { method, body: JSON.stringify(body) });
  function clear() {
    handle?.dispose();
    handle = null;
    content.replaceChildren();
  }
  function ensure() {
    if (dialog) return;
    dialog = node("dialog", "dashboard-studio");
    dialog.setAttribute("aria-label", "Dashboard Studio");
    const heading = node("header", "studio-heading");
    heading.append(
      node("h2", "", "Dashboard Studio"),
      button("Close", () => dialog.close()),
    );
    status = node("p", "dashboard-status");
    status.setAttribute("role", "status");
    content = node("div", "studio-content");
    dialog.append(heading, status, content);
    document.body.append(dialog);
    dialog.addEventListener("close", () => {
      handle?.dispose();
      handle = null;
    });
  }
  async function library() {
    ensure();
    if (!dialog.open) dialog.showModal();
    await action(async () => {
      const items = await dependencies.api("/api/studio");
      clear();
      content.append(
        node(
          "p",
          "dashboard-caption",
          "Saved snapshots · no automatic refresh · private workspace",
        ),
      );
      if (!items.length) {
        content.append(
          node(
            "p",
            "dashboard-empty",
            "No saved dashboards. Create a dashboard from an accepted result, then choose Save dashboard.",
          ),
        );
      }
      for (const item of items) {
        const row = node("div", "studio-library-row");
        row.append(
          button(item.name, () => action(() => open(item.id))),
          node(
            "span",
            "dashboard-caption",
            `${item.status} · Version ${item.version} · ${
              new Date(item.updated_at * 1000).toLocaleString()
            }`,
          ),
        );
        content.append(row);
      }
    });
  }
  async function open(identity, version = null, headVersion = null) {
    const saved = await dependencies.api(
      `/api/studio/${encodeURIComponent(identity)}${
        version == null ? "" : `?version=${version}`
      }`,
    );
    clear();
    let preview = null;
    const expected = headVersion ?? saved.version,
      historical = saved.version !== expected;
    const editable = saved.permission !== "viewer" && !historical;
    const toolbar = node("div", "dashboard-actions");
    const name = node("input", "dashboard-revision-input");
    name.value = saved.name;
    name.maxLength = 160;
    name.setAttribute("aria-label", "Dashboard name");
    const details = node(
      "p",
      "dashboard-caption",
      `${
        historical ? "Historical snapshot" : saved.status
      } · Version ${saved.version} · ${saved.owner}`,
    );
    const view = node("div", "dashboard-view");
    const source = saved.document.source;
    const evidence = node("details", "studio-source");
    evidence.append(
      node("summary", "", "Source evidence (saved, not refreshed)"),
      node(
        "p",
        "dashboard-caption",
        `${source.db_id} · Result ${source.result_id} · ${source.row_count} retained rows · ${
          source.execution_timestamp
            ? new Date(source.execution_timestamp * 1000).toLocaleString()
            : "Execution timestamp unavailable"
        }`,
      ),
      node("pre", "", source.sql),
    );
    const redraw = (spec) => {
      handle?.dispose();
      handle = window.QueryflowDashboard.render(view, spec, dependencies);
    };
    const change = (route, extra = {}) =>
      action(async () => {
        await request(`/api/studio/${identity}/${route}`, {
          expected_version: expected,
          ...extra,
        });
        await open(identity);
      });
    const save = button(
      "Save",
      () =>
        change("save", {
          preview_id: preview?.preview_id || null,
          name: name.value,
        }),
    );
    const publish = button("Approve & publish", () => {
      if (
        window.confirm(
          "Approve and publish this saved version? External sharing remains disabled.",
        )
      ) change("publish", { approve: true });
    });
    const editText = node("input", "dashboard-revision-input");
    editText.placeholder = "Describe a design change…";
    editText.setAttribute("aria-label", "Saved dashboard design change");
    const edit = button("Update preview", () =>
      action(async () => {
        if (!editText.value.trim()) return;
        const result = await request(`/api/studio/${identity}/edit`, {
          expected_version: expected,
          message: editText.value.trim(),
          preview_id: preview?.preview_id || null,
        });
        if (!result.preview_id) {
          throw new Error(
            result.message || "The requested design is unavailable.",
          );
        }
        preview = result;
        redraw(result.spec);
        publish.disabled = true;
        details.textContent =
          "Unsaved draft preview · choose Save to retain it";
      }));
    toolbar.append(
      button("Library", () => library()),
      name,
      button("Rename", () => change("rename", { name: name.value })),
      save,
      button("Save as New", () =>
        action(async () => {
          const copy = await request(`/api/studio/${identity}/save`, {
            expected_version: expected,
            preview_id: preview?.preview_id || null,
            name: name.value,
            save_as_new: true,
            source_version: historical ? saved.version : null,
          });
          await open(copy.dashboard_id);
        })),
      publish,
      button("Undo", () => change("undo")),
      button("Versions", () =>
        action(async () => {
          const revisions = await dependencies.api(
            `/api/studio/${identity}/versions`,
          );
          history.replaceChildren();
          for (const revision of revisions) {
            history.append(
              button(`Version ${revision.version} · ${revision.action}`, () =>
                action(() => open(identity, revision.version, expected))),
            );
          }
        })),
      button("Delete", () =>
        action(async () => {
          if (
            !window.confirm(
              "Permanently delete this dashboard and all its saved versions?",
            )
          ) return;
          await dependencies.api(
            `/api/studio/${identity}?expected_version=${expected}`,
            { method: "DELETE" },
          );
          // Avoid nesting the action lock while opening the list.
          clear();
          content.append(
            node("p", "dashboard-caption", "Dashboard deleted."),
            button("Library", () => library()),
          );
        })),
    );
    const history = node("div", "dashboard-actions");
    const edits = node("div", "dashboard-actions");
    edits.append(editText, edit);
    for (const control of [name, save, publish, editText, edit]) {
      control.disabled = !editable;
    }
    if (saved.permission !== "owner") publish.disabled = true;
    if (!editable) {
      toolbar.children[2].disabled = true;
      toolbar.children[6].disabled = true;
      toolbar.children[8].disabled = true;
    }
    if (saved.permission === "viewer") toolbar.children[4].disabled = true;
    if (saved.permission !== "owner") toolbar.children[8].disabled = true;
    content.append(toolbar, details, history, edits, view, evidence);
    redraw(saved.document.spec);
  }
  function install(options) {
    dependencies = options;
    document.getElementById("dashboardLibraryButton")?.addEventListener(
      "click",
      () => library(),
    );
  }
  function mountSave(host, getSpec, { api, sessionId, body }) {
    const status = node("span", "dashboard-status");
    status.setAttribute("role", "status");
    const save = button("Save dashboard", async () => {
      const spec = getSpec();
      if (!spec || save.disabled) return;
      const name = window.prompt(
        "Save a private dashboard snapshot as:",
        spec.title || "Dashboard",
      );
      if (!name?.trim()) return;
      save.disabled = true;
      status.textContent = "Saving…";
      try {
        const saved = await api("/api/studio", {
          method: "POST",
          body: JSON.stringify({
            session_id: sessionId,
            db_id: body.generation.db_id,
            result_id: body.result_id,
            design_revision_id: spec.revision_id,
            name: name.trim(),
          }),
        });
        status.textContent = "Saved as Draft";
        if (dependencies) {
          ensure();
          if (!dialog.open) dialog.showModal();
          await open(saved.dashboard_id);
        }
      } catch (error) {
        status.textContent = error.message || "Could not save dashboard.";
      } finally {
        save.disabled = false;
      }
    });
    host.append(save, status);
  }
  return { install, mountSave, library };
})();
