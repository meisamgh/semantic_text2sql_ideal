/* Temporary, single-result dashboards. Backend owns every authoritative KPI.
   Existing renderer/table callbacks own rendering; this module never queries SQL. */
window.QueryflowDashboard = (() => {
  const sections = Object.freeze(["filters", "kpis", "charts", "data"]);
  function node(tag, className, text) {
    const element = document.createElement(tag);
    element.className = className;
    if (text != null) element.textContent = text;
    return element;
  }
  function render(
    container,
    spec,
    { renderChart, renderTable, onSelectWidget },
  ) {
    const sources = spec?.schema_version === 2 ? spec.sources : null;
    const known = (id) =>
      sources ? Boolean(sources[id]) : id === spec.result_id;
    const count = (id) => sources ? sources[id]?.row_count : spec.row_count;
    if (
      !spec?.complete || !spec.result_id || !Array.isArray(spec.widgets) ||
      !Number.isInteger(spec.row_count) || spec.row_count < 0 ||
      spec.widgets.length > 4 || (sources && !sources[spec.result_id]) ||
      spec.widgets.some((w) => !known(w.result_id)) ||
      spec.widgets.some((w) =>
        w.chart_spec?.source_row_count !== count(w.result_id)
      ) ||
      (spec.data && spec.data.rows.length !== spec.row_count) ||
      (spec.kpis || []).some((k) => !known(k.result_id))
    ) {
      throw new Error("The dashboard does not reference one complete result.");
    }
    container.replaceChildren();
    container.classList.add("dashboard-view");
    container.dataset.resultId = spec.result_id;
    container.dataset.theme = spec.theme || "green";
    container.setAttribute("aria-label", "Dashboard");
    const title = node(
      "h3",
      "dashboard-title",
      spec.title || "Result dashboard",
    );
    container.append(title);
    if (spec.description) {
      container.append(node("p", "dashboard-caption", spec.description));
    }
    const handles = [];
    try {
      for (const kind of sections) {
        if (kind === "data" && !spec.data) continue;
        const section = node("section", `dashboard-section dashboard-${kind}`);
        section.dataset.dashboardSection = kind;
        section.append(node(
          "h4",
          "dashboard-section-title",
          {
            filters: "Scope",
            kpis: "KPIs",
            charts: "Charts",
            data: "Data",
          }[kind],
        ));
        container.append(section);
        if (kind === "filters") {
          const constraints = sources ? [] : spec.applied_constraints || [];
          if (sources) {
            for (const source of Object.values(sources)) {
              section.append(
                node("p", "dashboard-scope-label", source.question),
              );
              if (source.execution_timestamp) {
                section.append(
                  node(
                    "small",
                    "dashboard-caption",
                    "Data as of " +
                      new Date(source.execution_timestamp * 1000)
                        .toLocaleString(),
                  ),
                );
              }
            }
            const lineage = node("details", "dashboard-rationale");
            lineage.append(node("summary", "dashboard-caption", "Details"));
            for (const source of Object.values(sources)) {
              lineage.append(
                node("h5", "dashboard-caption", source.question),
                node("p", "dashboard-caption", source.result_id),
              );
              if (source.source_sql) {
                lineage.append(node("pre", "", source.source_sql));
              } else {for (
                  const condition of source.applied_constraints || []
                ) lineage.append(node("p", "dashboard-caption", condition));}
            }
            section.append(lineage);
          }
          if (!sources) {
            section.append(node("p", "dashboard-caption", "Current analysis"));
          }
          if (constraints.length) {
            const details = node("details", "dashboard-rationale");
            details.append(
              node("summary", "dashboard-caption", "Scope details"),
            );
            for (const constraint of constraints) {
              details.append(node("span", "dashboard-constraint", constraint));
            }
            section.append(details);
          }
          if (!sources && spec.execution_timestamp) {
            section.append(
              node(
                "small",
                "dashboard-caption",
                "Data as of " +
                  new Date(spec.execution_timestamp * 1000).toLocaleString(),
              ),
            );
          }
          const fields = spec.verified_filter_metadata || [];
          if (fields.length) {
            section.append(node(
              "p",
              "dashboard-caption",
              fields.map((f) => f.field).join(" · "),
            ));
          }
        } else if (kind === "kpis") {
          const grid = node("div", "dashboard-kpi-grid");
          section.append(grid);
          const count = spec.kpis?.length || 0;
          grid.style.setProperty(
            "--dashboard-kpi-span",
            String(count === 1 ? 12 : count === 2 ? 6 : count === 3 ? 4 : 3),
          );
          for (const metric of spec.kpis || []) {
            const card = node("article", "chart-kpi-card dashboard-kpi");
            if (spec.design_mode === "ai") {
              card.style.setProperty(
                "--dashboard-kpi-span",
                String(metric.column_span),
              );
            }
            card.dataset.resultId = metric.result_id;
            card.dataset.widgetId = metric.kpi_id || "";
            const exact = window.QueryflowFormat?.formatNumber(
              metric.value,
              metric.unit,
              undefined,
              { exact: true },
            ) ?? String(metric.value);
            card.title = `Exact value: ${exact}. ${metric.definition}`;
            card.append(node("span", "kpi-card-title", metric.title));
            const value = window.QueryflowFormat?.formatNumber(
              metric.value,
              metric.unit,
              undefined,
              { compact: Math.abs(metric.value) >= 1000 },
            ) ?? String(metric.value);
            card.append(node("strong", "dashboard-kpi-value", value));
            const context = (metric.periods || []).join(" — ") || metric.unit ||
              "";
            if (context) {
              card.append(node("small", "dashboard-caption", context));
            }
            if (onSelectWidget && metric.kpi_id) {
              const edit = node("button", "dashboard-widget-edit", "Edit");
              edit.type = "button";
              edit.setAttribute("aria-label", "Edit " + metric.title);
              edit.addEventListener(
                "click",
                () => onSelectWidget(metric.kpi_id),
              );
              card.append(edit);
            }
            grid.append(card);
          }
          if (!spec.kpis?.length) {
            section.append(
              node(
                "p",
                "dashboard-empty",
                "There is not enough information to show a metric yet.",
              ),
            );
          }
        } else if (kind === "charts") {
          const grid = node("div", "dashboard-chart-grid");
          section.append(grid);
          for (const widget of spec.widgets) {
            const card = node("article", "dashboard-chart-card");
            card.style.setProperty(
              "--dashboard-span",
              String(
                [4, 6, 8, 12].includes(widget.column_span)
                  ? widget.column_span
                  : 6,
              ),
            );
            card.dataset.resultId = widget.result_id;
            card.dataset.widgetId = widget.widget_id;
            if (widget === spec.widgets[0]) {
              card.classList.add("dashboard-primary");
            }
            const chart = widget.chart_spec;
            card.append(
              node(
                "h5",
                "dashboard-chart-title",
                window.QueryflowChartTheme?.title(chart) || chart.title,
              ),
            );
            if (onSelectWidget) {
              const edit = node("button", "dashboard-widget-edit", "Edit");
              edit.type = "button";
              edit.setAttribute("aria-label", "Edit " + chart.title);
              edit.addEventListener(
                "click",
                () => onSelectWidget(widget.widget_id),
              );
              card.append(edit);
            }
            const slot = node("div", "chart-slot");
            card.append(slot);
            grid.append(card);
            const handle = renderChart(slot, { chart_spec: chart }, null)
              ?.echarts;
            if (handle) handles.push(handle);
          }
          if (!spec.widgets.length) {
            section.append(
              node(
                "p",
                "dashboard-empty",
                "No suitable chart is available for this analysis.",
              ),
            );
          }
        } else {
          const table = node("div", "dashboard-table");
          section.append(table);
          renderTable(table, spec.data.columns, spec.data.rows, {
            truncated: false,
          });
        }
      }
      if (spec.design_mode === "fallback") {
        container.append(
          node(
            "p",
            "dashboard-caption",
            "Some design changes could not be applied. Your data is unchanged.",
          ),
        );
      }
      if (spec.notices?.length) {
        const details = node("details", "dashboard-rationale");
        details.append(node("summary", "dashboard-caption", "Details"));
        for (const message of spec.notices) {
          details.append(node("p", "dashboard-caption", message));
        }
        container.append(details);
      }
      if (spec.rationale) {
        const details = node("details", "dashboard-rationale");
        details.append(
          node("summary", "dashboard-caption", "Design rationale"),
          node("p", "dashboard-caption", spec.rationale),
        );
        container.append(details);
      }
    } catch (error) {
      handles.forEach((handle) => handle.dispose?.());
      container.replaceChildren();
      throw error;
    }
    return {
      dispose() {
        handles.forEach((handle) => handle.dispose?.());
        container.replaceChildren();
      },
    };
  }
  function mountAction(article, body, dependencies) {
    const generation = body.generation;
    if (
      !body.dashboard || !body.result_id || !generation?.accepted ||
      generation.execution_status !== "ACCEPTED"
    ) return;
    const host = article.querySelector?.(".response-card") || article;
    const view = node("div", "dashboard-view"),
      edits = node("div", "dashboard-actions");
    const input = node("input", "dashboard-revision-input");
    input.placeholder = "How would you like to improve this dashboard?";
    input.setAttribute("aria-label", "Dashboard design change");
    const edit = node("button", "secondary-button", "Edit with AI");
    edit.type = "button";
    const clear = node("button", "secondary-button", "Whole dashboard");
    clear.type = "button";
    const undo = node("button", "secondary-button", "Undo");
    undo.type = "button";
    const status = node("span", "dashboard-status");
    status.setAttribute("role", "status");
    let spec = body.dashboard, selected = null, handle = null;
    const redraw = () => {
      handle?.dispose();
      handle = render(view, spec, {
        ...dependencies,
        onSelectWidget: (id) => {
          selected = id;
          input.placeholder = "Describe a change to this component…";
          input.focus?.();
        },
      });
    };
    edit.addEventListener("click", () => {
      if (!input.value.trim()) return;
      dependencies.reviseDashboard?.(
        input.value.trim(),
        {
          dashboard_id: spec.dashboard_id,
          revision_id: spec.revision_id,
          widget_id: selected,
        },
      );
    });
    clear.addEventListener("click", () => {
      selected = null;
      input.placeholder = "How would you like to improve this dashboard?";
    });
    undo.addEventListener("click", async () => {
      if (undo.disabled) return;
      undo.disabled = true;
      try {
        const restored = await dependencies.api("/api/dashboards/undo", {
          method: "POST",
          body: JSON.stringify({
            session_id: dependencies.sessionId,
            db_id: generation.db_id,
            result_id: body.result_id,
          }),
        });
        if (restored.result_id !== body.result_id) {
          throw new Error("Dashboard data changed; reopen it.");
        }
        spec = restored;
        selected = null;
        redraw();
        status.textContent = "";
      } catch (error) {
        status.textContent = error.message || "Undo is unavailable.";
      } finally {
        undo.disabled = false;
      }
    });
    edits.append(input, edit, clear, undo, status);
    host.append(view, edits);
    window.QueryflowDashboardStudio?.mountSave(edits, () => spec, {
      ...dependencies,
      body,
    });
    try {
      redraw();
      const oldViews = Array.from(
        host.querySelectorAll?.(
          ".response-tabs,.answer-tab-panel,.sql-result-split",
        ) || [],
      );
      if (oldViews.length) {
        const details = node("details", "dashboard-technical-details");
        details.append(
          node("summary", "dashboard-caption", "Analysis details"),
          ...oldViews,
        );
        host.append(details);
      }
      const execution = host.querySelector?.(".result-head");
      if (execution) execution.hidden = true;
    } catch (error) {
      view.hidden = true;
      status.textContent = error.message || "Dashboard unavailable.";
    }
  }
  return { render, mountAction, sections };
})();
