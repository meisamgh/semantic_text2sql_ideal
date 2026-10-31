/* Temporary, single-result dashboards. Backend owns every authoritative KPI.
   Existing renderer/table callbacks own rendering; this module never queries SQL. */
window.QueryflowDashboard = (() => {
  const sections = Object.freeze(["filters", "kpis", "charts", "data"]);
  function node(tag, className, text) {
    const element=document.createElement(tag);element.className=className;
    if(text!=null) element.textContent=text;
    return element;
  }
  function render(container, spec, {renderChart,renderTable}) {
    if(!spec?.complete || !spec.result_id || !Array.isArray(spec.widgets)
      || !Number.isInteger(spec.row_count) || spec.row_count<0
      || spec.widgets.length>4 || spec.widgets.some(w=>w.result_id!==spec.result_id)
      || spec.widgets.some(w=>w.chart_spec?.source_row_count!==spec.row_count)
      || (spec.data && spec.data.rows.length!==spec.row_count)
      || (spec.kpis||[]).some(k=>k.result_id!==spec.result_id))
      throw new Error("The dashboard does not reference one complete result.");
    container.replaceChildren();container.classList.add("dashboard-view");
    container.dataset.resultId=spec.result_id;
    container.dataset.theme=spec.theme||"green";
    container.setAttribute("aria-label","Temporary result dashboard");
    const title=node("h3","dashboard-title",spec.title || "Result dashboard");
    container.append(title);
    if(spec.description) container.append(node("p","dashboard-caption",spec.description));
    const handles=[];
    try {
    for(const kind of sections){
      if(kind==="data" && !spec.data) continue;
      const section=node("section",`dashboard-section dashboard-${kind}`);
      section.dataset.dashboardSection=kind;
      section.append(node("h4","dashboard-section-title",{
        filters:"Filters · read-only",kpis:"KPIs and metrics",charts:"Analytical charts",data:"Source data",
      }[kind]));
      container.append(section);
      if(kind==="filters"){
        const constraints=spec.applied_constraints||[];
        if(!constraints.length) section.append(node("p","dashboard-empty","No explicit SQL filters."));
        for(const constraint of constraints) section.append(node("span","dashboard-constraint",constraint));
        const fields=spec.verified_filter_metadata||[];
        if(fields.length) section.append(node("p","dashboard-caption",
          `Verified filter fields: ${[...new Set(fields.map(f=>f.field))].join(", ")}. Editing is not enabled.`));
      } else if(kind==="kpis"){
        const grid=node("div","dashboard-kpi-grid");section.append(grid);
        const count=spec.kpis?.length||0;
        grid.style.setProperty("--dashboard-kpi-span",String(count===1?12:count===2?6:count===3?4:3));
        for(const metric of spec.kpis||[]){
          const card=node("article","chart-kpi-card dashboard-kpi");
          if(spec.design_mode==="ai") card.style.setProperty("--dashboard-kpi-span",String(metric.column_span));
          card.dataset.resultId=spec.result_id;
          const exact=window.QueryflowFormat?.formatNumber(metric.value,metric.unit,undefined,{exact:true}) ?? String(metric.value);
          card.title=`Exact value: ${exact}. ${metric.definition}`;
          card.append(node("span","kpi-card-title",metric.title));
          const value=window.QueryflowFormat?.formatNumber(metric.value,metric.unit,undefined,
            {compact:Math.abs(metric.value)>=1000}) ?? String(metric.value);
          card.append(node("strong","dashboard-kpi-value",value));
          card.append(node("small","dashboard-caption",metric.definition));
          grid.append(card);
        }
        if(!spec.kpis?.length) section.append(node("p","dashboard-empty",
          "No KPI calculation can be justified from this result's definitions and grain."));
      } else if(kind==="charts"){
        const grid=node("div","dashboard-chart-grid");section.append(grid);
        for(const widget of spec.widgets){
          const card=node("article","dashboard-chart-card");
          card.style.setProperty("--dashboard-span",String([4,6,8,12].includes(widget.column_span)?widget.column_span:6));
          card.dataset.resultId=widget.result_id;
          const chart=widget.chart_spec;
          card.append(node("h5","dashboard-chart-title",window.QueryflowChartTheme?.title(chart)||chart.title));
          const slot=node("div","chart-slot");card.append(slot);grid.append(card);
          const handle=renderChart(slot,{chart_spec:chart},null)?.echarts;
          if(handle) handles.push(handle);
        }
        if(!spec.widgets.length) section.append(node("p","dashboard-empty",
          "This result cannot support a validated analytical chart."));
      } else {
        const table=node("div","dashboard-table");section.append(table);
        renderTable(table,spec.data.columns,spec.data.rows,{truncated:false});
      }
    }
    for(const message of spec.notices||[]) container.append(node("p","dashboard-caption",message));
    if(spec.rationale){const details=node("details","dashboard-rationale");
      details.append(node("summary","dashboard-caption","Design rationale"),node("p","dashboard-caption",spec.rationale));container.append(details);}
    } catch(error){
      handles.forEach(handle=>handle.dispose?.());container.replaceChildren();throw error;
    }
    return {dispose(){handles.forEach(handle=>handle.dispose?.());container.replaceChildren();}};
  }
  function mountAction(article, body, dependencies) {
    const generation=body.generation;
    if(!body.result_id || !generation?.accepted || generation.execution_status!=="ACCEPTED") return;
    const toolbar=node("div","dashboard-actions");
    const button=node("button","secondary-button","Create dashboard");button.type="button";
    const complete=!generation.truncated && generation.rows?.length===generation.row_count;
    button.disabled=!complete;
    if(!complete) button.title="A dashboard requires the complete retained result, not a preview.";
    button.setAttribute("aria-expanded","false");
    const label=node("label","dashboard-table-option");
    const checkbox=document.createElement("input");checkbox.type="checkbox";
    checkbox.disabled=!complete;
    label.append(checkbox,document.createTextNode(" Include source table"));
    const status=node("span","dashboard-status");status.setAttribute("role","status");
    const mode=document.createElement("select");mode.setAttribute("aria-label","Dashboard design mode");
    for(const [value,text] of [["deterministic","Verified layout"],["ai","AI design"]]){
      const option=document.createElement("option");option.value=value;option.textContent=text;mode.append(option);
    }
    mode.value="deterministic";mode.disabled=!complete;
    const revision=node("input","dashboard-revision-input");revision.placeholder="Describe a dashboard change…";
    revision.setAttribute("aria-label","Dashboard design change");
    const revise=node("button","secondary-button","Update design");revise.type="button";
    const undo=node("button","secondary-button","Undo design");undo.type="button";
    const edits=node("div","dashboard-actions");edits.hidden=true;edits.append(revision,revise,undo);
    const view=node("div","dashboard-view");view.hidden=true;
    let spec=null,handle=null,loading=false;
    const host=article.querySelector?.(".response-card") || article;
    toolbar.append(button,label,status,mode);host.append(toolbar,view,edits);
    window.QueryflowDashboardStudio?.mountSave(edits,()=>spec,{...dependencies,body});
    revise.addEventListener("click",()=>{
      if(revision.value.trim() && dependencies.reviseDashboard) dependencies.reviseDashboard(revision.value.trim());
    });
    undo.addEventListener("click",async()=>{
      if(loading)return;loading=true;undo.disabled=true;
      try{
        const restored=await dependencies.api("/api/dashboards/undo",{method:"POST",body:JSON.stringify({
          session_id:dependencies.sessionId,db_id:generation.db_id,result_id:body.result_id,
        })});
        if(restored.result_id!==body.result_id)throw new Error("Dashboard source mismatch.");
        handle?.dispose();spec=restored;handle=render(view,spec,dependencies);
      }catch(error){status.textContent=error.message||"Undo unavailable.";}
      finally{loading=false;undo.disabled=false;}
    });
    button.addEventListener("click",async()=>{
      if(loading || !complete) return;
      if(!view.hidden){
        handle?.dispose();handle=null;view.hidden=true;edits.hidden=true;
        button.textContent="Open dashboard";button.setAttribute("aria-expanded","false");return;
      }
      loading=true;button.disabled=true;status.textContent="Preparing dashboard…";
      try {
        if(!spec) spec=await dependencies.api("/api/dashboards",{method:"POST",body:JSON.stringify({
          session_id:dependencies.sessionId,db_id:generation.db_id,result_id:body.result_id,
          include_data:checkbox.checked,
          design_mode:mode.value,
        })});
        if(spec.result_id!==body.result_id) throw new Error("Dashboard source does not match this result.");
        if(article.isConnected===false) return;
        view.hidden=false;handle=render(view,spec,dependencies);checkbox.disabled=true;mode.disabled=true;
        edits.hidden=false;
        button.textContent="Close dashboard";button.setAttribute("aria-expanded","true");
        status.textContent="";
      } catch(error){
        handle?.dispose();handle=null;view.hidden=true;spec=null;checkbox.disabled=false;
        status.textContent=error.message||"Dashboard unavailable.";
      }
      finally{loading=false;button.disabled=false;}
    });
    if(body.dashboard){spec=body.dashboard;button.click();}
  }
  return {render,mountAction,sections};
})();
