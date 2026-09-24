/* Declarative, data-bound charts: never execute model-generated markup or code. */
function renderResultChart(container, presentation, generation) {
  if (!presentation || !container) return;
  const note = document.createElement('p');
  note.className = 'chart-note';
  note.textContent = presentation.note;
  container.append(note);
  if (!['bar', 'line', 'scatter'].includes(presentation.chart_type)) return;
  const columns = generation.columns || [];
  const xi = columns.indexOf(presentation.x), yi = columns.indexOf(presentation.y);
  if (xi < 0 || yi < 0 || xi === yi) return;
  const rows = (generation.rows || []).slice(0, 40);
  if (presentation.chart_type === 'line') rows.sort((a, b) => String(a[xi]).localeCompare(String(b[xi])));
  const valid = rows.filter(r => typeof r[yi] === 'number' && Number.isFinite(r[yi]));
  if (valid.length < 2) return;
  const figure = document.createElement('figure');
  figure.className = 'result-chart';
  const title = document.createElement('figcaption');
  title.textContent = presentation.title || `${presentation.y} by ${presentation.x}`;
  figure.append(title);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 760 380');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', title.textContent);
  function mark(tag, attrs, text) {
    const el = document.createElementNS(svg.namespaceURI, tag);
    Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k, String(v)));
    if (text !== undefined) el.textContent = text;
    svg.append(el); return el;
  }
  const lo = Math.min(0, ...valid.map(r => r[yi])), hi = Math.max(0, ...valid.map(r => r[yi]));
  const y = v => 300 - (v - lo) / (hi - lo || 1) * 260;
  function temporal(value) {
    const text = String(value);
    const iso = /^\d{6}$/.test(text) ? `${text.slice(0,4)}-${text.slice(4)}-01`
      : text.length === 4 ? `${text}-01-01` : text.length === 7 ? `${text}-01` : text;
    return Date.parse(`${iso}T00:00:00Z`);
  }
  const xs = valid.map(r => presentation.chart_type === 'line' ? temporal(r[xi]) : r[xi]);
  const xmin = Math.min(...xs.map(Number)), xmax = Math.max(...xs.map(Number));
  const x = (r, i) => presentation.chart_type === 'scatter' || presentation.chart_type === 'line'
    ? 90 + ((presentation.chart_type === 'line' ? temporal(r[xi]) : r[xi]) - xmin) / (xmax - xmin || 1) * 620
    : 90 + (i + 0.5) / rows.length * 620;
  for (let i=0; i<=4; i++) {
    const v = lo + (hi-lo)*i/4;
    mark('line', {x1:80,x2:720,y1:y(v),y2:y(v),stroke:'#e2e8f0'});
    mark('text', {x:72,y:y(v)+4,'text-anchor':'end',fill:'#475569','font-size':11}, v.toLocaleString(undefined,{maximumFractionDigits:2}));
  }
  let previous = null;
  rows.forEach((r,i) => {
    if (typeof r[yi] !== 'number' || !Number.isFinite(r[yi])) { previous=null; return; }
    const px=x(r,i), py=y(r[yi]);
    if (presentation.chart_type === 'line' && previous) mark('line',{x1:previous[0],y1:previous[1],x2:px,y2:py,stroke:'#2563eb','stroke-width':2});
    const point = presentation.chart_type === 'bar'
      ? mark('rect',{x:px-220/rows.length,y:Math.min(py,y(0)),width:440/rows.length,height:Math.max(1,Math.abs(y(0)-py)),fill:'#2563eb'})
      : mark('circle',{cx:px,cy:py,r:4,fill:'#2563eb'});
    const tooltip=document.createElementNS(svg.namespaceURI,'title');
    tooltip.textContent=`${r[xi]}: ${r[yi]}`; point.append(tooltip);
    if (i % Math.max(1,Math.ceil(rows.length/8)) === 0) mark('text',{x:px,y:322,'text-anchor':'middle',fill:'#475569','font-size':11},String(r[xi]).slice(0,18));
    previous=[px,py];
  });
  mark('text',{x:400,y:363,'text-anchor':'middle',fill:'#334155','font-size':13},presentation.x);
  mark('text',{x:80,y:20,fill:'#334155','font-size':13},presentation.y);
  figure.append(svg);
  if (presentation.chart_reason) {
    const reason = document.createElement('p');
    reason.className = 'chart-reason';
    reason.textContent = presentation.chart_reason;
    figure.append(reason);
  }
  container.append(figure);
}
