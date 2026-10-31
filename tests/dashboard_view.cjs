const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('web/dashboard_view.js','utf8');
class Element{
  constructor(tag){this.tagName=tag;this.children=[];this.attributes={};this.dataset={};this.hidden=false;
    this.style={setProperty:(key,value)=>{this.style[key]=value;}};
    this.classList={add:()=>{}};this.listeners={};this.checked=false;}
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=[...items];}
  setAttribute(name,value){this.attributes[name]=value;}
  addEventListener(name,callback){this.listeners[name]=callback;}
  click(){return this.listeners.click?.();}
}
function setup(){
  const context=vm.createContext({window:{},document:{createElement:tag=>new Element(tag),
    createTextNode:text=>({textContent:text})}});
  vm.runInContext(source,context);return context.window.QueryflowDashboard;
}
function spec(includeData=true){return {result_id:'result',complete:true,row_count:2,title:'Monthly revenue',
  applied_constraints:["country = 'DE'"],kpis:[{result_id:'result',metric:'revenue',title:'Total',
    value:300,definition:'Sum of disjoint returned groups',calculation:'TOTAL'}],
  widgets:[{widget_id:'chart-0',result_id:'result',column_span:6,chart_spec:{chart_type:'line',
    x:'month',y:'revenue',title:'Revenue by month',source_row_count:2,
    data:[{month:'202601',revenue:100},{month:'202602',revenue:200}]}}],
  data:includeData?{columns:['month','revenue'],rows:[['202601',100],['202602',200]]}:null,
  notices:['Temporary view of one retained result. Filters are read-only.']};}
test('section order, backend KPI values and exact source data use existing callbacks',()=>{
  const view=setup(),container=new Element('div'),data=spec(),before=JSON.stringify(data);
  let charts=0,tables=0,disposals=0;
  const handle=view.render(container,data,{
    renderChart(slot,presentation){charts++;assert.equal(presentation.chart_spec,data.widgets[0].chart_spec);
      return {echarts:{dispose(){disposals++;}}};},
    renderTable(slot,columns,rows){tables++;assert.equal(rows,data.data.rows);assert.equal(columns,data.data.columns);},
  });
  assert.deepEqual(container.children.filter(n=>n.dataset.dashboardSection).map(n=>n.dataset.dashboardSection),
    ['filters','kpis','charts','data']);
  const kpis=container.children.find(n=>n.dataset.dashboardSection==='kpis');
  const card=kpis.children[1].children[0];
  assert.equal(card.children[1].textContent,'300');
  assert.ok(card.title.includes('300'));
  assert.equal(charts,1);assert.equal(tables,1);
  assert.equal(JSON.stringify(data),before);
  handle.dispose();assert.equal(disposals,1);assert.equal(container.children.length,0);
});
test('AI design, scoped undo and collaborative changes reuse existing actions',async()=>{
  const article=new Element('article'),calls=[],messages=[];
  const data=spec(false);data.design_mode='ai';data.revision_id='revision-a';
  const body={result_id:'result',generation:{accepted:true,execution_status:'ACCEPTED',
    db_id:'sales',rows:[[1],[2]],row_count:2,truncated:false}};
  setup().mountAction(article,body,{
    sessionId:'owner',api:async(path,request)=>{calls.push([path,JSON.parse(request.body)]);return data;},
    reviseDashboard:message=>messages.push(message),renderChart(){},renderTable(){},
  });
  const [toolbar,view,edits]=article.children;
  toolbar.children[3].value='ai';
  await toolbar.children[0].click();
  assert.equal(calls[0][0],'/api/dashboards');assert.equal(calls[0][1].design_mode,'ai');
  assert.equal(view.dataset.theme,'green');assert.equal(edits.hidden,false);
  edits.children[0].value='Make the primary chart wider';edits.children[1].click();
  assert.deepEqual(messages,['Make the primary chart wider']);
  await edits.children[2].click();assert.equal(calls[1][0],'/api/dashboards/undo');
  assert.equal(calls.length,2);
});
test('a validated dashboard response opens without another API, SQL or model call',async()=>{
  const article=new Element('article'),data=spec(false);
  setup().mountAction(article,{result_id:'result',dashboard:data,
    generation:{accepted:true,execution_status:'ACCEPTED',rows:[[1],[2]],row_count:2}},
    {api(){throw Error('No request expected');},renderChart(){},renderTable(){}});
  assert.equal(article.children[1].hidden,false);
});
test('empty state has no invented KPI, chart, data table or filter controls',()=>{
  const data=spec(false);data.kpis=[];data.widgets=[];data.applied_constraints=[];
  data.row_count=0;
  const container=new Element('div');
  setup().render(container,data,{renderChart(){throw Error('No chart expected');},
    renderTable(){throw Error('No table expected');}});
  assert.deepEqual(container.children.filter(n=>n.dataset.dashboardSection).map(n=>n.dataset.dashboardSection),
    ['filters','kpis','charts']);
  assert.ok(container.children.find(n=>n.dataset.dashboardSection==='filters').children[1].textContent.includes('No explicit'));
});
test('invalid source result cannot render mixed-result dashboards',()=>{
  const data=spec();data.widgets[0].result_id='wrong';
  assert.throws(()=>setup().render(new Element('div'),data,{}),/one complete result/);
});
test('explicit create, close and reopen need only one dashboard request, zero query/model calls',async()=>{
  const view=setup(),article=new Element('article');let requests=0,renders=0,disposals=0;
  view.mountAction(article,{result_id:'result',generation:{accepted:true,execution_status:'ACCEPTED',
    db_id:'sales',rows:[['202601',100],['202602',200]],row_count:2,truncated:false}}, {
    sessionId:'owner',async api(path,options){
      requests++;assert.equal(path,'/api/dashboards');
      const request=JSON.parse(options.body);assert.equal(request.result_id,'result');
      assert.equal(request.session_id,'owner');assert.equal(request.include_data,false);
      return spec(false);
    },renderChart(){renders++;return {echarts:{dispose(){disposals++;}}};},
    renderTable(){throw Error('Table was not requested');},
  });
  const button=article.children[0].children[0],panel=article.children[1];
  assert.equal(requests,0);
  await button.listeners.click();assert.equal(panel.hidden,false);
  assert.equal(button.attributes['aria-expanded'],'true');
  await button.listeners.click();assert.equal(panel.hidden,true);assert.equal(disposals,1);
  await button.listeners.click();assert.equal(panel.hidden,false);
  assert.equal(requests,1);assert.equal(renders,2);
});
test('API errors are recoverable and do not render a fabricated dashboard',async()=>{
  const article=new Element('article');
  setup().mountAction(article,{result_id:'result',generation:{db_id:'sales',accepted:true,
    execution_status:'ACCEPTED',rows:[[1]],row_count:1,truncated:false}},
    {async api(){throw Error('Result no longer retained');}});
  const toolbar=article.children[0],button=toolbar.children[0];
  await button.listeners.click();
  assert.equal(article.children[1].hidden,true);
  assert.equal(button.disabled,false);
  assert.equal(toolbar.children[2].textContent,'Result no longer retained');
});
test('dashboard reuses application rendering fallback and responsive tokens without new engines',()=>{
  const app=fs.readFileSync('web/app.js','utf8'),css=fs.readFileSync('web/styles.css','utf8');
  assert.ok(app.includes('renderChart:renderChartInto,renderTable'));
  assert.ok(css.includes('grid-template-columns:repeat(12,minmax(0,1fr))'));
  assert.ok(css.includes('.dashboard-chart-card { grid-column:span 12; }'));
  assert.ok(css.includes('background:var(--surface)'));
  assert.ok(!source.includes('newPlot')&&!source.includes('echarts.init')&&!source.includes('/api/chat'));
});
test('incomplete previews and unexecuted SQL cannot start dashboard creation',async()=>{
  const view=setup();let calls=0;
  const article=new Element('article');
  view.mountAction(article,{result_id:'result',generation:{db_id:'sales',accepted:true,
    execution_status:'ACCEPTED',rows:[[1]],row_count:10,truncated:true}},
    {async api(){calls++;}});
  const button=article.children[0].children[0];assert.equal(button.disabled,true);
  await button.listeners.click();assert.equal(calls,0);
  const unexecuted=new Element('article');
  view.mountAction(unexecuted,{result_id:'result',generation:{accepted:true,
    execution_status:'SAFETY_VALIDATED'}},{});
  assert.equal(unexecuted.children.length,0);
});
test('optional table is explicitly requested and double clicks do not duplicate requests',async()=>{
  const article=new Element('article');let requests=0,tables=0,resolve;
  setup().mountAction(article,{result_id:'result',generation:{db_id:'sales',accepted:true,
    execution_status:'ACCEPTED',rows:[[1],[2]],row_count:2,truncated:false}},
    {async api(path,options){requests++;
      assert.equal(JSON.parse(options.body).include_data,true);
      return await new Promise(done=>{resolve=done;});
    },renderChart(){return null;},renderTable(){tables++;}});
  const toolbar=article.children[0],button=toolbar.children[0];
  toolbar.children[1].children[0].checked=true;
  const pending=button.listeners.click();
  assert.equal(button.disabled,true);
  await button.listeners.click();assert.equal(requests,1);
  resolve(spec(true));await pending;
  assert.equal(tables,1);assert.equal(button.disabled,false);
});
test('a partial renderer failure disposes earlier charts instead of leaking handles',()=>{
  const data=spec(false);data.widgets.push({...data.widgets[0],widget_id:'chart-1'});
  let calls=0,disposed=0;
  const container=new Element('div');
  assert.throws(()=>setup().render(container,data,{
    renderChart(){if(calls++)throw Error('Renderer unavailable');
      return {echarts:{dispose(){disposed++;}}};},
  }),/Renderer unavailable/);
  assert.equal(disposed,1);assert.equal(container.children.length,0);
});
test('dashboard opens inside the existing response card, aligned with its answer',()=>{
  const article=new Element('article'),card=new Element('section');
  article.querySelector=selector=>selector==='.response-card'?card:null;
  setup().mountAction(article,{result_id:'result',generation:{db_id:'sales',accepted:true,
    execution_status:'ACCEPTED',rows:[[1]],row_count:1,truncated:false}},{});
  assert.equal(article.children.length,0);
  assert.equal(card.children.length,3);
  assert.equal(card.children[1].hidden,true);
});
