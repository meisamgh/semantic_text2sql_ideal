// Run with node --test tests/frontend_rendering.cjs. No browser dependencies.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(`${__dirname}/../web/app.js`, 'utf8');
class Element {
  constructor() { this.children = []; }
  append(...children) { this.children.push(...children); }
  createTHead() { const child = new Element(); this.append(child); return child; }
  createTBody() { return this.createTHead(); }
  insertRow() { return this.createTHead(); }
  insertCell() { return this.createTHead(); }
  classList = { add() {} };
  setAttribute(name, value) { this[name] = value; }
}
const context = vm.createContext({ document: { createElement: () => new Element() } });
context.document.createElementNS = () => new Element();
vm.runInContext(fs.readFileSync(`${__dirname}/../web/charts.js`, 'utf8'), context);
test('charts render actual values, preserve negative values, and skip NULL', () => {
  const container = new Element();
  context.renderResultChart(container, {
    chart_type: 'bar', x: 'Category', y: 'Value', title: 'Comparison', note: 'Returned rows',
  }, {columns: ['Category', 'Value'], rows: [['A', -2], ['B', 4], ['C', null]]});
  const svg = container.children[1].children[1];
  const bars = svg.children.filter(e => e.width !== undefined);
  assert.equal(bars.length, 2);
  assert.equal(bars[0].children[0].textContent, 'A: -2');
  assert.ok(Number(bars[0].height) > 0);
});
test('scalar presentation does not produce a misleading chart', () => {
  const container = new Element();
  context.renderResultChart(container, {chart_type: 'none', note: 'Returned rows'}, {});
  assert.equal(container.children.length, 1);
});

vm.runInContext(source.slice(source.indexOf('function renderTable('), source.indexOf('function downloadCsv(')), context);
test('result table renders headers, rows and NULL explicitly', () => {
  const container = new Element();
  context.renderTable(container, ['customer', 'amount'], [['Alice', null]]);
  const table = container.children[0];
  assert.equal(table.children[0].children[0].children[0].textContent, 'customer');
  const cells = table.children[1].children[0].children;
  assert.equal(cells[0].textContent, 'Alice');
  assert.equal(cells[1].children[0].textContent, 'NULL');
});

vm.runInContext(source.slice(source.indexOf('function visiblePipelineStage('), source.indexOf('async function send(')), context);
test('progress supports recovery tools without marking unobserved stages complete', () => {
  const classes = new Map();
  const stages = ['retrieval', 'generation', 'recovery'].map(stage => ({
    dataset: { stage },
    classList: { toggle(name, enabled) { classes.set(`${stage}:${name}`, enabled); } },
  }));
  const progress = { dataset: {}, querySelectorAll: () => stages };
  context.updatePipeline(progress, 'generation', 'running');
  context.updatePipeline(progress, 'recovery_values', 'running');
  assert.equal(classes.get('recovery:active'), true);
  assert.equal(classes.get('generation:complete'), true);
  assert.equal(classes.get('retrieval:complete'), false);
});

test('progress colors fast stages captured only in backend event history', () => {
  const classes = new Map();
  const stages = ['conversation', 'retrieval', 'grounding', 'generation'].map(stage => ({
    dataset: { stage },
    classList: { toggle(name, enabled) { classes.set(`${stage}:${name}`, enabled); } },
  }));
  const progress = { dataset: {}, querySelectorAll: () => stages };
  context.updatePipeline(progress, 'generation', 'running', [
    { stage: 'conversation' },
    { stage: 'retrieval' },
    { stage: 'grounding' },
    { stage: 'generation' },
  ]);
  assert.equal(classes.get('conversation:complete'), true);
  assert.equal(classes.get('grounding:complete'), true);
  assert.equal(classes.get('generation:active'), true);
});
vm.runInContext(source.slice(source.indexOf('function usageTotal('), source.indexOf('function renderAttempts(')), context);
for (const [name, body, generation, expected] of [
  ['successful result', { token_usage: { input_tokens: 10, output_tokens: 5 } }, { accepted: true, attempts: [] }, '15'],
  ['failure without usage', {}, { accepted: false, attempts: [] }, 'Unavailable'],
  ['generation usage fallback', {}, { token_usage: { input_tokens: 7, output_tokens: 2 } }, '9'],
]) {
  test(`token rendering: ${name}`, () => {
    const elements = new Map();
    const fragment = { querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, new Element());
      return elements.get(selector);
    } };
    context.renderTokenAccounting(fragment, body, generation);
    assert.equal(elements.get('.token-grid').children[0].children[1].textContent, expected);
  });
}
