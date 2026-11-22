// cupel.run/bench — read-only public leaderboard.
//
// Renders bench.json, written by `cupel/publish.py`. Every number that changes with
// a toggle is precomputed by the exporter; this page only selects a variant:
//   entry.variants[text ? 'text' : 'all'][metric]
// No ranking, interval or percentage logic lives here. Viewer state is in memory
// only: no localStorage, no URL params.

import { h, render } from 'preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import htm from 'htm';

const html = htm.bind(h);
const SCHEMA_VERSION = 1;
const TEXT_EXCLUDED = 'multimodal';
const REPO_URL = 'https://github.com/tolitius/cupel';

// copied once from cupel/ui/pages/dashboard.js
const CHART_PALETTE = [
  '#d46a3a',  // accent (burnt orange)
  '#6aaa6a',  // green
  '#d4a844',  // gold
  '#5b9bd5',  // blue
  '#c77dba',  // purple
  '#c45050',  // red
  '#4cb89a',  // teal
  '#d46b7a',  // rose
];
const FONT = "'IBM Plex Mono', monospace";
const TOOLTIP_STYLE = {
  backgroundColor: '#262523',
  titleColor: '#e4e0da',
  bodyColor: '#a09a92',
  borderColor: '#3e3b38',
  borderWidth: 1,
  titleFont: { family: FONT },
  bodyFont: { family: FONT },
};

const color = (i) => CHART_PALETTE[i % CHART_PALETTE.length];
const catLabel = (c) => (c || '').replace(/_/g, ' ');
const deltaStr = (d) => d > 0 ? `+${d}` : d === 0 ? '—' : `${d}`;
const deltaClass = (d) => d > 0 ? 'up' : d < 0 ? 'down' : 'flat';
const hwStr = (hw) => hw && hw.name ? `${hw.name}${hw.memory ? ' ' + hw.memory : ''}` : '';

function fmtDate(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return '';
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
}

const LOGO = html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="20" height="20" aria-hidden="true"><circle cx="24" cy="24" r="22" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="9" cy="12" r="2" fill="#7A8899" opacity="0.5"/><circle cx="6" cy="28" r="1.8" fill="#7A8899" opacity="0.45"/><circle cx="12" cy="40" r="1.8" fill="#7A8899" opacity="0.48"/><circle cx="28" cy="43" r="1.6" fill="#7A8899" opacity="0.42"/><circle cx="40" cy="32" r="1.8" fill="#7A8899" opacity="0.48"/><circle cx="40" cy="14" r="1.6" fill="#7A8899" opacity="0.42"/><circle cx="28" cy="6" r="1.8" fill="#7A8899" opacity="0.45"/><circle cx="16" cy="7" r="1.6" fill="#7A8899" opacity="0.4"/><circle cx="24" cy="24" r="7" fill="#D4A030"/><circle cx="21.5" cy="21" r="2.5" fill="#E8C25C" opacity="0.45"/></svg>`;

const Header = () => html`
  <header class="topbar"><a class="logo" href="/">${LOGO}cupel</a></header>`;

// ── Chart.js, loaded on first chart tab from the vendored UMD build ──

let chartPromise = null;
function loadChartJS() {
  if (window.Chart) return Promise.resolve(window.Chart);
  if (!chartPromise) {
    chartPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = new URL('./vendor/chart.umd.js', import.meta.url).href;
      s.onload = () => resolve(window.Chart);
      s.onerror = () => { chartPromise = null; reject(new Error('chart.js failed to load')); };
      document.head.appendChild(s);
    });
  }
  return chartPromise;
}

// ── chart configs, adapted from dashboard.js; rows are { e, v } ──

const modelLabelPlugin = {
  id: 'modelLabels',
  afterDatasetsDraw(chart) {
    const ctx = chart.ctx;
    ctx.save();
    chart.data.datasets.forEach((ds, i) => {
      const pt = chart.getDatasetMeta(i).data[0];
      if (!pt) return;
      ctx.font = '500 10px ' + FONT;
      ctx.fillStyle = color(i);
      ctx.globalAlpha = 0.9;
      ctx.save();
      ctx.translate(pt.x, pt.y + (i % 2 === 0 ? -14 : 18));
      ctx.rotate(-0.3);
      ctx.textAlign = 'center';
      ctx.fillText(ds.label, 0, 0);
      ctx.restore();
    });
    ctx.restore();
  },
};

function makePctLabelPlugin(sorted) {
  return {
    id: 'barPctLabels',
    afterDatasetsDraw(chart) {
      const ctx = chart.ctx;
      ctx.save();
      chart.getDatasetMeta(0).data.forEach((bar, i) => {
        ctx.font = '600 11px ' + FONT;
        ctx.fillStyle = '#e4e0da';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(sorted[i].v.pct.toFixed(1) + '%', bar.x + 8, bar.y);
      });
      ctx.restore();
    },
  };
}

const axis = (title, extra = {}) => ({
  title: title ? { display: true, text: title, color: '#6a6560', font: { family: FONT, size: 11 } } : undefined,
  grid: { color: '#332f2d' },
  ticks: { color: '#6a6560', font: { family: FONT, size: 10 } },
  ...extra,
});

function buildScatterConfig(rows) {
  const pcts = rows.map(r => r.v.pct);
  const yMin = pcts.length ? Math.floor(Math.min(...pcts) / 10) * 10 : 0;
  return {
    type: 'scatter',
    plugins: [modelLabelPlugin],
    data: {
      datasets: rows.map((r, i) => ({
        label: r.e.model,
        data: [{ x: r.e.tok_per_sec || 0, y: r.v.pct }],
        backgroundColor: color(i),
        borderColor: color(i),
        pointRadius: r.e.is_local ? 7 : 9,
        pointStyle: r.e.is_local ? 'circle' : 'rectRounded',
        pointHoverRadius: 12,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      layout: { padding: { top: 24, right: 24, left: 8, bottom: 4 } },
      scales: { x: axis('tok/s →'), y: axis('Score %', { min: yMin, max: 100 }) },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_STYLE,
          callbacks: {
            title: (items) => rows[items[0].datasetIndex]?.e.model || '',
            label: (ctx) => {
              const r = rows[ctx.datasetIndex];
              return [`${r.v.pct.toFixed(1)}%  ·  ${(r.e.tok_per_sec || 0).toFixed(0)} tok/s`, hwStr(r.e.hardware)];
            },
          },
        },
      },
    },
  };
}

function buildBarConfig(rows, metric) {
  const sorted = [...rows].sort((a, b) => b.v.pct - a.v.pct);
  return {
    type: 'bar',
    plugins: [makePctLabelPlugin(sorted)],
    data: {
      labels: sorted.map(r => r.e.model),
      datasets: [{
        data: sorted.map(r => r.v.pct),
        backgroundColor: sorted.map(r => color(rows.indexOf(r)) + 'aa'),
        borderColor: sorted.map(r => color(rows.indexOf(r))),
        borderWidth: 1, borderRadius: 3, borderSkipped: false, barPercentage: 0.65,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      layout: { padding: { right: 50 } },
      scales: {
        x: axis(null, { min: 0, max: 100, ticks: { color: '#6a6560', font: { family: FONT, size: 10 }, callback: v => v + '%' } }),
        y: {
          grid: { display: false },
          ticks: {
            color: '#a09a92', font: { family: FONT, size: 11, weight: '500' },
            callback: function (value) {
              const label = this.getLabelForValue(value);
              return label.length > 24 ? label.substring(0, 22) + '…' : label;
            },
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_STYLE,
          callbacks: {
            label: (ctx) => {
              const r = sorted[ctx.dataIndex];
              const head = metric === 'score' ? `${r.v.pts}/${r.v.max} pts  ·  ` : '';
              return [
                `${head}${r.v.pct.toFixed(1)}%`,
                `${(r.e.tok_per_sec || 0).toFixed(0)} tok/s  ·  ${(r.e.avg_time || 0).toFixed(1)}s avg`,
              ];
            },
          },
        },
      },
    },
  };
}

function buildRadarConfig(rows, radarModels, cats) {
  return {
    type: 'radar',
    data: {
      labels: cats.map(catLabel),
      datasets: rows
        .map((r, i) => ({ r, i }))
        .filter(({ r }) => radarModels.includes(r.e.model))
        .map(({ r, i }) => ({
          label: r.e.model,
          // null (not 0) when untested: Chart.js leaves a gap rather than drawing a failure
          data: cats.map(c => r.v.by_category[c] ?? null),
          borderColor: color(i),
          backgroundColor: color(i) + '18',
          borderWidth: 2, pointRadius: 3, pointBackgroundColor: color(i),
        })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true, min: 0, max: 100,
          ticks: { stepSize: 25, color: '#6a6560', font: { family: FONT, size: 9 }, backdropColor: 'transparent' },
          grid: { color: '#3e3b38' },
          angleLines: { color: '#332f2d' },
          pointLabels: { color: '#a09a92', font: { family: FONT, size: 10 } },
        },
      },
      plugins: {
        legend: { position: 'bottom', labels: { color: '#a09a92', font: { family: FONT, size: 11 }, boxWidth: 12, padding: 16 } },
        tooltip: TOOLTIP_STYLE,
      },
    },
  };
}

// ── table sorting ──
// rank, score and % follow the exporter's rank order (`standing`); faster model
// wins ties, the same tiebreak as the local dashboard

const SORT_GETTERS = {
  rank: r => r.standing,
  model: r => r.e.model.toLowerCase(),
  score: r => -r.standing,
  pct: r => -r.standing,
  pts: r => r.v.pts ?? 0,
  speed: r => r.e.tok_per_sec || 0,
  avg: r => r.e.avg_time || 0,
};

function sortRows(rows, sortCol, sortDir) {
  const getter = SORT_GETTERS[sortCol];
  if (!getter) return rows;
  return [...rows].sort((a, b) => {
    const va = getter(a), vb = getter(b);
    let cmp = typeof va === 'string' ? va.localeCompare(vb) : va - vb;
    if (sortDir === 'desc') cmp = -cmp;
    return cmp || ((b.e.tok_per_sec || 0) - (a.e.tok_per_sec || 0));
  });
}

// ── the board ──

function Board({ data }) {
  const view = data.view;
  const radarAllowed = (text) =>
    data.categories.filter(c => !(text && c === TEXT_EXCLUDED)).length >= 3;

  const [text, setText] = useState(!!view.text);
  const [metric, setMetric] = useState(data.check_available && view.metric === 'check' ? 'check' : 'score');
  const [chartTab, setChartTab] = useState(
    view.chart_tab === 'radar' && !radarAllowed(!!view.text) ? null : view.chart_tab);
  const [sortCol, setSortCol] = useState(view.sort_col || 'score');
  const [sortDir, setSortDir] = useState(view.sort_dir || 'desc');
  const [search, setSearch] = useState('');
  const [chartReady, setChartReady] = useState(!!window.Chart);
  const [chartError, setChartError] = useState(false);

  const subset = text ? 'text' : 'all';

  // every row on the board under the active variant, in rank order
  const ranked = data.entries
    .map(e => ({ e, v: e.variants[subset] && e.variants[subset][metric] }))
    .filter(r => r.v)
    .sort((a, b) => (a.v.rank - b.v.rank) || (b.v.pct - a.v.pct)
      || ((b.e.tok_per_sec || 0) - (a.e.tok_per_sec || 0)))
    .map((r, i) => ({ ...r, standing: i + 1 }));

  const [radarModels, setRadarModels] = useState(() => ranked.slice(0, 3).map(r => r.e.model));

  const q = search.trim().toLowerCase();
  let rows = q ? ranked.filter(r => r.e.model.toLowerCase().includes(q)) : ranked;
  rows = sortRows(rows, sortCol, sortDir);

  // compare percentages to the third row in the current display order (the local rule)
  const baseline = rows.length >= 3 ? rows[2].v.pct : (rows[rows.length - 1]?.v.pct || 0);
  if (sortCol === 'delta') {
    rows = [...rows].sort((a, b) => {
      const da = a.v.pct - baseline, db = b.v.pct - baseline;
      return sortDir === 'asc' ? da - db : db - da;
    });
  }

  const cats = data.categories.filter(c => !(text && c === TEXT_EXCLUDED));
  const showRadar = cats.length >= 3;
  const activeTab = chartTab === 'radar' && !showRadar ? null : chartTab;
  const nPrompts = text ? data.eval_set.n_text_prompts : data.eval_set.n_prompts;
  const peak = Math.max(0, ...ranked.map(r => r.e.tok_per_sec || 0));

  // ── chart lifecycle ──
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  useEffect(() => {
    if (activeTab && !window.Chart) {
      loadChartJS().then(() => setChartReady(true)).catch(() => setChartError(true));
    }
  }, [activeTab]);

  const chartKey = JSON.stringify([activeTab, chartReady, subset, metric, sortCol, sortDir, q, radarModels]);
  useEffect(() => {
    if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }
    if (!activeTab || !chartReady || !window.Chart || !canvasRef.current || rows.length === 0) return;
    const config = activeTab === 'scatter' ? buildScatterConfig(rows)
      : activeTab === 'bar' ? buildBarConfig(rows, metric)
      : buildRadarConfig(rows, radarModels, cats);
    chartRef.current = new window.Chart(canvasRef.current, config);
    return () => { if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; } };
  }, [chartKey]);

  const toggleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    else { setSortCol(col); setSortDir(col === 'model' || col === 'rank' ? 'asc' : 'desc'); }
  };
  const arrow = (col) => sortCol === col ? (sortDir === 'desc' ? ' ▼' : ' ▲') : '';
  const toggleRadarModel = (name) => setRadarModels(prev => {
    if (prev.includes(name)) return prev.length > 1 ? prev.filter(n => n !== name) : prev;
    return prev.length < 4 ? [...prev, name] : prev;
  });

  // ── pieces ──
  const frozenNotes = [];
  if (data.frozen.local) frozenNotes.push('local models only');

  const statsStrip = html`
    <div class="stats-strip">
      <div class="stat-cell"><div class="stat-label">Models</div><div class="stat-val">${ranked.length}</div></div>
      <div class="stat-cell"><div class="stat-label">Prompts</div><div class="stat-val">${nPrompts}</div></div>
      <div class="stat-cell"><div class="stat-label">Peak Speed</div><div class="stat-val">${peak.toFixed(0)} <span class="u">tok/s</span></div></div>
      ${ranked.length > 0 ? html`
        <div class="chart-tabs">
          ${[['scatter', 'Score vs Speed'], ['bar', 'Overall Accuracy'], ...(showRadar ? [['radar', 'Category Fingerprint']] : [])]
            .map(([tab, label]) => html`
              <button class="chart-tab ${activeTab === tab ? 'active' : ''}"
                onClick=${() => setChartTab(activeTab === tab ? null : tab)}>${label}</button>`)}
        </div>` : null}
      <div class="stat-cell toggles">
        <div class="toggle-row">
          <label class="toggle" title="exclude multimodal prompts">
            <input type="checkbox" checked=${text} onChange=${(ev) => setText(ev.target.checked)} /> text
          </label>
          ${data.check_available ? html`
            <label class="toggle" title="Rank on the full criteria vector instead of the 0-3 collapse">
              <input type="checkbox" checked=${metric === 'check'}
                onChange=${(ev) => setMetric(ev.target.checked ? 'check' : 'score')} /> criteria detail
            </label>` : null}
        </div>
        ${frozenNotes.length ? html`<div class="frozen-line">${frozenNotes.join(' · ')}</div>` : null}
      </div>
    </div>`;

  let chartControls = null;
  if (activeTab === 'scatter') {
    chartControls = html`
      <div class="chart-controls">
        <span><span class="legend-dot" style="border-radius:50%"></span>local</span>
        <span><span class="legend-dot" style="border-radius:2px"></span>cloud</span>
        <span class="hint">top-right = fast + accurate</span>
      </div>`;
  } else if (activeTab === 'radar') {
    chartControls = html`
      <div class="chart-controls">
        <span style="text-transform:uppercase;letter-spacing:0.06em;font-size:10px">compare</span>
        ${rows.map((r, i) => {
          const on = radarModels.includes(r.e.model);
          const c = color(i);
          return html`<button class="radar-model-btn ${on ? 'on' : ''}"
            style=${on ? `background:${c}20;color:${c};border-color:${c}55` : ''}
            disabled=${!on && radarModels.length >= 4}
            onClick=${() => toggleRadarModel(r.e.model)}>${r.e.model.length > 20 ? r.e.model.substring(0, 18) + '…' : r.e.model}</button>`;
        })}
      </div>`;
  }
  const chartHeight = activeTab === 'bar' ? Math.max(320, rows.length * 28 + 40) : 320;
  const chartPanel = activeTab ? html`
    <div class="chart-panel" key=${activeTab}>
      ${chartControls}
      <div class="chart-canvas-wrap" style="height:${chartHeight}px">
        ${chartError ? html`<div class="msg">chart unavailable</div>` : html`<canvas ref=${canvasRef}></canvas>`}
      </div>
    </div>` : null;

  const tableRows = rows.map((r, i) => {
    const { e, v } = r;
    const delta = Math.round((v.pct - baseline) * 10) / 10;
    const covPct = Math.round((v.coverage || 0) * 100);
    const ticks = (e.per_prompt || [])
      .filter(p => !(text && p.category === TEXT_EXCLUDED))
      .map(p => html`<div class=${p.score != null ? `s${Math.round(p.score)}` : 's0'}
        title="#${p.id}: ${p.score != null ? p.score : '—'}"></div>`);
    const hw = hwStr(e.hardware);
    return html`
      <tr class=${i === 0 ? 'winner' : ''} key=${e.model + '|' + hw + '|' + e.judge_model}>
        <td class="td-rank" title=${v.tied ? 'tied: within the measured run-to-run noise of the rank leader' : ''}>
          ${v.rank}${v.tied ? html`<span class="tie">=</span>` : null}
        </td>
        <td class="td-model">
          <div class="td-model-name">${e.model}</div>
          <div class="td-model-meta">
            ${e.judge_model ? html`<span class=${e.self_judged ? 'self' : ''}>${e.self_judged ? 'self-judged' : e.judge_model}</span>` : null}
            ${e.judge_model && hw ? ' · ' : ''}${hw}
            ${e.n_runs > 1 ? html`<span title="pooled over ${e.n_runs} runs"> · n=${e.n_runs}</span>` : null}
          </div>
          ${e.notes ? html`<div class="td-notes">${e.notes}</div>` : null}
        </td>
        <td class="td-bar">
          <div class="bar-track"><div class="bar-fill" style="width:${v.pct.toFixed(1)}%"><div class="bar-ticks">${ticks}</div></div></div>
        </td>
        <td class="td-pct" title="CI ${v.ci_lo}–${v.ci_hi}%">${v.pct.toFixed(1)}%</td>
        <td class="td-score">
          ${metric === 'score' ? html`${v.pts}<small>/${v.max}</small>` : '—'}
          ${e.n_runs > 1 && metric === 'score' ? html`<small style="display:block">pooled over ${e.n_runs}</small>` : null}
          ${v.low_coverage ? html`<small class="low" title="only ${covPct}% of prompts were scored: not comparable to a full run">⚠ ${covPct}% scored</small>` : null}
        </td>
        <td class="td-speed">${(e.tok_per_sec || 0).toFixed(0)} <small>tok/s</small></td>
        <td class="td-speed">${(e.avg_time || 0).toFixed(1)}s</td>
        <td class="td-delta ${deltaClass(delta)}">${deltaStr(delta)}</td>
      </tr>`;
  });

  const th = (col, label, cls = 'r') => html`
    <th class="${cls} sortable" onClick=${() => toggleSort(col)}>${label}${arrow(col)}</th>`;

  const table = html`
    <div class="lb"><table>
      <thead><tr>
        ${th('rank', '#', '')}
        <th>
          <span class="sortable" style="cursor:pointer" onClick=${() => toggleSort('model')}>Model${arrow('model')}</span>
          <input class="search" type="search" placeholder="search" aria-label="search models"
            value=${search} onInput=${(ev) => setSearch(ev.target.value)} />
        </th>
        <th>Score</th>
        ${th('pct', '%')}
        ${th('pts', 'Pts')}
        ${th('speed', 'Speed')}
        ${th('avg', 'Avg')}
        ${th('delta', 'Δ vs #3')}
      </tr></thead>
      <tbody>${tableRows}</tbody>
    </table>
    ${rows.length === 0 ? html`<div class="empty">no models match “${search}”</div>` : null}
    </div>`;

  const repeats = data.entries.some(e => e.n_runs > 1);
  const footnote = html`
    <div class="footnote">
      <div>${repeats ? 'some models averaged over repeat runs' : 'single runs'} · ${nPrompts} prompts · judge: ${data.judge || 'unknown'} · published ${fmtDate(data.published_at)}</div>
      <div>Scores are self-reported by the publisher on a private eval set. Made with <a href=${REPO_URL}>cupel</a>.</div>
    </div>`;

  return html`
    ${statsStrip}
    ${chartPanel}
    ${table}
    ${footnote}`;
}

function App() {
  const [state, setState] = useState({ status: 'loading' });

  useEffect(() => {
    fetch(new URL('./bench.json', import.meta.url), { cache: 'no-cache' })
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(data => {
        if (!data || typeof data !== 'object') throw new Error('bad json');
        if (data.schema_version !== SCHEMA_VERSION) setState({ status: 'version' });
        else if (!Array.isArray(data.entries)) throw new Error('bad json');
        else setState({ status: 'ok', data });
      })
      .catch(() => setState({ status: 'error' }));
  }, []);

  if (state.status === 'version') {
    return html`<div class="msg">this leaderboard was published by a newer cupel; reload later</div>`;
  }
  if (state.status === 'loading') return html`${Header()}<div class="msg">loading…</div>`;
  if (state.status === 'error') return html`${Header()}<div class="msg">leaderboard unavailable</div>`;

  const { data } = state;
  return html`
    ${Header()}
    <div class="title-strip">
      <div class="title-main">cupel${data.cupel_version ? html`<span class="ver">v${data.cupel_version}</span>` : null}</div>
    </div>
    ${data.entries.length === 0
      ? html`<div class="msg">no results published yet</div>`
      : html`<${Board} data=${data} />`}`;
}

render(html`<${App} />`, document.getElementById('app'));
