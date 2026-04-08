const { h, useState, useEffect, useCallback, html } = window.__preact;

const CAT_COLORS = {
  multimodal: "#c77dba", security: "#d4845a", clojure_code: "#8b7ec8",
  distributed_systems: "#5b9bd5", ml_architecture: "#4cb89a", python_coding: "#c9a033",
  business_logic: "#c45050", clojure_ecosystem: "#9b6ec8", frontend_architecture: "#4aa3d5",
  domain_knowledge: "#d46b7a", system_design: "#3bbfa0", observability: "#8bb840",
  networking: "#b86ec8", math_estimation: "#c9b040", diagnostic_reasoning: "#5ab88b",
  meta: "#808080", chemistry: "#c87da0", assistant_competence: "#4ac4c4",
};

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

const TOOLTIP_STYLE = {
  backgroundColor: '#262523',
  titleColor: '#e4e0da',
  bodyColor: '#a09a92',
  borderColor: '#3e3b38',
  borderWidth: 1,
  titleFont: { family: "'IBM Plex Mono', monospace" },
  bodyFont: { family: "'IBM Plex Mono', monospace" },
};

const FONT = "'IBM Plex Mono', monospace";

function catLabel(c) { return (c || '').replace(/_/g, ' '); }

function deltaStr(d) { return d > 0 ? `+${d}` : d === 0 ? '\u2014' : `${d}`; }
function deltaClass(d) { return d > 0 ? 'up' : d < 0 ? 'down' : 'flat'; }

// Sort columns config
const SORT_COLS = [
  { key: 'rank',  label: '#',     getter: (e, i) => i },
  { key: 'model', label: 'Model', getter: e => e.model.toLowerCase() },
  { key: 'score', label: 'Score', getter: e => e.pct },
  { key: 'pts',   label: 'Pts',   getter: e => e.total_score },
  { key: 'pct',   label: '%',     getter: e => e.pct },
  { key: 'delta', label: '\u0394',getter: null },
  { key: 'speed', label: 'Speed', getter: e => e.tok_per_sec || 0 },
  { key: 'avg',   label: 'Avg',   getter: e => e.avg_time || 0 },
];

// ── Chart.js lazy loader ──
let ChartJS = null;
let currentChart = null;

async function loadChartJS() {
  if (ChartJS) return ChartJS;
  const mod = await import('https://esm.sh/chart.js@4.4.7/auto');
  ChartJS = mod.Chart || mod.default;
  return ChartJS;
}

function isLocal(e) {
  return !!(e.hardware && e.hardware.memory && e.hardware.memory.trim() !== '');
}

// ── Chart.js custom plugins ──

const modelLabelPlugin = {
  id: 'modelLabels',
  afterDatasetsDraw(chart) {
    const ctx = chart.ctx;
    ctx.save();
    chart.data.datasets.forEach((ds, i) => {
      const meta = chart.getDatasetMeta(i);
      if (!meta.data[0]) return;
      const pt = meta.data[0];
      ctx.font = "500 10px " + FONT;
      ctx.fillStyle = CHART_PALETTE[i % CHART_PALETTE.length];
      ctx.globalAlpha = 0.9;
      const above = i % 2 === 0;
      const yOff = above ? -14 : 18;
      ctx.save();
      ctx.translate(pt.x, pt.y + yOff);
      ctx.rotate(-0.3);
      ctx.textAlign = 'center';
      ctx.fillText(ds.label, 0, 0);
      ctx.restore();
    });
    ctx.restore();
  },
};

function makePctLabelPlugin(sortedEntries) {
  return {
    id: 'barPctLabels',
    afterDatasetsDraw(chart) {
      const ctx = chart.ctx;
      ctx.save();
      const meta = chart.getDatasetMeta(0);
      meta.data.forEach((bar, i) => {
        ctx.font = "600 11px " + FONT;
        ctx.fillStyle = '#e4e0da';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(sortedEntries[i].pct.toFixed(1) + '%', bar.x + 8, bar.y);
      });
      ctx.restore();
    },
  };
}

// ── Build chart configs ──

function buildChartConfig(tab, entries, radarModels, cats, catPrompts) {
  if (tab === 'scatter') return buildScatterConfig(entries);
  if (tab === 'bar') return buildBarConfig(entries);
  if (tab === 'radar') return buildRadarConfig(entries, radarModels, cats, catPrompts);
  return null;
}

function buildScatterConfig(entries) {
  const pcts = entries.map(e => e.pct || 0);
  const yMin = pcts.length > 0 ? Math.floor(Math.min(...pcts) / 10) * 10 : 0;
  return {
    type: 'scatter',
    plugins: [modelLabelPlugin],
    data: {
      datasets: entries.map((e, i) => ({
        label: e.model,
        data: [{ x: e.tok_per_sec || 0, y: e.pct || 0 }],
        backgroundColor: CHART_PALETTE[i % CHART_PALETTE.length],
        borderColor: CHART_PALETTE[i % CHART_PALETTE.length],
        pointRadius: isLocal(e) ? 7 : 9,
        pointStyle: isLocal(e) ? 'circle' : 'rectRounded',
        pointHoverRadius: 12,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      layout: { padding: { top: 24, right: 24, left: 8, bottom: 4 } },
      scales: {
        x: {
          title: { display: true, text: 'tok/s \u2192', color: '#6a6560', font: { family: FONT, size: 11 } },
          grid: { color: '#332f2d' },
          ticks: { color: '#6a6560', font: { family: FONT, size: 10 } },
        },
        y: {
          title: { display: true, text: 'Score %', color: '#6a6560', font: { family: FONT, size: 11 } },
          grid: { color: '#332f2d' },
          ticks: { color: '#6a6560', font: { family: FONT, size: 10 } },
          min: yMin, max: 100,
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_STYLE,
          callbacks: {
            title: (items) => entries[items[0].datasetIndex]?.model || '',
            label: (ctx) => {
              const e = entries[ctx.datasetIndex];
              return [
                `${(e.pct || 0).toFixed(1)}%  \u00b7  ${(e.tok_per_sec || 0).toFixed(0)} tok/s`,
                `${e.hardware?.name || ''}${e.hardware?.memory ? ' ' + e.hardware.memory : ''}`,
              ];
            },
          },
        },
      },
    },
  };
}

function buildBarConfig(entries) {
  const sorted = [...entries].sort((a, b) => b.pct - a.pct);
  const pctPlugin = makePctLabelPlugin(sorted);
  return {
    type: 'bar',
    plugins: [pctPlugin],
    data: {
      labels: sorted.map(e => e.model),
      datasets: [{
        data: sorted.map(e => e.pct),
        backgroundColor: sorted.map(e => {
          const idx = entries.indexOf(e);
          return CHART_PALETTE[idx % CHART_PALETTE.length] + 'aa';
        }),
        borderColor: sorted.map(e => {
          const idx = entries.indexOf(e);
          return CHART_PALETTE[idx % CHART_PALETTE.length];
        }),
        borderWidth: 1,
        borderRadius: 3,
        borderSkipped: false,
        barPercentage: 0.65,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      layout: { padding: { right: 50 } },
      scales: {
        x: {
          min: 0, max: 100,
          grid: { color: '#332f2d' },
          ticks: { color: '#6a6560', font: { family: FONT, size: 10 }, callback: v => v + '%' },
        },
        y: {
          grid: { display: false },
          ticks: {
            color: '#a09a92',
            font: { family: FONT, size: 11, weight: '500' },
            callback: function(value) {
              const label = this.getLabelForValue(value);
              return label.length > 24 ? label.substring(0, 22) + '\u2026' : label;
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
              const e = sorted[ctx.dataIndex];
              return [
                `${e.total_score}/${e.max_score} pts  \u00b7  ${(e.pct || 0).toFixed(1)}%`,
                `${(e.tok_per_sec || 0).toFixed(0)} tok/s  \u00b7  ${(e.avg_time || 0).toFixed(1)}s avg`,
              ];
            },
          },
        },
      },
    },
  };
}

function buildRadarConfig(entries, radarModels, cats, catPrompts) {
  const selectedEntries = entries.filter(e => radarModels.includes(e.model));

  function computeCategoryScores(entry) {
    const scores = {};
    cats.forEach(cat => {
      const cp = catPrompts[cat];
      const maxCat = cp.length * 3;
      const catScore = cp.reduce((acc, p) => {
        const sp = (entry.scores_by_prompt || []).find(s => s.id === p.id);
        return acc + (sp && sp.score != null ? sp.score : 0);
      }, 0);
      scores[cat] = maxCat > 0 ? Math.round(catScore / maxCat * 100) : 0;
    });
    return scores;
  }

  return {
    type: 'radar',
    data: {
      labels: cats.map(c => catLabel(c)),
      datasets: selectedEntries.map(entry => {
        const catScores = computeCategoryScores(entry);
        const colorIdx = entries.indexOf(entry);
        return {
          label: entry.model,
          data: cats.map(c => catScores[c]),
          borderColor: CHART_PALETTE[colorIdx % CHART_PALETTE.length],
          backgroundColor: CHART_PALETTE[colorIdx % CHART_PALETTE.length] + '18',
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: CHART_PALETTE[colorIdx % CHART_PALETTE.length],
        };
      }),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true, max: 100, min: 0,
          ticks: {
            stepSize: 25, color: '#6a6560',
            font: { family: FONT, size: 9 },
            backdropColor: 'transparent',
          },
          grid: { color: '#3e3b38' },
          angleLines: { color: '#332f2d' },
          pointLabels: {
            color: '#a09a92',
            font: { family: FONT, size: 10 },
          },
        },
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#a09a92',
            font: { family: FONT, size: 11 },
            boxWidth: 12, padding: 16,
          },
        },
        tooltip: TOOLTIP_STYLE,
      },
    },
  };
}

// ──────────────────────────────────────────────

function Dashboard({ providers, refreshProviders }) {
  const [leaderboard, setLeaderboard] = useState(null);
  const [hardware, setHardware] = useState(null);
  const [dismissed, setDismissed] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [detailData, setDetailData] = useState(null);
  const [showExamples, setShowExamples] = useState(() => {
    const saved = localStorage.getItem('cupel:dash-show-examples');
    return saved !== null ? JSON.parse(saved) : true;
  });
  const [localOnly, setLocalOnly] = useState(() => {
    const saved = localStorage.getItem('cupel:dash-local-only');
    return saved !== null ? JSON.parse(saved) : false;
  });
  const [state, setState] = useState(null);
  const [sortCol, setSortCol] = useState('score');
  const [sortDir, setSortDir] = useState('desc');
  const [refreshing, setRefreshing] = useState(false);

  // Chart state
  const [chartTab, setChartTab] = useState(() => localStorage.getItem('cupel:dash-chart-tab') || null);
  const [radarModels, setRadarModels] = useState([]);
  const [chartLoaded, setChartLoaded] = useState(false);

  // Load Chart.js on mount if a tab was persisted from a previous session
  useEffect(() => {
    if (chartTab && !chartLoaded) {
      loadChartJS().then(() => setChartLoaded(true));
    }
  }, []);

  const doRefresh = useCallback(() => {
    if (!refreshProviders || refreshing) return;
    setRefreshing(true);
    refreshProviders();
    setTimeout(() => setRefreshing(false), 1500);
  }, [refreshProviders, refreshing]);

  useEffect(() => {
    Promise.all([
      fetch('/api/results/leaderboard').then(r => r.json()),
      fetch('/api/state').then(r => r.json()),
      fetch('/api/hardware').then(r => r.json()),
    ]).then(([lb, st, hw]) => {
      setLeaderboard(lb);
      setState(st);
      setHardware(hw);
    }).catch(console.error);
  }, []);

  const openModelDetail = useCallback((entry) => {
    setSelectedModel(entry.model);
    if (entry.is_example) {
      setDetailData({
        model: entry.model,
        is_example: true,
        hardware: entry.hardware,
        judge_model: entry.judge_model,
        total_score: entry.total_score,
        max_score: entry.max_score,
        pct: entry.pct,
        tok_per_sec: entry.tok_per_sec,
        avg_time: entry.avg_time,
        results: entry.scores_by_prompt,
      });
    } else if (entry.filename) {
      fetch(`/api/results/${entry.filename}`).then(r => r.json())
        .then(data => {
          setDetailData({
            model: entry.model,
            is_example: false,
            hardware: entry.hardware,
            judge_model: data.judge || entry.judge_model,
            total_score: entry.total_score,
            max_score: entry.max_score,
            pct: entry.pct,
            tok_per_sec: entry.tok_per_sec,
            avg_time: entry.avg_time,
            results: data.results || [],
          });
        })
        .catch(console.error);
    }
  }, []);

  const closeDetail = useCallback(() => {
    setDetailData(null);
    setSelectedModel(null);
  }, []);

  const toggleSort = useCallback((colKey) => {
    if (sortCol === colKey) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    } else {
      setSortCol(colKey);
      setSortDir(colKey === 'model' ? 'asc' : 'desc');
    }
  }, [sortCol]);

  const handleChartTab = useCallback((tab) => {
    const next = chartTab === tab ? null : tab;
    setChartTab(next);
    if (next) {
      localStorage.setItem('cupel:dash-chart-tab', next);
    } else {
      localStorage.removeItem('cupel:dash-chart-tab');
    }
    if (next && !chartLoaded) {
      loadChartJS().then(() => setChartLoaded(true));
    }
  }, [chartTab, chartLoaded]);

  const toggleRadarModel = useCallback((name) => {
    setRadarModels(prev => {
      if (prev.includes(name)) {
        return prev.length > 1 ? prev.filter(n => n !== name) : prev;
      }
      return prev.length < 4 ? [...prev, name] : prev;
    });
  }, []);

  if (!leaderboard) return html`<div class="page" style="padding:40px;color:var(--text-2)">Loading...</div>`;

  let entries = showExamples ? [...leaderboard.entries] : leaderboard.entries.filter(e => !e.is_example);
  if (localOnly) {
    entries = entries.filter(e => {
      const hw = e.hardware || {};
      return hw.memory && hw.memory.trim() !== '';
    });
  }
  const prompts = leaderboard.prompts;
  const maxScore = prompts.length * 3;

  const judgeModel = (entries.length > 0 && entries[0].judge_model) ? entries[0].judge_model : 'self-judged';
  const hardwareStr = hardware ? `${hardware.name || ''} ${hardware.memory || ''}`.trim() : '';

  // Sort entries
  const colDef = SORT_COLS.find(c => c.key === sortCol);
  if (colDef && colDef.getter) {
    entries.sort((a, b) => {
      const va = colDef.getter(a, 0);
      const vb = colDef.getter(b, 0);
      if (typeof va === 'string') return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      return sortDir === 'asc' ? va - vb : vb - va;
    });
  }

  const baseline = entries.length >= 3 ? entries[2].total_score : (entries[entries.length - 1]?.total_score || 0);

  if (sortCol === 'delta') {
    entries.sort((a, b) => {
      const da = a.total_score - baseline;
      const db = b.total_score - baseline;
      return sortDir === 'asc' ? da - db : db - da;
    });
  }

  const peakTok = Math.max(0, ...entries.map(e => e.tok_per_sec || 0));
  const hasSelfJudged = entries.some(e => e.self_judged);

  // Build category data
  const cats = [...new Set(prompts.map(p => p.category).filter(Boolean))];
  const catPrompts = {};
  cats.forEach(c => { catPrompts[c] = prompts.filter(p => p.category === c); });

  // Sort indicator
  const sortArrow = (col) => {
    if (sortCol !== col) return '';
    return sortDir === 'desc' ? ' \u25bc' : ' \u25b2';
  };

  const thStyle = 'cursor:pointer;user-select:none';

  // -- Title strip --
  const titleStrip = html`
    <div class="title-strip" style="display:flex;align-items:baseline;justify-content:space-between">
      <div class="title-main" style="margin-bottom:0">cupel</div>
      <div style="font-family:var(--font-data);font-size:13px;color:var(--text-3)">${hardwareStr}</div>
    </div>`;

  // -- Chart tabs (only if entries exist) --
  const showRadarTab = cats.length >= 3;
  const chartTabs = entries.length > 0 ? html`
    <div class="chart-tabs">
      <button class="chart-tab ${chartTab === 'scatter' ? 'active' : ''}" onClick=${() => handleChartTab('scatter')}>Score vs Speed</button>
      <button class="chart-tab ${chartTab === 'bar' ? 'active' : ''}" onClick=${() => handleChartTab('bar')}>Overall Accuracy</button>
      ${showRadarTab ? html`
        <button class="chart-tab ${chartTab === 'radar' ? 'active' : ''}" onClick=${() => handleChartTab('radar')}>Category Fingerprint</button>
      ` : null}
    </div>
  ` : null;

  // -- Stats strip --
  const statsStrip = html`
    <div class="stats-strip">
      <div class="stat-cell">
        <div class="stat-label">Models</div>
        <div class="stat-val">${entries.length}</div>
      </div>
      <div class="stat-cell">
        <div class="stat-label">Prompts</div>
        <div class="stat-val">${prompts.length}</div>
      </div>
      <div class="stat-cell">
        <div class="stat-label">Peak Speed</div>
        <div class="stat-val">${peakTok.toFixed(0)} <span class="u">tok/s</span></div>
      </div>
      ${chartTabs}
      <div class="stat-cell" style="margin-left:${entries.length > 0 ? '0' : 'auto'};border-right:none;display:flex;flex-direction:column;gap:4px;align-items:flex-start">
        <label class="show-ex">
          <input type="checkbox" checked=${localOnly}
            onChange=${(e) => { setLocalOnly(e.target.checked); localStorage.setItem('cupel:dash-local-only', JSON.stringify(e.target.checked)); }} />
          local only
        </label>
        <label class="show-ex">
          <input type="checkbox" checked=${showExamples}
            onChange=${(e) => { setShowExamples(e.target.checked); localStorage.setItem('cupel:dash-show-examples', JSON.stringify(e.target.checked)); }} />
          show examples
        </label>
      </div>
    </div>`;

  const selfJudgeWarning = hasSelfJudged ? html`
    <div class="self-judge-warning">
      Self-judged scores are unreliable. A weak model will give itself 3/3 on answers it got wrong. Configure a stronger model as judge for accurate scores.
    </div>` : null;

  // -- Chart panel --
  // Initialize radarModels to top 3 if empty and entries exist
  if (radarModels.length === 0 && entries.length > 0) {
    const byScore = [...entries].sort((a, b) => b.pct - a.pct);
    // Use setTimeout to avoid setState during render
    setTimeout(() => setRadarModels(byScore.slice(0, 3).map(e => e.model)), 0);
  }

  // Chart rendering effect
  useEffect(() => {
    if (!chartTab || !chartLoaded || entries.length === 0) {
      if (currentChart) { currentChart.destroy(); currentChart = null; }
      return;
    }
    // Load Chart.js on first tab activation
    if (!ChartJS) {
      loadChartJS().then(() => setChartLoaded(true));
      return;
    }
    const canvas = document.getElementById('cupel-chart-canvas');
    if (!canvas) return;

    if (currentChart) currentChart.destroy();

    const config = buildChartConfig(chartTab, entries, radarModels, cats, catPrompts);
    if (config) currentChart = new ChartJS(canvas, config);

    return () => {
      if (currentChart) { currentChart.destroy(); currentChart = null; }
    };
  }, [chartTab, chartLoaded, entries.length, showExamples, localOnly, radarModels, sortCol, sortDir]);

  const barHeight = chartTab === 'bar' ? Math.max(320, entries.length * 28 + 40) : 320;

  let chartControls = null;
  if (chartTab === 'scatter') {
    chartControls = html`
      <div class="chart-controls">
        <span style="display:inline-flex;align-items:center;gap:5px">
          <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--text-3)"></span> local
        </span>
        <span style="display:inline-flex;align-items:center;gap:5px">
          <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--text-3)"></span> cloud
        </span>
        <span style="margin-left:auto;font-size:10px">top-right = fast + accurate</span>
      </div>`;
  } else if (chartTab === 'radar') {
    chartControls = html`
      <div class="chart-controls">
        <span style="text-transform:uppercase;letter-spacing:0.06em;font-size:10px">compare</span>
        ${entries.map((e, i) => {
          const color = CHART_PALETTE[i % CHART_PALETTE.length];
          const on = radarModels.includes(e.model);
          return html`<button class="radar-model-btn ${on ? 'on' : ''}"
            style="${on ? `background:${color}20;color:${color};border-color:${color}55` : ''}"
            onClick=${() => toggleRadarModel(e.model)}>${e.model.length > 20 ? e.model.substring(0, 18) + '\u2026' : e.model}</button>`;
        })}
      </div>`;
  }

  const chartPanel = chartTab ? html`
    <div class="chart-panel" key=${chartTab}>
      ${chartControls}
      <div class="chart-canvas-wrap" style="height:${barHeight}px">
        <canvas id="cupel-chart-canvas"></canvas>
      </div>
    </div>
  ` : null;

  // -- Leaderboard TABLE --
  const lbRows = entries.map((entry, i) => {
    const isWinner = i === 0;
    const isExample = entry.is_example;
    const pct = (entry.pct || 0).toFixed(1);
    const delta = entry.total_score - baseline;
    const hw = entry.hardware;
    const hwStr = hw && hw.name ? `${hw.name}${hw.memory ? ' ' + hw.memory : ''}` : '';

    const ticks = (entry.scores_by_prompt || []).map(sp => {
      const cls = sp.score != null ? `s${sp.score}` : 's0';
      return html`<div class="${cls}"></div>`;
    });

    return html`
      <tr class="${isWinner ? 'winner' : ''} ${isExample ? 'is-example' : ''} ${selectedModel === entry.model ? 'selected' : ''}"
          style="cursor:pointer;animation-delay:${(i * 0.04).toFixed(2)}s"
          onClick=${() => openModelDetail(entry)}>
        <td class="td-rank">${i + 1}</td>
        <td style="white-space:nowrap">
          <div class="td-model-name">${entry.model}${isExample ? html` <span class="ex-tag">example</span>` : null}</div>
          <div class="td-model-meta">${entry.judge_model ? html`<span style="${entry.self_judged ? 'color:var(--warn)' : ''}">${entry.self_judged ? 'self-judged' : entry.judge_model}</span>` : null}${entry.judge_model && hwStr ? ' \u00b7 ' : ''}${hwStr}</div>
        </td>
        <td class="td-bar">
          <div class="bar-track">
            <div class="bar-fill" style="width:${pct}%">
              <div class="bar-ticks">${ticks}</div>
            </div>
          </div>
        </td>
        <td class="td-pct">${pct}%</td>
        <td class="td-score">${entry.total_score}<small>/${entry.max_score}</small></td>
        <td class="td-speed">${(entry.tok_per_sec || 0).toFixed(0)} <small>tok/s</small></td>
        <td class="td-speed">${(entry.avg_time || 0).toFixed(1)}s</td>
        <td class="td-delta ${deltaClass(delta)}">${deltaStr(delta)}</td>
      </tr>`;
  });

  const lb = html`
    <div class="lb"><table>
      <thead><tr>
        <th style="width:36px;${thStyle}" onClick=${() => toggleSort('rank')}>#${sortArrow('rank')}</th>
        <th style="width:1%;white-space:nowrap;${thStyle}" onClick=${() => toggleSort('model')}>Model${sortArrow('model')}</th>
        <th>Score</th>
        <th class="r" style="width:50px;${thStyle}" onClick=${() => toggleSort('pct')}>\u0025${sortArrow('pct')}</th>
        <th class="r" style="width:80px;${thStyle}" onClick=${() => toggleSort('pts')}>Pts${sortArrow('pts')}</th>
        <th class="r" style="width:100px;${thStyle}" onClick=${() => toggleSort('speed')}>Speed${sortArrow('speed')}</th>
        <th class="r" style="width:65px;${thStyle}" onClick=${() => toggleSort('avg')}>Avg${sortArrow('avg')}</th>
        <th class="r" style="width:45px;${thStyle}" onClick=${() => toggleSort('delta')}>${sortCol === 'delta' ? (sortDir === 'desc' ? '\u25bc' : '\u25b2') : '\u0394'}</th>
      </tr></thead>
      <tbody>
        ${lbRows}
      </tbody>
    </table></div>`;

  // -- Category breakdown (top 6 models) --
  const top6 = entries.slice(0, 6);
  const catTable = cats.length > 0 ? html`
    <div class="cat-section">
      <div class="cat-head">Score by category</div>
      <table class="cat-table">
        <thead><tr>
          <th>Category</th>
          ${top6.map(e => {
            let s = e.model;
            if (s.length > 16) s = s.substring(0, 14) + '\u2026';
            return html`<th>${s}</th>`;
          })}
        </tr></thead>
        <tbody>
          ${cats.map(cat => {
            const cp = catPrompts[cat];
            const maxCat = cp.length * 3;
            return html`<tr>
              <td><div class="cat-name">
                <span class="cat-dot" style="background:${CAT_COLORS[cat] || 'var(--text-3)'}"></span>
                ${catLabel(cat)}
                <span class="cat-n">${cp.length}p</span>
              </div></td>
              ${top6.map(entry => {
                const catScore = cp.reduce((acc, p) => {
                  const sp = (entry.scores_by_prompt || []).find(s => s.id === p.id);
                  return acc + (sp && sp.score != null ? sp.score : 0);
                }, 0);
                const pctCat = maxCat > 0 ? catScore / maxCat * 100 : 0;
                const color = pctCat >= 75 ? 'var(--score-3-fg)' : pctCat >= 55 ? 'var(--score-2-fg)' : pctCat >= 35 ? 'var(--score-1-fg)' : 'var(--score-0-fg)';
                return html`<td class="cat-cell">
                  <div class="cat-fill" style="background:${color};width:${pctCat.toFixed(0)}%"></div>
                  <span class="cat-val" style="color:${color}">${catScore}/${maxCat}</span>
                </td>`;
              })}
            </tr>`;
          })}
        </tbody>
      </table>
    </div>` : null;

  // -- Detail panel --
  const detail = detailData ? html`
    <div class="detail-panel open">
      <div class="detail-header">
        <div>
          <div class="detail-pid">${detailData.hardware ? `${detailData.hardware.name || ''} ${detailData.hardware.memory || ''}`.trim() : ''}</div>
          <div class="detail-title">${detailData.model}</div>
          <div class="detail-model">${detailData.judge_model ? `judged by ${detailData.judge_model}` : ''}</div>
        </div>
        <button class="detail-close" onClick=${closeDetail}>\u2715</button>
      </div>
      <div class="detail-score-area">
        <div class="score-badge s${Math.min(3, Math.max(0, Math.round((detailData.total_score / detailData.max_score) * 3)))}">${detailData.total_score}</div>
        <div>
          <div class="detail-reason">Scored ${detailData.total_score}/${detailData.max_score} (${(detailData.pct || 0).toFixed(1)}%). ${(detailData.pct || 0) >= 66 ? 'Strong performance with correct, well-structured responses.' : 'Some prompts answered incorrectly or with shallow reasoning.'}</div>
          <div class="detail-metrics">
            <span>\u23f1 <b>${(detailData.avg_time || 0).toFixed(1)}s</b> avg</span>
            <span>\u26a1 <b>${(detailData.tok_per_sec || 0).toFixed(0)}</b> tok/s</span>
          </div>
        </div>
      </div>
      <div class="detail-body">
        ${(detailData.results || []).map(r => {
          const score = r.score != null ? r.score : null;
          const sc = score != null ? score : 0;
          return html`
            <div class="detail-sec">
              <div class="detail-sec-head">
                <span>#${r.id} ${r.title || ''}</span>
                <span>${score != null ? `${score}/3` : '\u2014'}</span>
              </div>
              <div class="detail-sec-body">
                ${r.category ? html`<div style="font-family:var(--font-data);font-size:12px;color:${CAT_COLORS[r.category] || 'var(--text-3)'};margin-bottom:6px">${catLabel(r.category)}</div>` : null}
                ${score != null ? html`
                  <div style="margin-bottom:8px">
                    ${[3, 2, 1, 0].map(s => html`
                      <div class="rubric-row ${s === sc ? 'matched' : ''}">
                        <span class="rubric-score s${s}">${s}</span>
                        <span class="rubric-text">${['Completely wrong or refuses to engage.', 'Partially correct but misses key detail.', 'Correct but shallow \u2014 covers main points.', 'Precise, thorough, includes nuance and insight.'][s]}</span>
                      </div>
                    `)}
                  </div>
                ` : null}
                ${r.judge_reason ? html`<div class="detail-reason" style="margin-bottom:6px">${r.judge_reason}</div>` : null}
                ${r.elapsed_seconds ? html`<div class="detail-metrics"><span>\u23f1 <b>${r.elapsed_seconds}s</b></span><span>\u26a1 <b>${r.completion_tokens || 0}</b> tok</span></div>` : null}
                ${r.response ? html`
                  <details style="margin-top:6px">
                    <summary style="font-family:var(--font-data);font-size:13px;color:var(--text-3);cursor:pointer">Response</summary>
                    <pre class="detail-pre">${r.response}</pre>
                  </details>
                ` : null}
                ${r.thinking ? html`
                  <details style="margin-top:4px">
                    <summary style="font-family:var(--font-data);font-size:13px;color:var(--text-3);cursor:pointer">Thinking</summary>
                    <pre class="detail-pre" style="color:var(--text-3)">${r.thinking}</pre>
                  </details>
                ` : null}
              </div>
            </div>`;
        })}
      </div>
    </div>` : null;

  // -- Welcome banner --
  const readyProviders = (providers || []).filter(p => p.status === 'online' && (p.models || []).length > 0);
  const blockedProviders = (providers || []).filter(p => p.status === 'online' && (p.models || []).length === 0 && p.auth_error);
  const PP_NAMES = { 8000: 'oMLX', 11434: 'Ollama', 1234: 'LM Studio', 30000: 'SGLang' };
  const PP_KEYS = { 8000: 'OMLX_API_KEY', 11434: 'OLLAMA_API_KEY', 1234: 'LM_STUDIO_API_KEY', 30000: 'SGLANG_API_KEY' };
  const provName = (p) => {
    const port = parseInt((p.url || '').split(':').pop(), 10);
    return PP_NAMES[port] || p.name || p.url;
  };
  const provKey = (p) => {
    const port = parseInt((p.url || '').split(':').pop(), 10);
    return PP_KEYS[port] || '';
  };

  const welcomeBanner = (state && state.first_run && !dismissed) ? html`
    <div style="background:var(--bg-panel);border:1px solid var(--border);padding:16px 20px;margin:0;border-radius:var(--radius-lg)">
      <div style="font-family:var(--font-label);font-size:15px;color:var(--text-0);margin-bottom:8px">
        \u25c6 Welcome to cupel \u2014 <span style="font-family:var(--font-data)">${entries.length}</span> models scored by Claude Opus on <span style="font-family:var(--font-data)">${prompts.length}</span> prompts. Real data, not mock.
      </div>
      ${readyProviders.length > 0 ? html`
        <div style="font-family:var(--font-label);font-size:14px;color:var(--text-2);margin-bottom:${blockedProviders.length > 0 ? '6' : '10'}px">
          cupel found: ${readyProviders.map((p, i) => html`${i > 0 ? ', ' : ''}<span style="font-family:var(--font-data);font-size:13px;color:var(--text-1)">${provName(p)}</span> <span style="font-family:var(--font-data);font-size:13px;color:var(--text-3)">(${(p.models || []).length} model${(p.models || []).length !== 1 ? 's' : ''})</span>`)}
        </div>
      ` : null}
      ${blockedProviders.length > 0 ? html`
        <div style="font-family:var(--font-data);font-size:13px;color:var(--warn);margin-bottom:10px">
          ${blockedProviders.map(p => {
            const key = provKey(p);
            const port = parseInt((p.url || '').split(':').pop(), 10);
            const refreshBtn = html`<button style="width:24px;height:24px;border-radius:50%;border:1px solid ${refreshing ? 'var(--accent)' : 'var(--border)'};background:transparent;color:var(--accent);cursor:pointer;display:inline-flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0;transition:transform 0.3s;${refreshing ? 'animation:spin 0.8s linear infinite' : ''}" onClick=${doRefresh} title="Re-read .env and refresh">\u21bb</button>`;
            if (p.auth_error === 'auth_failed') {
              return html`<div style="display:flex;align-items:center;gap:8px">${provName(p)} on :${port} \u2014 <span style="color:var(--bad)">${key} is set but rejected</span> (check the key value) ${refreshBtn}</div>`;
            }
            return html`<div style="display:flex;align-items:center;gap:8px">${provName(p)} on :${port} \u2014 set <span style="color:var(--text-1)">${key}</span> in .env or ~/.cupel/.env ${refreshBtn}</div>`;
          })}
        </div>
      ` : null}
      ${readyProviders.length === 0 && blockedProviders.length === 0 ? html`
        <div style="font-family:var(--font-label);font-size:14px;color:var(--text-2);margin-bottom:10px">
          No local servers found. Start Ollama, oMLX, or LM Studio to bench local models.
        </div>
      ` : null}
      <div style="display:flex;gap:12px;align-items:center">
        ${readyProviders.length > 0 ? html`
          <button class="btn-primary" onClick=${() => { location.hash = '#/run'; }}>Bench your models \u2192</button>
        ` : null}
        <button class="${readyProviders.length > 0 ? 'btn-ghost' : 'btn-primary'}" onClick=${() => { location.hash = '#/settings?addProvider'; }}>Add provider \u2192</button>
        <span style="font-family:var(--font-label);font-size:13px;color:var(--text-3);cursor:pointer;margin-left:auto"
          onClick=${() => setDismissed(true)}>dismiss</span>
      </div>
    </div>
  ` : null;

  return html`
    <div class="page" style="display:flex;height:100%">
      <div style="flex:1;overflow-y:auto">
        ${titleStrip}
        ${welcomeBanner}
        ${statsStrip}
        ${selfJudgeWarning}
        ${chartPanel}
        ${lb}
        ${catTable}
      </div>
      ${detail}
    </div>`;
}

export default Dashboard;
