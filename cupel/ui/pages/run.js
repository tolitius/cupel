const { h, useState, useEffect, useCallback, html } = window.__preact;

const CAT_COLORS = {
  multimodal: "#c77dba", security: "#d4845a", clojure_code: "#8b7ec8",
  distributed_systems: "#5b9bd5", ml_architecture: "#4cb89a", python_coding: "#c9a033",
  business_logic: "#c45050", clojure_ecosystem: "#9b6ec8", frontend_architecture: "#4aa3d5",
  domain_knowledge: "#d46b7a", system_design: "#3bbfa0", observability: "#8bb840",
  networking: "#b86ec8", math_estimation: "#c9b040", diagnostic_reasoning: "#5ab88b",
  meta: "#808080", chemistry: "#c87da0", assistant_competence: "#4ac4c4",
};

// Category groups — concise groupings of related categories
const CAT_GROUPS = [
  { key: 'coding', label: 'coding', cats: ['clojure_code', 'python_coding', 'clojure_ecosystem'], color: '#8b7ec8' },
  { key: 'systems', label: 'systems', cats: ['system_design', 'distributed_systems', 'frontend_architecture'], color: '#5b9bd5' },
  { key: 'reasoning', label: 'reasoning', cats: ['diagnostic_reasoning', 'math_estimation', 'business_logic'], color: '#5ab88b' },
  { key: 'science', label: 'science & ml', cats: ['ml_architecture', 'chemistry'], color: '#4cb89a' },
  { key: 'ops', label: 'ops & security', cats: ['security', 'networking', 'observability'], color: '#d4845a' },
  { key: 'general', label: 'general', cats: ['domain_knowledge', 'assistant_competence', 'meta', 'multimodal'], color: '#808080' },
];

// Build a reverse lookup: category -> group key
const CAT_TO_GROUP = {};
CAT_GROUPS.forEach(g => g.cats.forEach(c => { CAT_TO_GROUP[c] = g.key; }));

function connectSSE(jobId, onEvent) {
  console.log('[cupel] SSE connecting to job', jobId);
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'error') console.error('[cupel] job error:', data.error || data);
    else if (data.status === 'error') console.warn('[cupel] prompt error:', data.model, '#' + data.prompt_id, data);
    onEvent(data);
    if (data.type === 'complete' || data.type === 'error' || data.type === 'cancelled') {
      console.log('[cupel] SSE closed:', data.type);
      es.close();
    }
  };
  es.onerror = (err) => {
    console.error('[cupel] SSE connection error \u2014 stream dropped', err);
    es.close();
  };
  return es;
}

function catLabel(c) { return (c || '').replace(/_/g, ' '); }

function shortenModel(name) {
  let s = name;
  s = s.replace('Qwen3.5-', 'Q').replace('Qwen3-', 'Q');
  s = s.replace('Nemotron-Cascade-2-', 'Nemo\u00b7');
  s = s.replace('-MXFP4-Q8', '\u00b7MX');
  s = s.replace('-4bit', '\u00b74b').replace('-8bit', '\u00b78b').replace('-bf16', '\u00b7bf16').replace('-6bit', '\u00b76b').replace('-5bit', '\u00b75b');
  return s.length > 20 ? s.slice(0, 18) + '\u2026' : s;
}

function shortTitle(title, maxLen = 30) {
  if (!title) return '';
  return title.length > maxLen ? title.slice(0, maxLen - 2) + '\u2026' : title;
}

const SCORE_COLORS = {
  3: { bg: 'var(--score-3-bg)', fg: 'var(--score-3-fg)' },
  2: { bg: 'var(--score-2-bg)', fg: 'var(--score-2-fg)' },
  1: { bg: 'var(--score-1-bg)', fg: 'var(--score-1-fg)' },
  0: { bg: 'var(--score-0-bg)', fg: 'var(--score-0-fg)' },
};

const PROVIDER_INFO = {
  8000:  { name: 'oMLX',      avatar: '\u2B22' },
  11434: { name: 'Ollama',    avatar: '\u25CE' },
  1234:  { name: 'LM Studio', avatar: '\u25A3' },
  30000: { name: 'SGLang',    avatar: '\u25C8' },
};

function guessProvider(p) {
  const port = p.port || parseInt((p.url || '').split(':').pop(), 10);
  return PROVIDER_INFO[port] || { name: p.name || p.url || 'unknown', avatar: p.source === 'external' ? '\u2601' : '\u25CB' };
}

function RunPage({ providers: initProviders }) {
  const [providers, setProviders] = useState(initProviders || []);
  const [selected, setSelected] = useState(() => JSON.parse(localStorage.getItem('cupel:bench-models') || '[]'));
  const [selectedCats, setSelectedCats] = useState(() => { const s = localStorage.getItem('cupel:bench-cats'); return s ? new Set(JSON.parse(s)) : new Set(['all']); });
  const [pickedPromptIds, setPickedPromptIds] = useState(() => { const s = localStorage.getItem('cupel:bench-pick'); return s ? new Set(JSON.parse(s)) : new Set(); });
  const [pickMode, setPickMode] = useState(false);
  const [judgeModel, setJudgeModel] = useState(() => localStorage.getItem('cupel:judge-model') || null);
  const [thinkingMode, setThinkingMode] = useState(() => localStorage.getItem('cupel:thinking-mode') || 'default');
  const [thinkingBudget, setThinkingBudget] = useState(() => parseInt(localStorage.getItem('cupel:thinking-budget')) || 4096);
  const [jobId, setJobId] = useState(null);
  const [grid, setGrid] = useState({});
  const [complete, setComplete] = useState(null);
  const [running, setRunning] = useState(false);
  const [hardware, setHardware] = useState(null);
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [starterPrompts, setStarterPrompts] = useState([]);
  const [fullPrompts, setFullPrompts] = useState([]);
  const [benchPrompts, setBenchPrompts] = useState([]);
  const [benchModels, setBenchModels] = useState([]);
  const [hasEvalSet, setHasEvalSet] = useState(true);
  const [detailPid, setDetailPid] = useState(null);
  const [detailData, setDetailData] = useState(null);
  const [lastRun, setLastRun] = useState(() => {
    try { return JSON.parse(localStorage.getItem('cupel:last-run')); } catch { return null; }
  });

  useEffect(() => { if (initProviders) setProviders(initProviders); }, [initProviders]);

  // Fetch hardware + eval sets + config + state on mount
  useEffect(() => {
    fetch('/api/hardware').then(r => r.json()).then(setHardware).catch(() => {});
    fetch('/api/state').then(r => r.json()).then(st => setHasEvalSet(st.has_eval_set)).catch(() => {});
    Promise.all([
      fetch('/api/eval-set').then(r => r.json()).catch(() => ({ prompts: [] })),
      fetch('/api/eval-set?variant=starter').then(r => r.json()).catch(() => ({ prompts: [] })),
      fetch('/api/config').then(r => r.json()).catch(() => ({})),
    ]).then(([full, starter, cfg]) => {
      setFullPrompts(full.prompts || []);
      setStarterPrompts(starter.prompts || []);
      // Pre-populate judge from config only if user hasn't picked one
      if (!localStorage.getItem('cupel:judge-model') && cfg.judge && cfg.judge.model) {
        setJudgeModel(cfg.judge.model);
      }
    });
  }, []);

  // Reconnect to active job on mount (waits for prompt data to load)
  useEffect(() => {
    if (fullPrompts.length === 0) return;
    fetch('/api/jobs').then(r => r.json()).then(jobs => {
      const active = (jobs || []).find(j => j.status === 'running');
      if (active) {
        setJobId(active.id);
        setRunning(true);
        // Restore models from server
        if (active.models && active.models.length > 0) {
          setBenchModels(active.models);
        }
        // Restore prompt list from server
        if (active.prompt_ids && active.prompt_ids.length > 0) {
          const lookup = {};
          fullPrompts.forEach(p => { lookup[p.id] = p; });
          starterPrompts.forEach(p => { if (!lookup[p.id]) lookup[p.id] = p; });
          const restored = active.prompt_ids
            .map(id => lookup[id] || { id, title: `Prompt ${id}`, category: '' })
            .filter(Boolean);
          if (restored.length > 0) setBenchPrompts(restored);
        }
        // Replay existing progress into grid, then connect SSE
        if (active.progress_count > 0) {
          fetch(`/api/jobs/${active.id}`).then(r => r.json()).then(detail => {
            if (detail.progress) {
              detail.progress.forEach(ev => handleEvent(ev));
            }
            connectSSE(active.id, handleEvent);
          }).catch(() => connectSSE(active.id, handleEvent));
        } else {
          connectSSE(active.id, handleEvent);
        }
      }
    }).catch(() => {});
  }, [fullPrompts, starterPrompts]);

  // Auto-select one model per discovered local provider on first visit
  const [userChangedModels, setUserChangedModels] = useState(() => localStorage.getItem('cupel:bench-models') !== null);
  useEffect(() => {
    if (userChangedModels || selected.length > 0 || localStorage.getItem('cupel:bench-models')) return;
    const localOnline = providers.filter(p => p.status === 'online' && p.source === 'local');
    if (localOnline.length === 0) return;
    const autoSelected = localOnline
      .filter(p => (p.models || []).length > 0)
      .map(p => ({ model: p.models[0], url: p.url }));
    if (autoSelected.length > 0) setSelected(autoSelected);
  }, [providers]);

  // Default to "starter" chip when starter prompts are available
  const [userChangedCats, setUserChangedCats] = useState(() => localStorage.getItem('cupel:bench-cats') !== null);
  useEffect(() => {
    if (userChangedCats || localStorage.getItem('cupel:bench-cats')) return;
    if (starterPrompts.length > 0 && starterPrompts.length < fullPrompts.length) {
      setSelectedCats(new Set(['starter']));
    }
  }, [starterPrompts, fullPrompts]);

  useEffect(() => {
    if (!modelDropdownOpen) return;
    const handleClick = (e) => {
      const container = document.querySelector('.model-dropdown-container');
      if (container && !container.contains(e.target)) {
        setModelDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [modelDropdownOpen]);

  const refreshProviders = useCallback(() => {
    fetch('/api/providers').then(r => r.json()).then(setProviders).catch(() => {});
  }, []);

  // Category group counts (only groups that have prompts)
  const groupCounts = {};
  CAT_GROUPS.forEach(g => {
    groupCounts[g.key] = fullPrompts.filter(p => g.cats.includes(p.category)).length;
  });
  const activeGroups = CAT_GROUPS.filter(g => groupCounts[g.key] > 0);
  const starterIds = new Set(starterPrompts.map(p => p.id));

  const toggleCat = useCallback((cat) => {
    setUserChangedCats(true);
    setSelectedCats(prev => {
      let next;
      if (cat === 'all') { next = new Set(['all']); }
      else {
        next = new Set(prev);
        next.delete('all');
        if (next.has(cat)) next.delete(cat); else next.add(cat);
        if (next.size === 0) next = new Set(['all']);
      }
      localStorage.setItem('cupel:bench-cats', JSON.stringify([...next]));
      return next;
    });
  }, []);

  // Filtered prompts based on selected groups
  let filteredPrompts;
  if (selectedCats.has('all')) {
    filteredPrompts = fullPrompts;
  } else {
    const selectedCategories = new Set();
    CAT_GROUPS.forEach(g => {
      if (selectedCats.has(g.key)) g.cats.forEach(c => selectedCategories.add(c));
    });
    filteredPrompts = fullPrompts.filter(p => {
      if (selectedCats.has('starter') && starterIds.has(p.id)) return true;
      return selectedCategories.has(p.category);
    });
  }
  if (pickMode) {
    filteredPrompts = fullPrompts.filter(p => pickedPromptIds.has(p.id));
  }
  const promptCount = filteredPrompts.length;

  // Provider-aware model lists
  const availableProviders = providers.filter(p => p.status === 'online' || p.source === 'external');
  const allModels = availableProviders.flatMap(p =>
    (p.models || []).map(m => ({ model: m, url: p.url, provider: p.name, source: p.source }))
  );

  const toggleModel = useCallback((model, url) => {
    setUserChangedModels(true);
    setSelected(prev => {
      const exists = prev.find(s => s.model === model && s.url === url);
      const next = exists
        ? prev.filter(s => !(s.model === model && s.url === url))
        : [...prev, { model, url }];
      localStorage.setItem('cupel:bench-models', JSON.stringify(next));
      return next;
    });
  }, []);

  const stopRun = useCallback(() => {
    if (!jobId) return;
    fetch(`/api/jobs/${jobId}`, { method: 'DELETE' }).then(r => r.json()).catch(() => {});
  }, [jobId]);

  function handleEvent(data) {
    if (data.type === 'complete') { setRunning(false); setComplete(data); return; }
    if (data.type === 'cancelled') {
      setRunning(false);
      setComplete({ cancelled: true, result_files: data.result_files });
      // Clear any cells still stuck in 'running' status
      setGrid(prev => {
        const next = { ...prev };
        for (const k in next) { if (next[k].status === 'running') next[k] = { status: 'skip' }; }
        return next;
      });
      return;
    }
    if (data.type === 'error') { setRunning(false); setComplete({ error: data.error || data.message || 'Run failed' }); return; }
    const key = `${data.model}::${data.prompt_id}`;
    setBenchModels(prev => prev.includes(data.model) ? prev : [...prev, data.model]);
    if (data.status === 'running') setGrid(prev => ({ ...prev, [key]: { status: 'running' } }));
    else if (data.status === 'judging') setGrid(prev => ({ ...prev, [key]: { status: 'judging' } }));
    else if (data.status && data.status.startsWith && data.status.startsWith('scored:')) {
      const score = parseInt(data.status.split(':')[1], 10);
      setGrid(prev => ({ ...prev, [key]: { status: 'done', score, elapsed: data.elapsed } }));
    }
    else if (data.status === 'error' || data.status === 'skip') setGrid(prev => ({ ...prev, [key]: { status: data.status } }));
    else if (data.status && data.elapsed) setGrid(prev => ({ ...prev, [key]: { status: 'done_unscored', elapsed: data.elapsed } }));
  }

  const thinkingValue = thinkingMode === 'default' ? null : thinkingMode === 'off' ? 0 : thinkingBudget;

  const startRun = useCallback(() => {
    const models = selected.map(s => s.model);
    if (!models.length || !filteredPrompts.length) return;
    const model_urls = {};
    selected.forEach(s => { model_urls[s.model] = s.url; });
    const promptIds = filteredPrompts.map(p => p.id);
    setBenchPrompts(filteredPrompts);
    setBenchModels(models);
    setGrid({}); setComplete(null); setRunning(true); setDetailPid(null); setDetailData(null); setLastRun(null);
    // Note: selected, selectedCats, judgeModel are intentionally NOT cleared
    fetch('/api/jobs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'run', models, model_urls, eval_set: 'full', prompts: promptIds, thinking: thinkingValue, judge_model: judgeModel })
    })
    .then(r => r.json())
    .then(data => { setJobId(data.id); connectSSE(data.id, handleEvent); })
    .catch((err) => { console.error('[cupel] failed to start job:', err); setRunning(false); setComplete({ error: 'Failed to start job' }); });
  }, [selected, selectedCats, judgeModel, filteredPrompts, thinkingMode, thinkingBudget]);

  // Derive prompt list: explicit for new runs, derived from grid for reconnections
  const activePrompts = benchPrompts.length > 0 ? benchPrompts : (() => {
    if (!running && !complete) return [];
    const pids = [...new Set(Object.keys(grid).map(k => parseInt(k.split('::')[1])))].sort((a, b) => a - b);
    if (pids.length === 0) return [];
    const lookup = {};
    fullPrompts.forEach(p => { lookup[p.id] = p; });
    starterPrompts.forEach(p => { if (!lookup[p.id]) lookup[p.id] = p; });
    return pids.map(id => lookup[id] || { id, title: `Prompt ${id}`, category: '' });
  })();

  const activeModels = benchModels;

  // ── Prompt detail panel ──
  const fetchPromptDetail = useCallback((pid) => {
    if (!jobId || pid == null) return;
    fetch(`/api/jobs/${jobId}/prompt-detail/${pid}`)
      .then(r => r.json())
      .then(setDetailData)
      .catch(() => {});
  }, [jobId]);

  const closeDetail = useCallback(() => { setDetailPid(null); setDetailData(null); }, []);

  const detailRowVersion = detailPid
    ? activeModels.map(m => { const c = grid[`${m}::${detailPid}`]; return c ? c.status : '-'; }).join(',')
    : '';
  useEffect(() => { if (detailPid && jobId) fetchPromptDetail(detailPid); }, [detailPid, detailRowVersion]);

  // ── Last run: save on completion, restore on click ──
  useEffect(() => {
    if (!complete || complete.error) return;
    if (activePrompts.length === 0 || activeModels.length === 0) return;
    localStorage.setItem('cupel:last-run', JSON.stringify({
      models: activeModels,
      prompts: activePrompts.map(p => ({ id: p.id, title: p.title, category: p.category })),
      grid,
      completedAt: new Date().toISOString(),
      jobId,
      cancelled: !!complete.cancelled,
    }));
  }, [complete]);

  const restoreLastRun = useCallback(() => {
    if (!lastRun) return;
    setBenchModels(lastRun.models || []);
    const lookup = {};
    fullPrompts.forEach(p => { lookup[p.id] = p; });
    setBenchPrompts((lastRun.prompts || []).map(p => lookup[p.id] || p));
    setGrid(lastRun.grid || {});
    setJobId(lastRun.jobId || null);
    setComplete(lastRun.cancelled ? { cancelled: true } : { result_files: [] });
    setLastRun(null);
  }, [lastRun, fullPrompts]);

  // Progress counts
  let doneCount = 0;
  const totalCells = activePrompts.length * activeModels.length;
  Object.values(grid).forEach(cell => {
    if (['done', 'done_unscored', 'error', 'skip'].includes(cell.status)) doneCount++;
  });

  // \u2500\u2500 Render helpers \u2500\u2500

  const cellMinW = activeModels.length <= 2 ? 90 : 56;
  const cellBase = `text-align:center;min-width:${cellMinW}px;padding:5px 4px;font-family:var(--font-data);font-size:15px;font-weight:600;border-bottom:1px solid var(--border-subtle)`;

  const renderPgCell = (model, pid) => {
    const cell = grid[`${model}::${pid}`];
    if (!cell) return html`<td style="${cellBase};color:var(--text-3);opacity:0.3">\u00b7</td>`;
    if (cell.status === 'running') return html`<td style="${cellBase}"><span class="bench-spinner"></span></td>`;
    if (cell.status === 'judging') return html`<td style="${cellBase};color:var(--accent)">jdg</td>`;
    if (cell.status === 'done') {
      const c = SCORE_COLORS[cell.score] || SCORE_COLORS[0];
      return html`<td style="${cellBase};background:${c.bg};color:${c.fg}">${cell.score}</td>`;
    }
    if (cell.status === 'error') return html`<td title="prompt failed" style="${cellBase};color:var(--bad)">ERR</td>`;
    if (cell.status === 'skip') return html`<td style="${cellBase};color:var(--text-3)">\u2014</td>`;
    if (cell.status === 'done_unscored') {
      const t = cell.elapsed >= 100 ? `${Math.round(cell.elapsed)}s` : `${Number(cell.elapsed).toFixed(1)}s`;
      return html`<td style="${cellBase}"><span style="color:var(--good);font-weight:700">\u2713</span> <span style="color:var(--text-3);font-weight:400">${t}</span></td>`;
    }
    return html`<td style="${cellBase};color:var(--text-3);opacity:0.3">\u00b7</td>`;
  };

  const modelTotal = (model) => {
    let total = 0, count = 0;
    activePrompts.forEach(p => {
      const cell = grid[`${model}::${p.id}`];
      if (cell && cell.status === 'done') { total += cell.score; count++; }
    });
    if (count === 0) return html`<span style="color:var(--text-3)">\u2014</span>`;
    const pct = count * 3 > 0 ? (total / (count * 3) * 100).toFixed(0) : 0;
    return html`<span>${total}<small style="color:var(--text-3);font-weight:400">/${count * 3}</small> <small style="color:var(--text-3);font-weight:400">${pct}%</small></span>`;
  };

  const sectionLabel = 'font-family:var(--font-label);font-size:13px;font-weight:600;color:var(--text-2);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px';

  // \u2500\u2500 Category chip style \u2500\u2500
  const chipStyle = (active) => `display:inline-flex;align-items:center;gap:5px;padding:4px 10px;cursor:pointer;font-family:var(--font-data);font-size:13px;border:1px solid ${active ? 'var(--accent)' : 'var(--border)'};background:${active ? 'var(--accent-dim)' : 'var(--bg-alt)'};color:${active ? 'var(--accent)' : 'var(--text-2)'};margin:0 6px 6px 0;border-radius:var(--radius-md)`;

  // \u2500\u2500 Status indicator \u2500\u2500

  const statusDot = (color) => html`<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color}"></span>`;

  const runningKey = Object.entries(grid).find(([k, v]) => v.status === 'running');
  const runningPromptId = runningKey ? parseInt(runningKey[0].split('::')[1]) : null;
  const runningPrompt = runningPromptId ? activePrompts.find(p => p.id === runningPromptId) : null;
  const runningTitle = runningPrompt ? (runningPrompt.title.length > 35 ? runningPrompt.title.slice(0, 33) + '\u2026' : runningPrompt.title) : '';
  const runningLabel = runningPrompt ? ` \u00b7 #${runningPrompt.id} ${runningTitle}` : '';

  const statusBar = running ? [
      html`<span class="bench-pulse">${statusDot('var(--warn)')}</span>`,
      html`<span style="font-family:var(--font-label);font-size:14px;color:var(--warn)">running</span>`,
    ] : complete && !complete.error && !complete.cancelled ? [
      statusDot('var(--good)'),
      html`<span style="font-family:var(--font-label);font-size:14px;color:var(--good)">complete</span>`,
    ] : complete && complete.cancelled ? [
      statusDot('var(--warn)'),
      html`<span style="font-family:var(--font-label);font-size:14px;color:var(--warn)">stopped</span>`,
    ] : complete && complete.error ? [
      statusDot('var(--bad)'),
      html`<span style="font-family:var(--font-label);font-size:14px;color:var(--bad)">error</span>`,
    ] : null;

  // \u2500\u2500 Progress panel \u2500\u2500

  const showProgress = (running || complete) && (activePrompts.length > 0 || (complete && complete.error));

  const progressPanel = showProgress ? html`
    <div style="border-bottom:2px solid var(--accent);background:var(--bg-panel)">
      <!-- Status bar -->
      <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 20px ${(complete && complete.error && activePrompts.length === 0) ? '14px' : '10px'}">
        <div style="display:flex;align-items:center;gap:10px">
          ${statusBar}
          ${totalCells > 0 ? html`
            <span style="font-family:var(--font-data);font-size:13px;color:var(--text-3)">
              ${doneCount}/${totalCells} \u00b7 ${activePrompts.length} prompts \u00b7 ${activeModels.length} model${activeModels.length !== 1 ? 's' : ''}${runningLabel}
            </span>
          ` : null}
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          ${running ? html`
            <button class="btn-ghost" style="font-size:13px;padding:3px 10px;color:var(--bad);border-color:var(--bad)" onClick=${stopRun}>Stop</button>
          ` : null}
          ${complete && !complete.error ? html`
            <a href="#/dashboard" class="btn-primary" style="text-decoration:none;font-size:13px;padding:5px 14px">Dashboard</a>
            <button class="btn-ghost" style="font-size:13px;padding:3px 10px" onClick=${() => {
              setComplete(null); setRunning(false); setBenchPrompts([]); setBenchModels([]); setGrid({});
            }}>New run</button>
          ` : null}
          ${complete && complete.error ? html`
            <button class="btn-ghost" style="font-size:13px;padding:3px 10px" onClick=${() => { setComplete(null); setRunning(false); }}>Dismiss</button>
          ` : null}
        </div>
      </div>

      ${totalCells > 0 ? html`
        <div style="height:6px;background:var(--border-subtle);margin:0 20px;border-radius:3px;overflow:hidden">
          <div style="height:100%;background:var(--accent);transition:width 0.4s ease;border-radius:3px;width:${(doneCount / totalCells * 100).toFixed(1)}%"></div>
        </div>
      ` : null}

      ${complete && complete.error ? html`
        <div style="padding:0 20px 14px;font-family:var(--font-data);font-size:14px;color:var(--bad)">${complete.error}</div>
      ` : null}

      <!-- Progress grid -->
      ${activePrompts.length > 0 ? html`
        <div style="padding:0 20px 16px;overflow-x:auto">
          <table style="border-collapse:collapse;width:100%">
            <thead>
              <tr style="border-bottom:1px solid var(--border)">
                <th style="text-align:left;padding:6px 6px 6px 0;font-family:var(--font-data);font-size:12px;color:var(--text-3);width:24px">#</th>
                <th style="text-align:left;padding:6px 12px 6px 0;font-family:var(--font-label);font-size:12px;color:var(--text-3);text-transform:uppercase;letter-spacing:0.05em">Prompt</th>
                ${activeModels.map(m => html`
                  <th style="text-align:center;padding:6px 4px;font-family:var(--font-data);font-size:14px;color:var(--text-2);min-width:${cellMinW}px;white-space:nowrap">${activeModels.length <= 3 ? m : shortenModel(m)}</th>
                `)}
              </tr>
            </thead>
            <tbody>
              ${activePrompts.map(p => {
                const rowCells = activeModels.map(m => grid[`${m}::${p.id}`]);
                const allDone = rowCells.length > 0 && rowCells.every(c => c && ['done', 'done_unscored', 'error', 'skip'].includes(c.status));
                const anyRunning = rowCells.some(c => c && c.status === 'running');
                const isSelected = detailPid === p.id;
                const rowStyle = (allDone && !isSelected ? 'opacity:0.55;' : '') + (isSelected ? 'background:var(--bg-hover);' : '');
                const idBorder = anyRunning ? 'border-left:3px solid var(--warn);padding-left:6px' : '';
                return html`
                <tr style="${rowStyle}cursor:pointer" onClick=${() => {
                  if (detailPid === p.id) { setDetailPid(null); setDetailData(null); }
                  else { setDetailPid(p.id); fetchPromptDetail(p.id); }
                }}>
                  <td style="padding:3px 6px 3px 0;font-family:var(--font-data);font-size:12px;color:var(--text-3);border-bottom:1px solid var(--border-subtle);${idBorder}">${p.id}</td>
                  <td style="padding:3px 12px 3px 0;font-family:var(--font-data);font-size:13px;color:var(--text-2);white-space:nowrap;border-bottom:1px solid var(--border-subtle)" title="${p.title || ''}">
                    <span title="${catLabel(p.category)}" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:${CAT_COLORS[p.category] || 'var(--text-3)'};margin-right:6px;vertical-align:middle"></span>${shortTitle(p.title, activeModels.length <= 2 ? 60 : 30)}
                  </td>
                  ${activeModels.map(m => renderPgCell(m, p.id))}
                </tr>`;
              })}
            </tbody>
            <tfoot>
              <tr style="border-top:1px solid var(--border)">
                <td></td>
                <td style="padding:8px 12px 4px 0;font-family:var(--font-label);font-size:13px;font-weight:700;color:var(--text-2);text-transform:uppercase;letter-spacing:0.05em">Score</td>
                ${activeModels.map(m => html`
                  <td style="text-align:center;padding:8px 4px 4px;font-family:var(--font-data);font-size:15px;font-weight:700;color:var(--text)">${modelTotal(m)}</td>
                `)}
              </tr>
            </tfoot>
          </table>
        </div>
      ` : null}
    </div>
  ` : null;

  // \u2500\u2500 Main render \u2500\u2500

  return html`
    <div class="page" style="overflow-y:auto">
      <!-- Header: title + bench button -->
      <div style="display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--border)">
        <div style="font-family:var(--font-label);font-size:17px;font-weight:700;color:var(--text)">Bench It</div>
        <div style="display:flex;align-items:center;gap:16px">
          ${hardware ? html`
            <span style="font-family:var(--font-data);font-size:13px;color:var(--text-3)">
              ${hardware.name || ''}${hardware.memory ? ' \u00b7 ' + hardware.memory : ''}
            </span>
          ` : null}
          ${!running ? html`
            <button class="btn-primary" disabled=${selected.length === 0 || promptCount === 0} onClick=${startRun}>
              Bench ${promptCount} prompts on ${selected.length} model${selected.length !== 1 ? 's' : ''}
            </button>
          ` : null}
        </div>
      </div>

      <!-- Progress panel (above config, only when active) -->
      ${progressPanel}

      ${!running && !complete && lastRun ? html`
        <div style="padding:12px 20px;background:var(--bg-panel);border-bottom:1px solid var(--border);cursor:pointer;display:flex;align-items:center;gap:8px"
          onClick=${restoreLastRun}>
          <span style="color:var(--text-3);font-size:11px">\u25B6</span>
          <span style="font-family:var(--font-label);font-size:13px;color:var(--text-2)">
            Last run</span>
          <span style="font-family:var(--font-data);font-size:13px;color:var(--text-3)">
            ${(lastRun.models || []).join(', ')}
            \u00b7 ${(lastRun.prompts || []).length} prompts
            ${lastRun.completedAt ? '\u00b7 ' + new Date(lastRun.completedAt).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : ''}
            ${lastRun.cancelled ? ' (stopped)' : ''}
          </span>
        </div>
      ` : null}

      ${!hasEvalSet ? html`
        <div style="padding:14px 20px;background:var(--bg-panel);border-bottom:1px solid var(--border)">
          <div style="font-family:var(--font-label);font-size:14px;color:var(--text-2);margin-bottom:6px">
            No custom prompts yet \u2014 running on cupel's built-in set (${fullPrompts.length} prompts across coding, systems, reasoning, and more).
          </div>
          <div style="font-family:var(--font-label);font-size:14px;color:var(--text-3)">
            Want your own? <a href="#/author" style="color:var(--accent);text-decoration:none">Author prompts \u2192</a>
          </div>
        </div>
      ` : null}

      <!-- 1. MODELS (moved to top) -->
      <div style="padding:16px 20px;border-bottom:1px solid var(--border)">
        <div style="${sectionLabel}">Models</div>
        ${allModels.length === 0 ? html`
          <div style="color:var(--text-3);font-family:var(--font-label);font-size:14px">No models available \u2014 start a provider or add one in Settings</div>
        ` : html`
          <div class="model-dropdown-container" style="position:relative;max-width:500px">
            <div class="input" style="cursor:pointer;display:flex;align-items:center;justify-content:space-between;min-height:38px"
              onClick=${() => setModelDropdownOpen(!modelDropdownOpen)}>
              <span style="color:${selected.length > 0 ? 'var(--text)' : 'var(--text-3)'}">
                ${selected.length > 0
                  ? `${selected.length} model${selected.length !== 1 ? 's' : ''} selected`
                  : 'Select models to bench...'}
              </span>
              <span style="color:var(--text-3);font-size:12px">${modelDropdownOpen ? '\u25B2' : '\u25BC'}</span>
            </div>
            ${modelDropdownOpen ? html`
              <div style="position:absolute;top:100%;left:0;right:0;z-index:10;background:var(--bg-alt);border:1px solid var(--border);border-top:none;max-height:300px;overflow-y:auto;border-radius:0 0 var(--radius-md) var(--radius-md)">
                ${availableProviders.map(p => {
                  if (!p.models || p.models.length === 0) return null;
                  return html`
                    <div style="padding:5px 12px;font-family:var(--font-label);font-size:12px;color:var(--text-3);text-transform:uppercase;letter-spacing:0.05em;background:var(--bg-panel);border-bottom:1px solid var(--border)">
                      ${p.name || p.url}${p.source === 'external' ? ' (API)' : ''}
                    </div>
                    ${(p.models || []).map(model => {
                      const checked = !!selected.find(s => s.model === model && s.url === p.url);
                      return html`
                        <label style="display:flex;align-items:center;gap:8px;padding:7px 12px;cursor:pointer;font-family:var(--font-data);font-size:14px;color:var(--text);border-bottom:1px solid var(--border-subtle)"
                          onClick=${(e) => { e.stopPropagation(); }}>
                          <input class="checkbox" type="checkbox" checked=${checked}
                            onChange=${() => toggleModel(model, p.url)} />
                          <span style="color:${checked ? 'var(--text)' : 'var(--text-2)'}">${model}</span>
                        </label>`;
                    })}
                  `;
                })}
              </div>
            ` : null}
          </div>
          ${selected.length > 0 ? html`
            <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px">
              ${selected.map(s => html`
                <span style="font-family:var(--font-data);font-size:13px;color:var(--text);background:var(--bg-alt);padding:3px 8px;border:1px solid var(--border);display:flex;align-items:center;gap:6px;border-radius:var(--radius-sm)">
                  ${s.model}
                  <span style="cursor:pointer;color:var(--text-3);font-size:14px" onClick=${() => toggleModel(s.model, s.url)}>\u00d7</span>
                </span>
              `)}
            </div>
          ` : null}
        `}
      </div>

      <!-- 2. IQ TASKS \u2014 grouped category chips -->
      <div style="padding:16px 20px;border-bottom:1px solid var(--border)">
        <div style="${sectionLabel}">IQ Tasks</div>
        <div style="display:flex;flex-wrap:wrap;${pickMode ? 'opacity:0.35;pointer-events:none' : ''}">
          <span style="${chipStyle(selectedCats.has('all'))}" onClick=${() => toggleCat('all')}>
            all (${fullPrompts.length})
          </span>
          ${starterPrompts.length > 0 && starterPrompts.length < fullPrompts.length ? html`
            <span style="${chipStyle(selectedCats.has('starter'))}" onClick=${() => toggleCat('starter')}>
              starter (${starterPrompts.length})
            </span>
          ` : null}
          ${activeGroups.map(g => html`
            <span style="${chipStyle(selectedCats.has(g.key))}" onClick=${() => toggleCat(g.key)}>
              <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:${g.color}"></span>
              ${g.label} (${groupCounts[g.key]})
            </span>
          `)}
        </div>
        ${!pickMode ? html`
          <div style="margin-top:4px">
            <span style="font-family:var(--font-data);font-size:13px;color:var(--accent);cursor:pointer" onClick=${() => setPickMode(true)}>
              + pick specific tasks${pickedPromptIds.size > 0 ? ` (${pickedPromptIds.size})` : ''}
            </span>
          </div>
        ` : html`
          <div style="margin-top:8px;display:flex;align-items:center;gap:10px">
            <span style="font-family:var(--font-label);font-size:13px;font-weight:600;color:var(--text-2)">picking specific tasks</span>
            <span style="font-family:var(--font-data);font-size:13px;color:var(--accent);cursor:pointer" onClick=${() => setPickMode(false)}>
              × back to categories
            </span>
          </div>
        `}
        ${pickMode ? html`
          <div style="margin-top:10px">
            <div style="max-height:260px;overflow-y:auto;border:1px solid var(--border);border-radius:var(--radius-md)">
              ${fullPrompts.map(p => {
                const checked = pickedPromptIds.has(p.id);
                return html`
                  <label style="display:flex;align-items:center;gap:8px;padding:6px 10px;cursor:pointer;font-family:var(--font-data);font-size:13px;color:var(--text);border-bottom:1px solid var(--border-subtle);background:${checked ? 'var(--accent-dim)' : 'transparent'}">
                    <input class="checkbox" type="checkbox" checked=${checked}
                      onChange=${() => {
                        const next = new Set(pickedPromptIds);
                        if (next.has(p.id)) next.delete(p.id); else next.add(p.id);
                        setPickedPromptIds(next);
                        localStorage.setItem('cupel:bench-pick', JSON.stringify([...next]));
                      }} />
                    <span style="font-size:12px;color:var(--text-3);min-width:24px">${p.id}</span>
                    <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:${CAT_COLORS[p.category] || 'var(--text-3)'};flex-shrink:0"></span>
                    <span>${p.title || ''}</span>
                  </label>`;
              })}
            </div>
          </div>
        ` : null}
      </div>

      <!-- 3. JUDGE -->
      <div style="padding:16px 20px;border-bottom:1px solid var(--border)">
        <div style="${sectionLabel}">Judge</div>
        <select class="input" style="max-width:400px" value=${judgeModel || ''} onChange=${(e) => { const v = e.target.value || null; setJudgeModel(v); if (v) localStorage.setItem('cupel:judge-model', v); else localStorage.removeItem('cupel:judge-model'); }}>
          <option value="">Self-judge (same model)</option>
          ${availableProviders.map(p => html`
            <optgroup label="${p.name || p.url}">
              ${(p.models || []).map(m => html`<option value=${m}>${m}</option>`)}
            </optgroup>
          `)}
        </select>
        ${judgeModel === null ? html`
          <div class="self-judge-warning" style="margin-top:8px;max-width:400px">
            Self-judging is unreliable \u2014 configure a stronger model for accurate scores
          </div>
        ` : null}
      </div>

      <!-- 4. THINKING -->
      <div style="padding:16px 20px;border-bottom:1px solid var(--border)">
        <div style="${sectionLabel}">Thinking</div>
        <div style="display:flex;flex-wrap:wrap;align-items:center">
          <span style="${chipStyle(thinkingMode === 'default')}" onClick=${() => { setThinkingMode('default'); localStorage.setItem('cupel:thinking-mode', 'default'); }}>
            model default
          </span>
          <span style="${chipStyle(thinkingMode === 'off')}" onClick=${() => { setThinkingMode('off'); localStorage.setItem('cupel:thinking-mode', 'off'); }}>
            off
          </span>
          <span style="${chipStyle(thinkingMode === 'budget')};${thinkingMode === 'budget' ? 'padding-right:6px' : ''}" onClick=${() => { setThinkingMode('budget'); localStorage.setItem('cupel:thinking-mode', 'budget'); }}>
            budget${thinkingMode === 'budget' ? html` <span style="display:inline-flex;align-items:center;margin-left:6px;gap:0" onClick=${(e) => e.stopPropagation()}>
              <span style="cursor:pointer;padding:2px 6px;border:1px solid var(--border);border-right:none;background:var(--bg-alt);color:var(--text-2);font-size:14px;font-family:var(--font-data);line-height:1;user-select:none;border-radius:var(--radius-sm) 0 0 var(--radius-sm)"
                onClick=${() => setThinkingBudget(b => { const v = Math.max(512, b - 512); localStorage.setItem('cupel:thinking-budget', v); return v; })}>−</span>
              <input
                type="text" inputmode="numeric"
                value=${thinkingBudget}
                onChange=${(e) => { const v = parseInt(e.target.value); if (v >= 512 && v <= 65536) { setThinkingBudget(v); localStorage.setItem('cupel:thinking-budget', v); } }}
                style="width:52px;background:var(--bg);border:1px solid var(--border);color:var(--text);font-family:var(--font-data);font-size:13px;padding:2px 4px;text-align:center"
              />
              <span style="cursor:pointer;padding:2px 6px;border:1px solid var(--border);border-left:none;background:var(--bg-alt);color:var(--text-2);font-size:14px;font-family:var(--font-data);line-height:1;user-select:none;border-radius:0 var(--radius-sm) var(--radius-sm) 0"
                onClick=${() => setThinkingBudget(b => { const v = Math.min(65536, b + 512); localStorage.setItem('cupel:thinking-budget', v); return v; })}>+</span>
            </span> <span style="color:var(--text-3);font-size:12px">tokens</span>` : ''}
          </span>
        </div>
        <div style="margin-top:6px;font-family:var(--font-label);font-size:13px;color:var(--text-3)">
          ${thinkingMode === 'default' ? 'uses config.yml setting or provider default' :
            thinkingMode === 'off' ? 'explicitly disables extended thinking' :
            `model will use up to ${thinkingBudget.toLocaleString()} thinking tokens before answering`}
        </div>
      </div>

      <!-- 5. CONNECTIONS -->
      <div style="padding:16px 20px;border-bottom:1px solid var(--border)">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
          <div style="${sectionLabel};margin-bottom:0">Connections</div>
          <button class="btn-ghost" style="font-size:13px;padding:3px 10px" onClick=${refreshProviders}>Refresh</button>
        </div>
        ${providers.map(p => {
          const info = guessProvider(p);
          return html`
            <div style="display:flex;align-items:center;gap:10px;padding:6px 0">
              <span style="font-family:var(--font-data);font-size:15px;color:${p.status === 'online' ? 'var(--accent)' : p.source === 'external' ? 'var(--text-2)' : 'var(--text-3)'};width:20px;text-align:center">${info.avatar}</span>
              <span style="font-family:var(--font-data);font-size:14px;font-weight:600;color:${p.status === 'online' || p.source === 'external' ? 'var(--text)' : 'var(--text-3)'}">${info.name}</span>
              <span class="status-dot ${p.status === 'online' ? 'online' : ''}" style="${p.source === 'external' ? 'background:var(--text-2)' : p.status !== 'online' ? 'background:var(--text-3)' : ''}"></span>
              <span style="font-family:var(--font-data);font-size:13px;color:var(--text-2)">${p.url}</span>
              ${(p.status === 'online' || p.source === 'external') ? html`
                <span style="font-family:var(--font-data);font-size:13px;color:var(--text-3)">${(p.models || []).length} model${(p.models || []).length !== 1 ? 's' : ''}</span>
              ` : null}
            </div>`;
        })}
      </div>

      ${detailPid != null ? (() => {
        const promptObj = fullPrompts.find(p => p.id === detailPid) || activePrompts.find(p => p.id === detailPid);
        if (!promptObj) return null;
        const isMultiTurn = !!promptObj.turns;
        const models = detailData && detailData.models ? detailData.models : {};

        const renderModelStatus = (cell, result) => {
          if (!cell) return html`<span style="color:var(--text-3)">Pending</span>`;
          if (cell.status === 'running') return html`<span style="color:var(--warn)"><span class="bench-spinner" style="display:inline-block;vertical-align:middle;margin-right:4px"></span> Running\u2026</span>`;
          if (cell.status === 'judging') return html`<span style="color:var(--accent)">Judging\u2026</span>`;
          if (cell.status === 'done' && result && result.score != null) {
            const c = SCORE_COLORS[result.score] || SCORE_COLORS[0];
            return html`<span class="score-badge s${result.score}" style="display:inline-flex;width:24px;height:24px;font-size:14px">${result.score}</span>`;
          }
          if (cell.status === 'done_unscored') {
            const t = cell.elapsed >= 100 ? Math.round(cell.elapsed) + 's' : Number(cell.elapsed).toFixed(1) + 's';
            return html`<span style="color:var(--good)">\u2713</span> <span style="color:var(--text-3)">${t}</span>`;
          }
          if (cell.status === 'error') return html`<span style="color:var(--bad)">ERR</span>`;
          if (cell.status === 'skip') return html`<span style="color:var(--text-3)">Skipped</span>`;
          return html`<span style="color:var(--text-3)">Pending</span>`;
        };

        const renderModelBody = (cell, result) => {
          if (!cell) return html`<div style="color:var(--text-3);font-family:var(--font-data);font-size:13px;padding:4px 0">Waiting to be tested</div>`;
          if (cell.status === 'running') return html`<div style="color:var(--text-3);font-family:var(--font-data);font-size:13px;padding:4px 0">Waiting for response\u2026</div>`;
          if (!result) return html`<div style="color:var(--text-3);font-family:var(--font-data);font-size:13px;padding:4px 0">\u2026</div>`;
          if (result.error) return html`<div style="color:var(--bad);font-family:var(--font-data);font-size:13px;padding:4px 0">${result.error}</div>`;
          if (result.skipped) return html`<div style="color:var(--text-3);font-family:var(--font-data);font-size:13px;padding:4px 0">${result.reason || 'Skipped'}</div>`;
          return html`
            ${result.score != null ? html`
              <div style="margin-bottom:8px">
                ${[3, 2, 1, 0].map(s => html`
                  <div class="rubric-row ${s === result.score ? 'matched' : ''}">
                    <span class="rubric-score s${s}">${s}</span>
                    <span class="rubric-text">${promptObj.rubric ? (promptObj.rubric[String(s)] || '') : ['Completely wrong.', 'Partially correct.', 'Correct but shallow.', 'Precise and thorough.'][s]}</span>
                  </div>
                `)}
              </div>
            ` : null}
            ${result.judge_reason ? html`<div class="detail-reason" style="margin-bottom:6px">${result.judge_reason}</div>` : null}
            ${cell.status === 'judging' ? html`<div style="color:var(--accent);font-family:var(--font-data);font-size:13px;margin-bottom:6px">Scoring in progress\u2026</div>` : null}
            ${result.elapsed_seconds ? html`<div class="detail-metrics" style="margin-bottom:6px"><span>\u23f1 <b>${result.elapsed_seconds >= 100 ? Math.round(result.elapsed_seconds) : Number(result.elapsed_seconds).toFixed(1)}s</b></span><span>\u26a1 <b>${result.completion_tokens || 0}</b> tok</span></div>` : null}
            ${result.responses ? html`
              <details style="margin-top:4px">
                <summary style="font-family:var(--font-data);font-size:13px;color:var(--text-3);cursor:pointer">Responses (${result.responses.length} turns)</summary>
                ${result.responses.map((r, i) => html`
                  <div style="margin:6px 0">
                    <div style="font-family:var(--font-label);font-size:11px;color:var(--text-3);text-transform:uppercase;margin-bottom:2px">Turn ${i + 1}</div>
                    <pre class="detail-pre">${r}</pre>
                  </div>
                `)}
              </details>
            ` : result.response ? html`
              <details style="margin-top:4px">
                <summary style="font-family:var(--font-data);font-size:13px;color:var(--text-3);cursor:pointer">Response</summary>
                <pre class="detail-pre">${result.response}</pre>
              </details>
            ` : null}
            ${result.thinking ? html`
              <details style="margin-top:4px">
                <summary style="font-family:var(--font-data);font-size:13px;color:var(--text-3);cursor:pointer">Thinking</summary>
                <pre class="detail-pre" style="color:var(--text-3)">${result.thinking}</pre>
              </details>
            ` : null}
          `;
        };

        return html`
          <div class="detail-panel open" style="width:460px">
            <div class="detail-header">
              <div>
                <div class="detail-pid">#${promptObj.id} \u00b7 ${catLabel(promptObj.category)}</div>
                <div class="detail-title">${promptObj.title}</div>
              </div>
              <button class="detail-close" onClick=${closeDetail}>\u2715</button>
            </div>
            <div class="detail-body">
              <div class="detail-sec">
                <details open>
                  <summary class="detail-sec-head">Prompt</summary>
                  <div class="detail-sec-body">
                    ${isMultiTurn ? html`
                      ${(promptObj.turns || []).map((turn, i) => html`
                        <div style="margin-bottom:8px">
                          <div style="font-family:var(--font-label);font-size:11px;color:var(--text-3);text-transform:uppercase;margin-bottom:2px">Turn ${i + 1}</div>
                          ${(turn.messages || []).map(m => html`
                            <div style="margin-bottom:4px">
                              <span style="font-family:var(--font-data);font-size:11px;color:var(--accent);text-transform:uppercase">${m.role}</span>
                              <pre class="detail-pre">${m.content}</pre>
                            </div>
                          `)}
                        </div>
                      `)}
                    ` : html`<pre class="detail-pre">${promptObj.prompt}</pre>`}
                  </div>
                </details>
              </div>
              ${activeModels.map(model => {
                const cell = grid[`${model}::${detailPid}`];
                const result = models[model];
                return html`
                  <div class="detail-sec">
                    <div class="detail-sec-head">
                      <span>${model}</span>
                      ${renderModelStatus(cell, result)}
                    </div>
                    <div class="detail-sec-body">
                      ${renderModelBody(cell, result)}
                    </div>
                  </div>
                `;
              })}
            </div>
          </div>
        `;
      })() : null}
    </div>
  `;
}

export default RunPage;
