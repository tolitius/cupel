const { h, useState, useEffect, html } = window.__preact;

import { connectSSE, judgeProviders, startJudgeJob } from '../lib/jobs.js';

const CAT_COLORS = {
  multimodal: "#c77dba", security: "#d4845a", clojure_code: "#8b7ec8",
  distributed_systems: "#5b9bd5", ml_architecture: "#4cb89a", python_coding: "#c9a033",
  business_logic: "#c45050", clojure_ecosystem: "#9b6ec8", frontend_architecture: "#4aa3d5",
  domain_knowledge: "#d46b7a", system_design: "#3bbfa0", observability: "#8bb840",
  networking: "#b86ec8", math_estimation: "#c9b040", diagnostic_reasoning: "#5ab88b",
  meta: "#808080", chemistry: "#c87da0", assistant_competence: "#4ac4c4",
};

function catLabel(c) { return (c || '').replace(/_/g, ' '); }

function ResultsPage({ providers }) {
  const [results, setResults] = useState([]);
  const [sortCol, setSortCol] = useState('timestamp');
  const [sortDir, setSortDir] = useState('desc');
  const [selectedFile, setSelectedFile] = useState(null);
  const [detailData, setDetailData] = useState(null);
  const [showMuted, setShowMuted] = useState(() => {
    const saved = localStorage.getItem('cupel:results-show-muted');
    return saved !== null ? JSON.parse(saved) : true;
  });
  // re-judge state
  const [picked, setPicked] = useState([]);
  const [judgeModel, setJudgeModel] = useState(() => localStorage.getItem('cupel:judge-model') || '');
  const [judging, setJudging] = useState(null);   // {done, total, errors} while a job runs
  const [judgeError, setJudgeError] = useState('');
  const [judgeWarning, setJudgeWarning] = useState('');

  const loadResults = () =>
    fetch('/api/results').then(r => r.json()).then(setResults).catch(() => {});

  useEffect(() => { loadResults(); }, []);

  const togglePick = (filename) =>
    setPicked(p => p.includes(filename) ? p.filter(f => f !== filename) : [...p, filename]);

  const rejudge = () => {
    if (!judgeModel || !picked.length) return;
    setJudgeError('');
    // map each selectable judge model back to the provider URL it came from, so the
    // backend resolves the judge's own endpoint rather than the configured default
    const modelUrls = {};
    judgeProviders(providers).forEach(p => {
      (p.models || []).forEach(m => { if (!(m in modelUrls)) modelUrls[m] = p.url; });
    });

    setJudgeWarning('');
    // total = prompts across the selected runs, so progress has a denominator
    const total = picked.reduce((n, fn) => {
      const row = results.find(r => r.filename === fn);
      return n + (row ? row.num_prompts || 0 : 0);
    }, 0);
    setJudging({ done: 0, total, errors: 0 });

    const openFile = selectedFile;
    startJudgeJob({ files: picked, judgeModel, modelUrls })
      .then(({ id }) => {
        connectSSE(id, (ev) => {
          if (ev.type === 'complete' || ev.type === 'cancelled') {
            setJudging(null);
            setPicked([]);
            loadResults();
            if (openFile) openDetail(openFile, true);
          } else if (ev.type === 'error') {
            setJudging(null);
            setJudgeError(ev.error || 'judging failed');
          } else if (typeof ev.status === 'string') {
            // a warning is not a failure — only `error:` stops the run
            if (ev.status.startsWith('warning:')) {
              setJudgeWarning(ev.status.slice('warning:'.length));
            } else if (ev.status.startsWith('error:')) {
              setJudgeError(ev.status.slice('error:'.length));
            } else if (ev.status.startsWith('scored:') || ev.status === 'skip') {
              setJudging(j => j && { ...j, done: j.done + 1 });
            } else if (ev.status === 'error') {
              setJudging(j => j && { ...j, done: j.done + 1, errors: j.errors + 1 });
            }
          }
        });
      })
      .catch(e => { setJudging(null); setJudgeError(e.message); });
  };

  const deleteResult = (filename) => {
    if (!confirm(`Delete ${filename}?`)) return;
    fetch(`/api/results/${filename}`, { method: 'DELETE' })
      .then(() => {
        setResults(r => r.filter(x => x.filename !== filename));
        // drop it from the selection too — a stale pick would 404 on the next re-judge
        setPicked(p => p.filter(f => f !== filename));
        if (selectedFile === filename) { setSelectedFile(null); setDetailData(null); }
      });
  };

  const toggleMute = (filename) => {
    fetch(`/api/results/${filename}/mute`, { method: 'POST' })
      .then(r => r.json())
      .then(() => {
        setResults(rs => rs.map(r => r.filename === filename ? { ...r, muted: !r.muted } : r));
      });
  };

  const tagResult = (filename, tag) => {
    fetch(`/api/results/${filename}/tag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tag }),
    }).then(r => r.json()).then(() => {
      fetch('/api/results').then(r => r.json()).then(setResults);
    });
  };

  const toggleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    else { setSortCol(col); setSortDir(col === 'model' || col === 'judge' ? 'asc' : 'desc'); }
  };

  const sortArrow = (col) => sortCol !== col ? '' : sortDir === 'desc' ? ' \u25bc' : ' \u25b2';

  const openDetail = (filename, forceReload) => {
    // forceReload re-fetches the open row after a re-judge instead of collapsing it
    if (selectedFile === filename && !forceReload) {
      setSelectedFile(null); setDetailData(null); return;
    }
    setSelectedFile(filename);
    fetch(`/api/results/${filename}`).then(r => r.json()).then(setDetailData).catch(() => {});
  };

  const mutedCount = results.filter(r => r.muted).length;
  let sorted = showMuted ? [...results] : results.filter(r => !r.muted);
  sorted.sort((a, b) => {
    let va, vb;
    switch (sortCol) {
      case 'model': va = (a.model || '').toLowerCase(); vb = (b.model || '').toLowerCase(); break;
      case 'timestamp': va = a.timestamp || ''; vb = b.timestamp || ''; break;
      case 'total_score': va = a.total_score ?? 0; vb = b.total_score ?? 0; break;
      case 'judge': va = (a.judge || '').toLowerCase(); vb = (b.judge || '').toLowerCase(); break;
      default: return 0;
    }
    if (typeof va === 'string') return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
    return sortDir === 'asc' ? va - vb : vb - va;
  });

  const thBase = 'padding:6px 8px;font-family:var(--font-data);font-size:13px;color:var(--text-3);text-transform:uppercase;letter-spacing:0.06em';
  const thSort = 'cursor:pointer;user-select:none';

  const renderDetailRow = (r) => {
    if (!detailData) return null;
    const results = detailData.results || [];
    // only scored prompts count — an errored prompt is missing data, not a zero
    const scoredResults = results.filter(p => p.score != null);
    const totalScore = scoredResults.reduce((acc, p) => acc + p.score, 0);
    const maxScore = scoredResults.length * 3;
    const pct = maxScore > 0 ? (totalScore / maxScore * 100).toFixed(1) : '0.0';

    return html`
      <tr key="${r.filename}-detail">
        <td colspan="8" style="padding:0;border-bottom:1px solid var(--border)">
          <div style="padding:16px 20px;background:var(--bg-hover)">
            <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
              <div class="score-badge s${Math.min(3, Math.max(0, Math.round(totalScore / Math.max(1, maxScore) * 3)))}" style="width:36px;height:36px;display:flex;align-items:center;justify-content:center;font-family:var(--font-data);font-size:17px;font-weight:700">${totalScore}</div>
              <div>
                <div style="font-family:var(--font-data);font-size:15px;font-weight:600;color:var(--text)">${detailData.model || r.model}</div>
                <div style="font-family:var(--font-data);font-size:13px;color:var(--text-3)">
                  ${(detailData.judges && detailData.judges.length)
                    ? `judged by ${detailData.judges.map(j => j.model).join(', ')}`
                    : (detailData.judge ? `judged by ${detailData.judge}` : 'self-judged')}
                  ${' \u00b7 '}${totalScore}/${maxScore} (${pct}%)
                </div>
              </div>
            </div>
            <div style="display:flex;flex-direction:column;gap:6px">
              ${results.map(p => {
                const score = p.score != null ? p.score : null;
                const sc = score != null ? score : 0;
                const scoreColor = sc === 3 ? 'var(--score-3-fg)' : sc === 2 ? 'var(--score-2-fg)' : sc === 1 ? 'var(--score-1-fg)' : 'var(--score-0-fg)';
                const scoreBg = sc === 3 ? 'var(--score-3-bg)' : sc === 2 ? 'var(--score-2-bg)' : sc === 1 ? 'var(--score-1-bg)' : 'var(--score-0-bg)';
                return html`
                  <div style="padding:8px 12px;background:var(--bg-panel);border:1px solid var(--border-subtle);border-radius:var(--radius-md)">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
                      <div style="display:flex;align-items:center;gap:8px">
                        <span style="font-family:var(--font-data);font-size:13px;color:var(--text-3)">#${p.id}</span>
                        <span style="font-family:var(--font-data);font-size:14px;color:var(--text);font-weight:600">${p.title || ''}</span>
                        ${p.category ? html`<span style="font-family:var(--font-data);font-size:12px;color:${CAT_COLORS[p.category] || 'var(--text-3)'}">${catLabel(p.category)}</span>` : null}
                      </div>
                      <span style="font-family:var(--font-data);font-size:14px;font-weight:700;padding:2px 8px;background:${scoreBg};color:${scoreColor};border-radius:var(--radius-sm)">${score != null ? `${score}/3` : '\u2014'}</span>
                    </div>
                    ${(p.judgments && p.judgments.length > 1) ? html`
                      <div style="margin-bottom:6px;padding:6px 8px;border-left:2px solid var(--border);display:flex;flex-direction:column;gap:4px">
                        <div style="font-family:var(--font-data);font-size:12px;color:var(--text-3)">
                          ${p.judgments.length} judges${p.judge_agreement ? ` · disagree by ${p.judge_agreement}` : ' · agree'}
                        </div>
                        ${p.judgments.map(j => html`
                          <div style="font-family:var(--font-data);font-size:13px;color:var(--text-2)">
                            <span style="font-weight:700">${j.score}</span>
                            <span style="color:var(--text-3)"> ${j.judge_model}</span>
                            ${j.reason ? html`<div style="font-family:var(--font-label);color:var(--text-3);font-size:13px">${j.reason}</div>` : null}
                          </div>`)}
                      </div>
                    ` : p.judge_reason ? html`<div style="font-family:var(--font-label);font-size:14px;color:var(--text-2);margin-bottom:4px">${p.judge_reason}</div>` : null}
                    ${p.elapsed_seconds ? html`<div style="font-family:var(--font-data);font-size:13px;color:var(--text-3);margin-bottom:4px">\u23f1 ${p.elapsed_seconds}s \u00b7 ${p.completion_tokens || 0} tok${p.thinking_tokens > 0 ? ` \u00b7 \uD83E\uDDE0 ${p.thinking_tokens} think tok` : ''}</div>` : null}
                    ${p.response ? html`
                      <details style="margin-top:4px">
                        <summary style="font-family:var(--font-data);font-size:13px;color:var(--text-3);cursor:pointer">Response</summary>
                        <pre style="font-family:var(--font-data);font-size:13px;color:var(--text-2);white-space:pre-wrap;word-break:break-word;margin:6px 0 0;max-height:300px;overflow-y:auto;padding:8px;background:var(--bg-alt);border:1px solid var(--border-subtle);border-radius:var(--radius-sm)">${p.response}</pre>
                      </details>
                    ` : null}
                    ${p.thinking ? html`
                      <details style="margin-top:4px">
                        <summary style="font-family:var(--font-data);font-size:13px;color:var(--text-3);cursor:pointer">Thinking</summary>
                        <pre style="font-family:var(--font-data);font-size:13px;color:var(--text-3);white-space:pre-wrap;word-break:break-word;margin:6px 0 0;max-height:300px;overflow-y:auto;padding:8px;background:var(--bg-alt);border:1px solid var(--border-subtle);border-radius:var(--radius-sm)">${p.thinking}</pre>
                      </details>
                    ` : null}
                  </div>`;
              })}
            </div>
          </div>
        </td>
      </tr>`;
  };

  return html`
    <div class="page">
      <div class="page-header" style="display:flex;align-items:center;justify-content:space-between">
        <span>Result Files</span>
        ${mutedCount > 0 ? html`
          <label style="font-family:var(--font-data);font-size:13px;color:var(--text-3);font-weight:400;display:flex;align-items:center;gap:6px;cursor:pointer">
            <input type="checkbox" checked=${showMuted} onChange=${() => setShowMuted(s => { const next = !s; localStorage.setItem('cupel:results-show-muted', JSON.stringify(next)); return next; })} />
            show muted (${mutedCount})
          </label>
        ` : null}
      </div>
      ${picked.length > 0 ? html`
        <div style="padding:12px 20px;border-bottom:1px solid var(--border);background:var(--bg-hover);display:flex;align-items:center;gap:12px;flex-wrap:wrap">
          <span style="font-family:var(--font-data);font-size:14px;color:var(--text)">
            ${picked.length} selected
          </span>
          <select class="input" style="max-width:280px;font-size:13px"
                  value=${judgeModel}
                  onChange=${(e) => { setJudgeModel(e.target.value); if (e.target.value) localStorage.setItem('cupel:judge-model', e.target.value); }}>
            <option value="">Select a judge model</option>
            ${judgeProviders(providers).map(p => html`
              <optgroup label="${p.name || p.url}">
                ${(p.models || []).map(m => html`<option value=${m}>${m}</option>`)}
              </optgroup>
            `)}
          </select>
          <button class="btn" style="padding:4px 12px;font-size:13px"
                  disabled=${!judgeModel || !!judging}
                  onClick=${rejudge}>
            ${judging ? `judging… ${judging.done}${judging.total ? `/${judging.total}` : ''}` : 're-judge'}
          </button>
          <button class="btn-ghost" style="padding:4px 10px;font-size:13px"
                  onClick=${() => setPicked([])}>clear</button>
          <span style="font-family:var(--font-data);font-size:12px;color:var(--text-3)">
            every judgment is kept — the shown score becomes the median, so it can go down
          </span>
          ${judgeWarning ? html`
            <div style="width:100%;font-family:var(--font-data);font-size:13px;color:var(--warn, #c9a033)">
              ⚠ ${judgeWarning}
            </div>` : null}
          ${judgeError ? html`
            <div style="width:100%;font-family:var(--font-data);font-size:13px;color:var(--bad, #c45050)">
              ${judgeError}
            </div>` : null}
        </div>
      ` : null}
      <div style="padding: 16px 20px">
        ${sorted.length === 0
          ? html`<div style="color: var(--text-3); font-family: var(--font-label); font-size: 15px">
              No result files yet. Run an eval from the Run page.
            </div>`
          : html`
            <table style="width: 100%; border-collapse: collapse">
              <thead>
                <tr style="border-bottom: 1px solid var(--border)">
                  <th style="width:28px"></th>
                  <th style="text-align:left;${thBase};${thSort}" onClick=${() => toggleSort('model')}>Model${sortArrow('model')}</th>
                  <th style="text-align:left;${thBase};${thSort}" onClick=${() => toggleSort('timestamp')}>Date${sortArrow('timestamp')}</th>
                  <th style="text-align:right;${thBase};${thSort}" onClick=${() => toggleSort('total_score')}>Score${sortArrow('total_score')}</th>
                  <th style="text-align:center;${thBase}">Scored</th>
                  <th style="text-align:left;${thBase};${thSort}" onClick=${() => toggleSort('judge')}>Judge${sortArrow('judge')}</th>
                  <th style="text-align:left;${thBase}">Tags</th>
                  <th style="width:80px"></th>
                </tr>
              </thead>
              <tbody>
                ${sorted.map(r => html`
                  <tr style="border-bottom:1px solid var(--border-subtle);cursor:pointer;background:${selectedFile === r.filename ? 'var(--bg-hover)' : ''};${r.muted ? 'opacity:0.45' : ''}"
                      key=${r.filename}
                      onClick=${() => openDetail(r.filename)}>
                    <td style="padding:8px;text-align:center" onClick=${(e) => e.stopPropagation()}>
                      <input type="checkbox" checked=${picked.includes(r.filename)}
                             onChange=${() => togglePick(r.filename)} />
                    </td>
                    <td style="padding:8px;font-family:var(--font-data);font-size:14px;color:var(--text);font-weight:600">${r.model}${r.muted ? html` <span style="font-size:12px;font-weight:400;color:var(--text-3)">muted</span>` : null}${r.notes ? html`<div style="font-size:12px;font-weight:400;color:var(--text-3);font-style:italic">${r.notes}</div>` : null}</td>
                    <td style="padding:8px;font-family:var(--font-data);font-size:14px;color:var(--text-2)">${r.timestamp || ''}</td>
                    <td style="padding:8px;font-family:var(--font-data);font-size:15px;color:var(--text);text-align:right;font-weight:700">${r.total_score != null ? `${r.total_score}/${r.max_score}` : '\u2014'}</td>
                    <td style="padding:8px;text-align:center;font-family:var(--font-data);font-size:14px;color:${r.num_scored === r.num_prompts ? 'var(--accent)' : 'var(--text-3)'}">${r.num_scored}/${r.num_prompts}</td>
                    <td style="padding:8px;font-family:var(--font-data);font-size:13px;color:var(--text-2)">
                      ${(r.judges && r.judges.length) ? r.judges.map((j, i) => html`
                        <div style="${i > 0 ? 'color:var(--text-3)' : ''}">${j}</div>`) : (r.judge || '\u2014')}
                    </td>
                    <td style="padding:8px;font-family:var(--font-data);font-size:13px;color:var(--text-2)">
                      ${(r.tags || []).map(t => html`<span style="padding:1px 4px;border:1px solid var(--border);margin-right:4px">${t}</span>`)}
                      <button class="btn-ghost" style="padding:2px 6px;font-size:13px;margin-left:4px" onClick=${(e) => {
                        e.stopPropagation();
                        const tag = prompt('Tag name:');
                        if (tag) tagResult(r.filename, tag);
                      }}>+ tag</button>
                    </td>
                    <td style="padding:8px;text-align:right;white-space:nowrap">
                      <button class="btn-ghost" style="padding:2px 8px;font-size:13px;margin-right:8px" onClick=${(e) => { e.stopPropagation(); toggleMute(r.filename); }}>${r.muted ? 'unmute' : 'mute'}</button>
                      <button class="btn-ghost" style="padding:2px 8px;font-size:13px;color:var(--bad)" onClick=${(e) => { e.stopPropagation(); deleteResult(r.filename); }}>delete</button>
                    </td>
                  </tr>
                  ${selectedFile === r.filename ? renderDetailRow(r) : null}
                `)}
              </tbody>
            </table>
          `}
      </div>
    </div>
  `;
}

export default ResultsPage;
