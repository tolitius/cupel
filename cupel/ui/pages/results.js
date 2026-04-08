const { h, useState, useEffect, html } = window.__preact;

const CAT_COLORS = {
  multimodal: "#c77dba", security: "#d4845a", clojure_code: "#8b7ec8",
  distributed_systems: "#5b9bd5", ml_architecture: "#4cb89a", python_coding: "#c9a033",
  business_logic: "#c45050", clojure_ecosystem: "#9b6ec8", frontend_architecture: "#4aa3d5",
  domain_knowledge: "#d46b7a", system_design: "#3bbfa0", observability: "#8bb840",
  networking: "#b86ec8", math_estimation: "#c9b040", diagnostic_reasoning: "#5ab88b",
  meta: "#808080", chemistry: "#c87da0", assistant_competence: "#4ac4c4",
};

function catLabel(c) { return (c || '').replace(/_/g, ' '); }

function ResultsPage() {
  const [results, setResults] = useState([]);
  const [sortCol, setSortCol] = useState('timestamp');
  const [sortDir, setSortDir] = useState('desc');
  const [selectedFile, setSelectedFile] = useState(null);
  const [detailData, setDetailData] = useState(null);
  const [showMuted, setShowMuted] = useState(() => {
    const saved = localStorage.getItem('cupel:results-show-muted');
    return saved !== null ? JSON.parse(saved) : false;
  });

  useEffect(() => {
    fetch('/api/results').then(r => r.json()).then(setResults).catch(() => {});
  }, []);

  const deleteResult = (filename) => {
    if (!confirm(`Delete ${filename}?`)) return;
    fetch(`/api/results/${filename}`, { method: 'DELETE' })
      .then(() => {
        setResults(r => r.filter(x => x.filename !== filename));
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

  const openDetail = (filename) => {
    if (selectedFile === filename) { setSelectedFile(null); setDetailData(null); return; }
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
    const totalScore = results.reduce((acc, p) => acc + (p.score != null ? p.score : 0), 0);
    const maxScore = results.length * 3;
    const pct = maxScore > 0 ? (totalScore / maxScore * 100).toFixed(1) : '0.0';

    return html`
      <tr key="${r.filename}-detail">
        <td colspan="7" style="padding:0;border-bottom:1px solid var(--border)">
          <div style="padding:16px 20px;background:var(--bg-hover)">
            <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
              <div class="score-badge s${Math.min(3, Math.max(0, Math.round(totalScore / Math.max(1, maxScore) * 3)))}" style="width:36px;height:36px;display:flex;align-items:center;justify-content:center;font-family:var(--font-data);font-size:17px;font-weight:700">${totalScore}</div>
              <div>
                <div style="font-family:var(--font-data);font-size:15px;font-weight:600;color:var(--text)">${detailData.model || r.model}</div>
                <div style="font-family:var(--font-data);font-size:13px;color:var(--text-3)">
                  ${detailData.judge ? `judged by ${detailData.judge}` : 'self-judged'}
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
                    ${p.judge_reason ? html`<div style="font-family:var(--font-label);font-size:14px;color:var(--text-2);margin-bottom:4px">${p.judge_reason}</div>` : null}
                    ${p.elapsed_seconds ? html`<div style="font-family:var(--font-data);font-size:13px;color:var(--text-3);margin-bottom:4px">\u23f1 ${p.elapsed_seconds}s \u00b7 ${p.completion_tokens || 0} tok</div>` : null}
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
      <div style="padding: 16px 20px">
        ${sorted.length === 0
          ? html`<div style="color: var(--text-3); font-family: var(--font-label); font-size: 15px">
              No result files yet. Run an eval from the Run page.
            </div>`
          : html`
            <table style="width: 100%; border-collapse: collapse">
              <thead>
                <tr style="border-bottom: 1px solid var(--border)">
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
                    <td style="padding:8px;font-family:var(--font-data);font-size:14px;color:var(--text);font-weight:600">${r.model}${r.muted ? html` <span style="font-size:12px;font-weight:400;color:var(--text-3)">muted</span>` : null}</td>
                    <td style="padding:8px;font-family:var(--font-data);font-size:14px;color:var(--text-2)">${r.timestamp || ''}</td>
                    <td style="padding:8px;font-family:var(--font-data);font-size:15px;color:var(--text);text-align:right;font-weight:700">${r.total_score != null ? `${r.total_score}/${r.max_score}` : '\u2014'}</td>
                    <td style="padding:8px;text-align:center;font-family:var(--font-data);font-size:14px;color:${r.num_scored === r.num_prompts ? 'var(--accent)' : 'var(--text-3)'}">${r.num_scored}/${r.num_prompts}</td>
                    <td style="padding:8px;font-family:var(--font-data);font-size:13px;color:var(--text-2)">${r.judge || '\u2014'}</td>
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
