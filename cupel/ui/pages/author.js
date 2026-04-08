const { h, useState, useEffect, useCallback, html } = window.__preact;

const CAT_GROUPS = [
  { label: 'coding', cats: ['clojure_code', 'python_coding', 'clojure_ecosystem'] },
  { label: 'systems', cats: ['system_design', 'distributed_systems', 'frontend_architecture'] },
  { label: 'reasoning', cats: ['diagnostic_reasoning', 'math_estimation', 'business_logic'] },
  { label: 'science & ml', cats: ['ml_architecture', 'chemistry'] },
  { label: 'ops & security', cats: ['security', 'networking', 'observability'] },
  { label: 'general', cats: ['domain_knowledge', 'assistant_competence', 'meta', 'multimodal'] },
];

function catLabel(c) { return (c || '').replace(/_/g, ' '); }

function AuthorPage({ providers: initProviders }) {
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState(() => localStorage.getItem('cupel:author-category') || 'system_design');
  const [customCat, setCustomCat] = useState('');
  const [difficulty, setDifficulty] = useState(() => localStorage.getItem('cupel:author-difficulty') || 'medium');
  const [generated, setGenerated] = useState(null);
  const [generating, setGenerating] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState(null);
  const [providers, setProviders] = useState([]);
  const [genModel, setGenModel] = useState(() => localStorage.getItem('cupel:author-gen-model') || '');
  const [genStats, setGenStats] = useState(null);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    Promise.all([
      fetch('/api/providers').then(r => r.json()),
      fetch('/api/config').then(r => r.json()).catch(() => ({})),
    ]).then(([pp, cfg]) => {
      setProviders(pp);
      // Config default always wins; localStorage is fallback when no config default
      const configDefault = cfg.generation_model;
      if (configDefault) {
        setGenModel(configDefault);
      } else if (!localStorage.getItem('cupel:author-gen-model')) {
        const avail = pp.filter(p => p.status === 'online' || p.source === 'external');
        const first = avail.flatMap(p => p.models || []);
        if (first.length > 0) setGenModel(first[0]);
      }
    }).catch(() => {});
  }, []);

  useEffect(() => { if (initProviders) setProviders(initProviders); }, [initProviders]);

  // Live timer — ticks every second while generating
  useEffect(() => {
    if (!generating) return;
    setElapsed(0);
    const t = setInterval(() => setElapsed(e => e + 1), 1000);
    return () => clearInterval(t);
  }, [generating]);

  const actualCategory = category === '__custom__' ? customCat : category;

  const generate = () => {
    setGenerating(true);
    setSaved(false);
    setError(null);
    setGenStats(null);
    setGenerated(null);
    fetch('/api/generate-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description, category: actualCategory, difficulty, model: genModel || undefined }),
    })
      .then(r => {
        if (!r.ok) return r.json().catch(() => ({ detail: `HTTP ${r.status}` })).then(body => { throw new Error(body.detail || `HTTP ${r.status}`); });
        return r.json();
      })
      .then(data => {
        console.log('[cupel] generate response:', JSON.stringify(data).slice(0, 500));

        setGenStats({
          elapsed: data.elapsed || 0,
          prompt_tokens: data.prompt_tokens || 0,
          completion_tokens: data.completion_tokens || 0,
        });

        // Server normalizes the prompt into {title, prompt, category, rubric}
        // so data.prompt should always have the right shape
        let gen = data.prompt;
        if (!gen || typeof gen !== 'object') {
          gen = { title: '', prompt: data.raw || JSON.stringify(data, null, 2), rubric: {} };
        }
        setGenerated(gen);
        setGenerating(false);
      })
      .catch(err => {
        console.error('[cupel] generate error:', err);
        setError(err.message || 'Generation failed');
        setGenerating(false);
        setGenStats(null);
      });
  };

  const save = () => {
    if (!generated) return;
    setError(null);
    fetch('/api/eval-set/prompts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(generated),
    })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(() => setSaved(true))
      .catch(err => { console.error('[cupel] save error:', err); setError(err.message || 'Save failed'); });
  };

  const availableProviders = providers.filter(p => p.status === 'online' || p.source === 'external');

  // Generation status line
  const genStatusLine = (() => {
    if (generating) {
      return html`
        <div style="display:flex;align-items:center;gap:8px;margin-top:10px;font-family:var(--font-data);font-size:14px;color:var(--warn)">
          <span class="bench-pulse">\u25cf</span>
          thinking\u2026 ${elapsed}s
        </div>`;
    }
    if (genStats && !error) {
      return html`
        <div style="margin-top:10px;font-family:var(--font-data);font-size:14px;color:var(--good)">
          \u2713 ${genStats.completion_tokens} tokens in ${genStats.elapsed.toFixed(1)}s
          <span style="color:var(--text-3);margin-left:6px">(${genStats.prompt_tokens} prompt \u2192 ${genStats.completion_tokens} completion)</span>
        </div>`;
    }
    return null;
  })();

  return html`
    <div class="page">
      <div class="page-header">Author Prompts</div>
      <div style="padding: 20px; max-width: 700px">
        <div style="margin-bottom: 16px">
          <label style="font-family: var(--font-label); font-size: 14px; color: var(--text-2); display: block; margin-bottom: 6px">Describe what you want to test</label>
          <textarea class="input" style="height: 80px; resize: vertical; font-family: var(--font-label)"
            value=${description}
            onInput=${e => setDescription(e.target.value)}
            placeholder="e.g. Test if the model can design a rate limiter for a REST API"></textarea>
        </div>
        <div style="display: flex; gap: 16px; margin-bottom: 16px">
          <div style="flex: 1">
            <label style="font-family: var(--font-label); font-size: 14px; color: var(--text-2); display: block; margin-bottom: 6px">Category</label>
            <select class="input" value=${category} onChange=${e => { setCategory(e.target.value); localStorage.setItem('cupel:author-category', e.target.value); }}>
              <option value="__custom__">enter your own\u2026</option>
              ${CAT_GROUPS.map(g => html`
                <optgroup label=${g.label}>
                  ${g.cats.map(c => html`<option value=${c}>${catLabel(c)}</option>`)}
                </optgroup>
              `)}
            </select>
            ${category === '__custom__' ? html`
              <input class="input" style="margin-top:6px" value=${customCat}
                onInput=${e => setCustomCat(e.target.value)}
                placeholder="Enter custom category name" />
            ` : null}
          </div>
          <div style="flex: 1">
            <label style="font-family: var(--font-label); font-size: 14px; color: var(--text-2); display: block; margin-bottom: 6px">Difficulty</label>
            <select class="input" value=${difficulty} onChange=${e => { setDifficulty(e.target.value); localStorage.setItem('cupel:author-difficulty', e.target.value); }}>
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>
        </div>

        <div style="margin-bottom: 16px">
          <label style="font-family: var(--font-label); font-size: 14px; color: var(--text-2); display: block; margin-bottom: 6px">Generation Model</label>
          <select class="input" style="max-width:400px" value=${genModel} onChange=${e => { setGenModel(e.target.value); localStorage.setItem('cupel:author-gen-model', e.target.value); }}>
            ${availableProviders.length === 0 ? html`
              <option value="" disabled>No providers available</option>
            ` : null}
            ${availableProviders.map(p => html`
              <optgroup label="${p.name || p.url}${p.source === 'external' ? ' (API)' : ''}">
                ${(p.models || []).map(m => html`<option value=${m}>${m}</option>`)}
              </optgroup>
            `)}
          </select>
          ${genStatusLine}
        </div>

        <button class="btn-primary" onClick=${generate} disabled=${!description || generating || !genModel}>
          ${generating ? 'Generating\u2026' : 'Generate Prompt'}
        </button>

        ${error ? html`
          <div style="margin-top: 12px; padding: 8px 14px; background: rgba(196, 80, 80, 0.1); border: 1px solid var(--bad); font-family: var(--font-label); font-size: 14px; color: var(--bad); border-radius: var(--radius-md)">
            ${error}
          </div>
        ` : null}

        ${generated ? html`
          <div style="margin-top: 24px; border-top: 1px solid var(--border); padding-top: 20px">
            <div style="font-family: var(--font-label); font-size: 15px; font-weight: 600; color: var(--text); margin-bottom: 12px">Generated Prompt</div>
            <div style="margin-bottom: 12px">
              <label style="font-family: var(--font-label); font-size: 14px; color: var(--text-2); display: block; margin-bottom: 6px">Title</label>
              <input class="input" value=${generated.title || ''} onInput=${e => setGenerated({...generated, title: e.target.value})} />
            </div>
            <div style="margin-bottom: 12px">
              <label style="font-family: var(--font-label); font-size: 14px; color: var(--text-2); display: block; margin-bottom: 6px">Prompt</label>
              <textarea class="input" style="height: 120px; resize: vertical; font-family: var(--font-data); font-size: 14px"
                value=${generated.prompt || ''}
                onInput=${e => setGenerated({...generated, prompt: e.target.value})}></textarea>
            </div>
            <div style="margin-bottom: 12px">
              <label style="font-family: var(--font-label); font-size: 14px; color: var(--text-2); display: block; margin-bottom: 6px">Rubric (JSON)</label>
              <textarea class="input" style="height: 100px; resize: vertical; font-family: var(--font-data); font-size: 14px"
                value=${JSON.stringify(generated.rubric || {}, null, 2)}
                onInput=${e => { try { setGenerated({...generated, rubric: JSON.parse(e.target.value)}); } catch {} }}></textarea>
            </div>
            <div style="display: flex; gap: 12px">
              <button class="btn-primary" onClick=${save} disabled=${saved}>
                ${saved ? 'Saved' : 'Save to Eval Set'}
              </button>
            </div>
          </div>
        ` : null}
      </div>
    </div>
  `;
}

export default AuthorPage;
