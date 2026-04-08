const { h, useState, useEffect, useCallback, html } = window.__preact;

const KNOWN_PROVIDERS = [
  // Local inference servers
  { name: 'omlx', label: 'oMLX', group: 'local',
    api_url: 'http://localhost:8000/v1/chat/completions',
    api_key_env: 'OMLX_API_KEY', fallback_models: [] },
  { name: 'ollama', label: 'Ollama', group: 'local',
    api_url: 'http://localhost:11434/v1/chat/completions',
    api_key_env: 'OLLAMA_API_KEY', fallback_models: [] },
  { name: 'lm-studio', label: 'LM Studio', group: 'local',
    api_url: 'http://localhost:1234/v1/chat/completions',
    api_key_env: 'LM_STUDIO_API_KEY', fallback_models: [] },
  { name: 'sglang', label: 'SGLang', group: 'local',
    api_url: 'http://localhost:30000/v1/chat/completions',
    api_key_env: 'SGLANG_API_KEY', fallback_models: [] },
  // Cloud API providers
  { name: 'anthropic', label: 'Anthropic', group: 'cloud',
    api_url: 'https://api.anthropic.com/v1/messages',
    api_key_env: 'ANTHROPIC_API_KEY',
    fallback_models: ['claude-opus-4-6', 'claude-sonnet-4-6', 'claude-haiku-4-5'] },
  { name: 'openrouter', label: 'OpenRouter', group: 'cloud',
    api_url: 'https://openrouter.ai/api/v1/chat/completions',
    api_key_env: 'OPENROUTER_API_KEY',
    fallback_models: ['google/gemini-2.5-pro', 'deepseek/deepseek-r1'] },
  { name: 'openai', label: 'OpenAI', group: 'cloud',
    api_url: 'https://api.openai.com/v1/chat/completions',
    api_key_env: 'OPENAI_API_KEY',
    fallback_models: ['gpt-4o', 'gpt-4o-mini', 'o3-mini'] },
];

function SettingsPage({ params, refreshProviders }) {
  const [config, setConfig] = useState(null);
  const [hardware, setHardware] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [providers, setProviders] = useState([]);
  const [providerKeys, setProviderKeys] = useState({});
  const [addingProvider, setAddingProvider] = useState(false);
  const [newProv, setNewProv] = useState({ name: '', api_url: '', api_key_env: '', models: '' });
  const [presetType, setPresetType] = useState(null);
  const [envKeyStatus, setEnvKeyStatus] = useState(null); // null | true | false
  const [envKeyRefreshing, setEnvKeyRefreshing] = useState(false);
  const [fetchingModels, setFetchingModels] = useState(false);
  const [fetchedModels, setFetchedModels] = useState(null);
  const [fetchError, setFetchError] = useState(null);
  const [selectedModels, setSelectedModels] = useState(new Set());
  const [judgeTab, setJudgeTab] = useState(null);
  const [testResult, setTestResult] = useState(null); // null | {ok, detail}
  const [testing, setTesting] = useState(false);
  const [testingIdx, setTestingIdx] = useState(null); // index of provider being tested
  const [providerTestResults, setProviderTestResults] = useState({}); // idx -> {ok, detail}
  const [editingProviderIdx, setEditingProviderIdx] = useState(null); // index of provider being edited for models
  const [editFetchedModels, setEditFetchedModels] = useState(null);
  const [editSelectedModels, setEditSelectedModels] = useState(new Set());
  const [editFetching, setEditFetching] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch('/api/config').then(r => r.json()),
      fetch('/api/hardware').then(r => r.json()),
      fetch('/api/providers').then(r => r.json()),
      fetch('/api/providers/keys').then(r => r.json()).catch(() => ({})),
    ]).then(([cfg, hw, pp, keys]) => {
      setConfig(cfg);
      setHardware(hw);
      setProviders(pp);
      setProviderKeys(keys);
      setSaved(true);
      const j = cfg.judge || {};
      const isRemote = j.api_url && !j.api_url.includes('localhost') && !j.api_url.includes('127.0.0.1');
      const extModels = (cfg.providers || []).flatMap(p => p.models || []);
      const judgeIsExternal = j.model && extModels.includes(j.model);
      setJudgeTab((isRemote || judgeIsExternal) ? 'remote' : 'local');
    }).catch(console.error);
  }, []);

  useEffect(() => {
    if (params && params.includes('addProvider')) {
      setAddingProvider(true);
    }
  }, [params]);

  // Sync unsaved state to window for navigation guard in index.html
  useEffect(() => {
    window.__cupelUnsaved = !saved && config !== null;
    return () => { window.__cupelUnsaved = false; };
  }, [saved, config]);

  const saveConfig = () => {
    setSaving(true);
    setSaved(false);
    const toSave = { ...config };
    if (toSave.judge) {
      toSave.judge = { model: toSave.judge.model || '' };
    }
    fetch('/api/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(toSave),
    })
      .then(() => {
        setSaving(false); setSaved(true);
        fetch('/api/providers').then(r => r.json()).then(setProviders).catch(() => {});
        fetch('/api/providers/keys').then(r => r.json()).then(setProviderKeys).catch(() => {});
        if (refreshProviders) refreshProviders();
      })
      .catch(() => setSaving(false));
  };

  if (!config) return html`<div class="page"><div style="padding: 40px; color: var(--text-3)">Loading...</div></div>`;

  const configProviders = config.providers || [];
  const keyEntries = Object.entries(providerKeys);

  const localProviders = providers.filter(p => p.source === 'local' && p.status === 'online');
  const externalProvidersList = providers.filter(p => p.source === 'external');
  const allAvailableProviders = providers.filter(p => p.status === 'online' || p.source === 'external');

  // ── Preset / add provider logic ──

  const isLocalPreset = presetType && KNOWN_PROVIDERS.find(p => p.name === presetType)?.group === 'local';

  const selectPreset = (type) => {
    setPresetType(type);
    setAddingProvider(true);
    setEnvKeyStatus(null);
    setFetchedModels(null);
    setFetchError(null);
    setSelectedModels(new Set());
    setTestResult(null);
    if (type === 'custom') {
      setNewProv({ name: '', api_url: '', api_key_env: '', models: '' });
    } else {
      const preset = KNOWN_PROVIDERS.find(p => p.name === type);
      if (preset) {
        setNewProv({ name: preset.name, api_url: preset.api_url, api_key_env: preset.api_key_env, models: '' });
        if (preset.api_key_env) {
          fetch(`/api/env-check?key=${encodeURIComponent(preset.api_key_env)}`)
            .then(r => r.json())
            .then(data => setEnvKeyStatus(data.set))
            .catch(() => setEnvKeyStatus(null));
        }
      }
    }
  };

  const checkKeyStatus = (keyEnv) => {
    if (!keyEnv.trim()) { setEnvKeyStatus(null); return; }
    fetch(`/api/env-check?key=${encodeURIComponent(keyEnv.trim())}`)
      .then(r => r.json())
      .then(data => setEnvKeyStatus(data.set))
      .catch(() => setEnvKeyStatus(null));
  };

  const refreshEnvKey = useCallback(() => {
    const keyEnv = newProv.api_key_env;
    if (!keyEnv || !keyEnv.trim()) return;
    setEnvKeyRefreshing(true);
    fetch(`/api/env-check?key=${encodeURIComponent(keyEnv.trim())}`)
      .then(r => r.json())
      .then(data => { setEnvKeyStatus(data.set); setEnvKeyRefreshing(false); })
      .catch(() => { setEnvKeyStatus(null); setEnvKeyRefreshing(false); });
  }, [newProv.api_key_env]);

  const fmtPrice = (perToken) => {
    if (!perToken || perToken === '0') return 'free';
    const n = parseFloat(perToken);
    if (isNaN(n) || n === 0) return 'free';
    const perM = n * 1_000_000;
    if (perM >= 1) return `$${perM.toFixed(perM >= 10 ? 0 : 1)}`;
    return `$${perM.toFixed(2)}`;
  };

  const fetchModels = () => {
    setFetchingModels(true);
    setFetchError(null);
    fetch('/api/providers/fetch-models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_url: newProv.api_url, api_key_env: newProv.api_key_env }),
    })
      .then(r => {
        if (!r.ok) return r.json().then(d => { throw new Error(d.detail || `HTTP ${r.status}`); });
        return r.json();
      })
      .then(data => {
        const models = data.models || [];
        setFetchedModels(models);
        setSelectedModels(new Set(models.map(m => typeof m === 'string' ? m : m.id)));
        setFetchingModels(false);
      })
      .catch(e => {
        setFetchedModels(null);
        setFetchError(e.message || 'Failed to fetch models');
        setFetchingModels(false);
      });
  };

  const toggleFetchedModel = (id) => {
    setSelectedModels(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  // Test connection for the add-provider form
  const testConnection = () => {
    setTesting(true);
    setTestResult(null);
    fetch('/api/providers/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_url: newProv.api_url, api_key_env: newProv.api_key_env }),
    })
      .then(r => r.json())
      .then(result => { setTestResult(result); setTesting(false); })
      .catch(() => { setTestResult({ ok: false, detail: 'request failed' }); setTesting(false); });
  };

  // Test connection for an existing configured provider
  const testExistingProvider = (idx) => {
    const p = configProviders[idx];
    setTestingIdx(idx);
    fetch('/api/providers/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_url: p.api_url, api_key_env: p.api_key_env }),
    })
      .then(r => r.json())
      .then(result => {
        setProviderTestResults(prev => ({ ...prev, [idx]: result }));
        setTestingIdx(null);
      })
      .catch(() => {
        setProviderTestResults(prev => ({ ...prev, [idx]: { ok: false, detail: 'request failed' } }));
        setTestingIdx(null);
      });
  };

  const addProvider = () => {
    let models;
    if (fetchedModels && selectedModels.size > 0) {
      models = [...selectedModels];
    } else {
      models = newProv.models.split(',').map(m => m.trim()).filter(Boolean);
    }
    const prov = {
      name: newProv.name.trim(),
      api_url: newProv.api_url.trim(),
      api_key_env: newProv.api_key_env.trim(),
      models,
    };
    if (!prov.name || !prov.api_url) return;
    if (configProviders.some(p => p.api_url.trim() === prov.api_url)) return;
    setConfig({...config, providers: [...configProviders, prov]});
    setNewProv({ name: '', api_url: '', api_key_env: '', models: '' });
    setAddingProvider(false);
    setPresetType(null);
    setEnvKeyStatus(null);
    setFetchedModels(null);
    setFetchError(null);
    setSelectedModels(new Set());
    setTestResult(null);
    setSaved(false);
  };

  const removeProvider = (idx) => {
    setConfig({...config, providers: configProviders.filter((_, i) => i !== idx)});
    const next = { ...providerTestResults };
    delete next[idx];
    setProviderTestResults(next);
    if (editingProviderIdx === idx) setEditingProviderIdx(null);
    setSaved(false);
  };

  const editProviderModels = (idx) => {
    const p = configProviders[idx];
    setEditingProviderIdx(idx);
    setEditSelectedModels(new Set(p.models || []));
    setEditFetchedModels(null);
    setEditFetching(true);
    fetch('/api/providers/fetch-models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_url: p.api_url, api_key_env: p.api_key_env }),
    })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(data => { setEditFetchedModels(data.models || []); setEditFetching(false); })
      .catch(() => { setEditFetchedModels([]); setEditFetching(false); });
  };

  const toggleEditModel = (mid) => {
    setEditSelectedModels(prev => {
      const next = new Set(prev);
      if (next.has(mid)) next.delete(mid); else next.add(mid);
      return next;
    });
  };

  const applyEditModels = () => {
    const idx = editingProviderIdx;
    if (idx === null) return;
    const updated = configProviders.map((p, i) => i === idx ? { ...p, models: [...editSelectedModels] } : p);
    setConfig({ ...config, providers: updated });
    setEditingProviderIdx(null);
    setEditFetchedModels(null);
    setEditSelectedModels(new Set());
    setSaved(false);
  };

  const cancelEditModels = () => {
    setEditingProviderIdx(null);
    setEditFetchedModels(null);
    setEditSelectedModels(new Set());
  };

  const cancelAdd = () => {
    setAddingProvider(false);
    setPresetType(null);
    setEnvKeyStatus(null);
    setFetchedModels(null);
    setFetchError(null);
    setSelectedModels(new Set());
    setTestResult(null);
    setNewProv({ name: '', api_url: '', api_key_env: '', models: '' });
  };

  const handleJudgeModel = (model) => {
    setConfig({...config, judge: { model }});
    setSaved(false);
  };

  // Cloud providers require a key; local providers don't
  const isCloud = presetType ? KNOWN_PROVIDERS.find(p => p.name === presetType)?.group === 'cloud'
    : (newProv.api_url.trim() && !newProv.api_url.includes('localhost') && !newProv.api_url.includes('127.0.0.1'));

  // Normalize URL for comparison: strip /v1, /chat, trailing slashes
  const baseUrl = (u) => (u || '').split('/v1')[0].split('/chat')[0].replace(/\/+$/, '');
  const urlMatchesConfig = newProv.api_url.trim() && configProviders.some(p => baseUrl(p.api_url) === baseUrl(newProv.api_url));
  const urlMatchesDiscovered = newProv.api_url.trim() && providers.some(p => p.source === 'local' && p.status === 'online' && baseUrl(p.url) === baseUrl(newProv.api_url));
  const urlExists = urlMatchesConfig || urlMatchesDiscovered;
  const addDisabled = !newProv.name.trim() || !newProv.api_url.trim() || (isCloud && envKeyStatus === false) || urlExists;

  const inputLbl = 'font-family: var(--font-label); font-size: 14px; color: var(--text-2); display: block; margin-bottom: 6px';
  const sectionHead = 'font-family: var(--font-data); font-size: 13px; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px';
  const tabStyle = (active) => `padding:5px 14px;font-family:var(--font-data);font-size:13px;cursor:pointer;border:1px solid ${active ? 'var(--accent)' : 'var(--border)'};background:${active ? 'var(--accent-dim)' : 'var(--bg-alt)'};color:${active ? 'var(--accent)' : 'var(--text-2)'};border-radius:var(--radius-sm)`;
  const presetBtnStyle = (active) => `padding:5px 12px;font-family:var(--font-data);font-size:13px;cursor:pointer;border:1px solid ${active ? 'var(--accent)' : 'var(--border)'};background:${active ? 'var(--accent-dim)' : 'var(--bg-alt)'};color:${active ? 'var(--accent)' : 'var(--text-2)'};border-radius:var(--radius-sm)`;

  const localPresets = KNOWN_PROVIDERS.filter(p => p.group === 'local');
  const cloudPresets = KNOWN_PROVIDERS.filter(p => p.group === 'cloud');

  // Show "Fetch models" when key is found, or for local presets (key optional)
  const canFetchModels = presetType !== 'custom' && (envKeyStatus === true || isLocalPreset);

  return html`
    <div class="page">
      <div class="page-header">Settings</div>
      <div style="padding: 20px; max-width: 600px">

        ${hardware ? html`
          <div style="margin-bottom: 24px">
            <div style="${sectionHead}; margin-bottom: 8px">Hardware</div>
            <div style="padding: 12px 16px; background: var(--bg-hover); border: 1px solid var(--border-subtle); border-radius: var(--radius-md)">
              <div style="font-family: var(--font-data); font-size: 15px; font-weight: 600; color: var(--text)">${hardware.name}</div>
              <div style="font-family: var(--font-data); font-size: 14px; color: var(--text-2); margin-top: 4px">${hardware.memory}${hardware.spec ? ` \u00b7 ${hardware.spec}` : ''}</div>
            </div>
          </div>
        ` : null}

        <div style="margin-bottom: 16px">
          <label style="${inputLbl}">Eval Set Path</label>
          <input class="input" value=${config.eval_set || ''} onInput=${e => { setConfig({...config, eval_set: e.target.value}); setSaved(false); }} />
        </div>

        <div style="display: flex; gap: 16px; margin-bottom: 16px">
          <div style="flex: 1">
            <label style="${inputLbl}">Temperature</label>
            <input class="input" type="number" step="0.1" min="0" max="2"
              value=${config.temperature ?? 0}
              onInput=${e => { setConfig({...config, temperature: parseFloat(e.target.value)}); setSaved(false); }} />
          </div>
          <div style="flex: 1">
            <label style="${inputLbl}">Max Output Tokens</label>
            <input class="input" type="number" step="1024" min="256"
              value=${config.max_tokens ?? 16384}
              onInput=${e => { setConfig({...config, max_tokens: parseInt(e.target.value)}); setSaved(false); }} />
          </div>
        </div>

        <!-- Default Judge -->
        <div style="margin-bottom: 24px; border-top: 1px solid var(--border); padding-top: 16px">
          <div style="${sectionHead}">Default Judge</div>
          <div style="display:flex;gap:4px;margin-bottom:12px">
            <button style="${tabStyle(judgeTab === 'local')}" onClick=${() => setJudgeTab('local')}>Local</button>
            <button style="${tabStyle(judgeTab === 'remote')}" onClick=${() => setJudgeTab('remote')}>Remote</button>
          </div>
          ${judgeTab === 'local' ? html`
            <select class="input" style="max-width:400px" value=${config.judge?.model || ''}
              onChange=${e => handleJudgeModel(e.target.value)}>
              <option value="">None</option>
              ${localProviders.map(p => html`
                <optgroup label="${p.name || p.url}">
                  ${(p.models || []).map(m => html`<option value=${m}>${m}</option>`)}
                </optgroup>
              `)}
            </select>
          ` : html`
            <select class="input" style="max-width:400px" value=${config.judge?.model || ''}
              onChange=${e => handleJudgeModel(e.target.value)}>
              <option value="">None</option>
              ${externalProvidersList.map(p => html`
                <optgroup label="${p.name || p.url}">
                  ${(p.models || []).map(m => html`<option value=${m}>${m}</option>`)}
                </optgroup>
              `)}
            </select>
          `}
        </div>

        <!-- Default Generation Model -->
        <div style="margin-bottom: 24px; border-top: 1px solid var(--border); padding-top: 16px">
          <div style="${sectionHead}">Default Generation Model</div>
          <select class="input" style="max-width:400px" value=${config.generation_model || ''}
            onChange=${e => { setConfig({...config, generation_model: e.target.value}); setSaved(false); }}>
            <option value="">Auto (first available)</option>
            ${allAvailableProviders.map(p => html`
              <optgroup label="${p.name || p.url}${p.source === 'external' ? ' (API)' : ''}">
                ${(p.models || []).map(m => html`<option value=${m}>${m}</option>`)}
              </optgroup>
            `)}
          </select>
        </div>

        <!-- Provider Keys -->
        ${keyEntries.length > 0 ? html`
          <div style="margin-bottom: 24px; border-top: 1px solid var(--border); padding-top: 16px">
            <div style="${sectionHead}">Provider Keys</div>
            <div style="padding: 12px 16px; background: var(--bg-hover); border: 1px solid var(--border-subtle); border-radius: var(--radius-md)">
              ${keyEntries.map(([k, v]) => html`
                <div style="display:flex;align-items:center;gap:8px;padding:4px 0;font-family:var(--font-data);font-size:14px">
                  <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${v ? 'var(--good)' : 'var(--text-3)'}"></span>
                  <span style="color:var(--text)">${k}</span>
                  <span style="color:${v ? 'var(--good)' : 'var(--text-3)'}; margin-left:auto">${v ? 'set' : 'missing'}</span>
                </div>
              `)}
              <div style="margin-top:8px;font-family:var(--font-label);font-size:13px;color:var(--text-3)">edit .env or ~/.cupel/.env to add keys</div>
            </div>
          </div>
        ` : null}

        <!-- External Providers -->
        <div style="margin-bottom: 24px; border-top: 1px solid var(--border); padding-top: 16px">
          <div style="${sectionHead}">Providers</div>

          ${configProviders.length === 0 && !addingProvider ? html`
            <div style="font-family:var(--font-label);font-size:14px;color:var(--text-3);margin-bottom:12px">
              No providers configured. Add one to use cloud APIs or additional local servers.
            </div>
          ` : null}

          ${configProviders.map((p, i) => {
            const tr = providerTestResults[i];
            const isEditing = editingProviderIdx === i;
            return html`
              <div style="padding:10px 14px;background:var(--bg-alt);border:1px solid ${isEditing ? 'var(--accent)' : 'var(--border-subtle)'};margin-bottom:8px;border-radius:var(--radius-md)" key=${i}>
                <div style="display:flex;justify-content:space-between;align-items:center">
                  <span style="font-family:var(--font-data);font-size:14px;font-weight:600;color:var(--text)">${p.name}</span>
                  <div style="display:flex;gap:6px">
                    <button class="btn-ghost" style="padding:2px 8px;font-size:12px"
                      onClick=${() => editProviderModels(i)} disabled=${isEditing || editFetching}>
                      models
                    </button>
                    <button class="btn-ghost" style="padding:2px 8px;font-size:12px"
                      onClick=${() => testExistingProvider(i)} disabled=${testingIdx === i}>
                      ${testingIdx === i ? 'testing...' : 'test'}
                    </button>
                    <button class="btn-ghost" style="padding:2px 8px;font-size:12px;color:var(--bad);border-color:var(--bad)"
                      onClick=${() => removeProvider(i)}>remove</button>
                  </div>
                </div>
                <div style="font-family:var(--font-data);font-size:13px;color:var(--text-2);margin-top:4px">${p.api_url}</div>
                <div style="font-family:var(--font-data);font-size:13px;color:var(--text-3);margin-top:2px">
                  ${p.api_key_env || 'no key'}
                  ${p.api_key_env && providerKeys[p.api_key_env] !== undefined ? html`
                    <span style="margin-left:6px;color:${providerKeys[p.api_key_env] ? 'var(--good)' : 'var(--text-3)'}">\u25cf ${providerKeys[p.api_key_env] ? 'set' : 'not set'}</span>
                  ` : null}
                </div>
                <div style="font-family:var(--font-data);font-size:13px;color:var(--text-2);margin-top:4px">
                  ${(p.models || []).join(', ') || 'no models'}
                </div>
                ${tr ? html`
                  <div style="margin-top:6px;font-family:var(--font-data);font-size:13px;color:${tr.ok ? 'var(--good)' : 'var(--bad)'}">
                    ${tr.ok ? '\u25cf' : '\u25cb'} ${tr.detail}
                  </div>
                ` : null}
                ${isEditing ? html`
                  <div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border-subtle)">
                    ${editFetching ? html`
                      <div style="font-family:var(--font-data);font-size:13px;color:var(--text-3)">Fetching models...</div>
                    ` : editFetchedModels && editFetchedModels.length > 0 ? html`
                      <label style="${inputLbl}">Models (${editSelectedModels.size} selected of ${editFetchedModels.length})</label>
                      <div style="max-height:250px;overflow-y:auto;border:1px solid var(--border);background:var(--bg-panel);border-radius:var(--radius-md)">
                        ${editFetchedModels.map(entry => {
                          const mid = typeof entry === 'string' ? entry : entry.id;
                          const pricing = typeof entry === 'object' ? entry.pricing : null;
                          const priceLabel = pricing
                            ? `${fmtPrice(pricing.prompt)} in / ${fmtPrice(pricing.completion)} out`
                            : '';
                          return html`
                            <label style="display:flex;align-items:center;gap:8px;padding:5px 10px;cursor:pointer;font-family:var(--font-data);font-size:14px;color:var(--text);border-bottom:1px solid var(--border-subtle)"
                              onClick=${(e) => e.stopPropagation()}>
                              <input class="checkbox" type="checkbox" checked=${editSelectedModels.has(mid)}
                                onChange=${() => toggleEditModel(mid)} />
                              <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${mid}</span>
                              ${priceLabel ? html`
                                <span style="font-size:12px;color:var(--text-3);white-space:nowrap;margin-left:auto">${priceLabel}</span>
                              ` : null}
                            </label>`;
                        })}
                      </div>
                      <div style="display:flex;gap:8px;margin-top:8px">
                        <button class="btn-primary" style="font-size:13px;padding:4px 12px" onClick=${applyEditModels}
                          disabled=${editSelectedModels.size === 0}>Apply</button>
                        <button class="btn-ghost" style="font-size:13px;padding:4px 12px" onClick=${cancelEditModels}>Cancel</button>
                      </div>
                    ` : html`
                      <div style="font-family:var(--font-data);font-size:13px;color:var(--text-3)">Could not fetch models from this provider</div>
                      <button class="btn-ghost" style="font-size:13px;padding:4px 12px;margin-top:6px" onClick=${cancelEditModels}>Close</button>
                    `}
                  </div>
                ` : null}
              </div>`;
          })}

          ${addingProvider ? html`
            <div style="padding:14px;background:var(--bg-alt);border:1px solid var(--accent);margin-bottom:8px;border-radius:var(--radius-md)">
              <!-- Preset buttons: two labeled rows -->
              <div style="margin-bottom:12px">
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
                  <span style="font-family:var(--font-data);font-size:12px;color:var(--text-3);text-transform:uppercase;letter-spacing:0.05em;min-width:36px">local</span>
                  ${localPresets.map(kp => html`
                    <button style="${presetBtnStyle(presetType === kp.name)}" onClick=${() => selectPreset(kp.name)}>${kp.label}</button>
                  `)}
                </div>
                <div style="display:flex;align-items:center;gap:6px">
                  <span style="font-family:var(--font-data);font-size:12px;color:var(--text-3);text-transform:uppercase;letter-spacing:0.05em;min-width:36px">cloud</span>
                  ${cloudPresets.map(kp => html`
                    <button style="${presetBtnStyle(presetType === kp.name)}" onClick=${() => selectPreset(kp.name)}>${kp.label}</button>
                  `)}
                  <button style="${presetBtnStyle(presetType === 'custom')}" onClick=${() => selectPreset('custom')}>Custom</button>
                </div>
              </div>

              <div style="margin-bottom:10px">
                <label style="${inputLbl}">Name</label>
                <input class="input" value=${newProv.name} onInput=${e => setNewProv({...newProv, name: e.target.value})} placeholder="e.g. anthropic, openrouter, ollama" />
              </div>
              <div style="margin-bottom:10px">
                <label style="${inputLbl}">API URL</label>
                <input class="input" value=${newProv.api_url} onInput=${e => setNewProv({...newProv, api_url: e.target.value})} placeholder="e.g. https://api.anthropic.com/v1/messages" />
                ${urlMatchesConfig ? html`<div style="font-family:var(--font-data);font-size:12px;color:var(--warn);margin-top:4px">provider with this URL already configured</div>`
                  : urlMatchesDiscovered ? html`<div style="font-family:var(--font-data);font-size:12px;color:var(--warn);margin-top:4px">already discovered as a running provider</div>` : null}
              </div>
              <div style="margin-bottom:10px">
                <label style="${inputLbl}">API Key Env Var</label>
                <input class="input" value=${newProv.api_key_env}
                  onInput=${e => { setNewProv({...newProv, api_key_env: e.target.value}); checkKeyStatus(e.target.value); }}
                  placeholder="e.g. OPENROUTER_API_KEY" />
                ${envKeyStatus === true ? html`
                  <div style="margin-top:4px;font-family:var(--font-data);font-size:13px;color:var(--good)">\u25cf key found</div>
                ` : envKeyStatus === false ? html`
                  <div style="margin-top:4px;display:flex;align-items:center;gap:8px">
                    <span style="font-family:var(--font-data);font-size:13px;color:${isCloud ? 'var(--bad)' : 'var(--text-3)'}">
                      ${isCloud ? '\u25cb key missing \u2014 add to .env or ~/.cupel/.env' : '\u25cb not set \u2014 add to .env if your server requires a key'}
                    </span>
                    <button style="width:22px;height:22px;border-radius:50%;border:1px solid ${envKeyRefreshing ? 'var(--accent)' : 'var(--border)'};background:transparent;color:var(--accent);cursor:pointer;display:inline-flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;${envKeyRefreshing ? 'animation:spin 0.8s linear infinite' : ''}"
                      onClick=${refreshEnvKey} title="Re-read .env and check key">\u21bb</button>
                  </div>
                ` : null}
              </div>

              <!-- Model selection -->
              ${fetchedModels ? html`
                <div style="margin-bottom:12px">
                  <label style="${inputLbl}">Models (${selectedModels.size} selected of ${fetchedModels.length})</label>
                  <div style="max-height:300px;overflow-y:auto;border:1px solid var(--border);background:var(--bg-panel);border-radius:var(--radius-md)">
                    ${fetchedModels.length > 0 ? html`
                      <label style="display:flex;align-items:center;gap:8px;padding:5px 10px;cursor:pointer;font-family:var(--font-data);font-size:14px;color:var(--text);border-bottom:2px solid var(--border)"
                        onClick=${(e) => e.stopPropagation()}>
                        <input class="checkbox" type="checkbox" checked=${fetchedModels.length > 0 && fetchedModels.every(m => selectedModels.has(typeof m === 'string' ? m : m.id))}
                          onChange=${() => {
                            const allSel = fetchedModels.every(m => selectedModels.has(typeof m === 'string' ? m : m.id));
                            if (allSel) setSelectedModels(new Set());
                            else setSelectedModels(new Set(fetchedModels.map(m => typeof m === 'string' ? m : m.id)));
                          }} />
                        <span>all models (${fetchedModels.length})</span>
                      </label>
                    ` : null}
                    ${fetchedModels.map(entry => {
                      const mid = typeof entry === 'string' ? entry : entry.id;
                      const pricing = typeof entry === 'object' ? entry.pricing : null;
                      const priceLabel = pricing
                        ? `${fmtPrice(pricing.prompt)} in / ${fmtPrice(pricing.completion)} out`
                        : '';
                      return html`
                        <label style="display:flex;align-items:center;gap:8px;padding:5px 10px;cursor:pointer;font-family:var(--font-data);font-size:14px;color:var(--text);border-bottom:1px solid var(--border-subtle)"
                          onClick=${(e) => e.stopPropagation()}>
                          <input class="checkbox" type="checkbox" checked=${selectedModels.has(mid)}
                            onChange=${() => toggleFetchedModel(mid)} />
                          <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${mid}</span>
                          ${priceLabel ? html`
                            <span style="font-size:12px;color:var(--text-3);white-space:nowrap;margin-left:auto">${priceLabel}</span>
                          ` : null}
                        </label>`;
                    })}
                  </div>
                </div>
              ` : html`
                <div style="margin-bottom:12px">
                  <label style="${inputLbl}">Models</label>
                  ${canFetchModels ? html`
                    <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
                      <button class="btn-ghost" style="font-size:13px;padding:4px 10px" onClick=${fetchModels} disabled=${fetchingModels}>
                        ${fetchingModels ? 'Fetching...' : 'Fetch models'}
                      </button>
                    </div>
                    ${fetchError ? html`<div style="font-family:var(--font-data);font-size:13px;color:var(--bad);margin-bottom:8px">${fetchError}</div>` : null}
                  ` : null}
                  ${presetType === 'custom' || !canFetchModels ? html`
                    <input class="input" value=${newProv.models} onInput=${e => setNewProv({...newProv, models: e.target.value})} placeholder="e.g. claude-opus-4-6, claude-sonnet-4-6 (comma-separated)" />
                  ` : null}
                </div>
              `}

              <!-- Test result -->
              ${testResult ? html`
                <div style="margin-bottom:10px;font-family:var(--font-data);font-size:13px;color:${testResult.ok ? 'var(--good)' : 'var(--bad)'}">
                  ${testResult.ok ? '\u25cf' : '\u25cb'} ${testResult.detail}
                </div>
              ` : null}

              <div style="display:flex;gap:8px">
                <button class="btn-primary" style="font-size:13px;padding:6px 14px" onClick=${addProvider}
                  disabled=${addDisabled}>Add Provider</button>
                <button class="btn-ghost" style="font-size:13px;padding:5px 12px" onClick=${testConnection} disabled=${testing || !newProv.api_url.trim()}>
                  ${testing ? 'Testing...' : 'Test Connection'}
                </button>
                <button class="btn-ghost" style="font-size:13px;padding:5px 12px" onClick=${cancelAdd}>Cancel</button>
              </div>
            </div>
          ` : html`
            <button class="btn-ghost" style="font-size:13px;padding:5px 12px" onClick=${() => { setAddingProvider(true); setPresetType(null); }}><span style="font-size:28px;font-weight:400;color:var(--accent);line-height:0;position:relative;top:2px">+</span> Add Provider</button>
          `}
        </div>

        <button class="btn-primary" onClick=${saveConfig} disabled=${saving}>
          ${saving ? 'Saving...' : saved ? 'Saved' : 'Save Configuration'}
        </button>
      </div>
    </div>
  `;
}

export default SettingsPage;
