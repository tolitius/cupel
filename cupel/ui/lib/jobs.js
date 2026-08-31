// Shared job helpers — used by the Bench It page and the Results page.
//
// Pages otherwise import nothing from each other, but the run page and the
// results page now drive the same /api/jobs endpoint over the same SSE stream,
// and keeping two copies of this is how the score-writing logic on the backend
// ended up drifting in four places.

export function connectSSE(jobId, onEvent) {
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
    console.error('[cupel] SSE connection error — stream dropped', err);
    es.close();
  };
  return es;
}

// Providers whose models can act as a judge: discovered local servers that are
// up, plus any configured cloud provider.
export function judgeProviders(providers) {
  return (providers || []).filter(
    p => (p.status === 'online' || p.source === 'external') && (p.models || []).length
  );
}

export function startJudgeJob({ files, judgeModel, modelUrls, replace }) {
  return fetch('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      type: 'judge',
      files,
      judge_model: judgeModel,
      model_urls: modelUrls || {},
      replace: !!replace,
    }),
  }).then(r => {
    if (!r.ok) return r.json().then(e => { throw new Error(e.detail || 'judge failed'); });
    return r.json();
  });
}
