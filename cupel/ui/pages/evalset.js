const { h, useState, useEffect, html } = window.__preact;

const CAT_COLORS = {
  multimodal: "#c77dba", security: "#d4845a", clojure_code: "#8b7ec8",
  distributed_systems: "#5b9bd5", ml_architecture: "#4cb89a", python_coding: "#c9a033",
  business_logic: "#c45050", clojure_ecosystem: "#9b6ec8", frontend_architecture: "#4aa3d5",
  domain_knowledge: "#d46b7a", system_design: "#3bbfa0", observability: "#8bb840",
  networking: "#b86ec8", math_estimation: "#c9b040", diagnostic_reasoning: "#5ab88b",
  meta: "#808080", chemistry: "#c87da0", assistant_competence: "#4ac4c4",
};

function EvalSetPage() {
  const [evalSet, setEvalSet] = useState(null);
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    fetch('/api/eval-set').then(r => r.json()).then(setEvalSet).catch(() => {});
  }, []);

  if (!evalSet) return html`<div class="page"><div style="padding: 40px; color: var(--text-3)">Loading...</div></div>`;

  const prompts = evalSet.prompts || [];

  return html`
    <div class="page">
      <div class="page-header">
        ${evalSet.name || 'Eval Set'}
        <span style="font-family: var(--font-data); font-size: 14px; color: var(--text-2); margin-left: 12px">${prompts.length} prompts</span>
      </div>
      <div style="padding: 0">
        <table style="width: 100%; border-collapse: collapse">
          <thead>
            <tr style="background: var(--bg-panel); border-bottom: 1px solid var(--border)">
              <th style="text-align: left; padding: 6px 12px; font-family: var(--font-data); font-size: 13px; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.06em; width: 36px">#</th>
              <th style="text-align: left; padding: 6px 8px; font-family: var(--font-data); font-size: 13px; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.06em; width: 130px">Category</th>
              <th style="text-align: left; padding: 6px 8px; font-family: var(--font-data); font-size: 13px; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.06em">Title</th>
            </tr>
          </thead>
          <tbody>
            ${prompts.map(p => html`
              <tr key=${p.id} style="border-bottom: 1px solid var(--border-subtle); cursor: pointer"
                onClick=${() => setExpanded(expanded === p.id ? null : p.id)}>
                <td style="padding: 8px 12px; font-family: var(--font-data); font-size: 14px; color: var(--text-2); text-align: center">${p.id}</td>
                <td style="padding: 8px">
                  <div style="display: flex; align-items: center; gap: 8px">
                    <span style="width: 3px; height: 12px; background: ${CAT_COLORS[p.category] || 'var(--text-3)'}; flex-shrink: 0"></span>
                    <span style="font-family: var(--font-label); font-size: 14px; color: var(--text-2)">${(p.category || '').replace(/_/g, ' ')}</span>
                  </div>
                </td>
                <td style="padding: 8px; font-family: var(--font-data); font-size: 14px; color: var(--text)">${p.title}</td>
              </tr>
              ${expanded === p.id ? html`
                <tr key="${p.id}-detail">
                  <td colspan="3" style="padding: 16px 20px; background: var(--bg-hover); border-bottom: 1px solid var(--border)">
                    ${p.prompt ? html`
                      <div style="margin-bottom: 12px">
                        <div style="font-family: var(--font-data); font-size: 13px; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px">Prompt</div>
                        <pre style="font-family: var(--font-data); font-size: 14px; color: var(--text-2); white-space: pre-wrap; line-height: 1.65">${p.prompt}</pre>
                      </div>
                    ` : null}
                    ${p.turns ? html`
                      <div style="margin-bottom: 12px">
                        <div style="font-family: var(--font-data); font-size: 13px; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px">Multi-turn (${p.turns.length} turns)</div>
                        <pre style="font-family: var(--font-data); font-size: 14px; color: var(--text-2); white-space: pre-wrap; line-height: 1.65">${JSON.stringify(p.turns, null, 2)}</pre>
                      </div>
                    ` : null}
                    ${p.rubric ? html`
                      <div>
                        <div style="font-family: var(--font-data); font-size: 13px; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px">Rubric</div>
                        ${Object.entries(p.rubric).sort((a, b) => b[0] - a[0]).map(([score, desc]) => html`
                          <div style="display: flex; gap: 8px; padding: 3px 0; align-items: flex-start">
                            <span style="font-family: var(--font-data); font-size: 14px; font-weight: 700; color: var(--text-2); width: 16px; text-align: center; flex-shrink: 0">${score}</span>
                            <span style="font-family: var(--font-label); font-size: 14px; color: var(--text-2); line-height: 1.5">${desc}</span>
                          </div>
                        `)}
                      </div>
                    ` : null}
                  </td>
                </tr>
              ` : null}
            `)}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

export default EvalSetPage;
