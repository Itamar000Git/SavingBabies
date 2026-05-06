const IMPACT = {
  normal:   { icon: '✓', cls: 'impact-normal' },
  elevated: { icon: '⚠', cls: 'impact-elevated' },
  critical: { icon: '✗', cls: 'impact-critical' },
}

function splitSummary(summary) {
  if (!summary) return { rule: '', ai: null }
  const sep = '\n\n— Groq AI (LLaMA 3.3 70B) —\n'
  const idx = summary.indexOf(sep)
  if (idx === -1) return { rule: summary, ai: null }
  return {
    rule: summary.slice(0, idx).trim(),
    ai:   summary.slice(idx + sep.length).trim(),
  }
}

export default function ExplanationPanel({ explanation }) {
  if (!explanation) return null

  const { rule, ai } = splitSummary(explanation.summary)

  return (
    <div className="explanation-panel">
      <h3 className="explanation-title">Why this result?</h3>

      {explanation.missing_signal_warning && (
        <div className="missing-signal-warning">
          ⚠ {explanation.missing_signal_warning}
        </div>
      )}

      {explanation.table_note && (
        <p className="table-note">{explanation.table_note}</p>
      )}

      <table className="params-table">
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Value</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {explanation.important_parameters.map((p, i) => {
            const { icon, cls } = IMPACT[p.impact] ?? IMPACT.normal
            return (
              <>
                <tr key={`val-${i}`}>
                  <td>{p.name}</td>
                  <td>{p.value}</td>
                  <td className={cls}>{icon}</td>
                </tr>
                {p.description && (
                  <tr key={`desc-${i}`} className="param-description-row">
                    <td colSpan={3} className="param-description">{p.description}</td>
                  </tr>
                )}
              </>
            )
          })}
        </tbody>
      </table>

      {rule && (
        <div className="explanation-rule">
          {rule.split('\n').map((line, i) => (
            <p key={i} className={
              line.startsWith('⚠') ? 'rule-line rule-concern'
              : line.startsWith('✓') ? 'rule-line rule-reassuring'
              : line.startsWith('ℹ') ? 'rule-line rule-info'
              : line.startsWith('Model scores') || line.startsWith('Ensemble') ? 'rule-line rule-scores'
              : 'rule-line'
            }>{line}</p>
          ))}
        </div>
      )}

      {ai && (
        <div className="ai-explanation">
          <div className="ai-explanation-header">
            <span className="ai-badge">🤖 Groq AI — LLaMA 3.3 70B</span>
          </div>
          <p className="ai-explanation-body">{ai}</p>
        </div>
      )}
    </div>
  )
}
