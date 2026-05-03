const IMPACT = {
  normal:   { icon: '✓', cls: 'impact-normal' },
  elevated: { icon: '⚠', cls: 'impact-elevated' },
  critical: { icon: '✗', cls: 'impact-critical' },
}

export default function ExplanationPanel({ explanation }) {
  if (!explanation) return null

  return (
    <div className="explanation-panel">
      <h3 className="explanation-title">Why this result?</h3>
      {explanation.missing_signal_warning && (
        <div className="missing-signal-warning">
          ⚠ {explanation.missing_signal_warning}
        </div>
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
      <p className="explanation-summary">{explanation.summary}</p>
    </div>
  )
}
