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
              <tr key={i}>
                <td>{p.name}</td>
                <td>{p.value}</td>
                <td className={cls}>{icon}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="explanation-summary">{explanation.summary}</p>
    </div>
  )
}
