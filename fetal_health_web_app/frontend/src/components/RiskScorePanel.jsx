export default function RiskScorePanel({ prediction }) {
  if (prediction?.risk_score == null) return null

  const riskPct = (prediction.risk_score * 100).toFixed(1)
  const threshPct = Math.round(prediction.threshold * 100)
  const healthyCutoffPct = Math.round(prediction.healthy_cutoff * 100)
  const dangerCutoffPct = Math.round(prediction.danger_cutoff * 100)
  const label = prediction.label  // "Healthy" | "Borderline" | "Danger"

  const statusClass = {
    Healthy:    'rs-status-healthy',
    Borderline: 'rs-status-borderline',
    Danger:     'rs-status-danger',
  }[label] ?? 'rs-status-borderline'

  const recommendation = {
    Healthy:    'Continue routine monitoring',
    Borderline: 'Further review recommended',
    Danger:     'Immediate clinical review recommended',
  }[label] ?? 'Review recommended'

  return (
    <div className={`risk-score-panel ${statusClass}`}>
      <h3 className="rs-title">Model Output — Risk Score</h3>
      <table className="rs-table">
        <tbody>
          <tr>
            <td className="rs-label">Risk Score</td>
            <td className="rs-value rs-score">{riskPct}%</td>
          </tr>
          <tr>
            <td className="rs-label">Decision Threshold</td>
            <td className="rs-value">{threshPct}%</td>
          </tr>
          <tr>
            <td className="rs-label">Borderline Range</td>
            <td className="rs-value">{healthyCutoffPct}%–{dangerCutoffPct}%</td>
          </tr>
          <tr className={`rs-status-row ${statusClass}`}>
            <td className="rs-label">Status</td>
            <td className="rs-value rs-status-cell">
              <strong>{label}</strong> — {recommendation}
            </td>
          </tr>
        </tbody>
      </table>
      <p className="rs-note">
        Risk score = P(Danger) output from the CNN. Danger defined as pH &lt; 7.10 in training labels.
      </p>
    </div>
  )
}
