export default function ReliabilityPanel({ reliability, prediction }) {
  if (!reliability) return null

  const isPlaceholder = prediction?.placeholder === true

  const levelClass = {
    'High confidence': 'reliability-high',
    'Low confidence': 'reliability-low',
    'Borderline / Uncertain': 'reliability-borderline',
    'Not applicable': 'reliability-placeholder',
    'Unknown': 'reliability-unknown',
  }[reliability.level] ?? 'reliability-unknown'

  return (
    <div className={`reliability-panel ${levelClass}`}>
      {isPlaceholder && (
        <div className="placeholder-warning">
          ⚠ Placeholder prediction — real model weights are not loaded. Do not use this as a real prediction.
        </div>
      )}
      <div className="reliability-header">
        <span className="reliability-badge">{reliability.level}</span>
        {reliability.recommend_review && !isPlaceholder && (
          <span className="review-flag">Further review recommended</span>
        )}
      </div>
      <p className="reliability-message">{reliability.message}</p>
    </div>
  )
}
