export default function GroundTruthPanel({ groundTruth, prediction }) {
  if (!groundTruth) return null

  const correctnessClass = {
    Correct: 'gt-correct',
    Incorrect: 'gt-incorrect',
    Unknown: 'gt-unknown',
  }[groundTruth.correctness] ?? 'gt-unknown'

  return (
    <div className="ground-truth-panel">
      <h3 className="gt-title">Ground Truth</h3>
      <div className="gt-row">
        <span className="gt-label">Predicted</span>
        <span className="gt-value">{prediction?.display_label ?? prediction?.label ?? '—'}</span>
      </div>
      {groundTruth.available ? (
        <>
          <div className="gt-row">
            <span className="gt-label">Actual outcome</span>
            <span className="gt-value">{groundTruth.actual_label}</span>
          </div>
          {groundTruth.ph_value != null && (
            <div className="gt-row">
              <span className="gt-label">pH value</span>
              <span className="gt-value">{groundTruth.ph_value}</span>
            </div>
          )}
          <div className={`gt-correctness ${correctnessClass}`}>
            {groundTruth.correctness === 'Correct' && '✓ Correct prediction'}
            {groundTruth.correctness === 'Incorrect' && '✗ Incorrect prediction'}
            {groundTruth.correctness === 'Unknown' && '? Unknown'}
          </div>
        </>
      ) : (
        <p className="gt-unavailable">
          Actual outcome is not available for this uploaded recording, so correctness cannot be verified from this screen.
        </p>
      )}
    </div>
  )
}
