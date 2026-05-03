const LABEL_COLOR = {
  healthy: '#22c55e',
  normal: '#22c55e',
  risk: '#ef4444',
  danger: '#ef4444',
  suspicious: '#f59e0b',
}

function getColor(label) {
  if (!label) return '#94a3b8'
  const key = label.toLowerCase()
  for (const [k, v] of Object.entries(LABEL_COLOR)) {
    if (key.includes(k)) return v
  }
  return '#94a3b8'
}

export default function BabyVisual({ prediction }) {
  const color = getColor(prediction?.label)

  return (
    <div className="baby-visual">
      <svg viewBox="0 0 200 320" width="180" height="288" aria-label="Baby silhouette">
        {/* Head */}
        <ellipse cx="100" cy="52" rx="36" ry="40" fill="none" stroke={color} strokeWidth="3" />
        {/* Body */}
        <ellipse cx="100" cy="168" rx="52" ry="72" fill="none" stroke={color} strokeWidth="3" />
        {/* Left arm */}
        <path d="M50 135 Q22 165 18 195" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
        {/* Right arm */}
        <path d="M150 135 Q178 165 182 195" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
        {/* Left leg */}
        <path d="M80 235 Q70 270 65 300" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
        {/* Right leg */}
        <path d="M120 235 Q130 270 135 300" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
      </svg>

      {prediction ? (
        <div className="prediction-badge" style={{ borderColor: color, color }}>
          <span className="prediction-label">{prediction.label}</span>
          {prediction.confidence != null && (
            <span className="prediction-confidence">
              {(prediction.confidence * 100).toFixed(1)}% confidence
            </span>
          )}
        </div>
      ) : (
        <div className="prediction-badge prediction-badge--empty">
          <span className="prediction-label">Awaiting prediction</span>
        </div>
      )}
    </div>
  )
}
