export default function ErrorBanner({ error, onDismiss }) {
  if (!error) return null
  return (
    <div className="error-banner" role="alert">
      <span>{error}</span>
      <button className="dismiss-btn" onClick={onDismiss} aria-label="Dismiss">×</button>
    </div>
  )
}
