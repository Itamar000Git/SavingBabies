export default function LoadingOverlay({ visible }) {
  if (!visible) return null
  return (
    <div className="loading-overlay" role="status" aria-label="Loading">
      <div className="spinner" />
      <p>Running prediction…</p>
    </div>
  )
}
