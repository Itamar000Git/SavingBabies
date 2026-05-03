import { useState } from 'react'
import UploadPanel from './components/UploadPanel'
import BabyVisual from './components/BabyVisual'
import ExplanationPanel from './components/ExplanationPanel'
import MetadataPanel from './components/MetadataPanel'
import LoadingOverlay from './components/LoadingOverlay'
import ErrorBanner from './components/ErrorBanner'
import { runPrediction } from './api/client'

// States: 'idle' | 'loading' | 'result' | 'error'

export default function App() {
  const [appState, setAppState] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleSubmit(file, modelName) {
    setAppState('loading')
    setError(null)
    setResult(null)
    try {
      const data = await runPrediction(file, modelName)
      setResult(data)
      setAppState('result')
    } catch (err) {
      setError(err.message)
      setAppState('error')
    }
  }

  function handleDismissError() {
    setError(null)
    setAppState('idle')
  }

  const prediction = result?.prediction ?? null
  const metadata = result?.metadata ?? null
  const explanation = result?.explanation ?? null

  return (
    <div className="app">
      <LoadingOverlay visible={appState === 'loading'} />

      <UploadPanel onSubmit={handleSubmit} isLoading={appState === 'loading'} />

      <ErrorBanner error={error} onDismiss={handleDismissError} />

      <div className="dashboard">
        <div className="left-panel">
          <BabyVisual prediction={prediction} />
          {explanation && <ExplanationPanel explanation={explanation} />}
        </div>
        <div className="right-panel">
          <MetadataPanel metadata={metadata} />
        </div>
      </div>

      <footer className="disclaimer">
        ⚠ This tool is for research and decision-support only and is not a medical diagnosis.
      </footer>
    </div>
  )
}
