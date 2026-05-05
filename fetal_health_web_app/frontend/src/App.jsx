import { useState } from 'react'
import UploadPanel from './components/UploadPanel'
import CTGChartPanel from './components/CTGChartPanel'
import ClassificationCard from './components/ClassificationCard'
import ExplanationDashboard from './components/ExplanationDashboard'
import LoadingOverlay from './components/LoadingOverlay'
import ErrorBanner from './components/ErrorBanner'
import { runPrediction } from './api/client'

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

  return (
    <div className="app">
      <LoadingOverlay visible={appState === 'loading'} />
      <UploadPanel onSubmit={handleSubmit} isLoading={appState === 'loading'} />
      <ErrorBanner error={error} onDismiss={handleDismissError} />

      {result && (
        <>
          <CTGChartPanel
            signalData={result.signal_data}
            fhrEvents={result.fhr_events}
          />

          <div className="dashboard">
            <div className="left-panel">
              <ClassificationCard
                prediction={result.prediction}
                reliability={result.reliability}
              />
            </div>
            <div className="right-panel">
              <ExplanationDashboard
                explanation={result.explanation}
                signalFeatures={result.signal_features}
                fhrEvents={result.fhr_events}
                metadata={result.metadata}
                groundTruth={result.ground_truth}
                prediction={result.prediction}
              />
            </div>
          </div>
        </>
      )}

      <footer className="disclaimer">
        This tool is for research and decision-support only and is not a medical diagnosis.
      </footer>
    </div>
  )
}
