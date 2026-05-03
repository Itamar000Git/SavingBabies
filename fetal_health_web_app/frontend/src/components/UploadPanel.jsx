import { useState, useEffect } from 'react'
import { getModels } from '../api/client'

export default function UploadPanel({ onSubmit, isLoading }) {
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [file, setFile] = useState(null)

  useEffect(() => {
    getModels()
      .then(data => {
        setModels(data.models)
        if (data.models.length > 0) setSelectedModel(data.models[0])
      })
      .catch(err => console.error('Could not load models:', err))
  }, [])

  const canRun = file !== null && selectedModel !== '' && !isLoading

  return (
    <div className="upload-panel">
      <h1 className="app-title">Fetal Health Prediction</h1>
      <div className="controls">
        <label className="file-label">
          <input
            type="file"
            accept=".csv"
            onChange={e => setFile(e.target.files[0] ?? null)}
          />
          <span>{file ? file.name : 'Choose CSV file…'}</span>
        </label>

        <select
          value={selectedModel}
          onChange={e => setSelectedModel(e.target.value)}
          disabled={models.length === 0}
        >
          {models.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>

        <button
          className="run-btn"
          onClick={() => canRun && onSubmit(file, selectedModel)}
          disabled={!canRun}
        >
          {isLoading ? 'Running…' : 'Run Prediction'}
        </button>
      </div>
      {file && (
        <p className="file-info">
          {file.name} — {(file.size / 1024).toFixed(1)} KB
        </p>
      )}
    </div>
  )
}
