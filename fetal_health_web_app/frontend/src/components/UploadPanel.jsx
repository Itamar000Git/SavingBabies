import { useState } from 'react'

const MODELS = [
  { value: 'ensemble',  label: 'Ensemble (XGBoost + CNN + BiGRU) ★' },
  { value: 'binarycnn', label: 'Binary CNN' },
  { value: 'xgboost',   label: 'XGBoost' },
  { value: 'minirocket', label: 'MiniRocket' },
]

export default function UploadPanel({ onSubmit, isLoading }) {
  const [file, setFile] = useState(null)
  const [model, setModel] = useState('ensemble')

  const canRun = file !== null && !isLoading

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
          <span>{file ? file.name : 'Choose CTG CSV file…'}</span>
        </label>

        <select
          className="model-select"
          value={model}
          onChange={e => setModel(e.target.value)}
          disabled={isLoading}
        >
          {MODELS.map(m => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>

        <button
          className="run-btn"
          onClick={() => canRun && onSubmit(file, model)}
          disabled={!canRun}
        >
          {isLoading ? 'Analysing…' : 'Analyse CTG'}
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
