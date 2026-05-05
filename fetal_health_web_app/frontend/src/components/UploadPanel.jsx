import { useState } from 'react'

const MODEL = 'binarycnn'

export default function UploadPanel({ onSubmit, isLoading }) {
  const [file, setFile] = useState(null)

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

        <button
          className="run-btn"
          onClick={() => canRun && onSubmit(file, MODEL)}
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
