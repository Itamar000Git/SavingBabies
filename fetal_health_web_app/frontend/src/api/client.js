const BASE = '/api'

export async function getModels() {
  const res = await fetch(`${BASE}/models`)
  if (!res.ok) throw new Error('Failed to fetch model list')
  return res.json()
}

export async function runPrediction(file, modelName) {
  const form = new FormData()
  form.append('file', file)
  form.append('model_name', modelName)
  const res = await fetch(`${BASE}/predict`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown server error' }))
    throw new Error(err.detail || 'Prediction failed')
  }
  return res.json()
}
