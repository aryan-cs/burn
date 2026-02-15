import { useState } from 'react'
import { InfoTooltip } from '../InfoTooltip'

const TIPS: Record<string, string> = {
  'Feature Input': 'Enter a value for each feature. The model will use these to make a prediction.',
  'Prediction': 'The predicted class label or numeric value output by the model.',
  'Probabilities': 'Confidence the model assigns to each possible class (classification only).',
}

interface MlTestTabProps {
  featureNames: string[]
  targetNames: string[]
  isClassification: boolean
  jobId: string | null
  trainingComplete: boolean
  onBackToBuild: () => void
}

interface PredResponse {
  prediction: number | string
  probabilities?: number[]
  target_names?: string[]
}

export function MlTestTab({
  featureNames,
  targetNames,
  isClassification,
  jobId,
  trainingComplete,
  onBackToBuild,
}: MlTestTabProps) {
  const [values, setValues] = useState<string[]>(() => featureNames.map(() => '0'))
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<PredResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const canPredict = trainingComplete && jobId !== null && !busy

  const handleChange = (index: number, value: string) => {
    setValues((prev) => {
      const next = [...prev]
      next[index] = value
      return next
    })
  }

  const handlePredict = async () => {
    if (!canPredict) return
    setBusy(true)
    setError(null)

    try {
      const features = values.map((v) => {
        const n = Number(v)
        if (!Number.isFinite(n)) throw new Error(`Invalid number: "${v}"`)
        return n
      })

      const res = await fetch('/api/ml/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId, features }),
      })

      if (!res.ok) {
        const body = await res.text()
        throw new Error(body || `HTTP ${res.status}`)
      }

      const data = (await res.json()) as PredResponse
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="tab-panel">
      {/* Feature inputs */}
      <section className="panel-card">
        <h3 className="panel-subtitle">
          Feature Input
          <InfoTooltip title="Feature Input" text={TIPS['Feature Input']} position="right" />
        </h3>
        <div className="ml-feature-inputs">
          {featureNames.map((name, i) => (
            <label key={name} className="ml-feature-row">
              <span className="ml-feature-label">{name}</span>
              <input
                type="number"
                step="any"
                value={values[i] ?? '0'}
                onChange={(e) => handleChange(i, e.target.value)}
                className="ml-feature-input"
              />
            </label>
          ))}
        </div>
      </section>

      {/* Predict button */}
      <div className="panel-actions">
        <button
          onClick={handlePredict}
          disabled={!canPredict}
          className="btn btn-validate"
        >
          {busy ? 'Predicting...' : 'Predict'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <section className="panel-card ml-error-card">
          <p className="ml-error-text">{error}</p>
        </section>
      )}

      {/* Result */}
      {result && (
        <section className="panel-card">
          <h3 className="panel-subtitle">
            Prediction
            <InfoTooltip title="Prediction" text={TIPS['Prediction']} position="right" />
          </h3>
          <div className="ml-prediction-result">
            <span className="ml-prediction-value">
              {isClassification
                ? targetNames[Number(result.prediction)] ?? String(result.prediction)
                : Number(result.prediction).toFixed(4)}
            </span>
          </div>

          {isClassification && result.probabilities && (
            <>
              <h4 className="panel-subtitle ml-prob-title">
                Probabilities
                <InfoTooltip title="Probabilities" text={TIPS['Probabilities']} position="right" />
              </h4>
              <div className="ml-prob-list">
                {result.probabilities.map((prob, i) => {
                  const label = targetNames[i] ?? `Class ${i}`
                  return (
                    <div key={i} className="ml-prob-row">
                      <span className="ml-prob-label">{label}</span>
                      <div className="ml-prob-bar-bg">
                        <div
                          className="ml-prob-bar-fill"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                      <span className="ml-prob-val">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </section>
      )}

      {/* Back */}
      <div className="panel-actions">
        <button onClick={onBackToBuild} className="btn btn-neutral">
          Back To Build
        </button>
      </div>
    </div>
  )
}
