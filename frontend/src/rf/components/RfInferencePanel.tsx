import type { RFInferResponse } from '../types'

interface RfInferencePanelProps {
  featureCount: number
  featureNames: string[]
  values: number[]
  onChangeValues: (values: number[]) => void
  onLoadImage: (file: File) => void
  onInfer: () => void
  inferenceData: RFInferResponse | null
}

export function RfInferencePanel({
  featureCount,
  featureNames,
  values,
  onChangeValues,
  onLoadImage,
  onInfer,
  inferenceData,
}: RfInferencePanelProps) {
  return (
    <section className="rf-card">
      <div className="rf-card-title">Inference Input</div>
      <div className="rf-hint">
        Enter one sample feature vector or upload an image to auto-fill values, then run `POST /api/rf/infer`.
      </div>
      <div className="rf-row">
        <label className="rf-file-upload">
          <span>Upload Image</span>
          <input
            type="file"
            accept="image/*"
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) onLoadImage(file)
              event.target.value = ''
            }}
          />
        </label>
      </div>
      <div className="rf-grid-4">
        {Array.from({ length: featureCount }).map((_, index) => (
          <div key={index} className="rf-node-config">
            <label className="rf-label">{featureNames[index] ?? `feature_${index}`}</label>
            <input
              className="rf-input"
              type="number"
              step={0.0001}
              value={values[index] ?? 0}
              onChange={(event) => {
                const next = values.slice(0, featureCount)
                while (next.length < featureCount) next.push(0)
                next[index] = Number(event.target.value) || 0
                onChangeValues(next)
              }}
            />
          </div>
        ))}
      </div>
      <div className="rf-button-row rf-button-row-single">
        <button className="rf-btn rf-btn-indigo" onClick={onInfer}>
          Infer
        </button>
      </div>
      <div className="rf-card-subtitle">Latest Inference</div>
      <pre className="rf-json rf-json-small">
        {inferenceData
          ? JSON.stringify(
              {
                predictions: inferenceData.predictions,
                prediction_indices: inferenceData.prediction_indices,
                probabilities: inferenceData.probabilities,
              },
              null,
              2
            )
          : 'No inference output yet.'}
      </pre>
    </section>
  )
}
