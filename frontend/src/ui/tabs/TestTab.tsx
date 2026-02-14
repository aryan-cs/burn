import type { Dispatch, SetStateAction } from 'react'
import { InferencePixelPad } from '../InferencePixelPad'

interface TestTabProps {
  trainingStatus: string
  trainingStatusClass: string
  trainingJobId: string | null
  backendMessage: string
  inferenceGrid: number[][]
  setInferenceGrid: Dispatch<SetStateAction<number[][]>>
  padDisabled: boolean
  inferenceTopPrediction: number | null
  inferenceOutput: string
  onInferModel: () => void
  inferDisabled: boolean
  inferLabel: string
  onBackToBuild: () => void
}

export function TestTab({
  trainingStatus,
  trainingStatusClass,
  trainingJobId,
  backendMessage,
  inferenceGrid,
  setInferenceGrid,
  padDisabled,
  inferenceTopPrediction,
  inferenceOutput,
  onInferModel,
  inferDisabled,
  inferLabel,
  onBackToBuild,
}: TestTabProps) {
  return (
    <div className="tab-panel">
      <section className="panel-card">
        <div className="panel-card-header">
          <h2 className="panel-title">Inference</h2>
          <span className={trainingStatusClass}>{trainingStatus.toUpperCase()}</span>
        </div>
        <p className="panel-muted-text">
          {trainingJobId ? `Using job ${trainingJobId}` : 'No trained model job available yet.'}
        </p>
        <p className="panel-muted-text panel-muted-text-tight">{backendMessage}</p>
      </section>

      <section className="panel-card panel-card-fill">
        <InferencePixelPad grid={inferenceGrid} setGrid={setInferenceGrid} disabled={padDisabled} />
        <div className="inference-top-prediction">
          Top Prediction: {inferenceTopPrediction ?? 'none'}
        </div>
        <pre className="inference-output">
          {inferenceOutput}
        </pre>
      </section>

      <div className="panel-actions panel-actions-split">
        <button
          onClick={onInferModel}
          disabled={inferDisabled}
          className="btn btn-indigo"
        >
          {inferLabel}
        </button>
        <button
          onClick={onBackToBuild}
          className="btn btn-neutral"
        >
          Back To Build
        </button>
      </div>
    </div>
  )
}
