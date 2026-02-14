import type { Dispatch, SetStateAction } from 'react'
import { InferencePixelPad } from '../InferencePixelPad'

interface TestTabProps {
  inferenceGrid: number[][]
  setInferenceGrid: Dispatch<SetStateAction<number[][]>>
  padDisabled: boolean
  inferenceTopPrediction: number | null
  onInferModel: () => void
  inferDisabled: boolean
  inferLabel: string
}

export function TestTab({
  inferenceGrid,
  setInferenceGrid,
  padDisabled,
  inferenceTopPrediction,
  onInferModel,
  inferDisabled,
  inferLabel,
}: TestTabProps) {
  return (
    <div className="tab-panel">
      <section className="panel-card panel-card-fill">
        <InferencePixelPad grid={inferenceGrid} setGrid={setInferenceGrid} disabled={padDisabled} />
        <div className="inference-top-prediction">
          Top Prediction: {inferenceTopPrediction ?? 'none'}
        </div>
      </section>

      <div className="panel-actions">
        <button
          onClick={onInferModel}
          disabled={inferDisabled}
          className="btn btn-validate"
        >
          {inferLabel}
        </button>
      </div>
    </div>
  )
}
