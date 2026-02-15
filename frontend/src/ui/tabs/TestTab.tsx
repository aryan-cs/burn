import type { Dispatch, SetStateAction } from 'react'
import { InferencePixelPad } from '../InferencePixelPad'

interface InferenceDatasetSample {
  id: string
  filename: string
  label: number
  label_name: string
  image_data_url: string
  inputs: number[][][]
}

interface TestTabProps {
  datasetId: string
  inferenceGrid: number[][]
  setInferenceGrid: Dispatch<SetStateAction<number[][]>>
  padDisabled: boolean
  inferenceTopPrediction: number | null
  imageSamples: InferenceDatasetSample[]
  imageSamplePredictions: Record<string, number | null>
  imageSamplesLoading: boolean
  imageSamplesError: string | null
  onInferModel: () => void
  inferDisabled: boolean
  inferLabel: string
}

export function TestTab({
  datasetId,
  inferenceGrid,
  setInferenceGrid,
  padDisabled,
  inferenceTopPrediction,
  imageSamples,
  imageSamplePredictions,
  imageSamplesLoading,
  imageSamplesError,
  onInferModel,
  inferDisabled,
  inferLabel,
}: TestTabProps) {
  const isCatsVsDogs = datasetId === 'cats_vs_dogs'
  const getPredictionLabel = (prediction: number | null | undefined) => {
    if (prediction === undefined) return 'running...'
    if (prediction === null) return 'none'
    if (!isCatsVsDogs) return String(prediction)
    if (prediction === 0) return 'cat'
    if (prediction === 1) return 'dog'
    return String(prediction)
  }
  const predictionLabel =
    inferenceTopPrediction === null
      ? 'none'
      : isCatsVsDogs
        ? inferenceTopPrediction === 0
          ? 'cat (0)'
          : inferenceTopPrediction === 1
            ? 'dog (1)'
            : String(inferenceTopPrediction)
        : String(inferenceTopPrediction)
  const evaluatedCount = imageSamples.reduce((count, sample) => {
    const prediction = imageSamplePredictions[sample.id]
    return typeof prediction === 'number' ? count + 1 : count
  }, 0)
  const correctCount = imageSamples.reduce((count, sample) => {
    const prediction = imageSamplePredictions[sample.id]
    return typeof prediction === 'number' && prediction === sample.label ? count + 1 : count
  }, 0)
  const accuracyLabel =
    evaluatedCount > 0 ? `${((correctCount / evaluatedCount) * 100).toFixed(1)}%` : 'N/A'

  return (
    <div className="tab-panel test-tab-panel">
      <section className="panel-card panel-card-fill test-main-card">
        {isCatsVsDogs ? (
          <div className="inference-image-samples">
            <div className="inference-image-samples-head">
              <p className="inference-image-samples-title">Cats vs Dogs Samples (96x96)</p>
              {imageSamplesLoading ? (
                <p className="inference-image-samples-meta">Loading samples...</p>
              ) : imageSamplesError ? (
                <p className="inference-image-samples-error">{imageSamplesError}</p>
              ) : (
                <p className="inference-image-samples-meta">{imageSamples.length} samples loaded.</p>
              )}
            </div>

            <div className="inference-image-grid">
              {imageSamples.map((sample) => (
                <div
                  key={sample.id}
                  className="inference-image-tile"
                >
                  <img src={sample.image_data_url} alt={sample.filename} className="inference-image-preview" />
                  <span className="inference-image-label">{sample.label_name}</span>
                  <span
                    className={`inference-image-guess ${
                      imageSamplePredictions[sample.id] === undefined
                        ? 'inference-image-guess-pending'
                        : imageSamplePredictions[sample.id] === sample.label
                          ? 'inference-image-guess-correct'
                          : 'inference-image-guess-wrong'
                    }`}
                  >
                    Guess: {getPredictionLabel(imageSamplePredictions[sample.id])}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <InferencePixelPad grid={inferenceGrid} setGrid={setInferenceGrid} disabled={padDisabled} />
        )}
        <div className="inference-top-prediction">
          {isCatsVsDogs ? `Accuracy: ${accuracyLabel}` : `Top Prediction: ${predictionLabel}`}
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
