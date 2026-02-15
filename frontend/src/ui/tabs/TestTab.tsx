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
  selectedImageSampleId: string | null
  onSelectImageSample: (sampleId: string) => void
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
  selectedImageSampleId,
  onSelectImageSample,
  imageSamplesLoading,
  imageSamplesError,
  onInferModel,
  inferDisabled,
  inferLabel,
}: TestTabProps) {
  const isCatsVsDogs = datasetId === 'cats_vs_dogs'
  const selectedSample =
    imageSamples.find((sample) => sample.id === selectedImageSampleId) ?? null
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

  return (
    <div className="tab-panel">
      <section className="panel-card panel-card-fill">
        {isCatsVsDogs ? (
          <div className="inference-image-samples">
            <div className="inference-image-samples-head">
              <p className="inference-image-samples-title">Cats vs Dogs Samples (96x96)</p>
              <p className="inference-image-samples-subtitle">
                Choose a real dataset image for inference.
              </p>
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
                <button
                  key={sample.id}
                  type="button"
                  onClick={() => onSelectImageSample(sample.id)}
                  className={`inference-image-tile ${
                    selectedImageSampleId === sample.id
                      ? 'inference-image-tile-selected'
                      : 'inference-image-tile-idle'
                  }`}
                >
                  <img src={sample.image_data_url} alt={sample.filename} className="inference-image-preview" />
                  <span className="inference-image-label">{sample.label_name}</span>
                </button>
              ))}
            </div>

            <div className="inference-image-selection">
              {selectedSample
                ? `Selected: ${selectedSample.filename} (${selectedSample.label_name})`
                : 'No sample selected.'}
            </div>
          </div>
        ) : (
          <InferencePixelPad grid={inferenceGrid} setGrid={setInferenceGrid} disabled={padDisabled} />
        )}
        <div className="inference-top-prediction">
          Top Prediction: {predictionLabel}
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
