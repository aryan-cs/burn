import type { RFDatasetMeta } from '../types'

interface RfDatasetPanelProps {
  datasets: RFDatasetMeta[]
  datasetId: string
  onChangeDataset: (dataset: string) => void
}

export function RfDatasetPanel({ datasets, datasetId, onChangeDataset }: RfDatasetPanelProps) {
  return (
    <section className="rf-card">
      <div className="rf-card-title">Dataset</div>
      <div className="rf-row">
        <select
          className="rf-select"
          value={datasetId}
          onChange={(event) => onChangeDataset(event.target.value)}
        >
          {datasets.map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.name}
            </option>
          ))}
        </select>
      </div>
      {datasets.length > 0 && (
        <div className="rf-hint">
          Kaggle source:{' '}
          {datasets.find((dataset) => dataset.id === datasetId)?.kaggle_dataset ?? 'unknown'}
        </div>
      )}
    </section>
  )
}
