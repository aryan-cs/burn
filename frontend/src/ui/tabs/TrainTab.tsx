import { MetricLineChart, MetricTile } from './Metrics'

interface TrainingConfigView {
  dataset: string
  epochs: number
  batchSize: number
  learningRate: number
}

interface TrainTabProps {
  trainingConfig: TrainingConfigView
  isBackendBusy: boolean
  onDatasetChange: (value: string) => void
  onEpochsChange: (value: number) => void
  onBatchSizeChange: (value: number) => void
  onLearningRateChange: (value: number) => void
  currentEpoch: number
  totalEpochs: number
  latestTrainLoss: number | null
  latestTrainAccuracy: number | null
  latestTestLoss: number | null
  latestTestAccuracy: number | null
  trainLossSeries: number[]
  testLossSeries: number[]
  trainAccuracySeries: number[]
  testAccuracySeries: number[]
  isTraining: boolean
  onStopModel: () => void
  stopDisabled: boolean
  stopLabel: string
  onTrainModel: () => void
  trainDisabled: boolean
  trainLabel: string
}

export function TrainTab({
  trainingConfig,
  isBackendBusy,
  onDatasetChange,
  onEpochsChange,
  onBatchSizeChange,
  onLearningRateChange,
  currentEpoch,
  totalEpochs,
  latestTrainLoss,
  latestTrainAccuracy,
  latestTestLoss,
  latestTestAccuracy,
  trainLossSeries,
  testLossSeries,
  trainAccuracySeries,
  testAccuracySeries,
  isTraining,
  onStopModel,
  stopDisabled,
  stopLabel,
  onTrainModel,
  trainDisabled,
  trainLabel,
}: TrainTabProps) {
  return (
    <div className="tab-panel">
      <section className="panel-card panel-card-fill">
        <TrainGraphsBlock
          trainAccuracySeries={trainAccuracySeries}
          testAccuracySeries={testAccuracySeries}
          trainLossSeries={trainLossSeries}
          testLossSeries={testLossSeries}
        />

        <div className="summary-grid">
          <MetricTile
            label="Epoch"
            value={totalEpochs > 0 ? `${currentEpoch}/${totalEpochs}` : '0/0'}
            compact
          />
          <MetricTile
            label="Train"
            value={formatTrainMetricText(latestTrainLoss, latestTrainAccuracy)}
            compact
          />
          <MetricTile
            label="Test"
            value={formatTrainMetricText(latestTestLoss, latestTestAccuracy)}
            compact
          />
        </div>
      </section>

      <section className="panel-card train-settings-card">
        <div className="config-grid train-config-grid">
          <label className="config-row">
            <span className="config-label">Dataset</span>
            <select
              value={trainingConfig.dataset}
              disabled={isBackendBusy}
              onChange={(e) => onDatasetChange(e.target.value)}
              className="config-control"
            >
              <option value="mnist">MNIST</option>
            </select>
          </label>

          <label className="config-row">
            <span className="config-label">Epochs</span>
            <input
              type="number"
              min={1}
              value={trainingConfig.epochs}
              disabled={isBackendBusy}
              onChange={(e) => onEpochsChange(Math.max(1, Number(e.target.value) || 1))}
              className="config-control config-control-numeric"
            />
          </label>

          <label className="config-row">
            <span className="config-label">Batch</span>
            <input
              type="number"
              min={1}
              value={trainingConfig.batchSize}
              disabled={isBackendBusy}
              onChange={(e) => onBatchSizeChange(Math.max(1, Number(e.target.value) || 1))}
              className="config-control config-control-numeric"
            />
          </label>

          <label className="config-row">
            <span className="config-label">Learning Rate</span>
            <input
              type="number"
              min={0.000001}
              step={0.0001}
              value={trainingConfig.learningRate}
              disabled={isBackendBusy}
              onChange={(e) =>
                onLearningRateChange(Math.max(0.000001, Number(e.target.value) || 0.000001))
              }
              className="config-control config-control-numeric"
            />
          </label>
        </div>
      </section>

      <div className="panel-actions">
        {isTraining ? (
          <button
            onClick={onStopModel}
            disabled={stopDisabled}
            className="btn btn-validate btn-danger"
          >
            {stopLabel}
          </button>
        ) : (
          <button
            onClick={onTrainModel}
            disabled={trainDisabled}
            className="btn btn-validate"
          >
            {trainLabel}
          </button>
        )}
      </div>
    </div>
  )
}

interface TrainGraphsBlockProps {
  trainLossSeries: number[]
  testLossSeries: number[]
  trainAccuracySeries: number[]
  testAccuracySeries: number[]
}

function TrainGraphsBlock({
  trainLossSeries,
  testLossSeries,
  trainAccuracySeries,
  testAccuracySeries,
}: TrainGraphsBlockProps) {
  return (
    <div className="train-graphs train-graph-block">
      <div>
        <h3 className="panel-subtitle">Accuracy</h3>
        <div className="panel-chart">
          <MetricLineChart
            primaryLabel="Train Acc"
            secondaryLabel="Test Acc"
            primaryValues={trainAccuracySeries}
            secondaryValues={testAccuracySeries}
            maxValue={1}
            xAxisLabel="Epoch"
            yAxisLabel="Accuracy (%)"
            xTickStep={0.05}
            yTickStep={0.05}
          />
        </div>
      </div>

      <div>
        <h3 className="panel-subtitle">Loss</h3>
        <div className="panel-chart">
          <MetricLineChart
            primaryLabel="Train Loss"
            secondaryLabel="Test Loss"
            primaryValues={trainLossSeries}
            secondaryValues={testLossSeries}
            maxValue={Math.max(...trainLossSeries, ...testLossSeries, 0.01)}
            xAxisLabel="Epoch"
            yAxisLabel="Loss"
            xTickStep={0.05}
            yTickStep={0.05}
          />
        </div>
      </div>
    </div>
  )
}

function formatTrainMetricText(
  loss: number | null,
  accuracy: number | null
): string {
  if (loss === null || accuracy === null) {
    return 'N/A'
  }
  return `${(loss * 100).toFixed(2)}% Loss, ${(accuracy * 100).toFixed(1)}% Up`
}
