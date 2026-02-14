import { MetricLineChart, MetricTile } from './Metrics'

interface TrainingConfigView {
  dataset: string
  epochs: number
  batchSize: number
  learningRate: number
}

interface TrainTabProps {
  trainingStatus: string
  trainingStatusClass: string
  trainingErrorMessage: string | null
  backendMessage: string
  trainingConfig: TrainingConfigView
  isBackendBusy: boolean
  onDatasetChange: (value: string) => void
  onEpochsChange: (value: number) => void
  onBatchSizeChange: (value: number) => void
  onLearningRateChange: (value: number) => void
  trainingJobId: string | null
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
  onGoToTest: () => void
  goToTestDisabled: boolean
}

export function TrainTab({
  trainingStatus,
  trainingStatusClass,
  trainingErrorMessage,
  backendMessage,
  trainingConfig,
  isBackendBusy,
  onDatasetChange,
  onEpochsChange,
  onBatchSizeChange,
  onLearningRateChange,
  trainingJobId,
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
  onGoToTest,
  goToTestDisabled,
}: TrainTabProps) {
  return (
    <div className="tab-panel">
      <section className="panel-card">
        <div className="panel-card-header">
          <h2 className="panel-title">Training</h2>
          <span className={trainingStatusClass}>{trainingStatus.toUpperCase()}</span>
        </div>

        <p className="panel-muted-text">{trainingErrorMessage ?? backendMessage}</p>

        <div className="config-grid">
          <label className="config-row">
            <span className="config-label">Dataset</span>
            <select
              value={trainingConfig.dataset}
              disabled={isBackendBusy}
              onChange={(e) => onDatasetChange(e.target.value)}
              className="config-control"
            >
              <option value="mnist">mnist</option>
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
            <span className="config-label">LR</span>
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

        <div className="summary-grid">
          <MetricTile label="Job ID" value={trainingJobId ?? 'none'} compact />
          <MetricTile
            label="Epoch"
            value={totalEpochs > 0 ? `${currentEpoch}/${totalEpochs}` : '0/0'}
            compact
          />
          <MetricTile
            label="Train"
            value={
              latestTrainLoss !== null && latestTrainAccuracy !== null
                ? `L ${latestTrainLoss.toFixed(4)} A ${(latestTrainAccuracy * 100).toFixed(1)}%`
                : 'n/a'
            }
            compact
          />
          <MetricTile
            label="Test"
            value={
              latestTestLoss !== null && latestTestAccuracy !== null
                ? `L ${latestTestLoss.toFixed(4)} A ${(latestTestAccuracy * 100).toFixed(1)}%`
                : 'n/a'
            }
            compact
          />
        </div>
      </section>

      <section className="panel-card">
        <h2 className="panel-title">Loss</h2>
        <div className="panel-chart">
          <MetricLineChart
            primaryLabel="Train Loss"
            secondaryLabel="Test Loss"
            primaryValues={trainLossSeries}
            secondaryValues={testLossSeries}
            maxValue={Math.max(...trainLossSeries, ...testLossSeries, 0.01)}
          />
        </div>
      </section>

      <section className="panel-card">
        <h2 className="panel-title">Accuracy</h2>
        <div className="panel-chart">
          <MetricLineChart
            primaryLabel="Train Acc"
            secondaryLabel="Test Acc"
            primaryValues={trainAccuracySeries}
            secondaryValues={testAccuracySeries}
            maxValue={1}
          />
        </div>
      </section>

      <div className="panel-actions panel-actions-split">
        {isTraining ? (
          <button
            onClick={onStopModel}
            disabled={stopDisabled}
            className="btn btn-danger"
          >
            {stopLabel}
          </button>
        ) : (
          <button
            onClick={onTrainModel}
            disabled={trainDisabled}
            className="btn btn-success"
          >
            {trainLabel}
          </button>
        )}
        <button
          onClick={onGoToTest}
          disabled={goToTestDisabled}
          className="btn btn-indigo"
        >
          Go To Test
        </button>
      </div>
    </div>
  )
}
