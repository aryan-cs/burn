import { MetricLineChart } from './Metrics'
import { LossSurfaceGraph } from './LossSurfaceGraph'

const TRAIN_ACC_COLOR = '#ffb429'
const TRAIN_LOSS_COLOR = '#ffd89c'
const TEST_ACC_COLOR = '#ff9f1a'
const TEST_LOSS_COLOR = '#ffe8be'

interface TrainingConfigView {
  dataset: string
  epochs: number
  batchSize: number
  optimizer: string
  learningRate: number
  loss: string
}

interface TrainTabProps {
  trainingConfig: TrainingConfigView
  isBackendBusy: boolean
  onDatasetChange: (value: string) => void
  onEpochsChange: (value: number) => void
  onBatchSizeChange: (value: number) => void
  onOptimizerChange: (value: string) => void
  onLearningRateChange: (value: number) => void
  onLossChange: (value: string) => void
  currentEpoch: number
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
  onOptimizerChange,
  onLearningRateChange,
  onLossChange,
  currentEpoch,
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
          latestTrainLoss={latestTrainLoss}
          latestTrainAccuracy={latestTrainAccuracy}
          latestTestLoss={latestTestLoss}
          latestTestAccuracy={latestTestAccuracy}
          currentEpoch={currentEpoch}
          totalEpochs={trainingConfig.epochs}
          optimizer={trainingConfig.optimizer}
        />
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
              <option value="digits">Digits (8x8)</option>
            </select>
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
            <span className="config-label">Optimizer</span>
            <select
              value={trainingConfig.optimizer}
              disabled={isBackendBusy}
              onChange={(e) => onOptimizerChange(e.target.value)}
              className="config-control"
            >
              {OPTIMIZER_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {formatOptionLabel(option)}
                </option>
              ))}
            </select>
          </label>

          <label className="config-row">
            <span className="config-label">Loss Function</span>
            <select
              value={trainingConfig.loss}
              disabled={isBackendBusy}
              onChange={(e) => onLossChange(e.target.value)}
              className="config-control"
            >
              {LOSS_FUNCTION_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {formatOptionLabel(option)}
                </option>
              ))}
            </select>
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
  latestTrainLoss: number | null
  latestTrainAccuracy: number | null
  latestTestLoss: number | null
  latestTestAccuracy: number | null
  currentEpoch: number
  totalEpochs: number
  optimizer: string
}

function TrainGraphsBlock({
  trainLossSeries,
  testLossSeries,
  trainAccuracySeries,
  testAccuracySeries,
  latestTrainLoss,
  latestTrainAccuracy,
  latestTestLoss,
  latestTestAccuracy,
  currentEpoch,
  totalEpochs,
  optimizer,
}: TrainGraphsBlockProps) {
  const lossSeries = trainLossSeries.length > 0 ? trainLossSeries : testLossSeries
  const trainChartBounds = getChartBounds(trainAccuracySeries, trainLossSeries)
  const testChartBounds = getChartBounds(testAccuracySeries, testLossSeries)

  return (
    <div className="train-graphs train-graph-block">
      <div className="train-accuracy-combined">
        <div className="train-accuracy-charts">
          <div className="train-graph-pane">
            <h3 className="panel-subtitle">Train Accuracy</h3>
            <div className="panel-chart">
              <MetricLineChart
                primaryLabel="Train Accuracy"
                secondaryLabel="Train Loss"
              primaryValues={trainAccuracySeries}
              secondaryValues={trainLossSeries}
              minValue={trainChartBounds.min}
              maxValue={trainChartBounds.max}
              primaryColor={TRAIN_ACC_COLOR}
              secondaryColor={TRAIN_LOSS_COLOR}
              xAxisLabel="Epoch"
              yAxisLabel="Value"
              xTickStep={0.05}
              yTickStep={0.05}
              />
            </div>
          </div>

          <div className="train-graph-pane">
            <h3 className="panel-subtitle">Test Accuracy</h3>
            <div className="panel-chart">
              <MetricLineChart
                primaryLabel="Test Accuracy"
                secondaryLabel="Test Loss"
              primaryValues={testAccuracySeries}
              secondaryValues={testLossSeries}
              minValue={testChartBounds.min}
              maxValue={testChartBounds.max}
              primaryColor={TEST_ACC_COLOR}
              secondaryColor={TEST_LOSS_COLOR}
              xAxisLabel="Epoch"
              yAxisLabel="Value"
              xTickStep={0.05}
              yTickStep={0.05}
              />
            </div>
          </div>
        </div>

        <div className="train-accuracy-stats">
          <div className="train-accuracy-stat-group">
            <p className="train-accuracy-stat-title">Train</p>
            <p className="train-accuracy-stat-value train-accuracy-stat-train-acc">
              {formatAccuracyValue(latestTrainAccuracy)} Accuracy
            </p>
            <p className="train-accuracy-stat-value train-accuracy-stat-train-loss">
              {formatLossValue(latestTrainLoss)} Loss
            </p>
          </div>
          <div className="train-accuracy-stat-group">
            <p className="train-accuracy-stat-title">Test</p>
            <p className="train-accuracy-stat-value train-accuracy-stat-test-acc">
              {formatAccuracyValue(latestTestAccuracy)} Accuracy
            </p>
            <p className="train-accuracy-stat-value train-accuracy-stat-test-loss">
              {formatLossValue(latestTestLoss)} Loss
            </p>
          </div>
        </div>
      </div>

      <div className="train-graph-pane">
        <h3 className="panel-subtitle">Loss Landscape</h3>
        <div className="panel-chart">
          <LossSurfaceGraph
            lossValues={lossSeries}
            optimizer={optimizer}
            currentEpoch={currentEpoch}
            totalEpochs={totalEpochs}
          />
        </div>
      </div>
    </div>
  )
}

function formatLossValue(loss: number | null): string {
  if (loss === null) return 'N/A'
  return `${(loss * 100).toFixed(2)}%`
}

function formatAccuracyValue(accuracy: number | null): string {
  if (accuracy === null) return 'N/A'
  return `${(accuracy * 100).toFixed(1)}%`
}

function getChartBounds(
  primaryValues: number[],
  secondaryValues: number[]
): { min: number; max: number } {
  const values = [...primaryValues, ...secondaryValues].filter((value) =>
    Number.isFinite(value)
  )
  if (values.length === 0) {
    return { min: 0, max: 1 }
  }

  const rawMin = Math.min(...values)
  const rawMax = Math.max(...values)
  const span = Math.max(rawMax - rawMin, 0.05)
  const paddedMin = Math.max(0, rawMin - span * 0.06)
  const paddedMax = rawMax + span * 0.02

  return {
    min: paddedMin,
    max: Math.max(paddedMax, paddedMin + 0.1),
  }
}

const OPTIMIZER_OPTIONS = ['adam', 'sgd']
const LOSS_FUNCTION_OPTIONS = ['cross_entropy', 'mse']

function formatOptionLabel(value: string): string {
  if (!value) return ''
  if (value === 'mse') return 'MSE'
  return value
    .split('_')
    .map((part) => {
      if (part.length === 0) return part
      if (part === 'sgd') return part.toUpperCase()
      return part.charAt(0).toUpperCase() + part.slice(1)
    })
    .join(' ')
}
