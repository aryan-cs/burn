import type { MlMetrics, MlModelType, MlProgressUpdate } from '../../store/mlStore'
import { InfoTooltip } from '../InfoTooltip'

const METRIC_TIPS: Record<string, string> = {
  'Progress': 'Training progress from 0% to 100%. For logistic regression this tracks iterations; for random forest it tracks trees built.',
  'Accuracy': 'Percentage of samples correctly classified.',
  'Precision': 'Of all predicted positive cases, how many were actually positive.',
  'Recall': 'Of all actual positive cases, how many were correctly predicted.',
  'F1': 'Harmonic mean of precision and recall. Balances both metrics.',
  'R²': 'Coefficient of determination — how much variance the model explains. 1.0 is perfect.',
  'MSE': 'Mean Squared Error — average of squared differences between predicted and actual values.',
  'RMSE': 'Root Mean Squared Error — square root of MSE, in the same units as the target.',
  'MAE': 'Mean Absolute Error — average absolute difference between predicted and actual values.',
  'Training Time': 'Total wall-clock time spent training the model.',
  'Feature Importances': 'How much each feature contributes to predictions. Higher = more important.',
}

interface MlTrainTabProps {
  modelType: MlModelType
  status: string
  progress: MlProgressUpdate | null
  progressHistory: MlProgressUpdate[]
  trainMetrics: MlMetrics | null
  testMetrics: MlMetrics | null
  featureImportances: Record<string, number> | null
  trainingTime: number | null
  isTraining: boolean
  onStop: () => void
  stopDisabled: boolean
  stopLabel: string
  onBackToBuild: () => void
}

export function MlTrainTab({
  modelType,
  status,
  progress,
  progressHistory,
  trainMetrics,
  testMetrics,
  featureImportances,
  trainingTime,
  isTraining,
  onStop,
  stopDisabled,
  stopLabel,
  onBackToBuild,
}: MlTrainTabProps) {
  const isClassification = trainMetrics
    ? 'accuracy' in trainMetrics
    : modelType !== 'linear_regression'

  return (
    <div className="tab-panel">
      {/* Progress bar */}
      <section className="panel-card">
        <div className="ml-progress-header">
          <h3 className="panel-subtitle">
            Training Progress
            <InfoTooltip title="Progress" text={METRIC_TIPS['Progress']} position="right" />
          </h3>
          <span className={`status-pill status-pill-${status === 'complete' ? 'complete' : status === 'error' ? 'error' : status === 'training' ? 'training' : 'idle'}`}>
            {status.toUpperCase().replace('_', ' ')}
          </span>
        </div>
        <div className="ml-progress-bar-container">
          <div
            className="ml-progress-bar-fill"
            style={{ width: `${(progress?.progress ?? 0) * 100}%` }}
          />
        </div>
        <div className="ml-progress-detail">
          {progress
            ? `Step ${progress.step} / ${progress.totalSteps} — ${(progress.progress * 100).toFixed(0)}%`
            : 'Waiting to start...'}
        </div>

        {/* Live accuracy / R² chart area */}
        {progressHistory.length > 0 && (
          <div className="ml-progress-chart">
            <div className="ml-sparkline-container">
              {isClassification ? (
                <>
                  <MlSparkline
                    label="Train Accuracy"
                    values={progressHistory.map((p) => p.trainAccuracy ?? 0)}
                    color="#4da3ff"
                  />
                  <MlSparkline
                    label="Test Accuracy"
                    values={progressHistory.map((p) => p.testAccuracy ?? 0)}
                    color="#ff8c2b"
                  />
                </>
              ) : (
                <>
                  <MlSparkline
                    label="Train R²"
                    values={progressHistory.map((p) => p.trainR2 ?? 0)}
                    color="#4da3ff"
                  />
                  <MlSparkline
                    label="Test R²"
                    values={progressHistory.map((p) => p.testR2 ?? 0)}
                    color="#ff8c2b"
                  />
                </>
              )}
            </div>
          </div>
        )}
      </section>

      {/* Final metrics */}
      {trainMetrics && testMetrics && (
        <section className="panel-card">
          <h3 className="panel-subtitle">Final Metrics</h3>
          <div className="ml-metrics-grid">
            <div className="ml-metrics-column">
              <h4 className="ml-metrics-column-title">Train</h4>
              {Object.entries(trainMetrics).map(([key, val]) => (
                <div key={key} className="ml-metric-row">
                  <span className="ml-metric-label">
                    {formatMetricLabel(key)}
                    <InfoTooltip
                      title={formatMetricLabel(key)}
                      text={METRIC_TIPS[formatMetricLabel(key)] ?? `${formatMetricLabel(key)} on training data.`}
                      position="right"
                    />
                  </span>
                  <span className="ml-metric-value">{formatMetricValue(key, val)}</span>
                </div>
              ))}
            </div>
            <div className="ml-metrics-column">
              <h4 className="ml-metrics-column-title">Test</h4>
              {Object.entries(testMetrics).map(([key, val]) => (
                <div key={key} className="ml-metric-row">
                  <span className="ml-metric-label">{formatMetricLabel(key)}</span>
                  <span className="ml-metric-value">{formatMetricValue(key, val)}</span>
                </div>
              ))}
            </div>
          </div>

          {trainingTime !== null && (
            <div className="ml-metric-row ml-metric-row-highlight">
              <span className="ml-metric-label">
                Training Time
                <InfoTooltip title="Training Time" text={METRIC_TIPS['Training Time']} position="right" />
              </span>
              <span className="ml-metric-value">{trainingTime.toFixed(2)}s</span>
            </div>
          )}
        </section>
      )}

      {/* Feature importances */}
      {featureImportances && (
        <section className="panel-card">
          <h3 className="panel-subtitle">
            Feature Importances
            <InfoTooltip title="Feature Importances" text={METRIC_TIPS['Feature Importances']} position="right" />
          </h3>
          <div className="ml-importance-list">
            {sortedImportances(featureImportances).map(([name, imp]) => (
              <div key={name} className="ml-importance-row">
                <span className="ml-importance-name">{name}</span>
                <div className="ml-importance-bar-bg">
                  <div
                    className="ml-importance-bar-fill"
                    style={{ width: `${imp * 100}%` }}
                  />
                </div>
                <span className="ml-importance-val">{(imp * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Actions */}
      <div className="panel-actions panel-actions-split">
        {isTraining ? (
          <button onClick={onStop} disabled={stopDisabled} className="btn btn-validate btn-danger">
            {stopLabel}
          </button>
        ) : (
          <button onClick={onBackToBuild} className="btn btn-neutral">
            Back To Build
          </button>
        )}
      </div>
    </div>
  )
}

// ── Sparkline ──

function MlSparkline({ label, values, color }: { label: string; values: number[]; color: string }) {
  if (values.length === 0) return null
  const max = Math.max(...values, 0.01)
  const w = 200
  const h = 40
  const points = values
    .map((v, i) => `${(i / Math.max(values.length - 1, 1)) * w},${h - (v / max) * h}`)
    .join(' ')

  return (
    <div className="ml-sparkline">
      <span className="ml-sparkline-label" style={{ color }}>{label}</span>
      <svg viewBox={`0 0 ${w} ${h}`} className="ml-sparkline-svg">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinejoin="round"
        />
      </svg>
      <span className="ml-sparkline-value">{(values[values.length - 1] * 100).toFixed(1)}%</span>
    </div>
  )
}

// ── Helpers ──

function sortedImportances(imp: Record<string, number>): [string, number][] {
  const maxVal = Math.max(...Object.values(imp), 1e-10)
  return Object.entries(imp)
    .map(([k, v]) => [k, v / maxVal] as [string, number])
    .sort((a, b) => b[1] - a[1])
}

function formatMetricLabel(key: string): string {
  const map: Record<string, string> = {
    accuracy: 'Accuracy',
    precision: 'Precision',
    recall: 'Recall',
    f1: 'F1',
    r2: 'R²',
    mse: 'MSE',
    rmse: 'RMSE',
    mae: 'MAE',
  }
  return map[key] ?? key
}

function formatMetricValue(key: string, value: number): string {
  if (key === 'accuracy' || key === 'precision' || key === 'recall' || key === 'f1') {
    return `${(value * 100).toFixed(1)}%`
  }
  if (key === 'r2') {
    return value.toFixed(4)
  }
  return value.toFixed(4)
}
