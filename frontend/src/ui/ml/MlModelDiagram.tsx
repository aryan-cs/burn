import type { MlModelType, MlTrainingStatus } from '../../store/mlStore'

/**
 * A visual pipeline diagram shown in the viewport area when
 * the user is in Classical ML mode. Shows:
 *   Data → Preprocessing → Model → Output
 */

interface MlModelDiagramProps {
  modelType: MlModelType
  datasetName: string
  task: 'classification' | 'regression' | null
  featureCount: number
  status: MlTrainingStatus
}

const MODEL_LABELS: Record<MlModelType, string> = {
  linear_regression: 'Linear Regression',
  logistic_regression: 'Logistic Regression',
  random_forest: 'Random Forest',
}

const MODEL_ICONS: Record<MlModelType, string> = {
  linear_regression: '📈',
  logistic_regression: '🔀',
  random_forest: '🌲',
}

const STATUS_GLOW: Record<string, string> = {
  idle: 'var(--c-surface-bright, #444)',
  loading_data: '#f5a623',
  training: '#4da3ff',
  complete: '#2ecc71',
  error: '#e74c3c',
  stopped: '#e67e22',
}

export function MlModelDiagram({
  modelType,
  datasetName,
  task,
  featureCount,
  status,
}: MlModelDiagramProps) {
  const glow = STATUS_GLOW[status] ?? STATUS_GLOW.idle

  return (
    <div className="ml-diagram-wrapper">
      <div className="ml-diagram-pipeline">
        {/* Step 1: Dataset */}
        <div className="ml-diagram-node">
          <div className="ml-diagram-icon">📊</div>
          <div className="ml-diagram-label">Dataset</div>
          <div className="ml-diagram-sub">{datasetName || '—'}</div>
          {featureCount > 0 && (
            <div className="ml-diagram-sub">{featureCount} features</div>
          )}
        </div>

        <Arrow />

        {/* Step 2: Preprocessing */}
        <div className="ml-diagram-node">
          <div className="ml-diagram-icon">⚙️</div>
          <div className="ml-diagram-label">Preprocessing</div>
          <div className="ml-diagram-sub">StandardScaler</div>
          <div className="ml-diagram-sub">Train / Test Split</div>
        </div>

        <Arrow />

        {/* Step 3: Model */}
        <div
          className="ml-diagram-node ml-diagram-node-model"
          style={{ boxShadow: `0 0 20px ${glow}, 0 0 40px ${glow}44` }}
        >
          <div className="ml-diagram-icon ml-diagram-icon-large">
            {MODEL_ICONS[modelType]}
          </div>
          <div className="ml-diagram-label">{MODEL_LABELS[modelType]}</div>
          <div className="ml-diagram-sub ml-diagram-status">
            {formatStatus(status)}
          </div>
        </div>

        <Arrow />

        {/* Step 4: Output */}
        <div className="ml-diagram-node">
          <div className="ml-diagram-icon">🎯</div>
          <div className="ml-diagram-label">Output</div>
          <div className="ml-diagram-sub">
            {task === 'classification' ? 'Class Label' : task === 'regression' ? 'Numeric Value' : '—'}
          </div>
        </div>
      </div>
    </div>
  )
}

function Arrow() {
  return (
    <div className="ml-diagram-arrow">
      <svg width="48" height="24" viewBox="0 0 48 24">
        <line
          x1="0"
          y1="12"
          x2="38"
          y2="12"
          stroke="var(--c-text-secondary, #aaa)"
          strokeWidth="2"
        />
        <polygon
          points="36,6 48,12 36,18"
          fill="var(--c-text-secondary, #aaa)"
        />
      </svg>
    </div>
  )
}

function formatStatus(status: MlTrainingStatus): string {
  switch (status) {
    case 'idle':
      return 'Ready'
    case 'loading_data':
      return 'Loading Data…'
    case 'training':
      return 'Training…'
    case 'complete':
      return 'Complete ✓'
    case 'error':
      return 'Error ✗'
    case 'stopped':
      return 'Stopped'
    default:
      return status
  }
}
