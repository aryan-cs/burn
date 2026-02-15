import type { MlModelType, MlDatasetInfo, MlHyperparameters, MlTask } from '../../store/mlStore'
import { InfoTooltip } from '../InfoTooltip'

const MODEL_TYPE_OPTIONS: { value: MlModelType; label: string; description: string }[] = [
  {
    value: 'linear_regression',
    label: 'Linear Regression',
    description: 'Fits a straight line through the data to predict continuous values.',
  },
  {
    value: 'logistic_regression',
    label: 'Logistic Regression',
    description: 'A classification model that uses a sigmoid function to predict class probabilities.',
  },
  {
    value: 'random_forest',
    label: 'Random Forest',
    description: 'An ensemble of decision trees that vote on the prediction. Works for both classification and regression.',
  },
]

const FIELD_TIPS: Record<string, string> = {
  'Model Type': 'The machine learning algorithm to use. Each has different strengths and trade-offs.',
  'Dataset': 'The dataset to train on. Different datasets suit different model types.',
  'Test Size': 'Fraction of data reserved for testing (not used during training). Typically 0.2 (20%).',
  'Fit Intercept': 'Whether to include a bias/intercept term in the model.',
  'C': 'Inverse regularisation strength. Smaller values = stronger regularisation (prevents overfitting).',
  'Max Iterations': 'Maximum number of optimisation iterations. Increase if the model hasn\'t converged.',
  'Penalty': 'Type of regularisation: L1 (sparse), L2 (smooth), or none.',
  'Solver': 'Optimisation algorithm. LBFGS is good for small datasets; SAGA supports all penalties.',
  'N Estimators': 'Number of decision trees in the forest. More trees = better accuracy but slower training.',
  'Max Depth': 'Maximum depth of each tree. Limiting depth prevents overfitting. Blank = unlimited.',
  'Min Samples Split': 'Minimum samples needed to split a node. Higher = more conservative trees.',
  'Min Samples Leaf': 'Minimum samples in a leaf node. Higher = smoother predictions.',
  'Criterion': 'Function to measure split quality. Gini is faster; Entropy can be more thorough.',
  'Max Features': 'Number of features considered per split. "sqrt" works well for most cases.',
}

interface MlBuildTabProps {
  modelType: MlModelType
  dataset: string
  testSize: number
  hyperparams: MlHyperparameters
  datasets: MlDatasetInfo[]
  task: MlTask | null
  onModelTypeChange: (type: MlModelType) => void
  onDatasetChange: (dataset: string) => void
  onTestSizeChange: (size: number) => void
  onHyperparamChange: (patch: Partial<MlHyperparameters>) => void
  onTrain: () => void
  trainDisabled: boolean
  trainLabel: string
}

export function MlBuildTab({
  modelType,
  dataset,
  testSize,
  hyperparams,
  datasets,
  onModelTypeChange,
  onDatasetChange,
  onTestSizeChange,
  onHyperparamChange,
  onTrain,
  trainDisabled,
  trainLabel,
}: MlBuildTabProps) {
  const safeDatasets = Array.isArray(datasets) ? datasets : []

  // Filter datasets compatible with the model type
  const compatibleDatasets = safeDatasets.filter((d) => {
    if (modelType === 'linear_regression') return d.task === 'regression'
    if (modelType === 'logistic_regression') return d.task === 'classification'
    return true // RF works with both
  })

  const selectedDataset = safeDatasets.find((d) => d.id === dataset)

  return (
    <div className="tab-panel">
      {/* Model type selection */}
      <section className="panel-card">
        <div className="config-grid ml-config-grid">
          <label className="config-row">
            <span className="config-label">
              Model Type
              <InfoTooltip title="Model Type" text={FIELD_TIPS['Model Type']} position="right" />
            </span>
            <select
              value={modelType}
              onChange={(e) => onModelTypeChange(e.target.value as MlModelType)}
              className="config-control"
            >
              {MODEL_TYPE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>

          <label className="config-row">
            <span className="config-label">
              Dataset
              <InfoTooltip title="Dataset" text={FIELD_TIPS['Dataset']} position="right" />
            </span>
            <select
              value={dataset}
              onChange={(e) => onDatasetChange(e.target.value)}
              className="config-control"
            >
              {compatibleDatasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </label>

          <label className="config-row">
            <span className="config-label">
              Test Size
              <InfoTooltip title="Test Size" text={FIELD_TIPS['Test Size']} position="right" />
            </span>
            <input
              type="number"
              min={0.05}
              max={0.5}
              step={0.05}
              value={testSize}
              onChange={(e) => onTestSizeChange(Number(e.target.value))}
              className="config-control config-control-numeric"
            />
          </label>
        </div>
      </section>

      {/* Dataset info */}
      {selectedDataset && (
        <section className="panel-card">
          <p className="ml-dataset-desc">{selectedDataset.description}</p>
          <div className="ml-dataset-meta">
            <span>{selectedDataset.n_samples} samples</span>
            <span>{selectedDataset.n_features} features</span>
            {selectedDataset.task === 'classification' && (
              <span>{selectedDataset.n_classes} classes</span>
            )}
          </div>
        </section>
      )}

      {/* Hyperparameters */}
      <section className="panel-card">
        <h3 className="panel-subtitle">Hyperparameters</h3>
        <div className="config-grid ml-config-grid">
          {modelType === 'linear_regression' && (
            <label className="config-row">
              <span className="config-label">
                Fit Intercept
                <InfoTooltip title="Fit Intercept" text={FIELD_TIPS['Fit Intercept']} position="right" />
              </span>
              <select
                value={hyperparams.fit_intercept ? 'true' : 'false'}
                onChange={(e) => onHyperparamChange({ fit_intercept: e.target.value === 'true' })}
                className="config-control"
              >
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
            </label>
          )}

          {modelType === 'logistic_regression' && (
            <>
              <label className="config-row">
                <span className="config-label">
                  C (Regularisation)
                  <InfoTooltip title="C" text={FIELD_TIPS['C']} position="right" />
                </span>
                <input
                  type="number"
                  min={0.001}
                  step={0.1}
                  value={hyperparams.C ?? 1.0}
                  onChange={(e) => onHyperparamChange({ C: Number(e.target.value) })}
                  className="config-control config-control-numeric"
                />
              </label>
              <label className="config-row">
                <span className="config-label">
                  Max Iterations
                  <InfoTooltip title="Max Iterations" text={FIELD_TIPS['Max Iterations']} position="right" />
                </span>
                <input
                  type="number"
                  min={10}
                  step={10}
                  value={hyperparams.max_iter ?? 200}
                  onChange={(e) => onHyperparamChange({ max_iter: Number(e.target.value) })}
                  className="config-control config-control-numeric"
                />
              </label>
              <label className="config-row">
                <span className="config-label">
                  Penalty
                  <InfoTooltip title="Penalty" text={FIELD_TIPS['Penalty']} position="right" />
                </span>
                <select
                  value={hyperparams.penalty ?? 'l2'}
                  onChange={(e) => onHyperparamChange({ penalty: e.target.value })}
                  className="config-control"
                >
                  <option value="l2">L2</option>
                  <option value="l1">L1</option>
                  <option value="none">None</option>
                </select>
              </label>
              <label className="config-row">
                <span className="config-label">
                  Solver
                  <InfoTooltip title="Solver" text={FIELD_TIPS['Solver']} position="right" />
                </span>
                <select
                  value={hyperparams.solver ?? 'lbfgs'}
                  onChange={(e) => onHyperparamChange({ solver: e.target.value })}
                  className="config-control"
                >
                  <option value="lbfgs">LBFGS</option>
                  <option value="liblinear">Liblinear</option>
                  <option value="saga">SAGA</option>
                  <option value="sag">SAG</option>
                  <option value="newton-cg">Newton-CG</option>
                </select>
              </label>
            </>
          )}

          {modelType === 'random_forest' && (
            <>
              <label className="config-row">
                <span className="config-label">
                  N Estimators
                  <InfoTooltip title="N Estimators" text={FIELD_TIPS['N Estimators']} position="right" />
                </span>
                <input
                  type="number"
                  min={1}
                  step={10}
                  value={hyperparams.n_estimators ?? 100}
                  onChange={(e) => onHyperparamChange({ n_estimators: Number(e.target.value) })}
                  className="config-control config-control-numeric"
                />
              </label>
              <label className="config-row">
                <span className="config-label">
                  Max Depth
                  <InfoTooltip title="Max Depth" text={FIELD_TIPS['Max Depth']} position="right" />
                </span>
                <input
                  type="number"
                  min={1}
                  placeholder="Unlimited"
                  value={hyperparams.max_depth ?? ''}
                  onChange={(e) =>
                    onHyperparamChange({
                      max_depth: e.target.value ? Number(e.target.value) : null,
                    })
                  }
                  className="config-control config-control-numeric"
                />
              </label>
              <label className="config-row">
                <span className="config-label">
                  Min Samples Split
                  <InfoTooltip title="Min Samples Split" text={FIELD_TIPS['Min Samples Split']} position="right" />
                </span>
                <input
                  type="number"
                  min={2}
                  value={hyperparams.min_samples_split ?? 2}
                  onChange={(e) => onHyperparamChange({ min_samples_split: Number(e.target.value) })}
                  className="config-control config-control-numeric"
                />
              </label>
              <label className="config-row">
                <span className="config-label">
                  Criterion
                  <InfoTooltip title="Criterion" text={FIELD_TIPS['Criterion']} position="right" />
                </span>
                <select
                  value={hyperparams.criterion ?? 'gini'}
                  onChange={(e) => onHyperparamChange({ criterion: e.target.value })}
                  className="config-control"
                >
                  <option value="gini">Gini</option>
                  <option value="entropy">Entropy</option>
                  <option value="log_loss">Log Loss</option>
                </select>
              </label>
              <label className="config-row">
                <span className="config-label">
                  Max Features
                  <InfoTooltip title="Max Features" text={FIELD_TIPS['Max Features']} position="right" />
                </span>
                <select
                  value={hyperparams.max_features ?? 'sqrt'}
                  onChange={(e) => onHyperparamChange({ max_features: e.target.value })}
                  className="config-control"
                >
                  <option value="sqrt">sqrt</option>
                  <option value="log2">log2</option>
                </select>
              </label>
            </>
          )}
        </div>
      </section>

      {/* Train button */}
      <div className="panel-actions">
        <button
          onClick={onTrain}
          disabled={trainDisabled}
          className="btn btn-validate"
        >
          {trainLabel}
        </button>
      </div>
    </div>
  )
}
