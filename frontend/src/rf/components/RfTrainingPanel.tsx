import type { RFNode, RFTrainingConfig, RFVisualizationConfig } from '../types'
import type { RFRunStatus } from '../store/rfRunStore'
import { RfNodeEditor } from './RfNodeEditor'

function formatStrategy(strategy: string): string {
  if (!strategy) return 'Bagging'
  return strategy.charAt(0).toUpperCase() + strategy.slice(1)
}

interface RfTrainingPanelProps {
  training: RFTrainingConfig
  visualization: RFVisualizationConfig
  status: RFRunStatus
  activeJobId: string | null
  builtTrees: number
  totalTrees: number
  completionPercent: number
  onPatchTraining: (patch: Partial<RFTrainingConfig>) => void
  onPatchVisualization: (patch: Partial<RFVisualizationConfig>) => void
  onValidate: () => void
  onCompile: () => void
  onTrain: () => void
  onStop: () => void
  onStatus: () => void
  onExportPy: () => void
  onExportPkl: () => void
}

export function RfTrainingPanel({
  training,
  visualization,
  status,
  activeJobId,
  builtTrees,
  totalTrees,
  completionPercent,
  onPatchTraining,
  onPatchVisualization,
  onValidate,
  onCompile,
  onTrain,
  onStop,
  onStatus,
  onExportPy,
  onExportPkl,
}: RfTrainingPanelProps) {
  return (
    <section className="rf-card">
      <div className="rf-card-title">Training Controls</div>
      <div className="rf-grid-2">
        <div className="rf-node-config">
          <label className="rf-label">Test Size</label>
          <input
            className="rf-input"
            type="number"
            step={0.01}
            min={0.05}
            max={0.95}
            value={training.testSize}
            onChange={(event) =>
              onPatchTraining({ testSize: Math.min(0.95, Math.max(0.05, Number(event.target.value) || 0.2)) })
            }
          />
        </div>
        <div className="rf-node-config">
          <label className="rf-label">Random State</label>
          <input
            className="rf-input"
            type="number"
            value={training.randomState}
            onChange={(event) => onPatchTraining({ randomState: Number(event.target.value) || 0 })}
          />
        </div>
        <div className="rf-node-config">
          <label className="rf-label">Stratify</label>
          <select
            className="rf-select"
            value={training.stratify ? 'true' : 'false'}
            onChange={(event) => onPatchTraining({ stratify: event.target.value === 'true' })}
          >
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
        </div>
        <div className="rf-node-config">
          <label className="rf-label">Log Every Trees</label>
          <input
            className="rf-input"
            type="number"
            min={1}
            value={training.logEveryTrees}
            onChange={(event) => onPatchTraining({ logEveryTrees: Math.max(1, Number(event.target.value) || 1) })}
          />
        </div>
        <div className="rf-node-config">
          <label className="rf-label">Ensemble Strategy</label>
          <select
            className="rf-select"
            value={training.ensembleStrategy}
            onChange={(event) =>
              onPatchTraining({
                ensembleStrategy: event.target.value as 'bagging' | 'boosting' | 'stacking' | 'averaging',
              })
            }
          >
            <option value="bagging">Bagging</option>
            <option value="boosting">Boosting</option>
            <option value="stacking">Stacking</option>
            <option value="averaging">Averaging</option>
          </select>
        </div>
      </div>

      <div className="rf-card-subtitle">3D Visualization</div>
      <div className="rf-grid-2">
        <div className="rf-node-config">
          <label className="rf-label">Visible Trees</label>
          <input
            className="rf-input"
            type="number"
            min={3}
            max={48}
            value={visualization.visibleTrees}
            onChange={(event) =>
              onPatchVisualization({
                visibleTrees: Math.min(48, Math.max(3, Number(event.target.value) || 3)),
              })
            }
          />
        </div>
        <div className="rf-node-config">
          <label className="rf-label">Tree Depth</label>
          <input
            className="rf-input"
            type="number"
            min={2}
            max={6}
            value={visualization.treeDepth}
            onChange={(event) =>
              onPatchVisualization({
                treeDepth: Math.min(6, Math.max(2, Number(event.target.value) || 2)),
              })
            }
          />
        </div>
        <div className="rf-node-config">
          <label className="rf-label">Tree Spread</label>
          <input
            className="rf-input"
            type="number"
            min={0.6}
            max={2.2}
            step={0.1}
            value={visualization.treeSpread}
            onChange={(event) =>
              onPatchVisualization({
                treeSpread: Math.min(2.2, Math.max(0.6, Number(event.target.value) || 0.6)),
              })
            }
          />
        </div>
        <div className="rf-node-config">
          <label className="rf-label">Node Scale</label>
          <input
            className="rf-input"
            type="number"
            min={0.6}
            max={1.8}
            step={0.1}
            value={visualization.nodeScale}
            onChange={(event) =>
              onPatchVisualization({
                nodeScale: Math.min(1.8, Math.max(0.6, Number(event.target.value) || 0.6)),
              })
            }
          />
        </div>
      </div>

      <div className="rf-button-row">
        <button className="rf-btn" onClick={onValidate}>
          Validate
        </button>
        <button className="rf-btn" onClick={onCompile}>
          Compile
        </button>
        <button className="rf-btn rf-btn-green" onClick={onTrain}>
          Train
        </button>
        <button className="rf-btn rf-btn-red" onClick={onStop}>
          Stop
        </button>
        <button className="rf-btn" onClick={onStatus}>
          Status
        </button>
        <button className="rf-btn" onClick={onExportPy}>
          Export .py
        </button>
        <button className="rf-btn" onClick={onExportPkl}>
          Export .pkl
        </button>
      </div>

      <div className="rf-meta-grid">
        <div>
          <div className="rf-meta-label">Strategy</div>
          <div className="rf-meta-value">{formatStrategy(training.ensembleStrategy)}</div>
        </div>
        <div>
          <div className="rf-meta-label">Status</div>
          <div className="rf-meta-value">{status.toUpperCase()}</div>
        </div>
        <div>
          <div className="rf-meta-label">Job ID</div>
          <div className="rf-meta-value">{activeJobId ?? '—'}</div>
        </div>
        <div>
          <div className="rf-meta-label">Trees Built</div>
          <div className="rf-meta-value">
            {builtTrees}/{totalTrees}
          </div>
        </div>
        <div>
          <div className="rf-meta-label">Completion</div>
          <div className="rf-meta-value">{completionPercent}%</div>
        </div>
      </div>
    </section>
  )
}

interface RfForestConfigPanelProps {
  modelNode: RFNode | null
  onPatchNodeConfig: (nodeId: string, patch: Record<string, unknown>) => void
  onResetPreset: () => void
}

export function RfForestConfigPanel({
  modelNode,
  onPatchNodeConfig,
  onResetPreset,
}: RfForestConfigPanelProps) {
  if (!modelNode) return null

  return (
    <section className="rf-card">
      <div className="rf-card-title-row">
        <div className="rf-card-title">Forest Hyperparameters</div>
        <button className="rf-btn rf-btn-sm" onClick={onResetPreset}>
          Reset Defaults
        </button>
      </div>
      <div className="rf-row">
        <RfNodeEditor node={modelNode} onPatchConfig={onPatchNodeConfig} />
      </div>
    </section>
  )
}
