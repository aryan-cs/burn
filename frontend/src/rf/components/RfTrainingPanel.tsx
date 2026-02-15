import type { RFTrainingConfig } from '../types'
import type { RFRunStatus } from '../store/rfRunStore'

interface RfTrainingPanelProps {
  training: RFTrainingConfig
  status: RFRunStatus
  activeJobId: string | null
  onPatchTraining: (patch: Partial<RFTrainingConfig>) => void
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
  status,
  activeJobId,
  onPatchTraining,
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
          <div className="rf-meta-label">Status</div>
          <div className="rf-meta-value">{status.toUpperCase()}</div>
        </div>
        <div>
          <div className="rf-meta-label">Job ID</div>
          <div className="rf-meta-value">{activeJobId ?? '—'}</div>
        </div>
      </div>
    </section>
  )
}
