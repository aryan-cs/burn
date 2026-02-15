import type { RFNode } from '../types'

interface RfNodeEditorProps {
  node: RFNode
  onPatchConfig: (nodeId: string, patch: Record<string, unknown>) => void
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback
}

export function RfNodeEditor({ node, onPatchConfig }: RfNodeEditorProps) {
  if (node.type === 'RFInput') {
    const shape = Array.isArray(node.config.shape) ? node.config.shape : [4]
    const features = asNumber(shape[0], 4)
    return (
      <div className="rf-node-config">
        <label className="rf-label">Feature Count</label>
        <input
          className="rf-input"
          type="number"
          min={1}
          value={features}
          onChange={(event) =>
            onPatchConfig(node.id, { shape: [Math.max(1, Number(event.target.value) || 1)] })
          }
        />
      </div>
    )
  }

  if (node.type === 'RFOutput') {
    const numClasses = asNumber(node.config.num_classes, 2)
    return (
      <div className="rf-node-config">
        <label className="rf-label">Num Classes</label>
        <input
          className="rf-input"
          type="number"
          min={2}
          value={numClasses}
          onChange={(event) =>
            onPatchConfig(node.id, { num_classes: Math.max(2, Number(event.target.value) || 2) })
          }
        />
      </div>
    )
  }

  if (node.type === 'RandomForestClassifier') {
    const nEstimators = asNumber(node.config.n_estimators, 100)
    const maxDepthRaw = node.config.max_depth
    const maxDepth = typeof maxDepthRaw === 'number' ? String(maxDepthRaw) : ''
    const criterion = typeof node.config.criterion === 'string' ? node.config.criterion : 'gini'
    const maxFeatures = node.config.max_features ?? 'sqrt'
    const minSamplesSplit = asNumber(node.config.min_samples_split, 2)
    const minSamplesLeaf = asNumber(node.config.min_samples_leaf, 1)
    const bootstrap = asBoolean(node.config.bootstrap, true)
    const randomState = asNumber(node.config.random_state, 42)

    return (
      <div className="rf-node-grid">
        <div className="rf-node-config">
          <label className="rf-label">n_estimators</label>
          <input
            className="rf-input"
            type="number"
            min={1}
            value={nEstimators}
            onChange={(event) =>
              onPatchConfig(node.id, { n_estimators: Math.max(1, Number(event.target.value) || 1) })
            }
          />
        </div>
        <div className="rf-node-config">
          <label className="rf-label">max_depth (blank = null)</label>
          <input
            className="rf-input"
            type="number"
            min={1}
            value={maxDepth}
            onChange={(event) => {
              const next = event.target.value.trim()
              onPatchConfig(node.id, { max_depth: next === '' ? null : Math.max(1, Number(next) || 1) })
            }}
          />
        </div>
        <div className="rf-node-config">
          <label className="rf-label">criterion</label>
          <select
            className="rf-select"
            value={criterion}
            onChange={(event) => onPatchConfig(node.id, { criterion: event.target.value })}
          >
            <option value="gini">gini</option>
            <option value="entropy">entropy</option>
            <option value="log_loss">log_loss</option>
          </select>
        </div>
        <div className="rf-node-config">
          <label className="rf-label">max_features</label>
          <select
            className="rf-select"
            value={String(maxFeatures)}
            onChange={(event) => {
              const value = event.target.value
              onPatchConfig(node.id, { max_features: value === 'none' ? null : value })
            }}
          >
            <option value="sqrt">sqrt</option>
            <option value="log2">log2</option>
            <option value="none">none</option>
          </select>
        </div>
        <div className="rf-node-config">
          <label className="rf-label">min_samples_split</label>
          <input
            className="rf-input"
            type="number"
            min={2}
            value={minSamplesSplit}
            onChange={(event) =>
              onPatchConfig(node.id, {
                min_samples_split: Math.max(2, Number(event.target.value) || 2),
              })
            }
          />
        </div>
        <div className="rf-node-config">
          <label className="rf-label">min_samples_leaf</label>
          <input
            className="rf-input"
            type="number"
            min={1}
            value={minSamplesLeaf}
            onChange={(event) =>
              onPatchConfig(node.id, { min_samples_leaf: Math.max(1, Number(event.target.value) || 1) })
            }
          />
        </div>
        <div className="rf-node-config">
          <label className="rf-label">bootstrap</label>
          <select
            className="rf-select"
            value={bootstrap ? 'true' : 'false'}
            onChange={(event) => onPatchConfig(node.id, { bootstrap: event.target.value === 'true' })}
          >
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
        </div>
        <div className="rf-node-config">
          <label className="rf-label">random_state</label>
          <input
            className="rf-input"
            type="number"
            value={randomState}
            onChange={(event) =>
              onPatchConfig(node.id, { random_state: Number(event.target.value) || 0 })
            }
          />
        </div>
      </div>
    )
  }

  return <div className="rf-hint">No configuration fields for this node.</div>
}
