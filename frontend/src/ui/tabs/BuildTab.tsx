import type { LayerRole } from '../../utils/graphOrder'
import { MetricTile } from './Metrics'

export interface BuildLayerItem {
  id: string
  name: string
  role: LayerRole
  rows: number
  cols: number
}

interface BuildTabProps {
  layerItems: BuildLayerItem[]
  hasSelectedNode: boolean
  isEditingName: boolean
  draftName: string
  selectedDisplayName: string
  selectedRows: number
  selectedCols: number
  selectedActivation: string
  activationOptions: string[]
  layerCount: number
  neuronCount: number
  weightCount: number
  biasCount: number
  layerTypeSummary: string
  sharedNonOutputActivation: string
  onAddLayer: () => void
  onBeginNameEdit: () => void
  onDraftNameChange: (value: string) => void
  onNameCommit: () => void
  onNameCancel: () => void
  onRowsChange: (value: string) => void
  onColsChange: (value: string) => void
  onActivationChange: (value: string) => void
  onValidate: () => void
  validateDisabled: boolean
  validateLabel: string
}

export function BuildTab({
  layerItems,
  hasSelectedNode,
  isEditingName,
  draftName,
  selectedDisplayName,
  selectedRows,
  selectedCols,
  selectedActivation,
  activationOptions,
  layerCount,
  neuronCount,
  weightCount,
  biasCount,
  layerTypeSummary,
  sharedNonOutputActivation,
  onAddLayer,
  onBeginNameEdit,
  onDraftNameChange,
  onNameCommit,
  onNameCancel,
  onRowsChange,
  onColsChange,
  onActivationChange,
  onValidate,
  validateDisabled,
  validateLabel,
}: BuildTabProps) {
  return (
    <div className="tab-panel">
      <section className="panel-card panel-card-layers">
        <div className="layer-list-shell">
          <ol className="layer-list">
            {layerItems.length === 0 ? (
              <li className="layer-list-empty-item">No layers added yet.</li>
            ) : (
              layerItems.map((layerItem) => (
                <li
                  key={layerItem.id}
                  className="layer-list-item"
                >
                  <div className="layer-list-item-left">
                    <span className={getRoleDotClass(layerItem.role)} />
                    <span className="layer-list-item-name">{layerItem.name}</span>
                  </div>
                  <span className="layer-list-item-size">
                    {layerItem.rows} x {layerItem.cols}
                  </span>
                </li>
              ))
            )}
            <li className="layer-list-add-item">
              <button
                onClick={onAddLayer}
                aria-label="Add Layer"
                title="Add Layer"
                className="layer-list-add-button"
              >
                <span className="layer-list-add-plus">+</span>
                <span>Add Layer</span>
              </button>
            </li>
          </ol>
        </div>
      </section>

      {hasSelectedNode ? (
        <section className="panel-card layer-editor-card">
          <div className="layer-editor-header">
            {isEditingName ? (
              <input
                value={draftName}
                autoFocus
                onChange={(e) => onDraftNameChange(e.target.value)}
                onBlur={onNameCommit}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    onNameCommit()
                  } else if (e.key === 'Escape') {
                    e.preventDefault()
                    onNameCancel()
                  }
                }}
                className="layer-name-input"
              />
            ) : (
              <button
                type="button"
                onClick={onBeginNameEdit}
                className="layer-name-button"
              >
                {selectedDisplayName}
              </button>
            )}
          </div>
          <div className="layer-editor-fields-row">
            <div className="field-group field-group-inline field-group-size">
              <p className="field-label">Size</p>
              <div className="field-size-row">
                <input
                  type="number"
                  min={1}
                  value={selectedRows}
                  onChange={(e) => onRowsChange(e.target.value)}
                  className="size-input"
                />
                <span className="size-separator">x</span>
                <input
                  type="number"
                  min={1}
                  value={selectedCols}
                  onChange={(e) => onColsChange(e.target.value)}
                  className="size-input"
                />
              </div>
            </div>
            <span
              aria-hidden
              className="layer-editor-divider"
            />

            <div className="field-group field-group-inline field-group-activation">
              <label
                htmlFor="layer-activation"
                className="field-label"
              >
                Activation
              </label>
              <select
                id="layer-activation"
                value={selectedActivation}
                onChange={(e) => onActivationChange(e.target.value)}
                className="activation-select"
              >
                {activationOptions.map((option) => (
                  <option key={option} value={option}>
                    {formatActivationLabel(option)}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </section>
      ) : null}

      <section className="panel-card build-summary-card">
        <div className="summary-grid">
          <MetricTile label="Layers" value={String(layerCount)} />
          <MetricTile label="Neurons" value={String(neuronCount)} />
          <MetricTile label="Weights" value={String(weightCount)} />
          <MetricTile label="Biases" value={String(biasCount)} />
          <MetricTile label="Layer Type" value={layerTypeSummary} />
          <MetricTile
            label="Shared Activation Function"
            value={formatActivationLabel(sharedNonOutputActivation)}
          />
        </div>
      </section>

      <div className="panel-actions">
        <button
          onClick={onValidate}
          disabled={validateDisabled}
          className="btn btn-validate"
        >
          {validateLabel}
        </button>
      </div>
    </div>
  )
}

function getRoleDotClass(role: LayerRole): string {
  if (role === 'input') {
    return 'layer-dot layer-dot-input'
  }
  if (role === 'output') {
    return 'layer-dot layer-dot-output'
  }
  return 'layer-dot layer-dot-hidden'
}

function formatActivationLabel(value: string): string {
  if (!value) return ''
  return value
    .split('_')
    .map((part) => {
      if (part.length === 0) return part
      return part.charAt(0).toUpperCase() + part.slice(1)
    })
    .join(' ')
}
