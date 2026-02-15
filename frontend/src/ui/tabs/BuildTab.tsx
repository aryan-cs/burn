import type { LayerRole } from '../../utils/graphOrder'
import { InfoTooltip } from '../InfoTooltip'
import { FIELD_TOOLTIPS } from '../tooltipData'
import { MetricTile } from './Metrics'

export interface BuildLayerItem {
  id: string
  name: string
  role: LayerRole
  sizeLabel: string
}

interface BuildTabProps {
  layerItems: BuildLayerItem[]
  hasSelectedNode: boolean
  selectedNodeId: string | null
  selectedNodeType: string | null
  isEditingName: boolean
  draftName: string
  selectedDisplayName: string
  selectedRows: number
  selectedCols: number
  selectedChannels: number
  selectedUnits: number
  selectedDropoutRate: number
  selectedOutputClasses: number
  selectedActivation: string
  selectedShapeLabel: string
  canEditSize: boolean
  sizeFieldLabel: string
  canEditActivation: boolean
  canEditChannels: boolean
  canEditUnits: boolean
  canEditDropoutRate: boolean
  canEditOutputClasses: boolean
  activationOptions: string[]
  layerCount: number
  neuronCount: number
  weightCount: number
  biasCount: number
  layerTypeSummary: string
  sharedNonOutputActivation: string
  onAddLayer: () => void
  onSelectLayer: (nodeId: string) => void
  onBeginNameEdit: () => void
  onDraftNameChange: (value: string) => void
  onNameCommit: () => void
  onNameCancel: () => void
  onRowsChange: (value: string) => void
  onColsChange: (value: string) => void
  onChannelsChange: (value: string) => void
  onUnitsChange: (value: string) => void
  onDropoutRateChange: (value: string) => void
  onOutputClassesChange: (value: string) => void
  onActivationChange: (value: string) => void
  onValidate: () => void
  buildStatus: 'idle' | 'success' | 'error'
  buildStatusMessage: string
  buildIssues: string[]
  buildWarnings: string[]
  validateDisabled: boolean
  validateLabel: string
}

export function BuildTab({
  layerItems,
  hasSelectedNode,
  selectedNodeId,
  selectedNodeType,
  isEditingName,
  draftName,
  selectedDisplayName,
  selectedRows,
  selectedCols,
  selectedChannels,
  selectedUnits,
  selectedDropoutRate,
  selectedOutputClasses,
  selectedActivation,
  selectedShapeLabel,
  canEditSize,
  sizeFieldLabel,
  canEditActivation,
  canEditChannels,
  canEditUnits,
  canEditDropoutRate,
  canEditOutputClasses,
  activationOptions,
  layerCount,
  neuronCount,
  weightCount,
  biasCount,
  layerTypeSummary,
  sharedNonOutputActivation,
  onAddLayer,
  onSelectLayer,
  onBeginNameEdit,
  onDraftNameChange,
  onNameCommit,
  onNameCancel,
  onRowsChange,
  onColsChange,
  onChannelsChange,
  onUnitsChange,
  onDropoutRateChange,
  onOutputClassesChange,
  onActivationChange,
  onValidate,
  buildStatus,
  buildStatusMessage,
  buildIssues,
  buildWarnings,
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
              layerItems.map((layerItem) => {
                const isSelected = selectedNodeId === layerItem.id
                return (
                  <li
                    key={layerItem.id}
                    className={`layer-list-item ${isSelected ? 'layer-list-item-active' : ''}`}
                    role="button"
                    tabIndex={0}
                    onClick={() => onSelectLayer(layerItem.id)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault()
                        onSelectLayer(layerItem.id)
                      }
                    }}
                  >
                    <div className="layer-list-item-left">
                      <span className={getRoleDotClass(layerItem.role)} />
                      <span className="layer-list-item-name">{layerItem.name}</span>
                    </div>
                    <span className="layer-list-item-size">{layerItem.sizeLabel}</span>
                  </li>
                )
              })
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
              <button type="button" onClick={onBeginNameEdit} className="layer-name-button">
                {selectedDisplayName}
              </button>
            )}
          </div>

          <p className="panel-muted-text panel-muted-text-tight">Type: {selectedNodeType ?? '—'}</p>

          <div className="layer-editor-fields-row">
            {canEditSize ? (
              <div className="field-group field-group-inline field-group-size">
                <p className="field-label">
                  {sizeFieldLabel}
                  <InfoTooltip title="Size" text={FIELD_TOOLTIPS.Size} position="top" />
                </p>

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
            ) : (
              <div className="field-group field-group-inline field-group-size">
                <p className="field-label">Shape</p>
                <div className="field-readonly">{selectedShapeLabel}</div>
              </div>
            )}

            <span aria-hidden className="layer-editor-divider" />

            <div className="field-group field-group-inline field-group-activation">
              {canEditActivation ? (
                <>
                  <label htmlFor="layer-activation" className="field-label">
                    Activation
                    <InfoTooltip
                      title="Activation"
                      text={FIELD_TOOLTIPS.Activation}
                      position="top"
                    />
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
                </>
              ) : (
                <>
                  <p className="field-label">Activation</p>
                  <div className="field-readonly">Not applicable</div>
                </>
              )}
            </div>
          </div>

          {canEditChannels || canEditUnits || canEditDropoutRate || canEditOutputClasses ? (
            <div className="layer-editor-extra-grid">
              {canEditChannels ? (
                <label className="field-group field-group-inline">
                  <span className="field-label">Channels</span>
                  <input
                    type="number"
                    min={1}
                    value={selectedChannels}
                    onChange={(event) => onChannelsChange(event.target.value)}
                    className="size-input"
                  />
                </label>
              ) : null}

              {canEditUnits ? (
                <label className="field-group field-group-inline">
                  <span className="field-label">Units</span>
                  <input
                    type="number"
                    min={1}
                    value={selectedUnits}
                    onChange={(event) => onUnitsChange(event.target.value)}
                    className="size-input"
                  />
                </label>
              ) : null}

              {canEditDropoutRate ? (
                <label className="field-group field-group-inline">
                  <span className="field-label">Dropout Rate</span>
                  <input
                    type="number"
                    min={0}
                    max={0.99}
                    step={0.01}
                    value={selectedDropoutRate}
                    onChange={(event) => onDropoutRateChange(event.target.value)}
                    className="size-input"
                  />
                </label>
              ) : null}

              {canEditOutputClasses ? (
                <label className="field-group field-group-inline">
                  <span className="field-label">Output Classes</span>
                  <input
                    type="number"
                    min={1}
                    value={selectedOutputClasses}
                    onChange={(event) => onOutputClassesChange(event.target.value)}
                    className="size-input"
                  />
                </label>
              ) : null}
            </div>
          ) : null}
        </section>
      ) : null}

      <section className="panel-card build-summary-card">
        <div className="summary-grid">
          <MetricTile label="Layers" value={String(layerCount)} tooltip={FIELD_TOOLTIPS.Layers} />
          <MetricTile label="Neurons" value={String(neuronCount)} tooltip={FIELD_TOOLTIPS.Neurons} />
          <MetricTile label="Weights" value={String(weightCount)} tooltip={FIELD_TOOLTIPS.Weights} />
          <MetricTile label="Biases" value={String(biasCount)} tooltip={FIELD_TOOLTIPS.Biases} />
          <MetricTile label="Layer Type" value={layerTypeSummary} tooltip={FIELD_TOOLTIPS['Layer Type']} />
          <MetricTile
            label="Shared Activation Function"
            value={formatActivationLabel(sharedNonOutputActivation)}
            tooltip={FIELD_TOOLTIPS['Shared Activation Function']}
          />
        </div>
      </section>

      <section className="panel-card build-feedback-card">
        <div className={`build-feedback-status build-feedback-status-${buildStatus}`}>
          {buildStatusMessage}
        </div>

        {buildWarnings.length > 0 ? (
          <ul className="build-feedback-list">
            {buildWarnings.map((warning, index) => (
              <li key={`warning-${index}`} className="build-feedback-item build-feedback-item-warning">
                {warning}
              </li>
            ))}
          </ul>
        ) : null}

        {buildIssues.length > 0 ? (
          <ul className="build-feedback-list">
            {buildIssues.map((issue, index) => (
              <li key={`issue-${index}`} className="build-feedback-item build-feedback-item-error">
                {issue}
              </li>
            ))}
          </ul>
        ) : null}
      </section>

      <div className="panel-actions">
        <button onClick={onValidate} disabled={validateDisabled} className="btn btn-validate">
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

