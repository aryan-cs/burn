import { useState } from 'react'
import { Viewport } from './canvas/Viewport'
import { useConnectionDraw } from './hooks/useConnectionDraw'
import { useGraphStore, type Edge, type LayerNode } from './store/graphStore'
import {
  getLayerRolesForColoring,
  getNeuralNetworkOrder,
  type LayerRole,
} from './utils/graphOrder'

const DEFAULT_LAYER_ROWS = 4
const DEFAULT_LAYER_COLS = 6
const LAYER_SPACING = 4
const ALIGN_LAYER_SPACING = 2.25
const ALIGN_Y = 0.8
const DEFAULT_ACTIVATION = 'linear'
const ACTIVATION_OPTIONS = [
  'linear',
  'relu',
  'sigmoid',
  'tanh',
  'softmax',
  'gelu',
  'leaky_relu',
]

function App() {
  useConnectionDraw()

  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const addNode = useGraphStore((s) => s.addNode)
  const updateNodeConfig = useGraphStore((s) => s.updateNodeConfig)
  const setNodesPosition = useGraphStore((s) => s.setNodesPosition)
  const setDraggingNodeId = useGraphStore((s) => s.setDraggingNodeId)
  const cancelConnectionDrag = useGraphStore((s) => s.cancelConnectionDrag)
  const selectNode = useGraphStore((s) => s.selectNode)

  const selectedNode = selectedNodeId ? nodes[selectedNodeId] : null
  const nodeRoles = getLayerRolesForColoring(nodes, edges)
  const selectedRole: LayerRole = selectedNode
    ? nodeRoles.get(selectedNode.id) ?? 'hidden'
    : 'hidden'
  const selectedRows = toPositiveInt(selectedNode?.config.rows, DEFAULT_LAYER_ROWS)
  const selectedCols = toPositiveInt(selectedNode?.config.cols, DEFAULT_LAYER_COLS)
  const sharedNonOutputActivation = getSharedNonOutputActivation(
    nodes,
    nodeRoles,
    DEFAULT_ACTIVATION
  )
  const selectedActivation =
    selectedRole === 'output'
      ? toStringOrFallback(selectedNode?.config.activation, DEFAULT_ACTIVATION)
      : sharedNonOutputActivation
  const selectedCustomName = toStringOrFallback(selectedNode?.config.name, '')
  const selectedDisplayName = selectedNode
    ? getLayerDisplayName(selectedNode.id, selectedRole, selectedCustomName)
    : ''

  const layerCount = Object.keys(nodes).length
  const orderedNodeIds = getNeuralNetworkOrder(nodes, edges)
  const expandedConnectionCount = getExpandedConnectionCount(nodes, edges)
  const connectedSetCount = getConnectedSetCount(nodes, edges)
  const neuronCount = Object.values(nodes).reduce((total, node) => {
    return total + getLayerNeuronCount(node)
  }, 0)

  const [editingNodeId, setEditingNodeId] = useState<string | null>(null)
  const [draftName, setDraftName] = useState('')
  const isEditingName = Boolean(editingNodeId && editingNodeId === selectedNodeId)

  const handleAddLayer = () => {
    const nodeCount = Object.keys(useGraphStore.getState().nodes).length
    const position: [number, number, number] = [nodeCount * LAYER_SPACING, 0.8, 0]
    const nodeId = addNode('Dense', position)

    updateNodeConfig(nodeId, {
      rows: DEFAULT_LAYER_ROWS,
      cols: DEFAULT_LAYER_COLS,
      units: DEFAULT_LAYER_ROWS * DEFAULT_LAYER_COLS,
      activation: DEFAULT_ACTIVATION,
    })
    selectNode(nodeId)
  }

  const handleAlign = () => {
    if (orderedNodeIds.length === 0) return

    cancelConnectionDrag()
    setDraggingNodeId(null)

    const nextPositions: Record<string, [number, number, number]> = {}
    const center = (orderedNodeIds.length - 1) / 2
    orderedNodeIds.forEach((nodeId, index) => {
      const z = (center - index) * ALIGN_LAYER_SPACING
      nextPositions[nodeId] = [0, ALIGN_Y, z]
    })
    setNodesPosition(nextPositions)
  }

  const handleRowsChange = (nextRowsValue: string) => {
    if (!selectedNodeId) return
    const nextRows = Number(nextRowsValue)
    if (!Number.isInteger(nextRows) || nextRows <= 0) return
    updateNodeConfig(selectedNodeId, {
      rows: nextRows,
      units: nextRows * selectedCols,
    })
  }

  const handleColsChange = (nextColsValue: string) => {
    if (!selectedNodeId) return
    const nextCols = Number(nextColsValue)
    if (!Number.isInteger(nextCols) || nextCols <= 0) return
    updateNodeConfig(selectedNodeId, {
      cols: nextCols,
      units: selectedRows * nextCols,
    })
  }

  const handleActivationChange = (nextActivation: string) => {
    if (!selectedNodeId) return

    if (selectedRole === 'output') {
      updateNodeConfig(selectedNodeId, { activation: nextActivation })
      return
    }

    Object.keys(nodes).forEach((nodeId) => {
      const role = nodeRoles.get(nodeId) ?? 'hidden'
      if (role !== 'output') {
        updateNodeConfig(nodeId, { activation: nextActivation })
      }
    })
  }

  const beginNameEdit = () => {
    if (!selectedNodeId) return
    setDraftName(selectedDisplayName)
    setEditingNodeId(selectedNodeId)
  }

  const commitNameEdit = () => {
    if (!selectedNodeId || !editingNodeId || editingNodeId !== selectedNodeId) {
      setEditingNodeId(null)
      return
    }

    const nextName = draftName.trim()
    const defaultName = getDefaultLayerName(selectedNodeId, selectedRole)
    if (nextName.length === 0 || nextName.toLowerCase() === defaultName.toLowerCase()) {
      updateNodeConfig(selectedNodeId, { name: undefined })
    } else {
      updateNodeConfig(selectedNodeId, { name: nextName })
    }
    setEditingNodeId(null)
  }

  const cancelNameEdit = () => {
    setDraftName(selectedDisplayName)
    setEditingNodeId(null)
  }

  return (
    <div className="flex h-screen w-full bg-[#0a0a0f] text-white">
      <section className="h-full w-1/3 overflow-y-auto border-r border-white/10 bg-[#101219] p-5">
        <h1 className="text-lg font-semibold">NN Dashboard</h1>
        <p className="mt-1 text-sm text-white/60">Information about the designed neural network.</p>

        <div className="mt-4 flex items-center gap-2">
          <button
            onClick={handleAddLayer}
            className="flex-1 rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-500"
          >
            Add Layer
          </button>
          <button
            onClick={handleAlign}
            className="rounded-md border border-white/20 bg-[#12131c]/90 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-[#1a1c28]"
          >
            Align
          </button>
        </div>

        <section className="mt-5 rounded-lg border border-white/10 bg-white/5 p-4">
          <h2 className="text-sm font-semibold text-white/95">Model Summary</h2>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <MetricTile label="Layers" value={String(layerCount)} />
            <MetricTile label="Neurons" value={String(neuronCount)} />
            <MetricTile label="Edge Links" value={String(Object.keys(edges).length)} />
            <MetricTile label="Dense Links" value={String(expandedConnectionCount)} />
            <MetricTile label="Connected Sets" value={String(connectedSetCount)} />
            <MetricTile label="Shared Act" value={sharedNonOutputActivation} />
          </div>
        </section>

        <section className="mt-5 rounded-lg border border-white/10 bg-white/5 p-4">
          <h2 className="text-sm font-semibold text-white/95">Layer Flow</h2>
          {orderedNodeIds.length === 0 ? (
            <p className="mt-2 text-sm text-white/60">No layers added yet.</p>
          ) : (
            <ol className="mt-3 space-y-2">
              {orderedNodeIds.map((nodeId, index) => {
                const node = nodes[nodeId]
                if (!node) return null
                const role = nodeRoles.get(nodeId) ?? 'hidden'
                const name = getLayerDisplayName(
                  nodeId,
                  role,
                  toStringOrFallback(node.config.name, '')
                )
                return (
                  <li
                    key={nodeId}
                    className="flex items-center justify-between rounded-md border border-white/10 bg-white/[0.03] px-2 py-1.5 text-sm"
                  >
                    <span>
                      {index + 1}. {name}
                    </span>
                    <span className={getRoleBadgeClass(role)}>{getRoleLabel(role)}</span>
                  </li>
                )
              })}
            </ol>
          )}
        </section>

        {selectedNode ? (
          <section className="mt-5 rounded-lg border border-white/10 bg-white/5 p-4">
            {isEditingName ? (
              <input
                value={draftName}
                autoFocus
                onChange={(e) => setDraftName(e.target.value)}
                onBlur={commitNameEdit}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    commitNameEdit()
                  } else if (e.key === 'Escape') {
                    e.preventDefault()
                    cancelNameEdit()
                  }
                }}
                className="w-full rounded-md border border-white/15 bg-white/5 px-2 py-1.5 text-sm font-semibold text-white outline-none"
              />
            ) : (
              <button
                type="button"
                onClick={beginNameEdit}
                className="text-left text-sm font-semibold text-white hover:text-blue-300"
              >
                {selectedDisplayName}
              </button>
            )}

            <div className="mt-4">
              <p className="text-xs uppercase tracking-wide text-white/55">Size</p>
              <div className="mt-2 flex items-center gap-2">
                <input
                  type="number"
                  min={1}
                  value={selectedRows}
                  onChange={(e) => handleRowsChange(e.target.value)}
                  className="w-20 rounded-md border border-white/15 bg-white/5 px-2 py-1.5 text-sm text-white outline-none"
                />
                <span className="text-sm text-white/70">x</span>
                <input
                  type="number"
                  min={1}
                  value={selectedCols}
                  onChange={(e) => handleColsChange(e.target.value)}
                  className="w-20 rounded-md border border-white/15 bg-white/5 px-2 py-1.5 text-sm text-white outline-none"
                />
              </div>
            </div>

            <div className="mt-4">
              <label
                htmlFor="layer-activation"
                className="text-xs uppercase tracking-wide text-white/55"
              >
                Activation
              </label>
              <select
                id="layer-activation"
                value={selectedActivation}
                onChange={(e) => handleActivationChange(e.target.value)}
                className="mt-2 w-full rounded-md border border-white/15 bg-white/5 px-2 py-1.5 text-sm text-white outline-none"
              >
                {ACTIVATION_OPTIONS.map((option) => (
                  <option key={option} value={option} className="bg-[#12131c] text-white">
                    {option}
                  </option>
                ))}
              </select>
              {selectedRole !== 'output' ? (
                <p className="mt-2 text-xs text-white/60">
                  Applies to all non-output layers.
                </p>
              ) : null}
            </div>
          </section>
        ) : (
          <section className="mt-5 rounded-lg border border-white/10 bg-white/5 p-4 text-sm text-white/65">
            Select a layer in the 3D view to edit its properties.
          </section>
        )}
      </section>

      <section className="h-full w-2/3">
        <Viewport />
      </section>
    </div>
  )
}

export default App

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-white/10 bg-black/20 px-2 py-2">
      <p className="text-[11px] uppercase tracking-wide text-white/50">{label}</p>
      <p className="mt-1 text-sm font-medium text-white">{value}</p>
    </div>
  )
}

function toPositiveInt(value: unknown, fallback: number): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback
  }
  return Math.floor(parsed)
}

function toStringOrFallback(value: unknown, fallback: string): string {
  if (typeof value !== 'string') return fallback
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : fallback
}

function getLayerNeuronCount(node: LayerNode): number {
  const rows = toPositiveInt(node.config.rows, DEFAULT_LAYER_ROWS)
  const cols = toPositiveInt(node.config.cols, DEFAULT_LAYER_COLS)
  return rows * cols
}

function getSharedNonOutputActivation(
  nodes: Record<string, LayerNode>,
  nodeRoles: Map<string, LayerRole>,
  fallback: string
): string {
  for (const [nodeId, node] of Object.entries(nodes)) {
    const role = nodeRoles.get(nodeId) ?? 'hidden'
    if (role === 'output') continue
    const activation = toStringOrFallback(node.config.activation, '')
    if (activation.length > 0) return activation
  }
  return fallback
}

function getLayerDisplayName(
  nodeId: string,
  role: LayerRole,
  customName: string
): string {
  if (customName.length > 0) {
    return customName
  }
  return getDefaultLayerName(nodeId, role)
}

function getDefaultLayerName(nodeId: string, role: LayerRole): string {
  if (role === 'input') return 'input layer'
  if (role === 'output') return 'output layer'

  const match = nodeId.match(/_(\d+)$/)
  if (!match) return 'Layer'
  return `Layer ${match[1]}`
}

function getRoleLabel(role: LayerRole): string {
  if (role === 'input') return 'Input'
  if (role === 'output') return 'Output'
  return 'Hidden'
}

function getRoleBadgeClass(role: LayerRole): string {
  if (role === 'input') {
    return 'rounded px-1.5 py-0.5 text-xs font-medium text-black bg-[#ff8c2b]'
  }
  if (role === 'output') {
    return 'rounded px-1.5 py-0.5 text-xs font-medium text-white bg-[#4da3ff]'
  }
  return 'rounded px-1.5 py-0.5 text-xs font-medium text-black bg-white'
}

function getExpandedConnectionCount(
  nodes: Record<string, LayerNode>,
  edges: Record<string, Edge>
): number {
  return Object.values(edges).reduce((total, edge) => {
    const source = nodes[edge.source]
    const target = nodes[edge.target]
    if (!source || !target) return total
    return total + getLayerNeuronCount(source) * getLayerNeuronCount(target)
  }, 0)
}

function getConnectedSetCount(
  nodes: Record<string, LayerNode>,
  edges: Record<string, Edge>
): number {
  const nodeIds = Object.keys(nodes)
  const adjacency = new Map<string, Set<string>>()
  nodeIds.forEach((nodeId) => adjacency.set(nodeId, new Set<string>()))

  Object.values(edges).forEach((edge) => {
    if (!nodes[edge.source] || !nodes[edge.target] || edge.source === edge.target) return
    adjacency.get(edge.source)?.add(edge.target)
    adjacency.get(edge.target)?.add(edge.source)
  })

  const visited = new Set<string>()
  let components = 0

  nodeIds.forEach((nodeId) => {
    if (visited.has(nodeId)) return
    if ((adjacency.get(nodeId)?.size ?? 0) === 0) return

    components += 1
    const queue = [nodeId]
    visited.add(nodeId)

    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const currentNodeId = queue[cursor]
      const neighbors = adjacency.get(currentNodeId) ?? new Set<string>()
      neighbors.forEach((neighborId) => {
        if (visited.has(neighborId)) return
        visited.add(neighborId)
        queue.push(neighborId)
      })
    }
  })

  return components
}
