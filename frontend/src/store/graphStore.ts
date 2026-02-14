import { create } from 'zustand'

// ── Types ──────────────────────────────────────────────

export type LayerType =
  | 'Input'
  | 'Dense'
  | 'Conv2D'
  | 'MaxPool2D'
  | 'LSTM'
  | 'GRU'
  | 'Dropout'
  | 'BatchNorm'
  | 'Flatten'
  | 'Reshape'
  | 'Output'

export interface LayerConfig {
  // Grid-based smart layer shape
  rows?: number
  cols?: number
  name?: string
  // Dense
  units?: number
  activation?: string
  // Conv2D
  filters?: number
  kernel_size?: number
  padding?: number
  in_channels?: number
  // Dropout
  rate?: number
  // Input
  shape?: number[]
  // Output
  num_classes?: number
  // Generic
  [key: string]: unknown
}

export interface LayerNode {
  id: string
  type: LayerType
  position: [number, number, number]
  rotation: [number, number, number]
  config: LayerConfig
  weights?: Float32Array
  shape: {
    input: number[] | null
    output: number[] | null
  }
}

export interface Edge {
  id: string
  source: string
  target: string
  weightStats?: {
    mean: number
    std: number
    min: number
    max: number
  }
}

export interface GraphJSON {
  nodes: Array<{
    id: string
    type: LayerType
    config: LayerConfig
  }>
  edges: Array<{
    id: string
    source: string
    target: string
  }>
}

interface GraphSnapshot {
  nodes: Record<string, LayerNode>
  edges: Record<string, Edge>
  nextNodeId: number
  nextEdgeId: number
}

// ── Default configs per layer type ─────────────────────

const DEFAULT_CONFIGS: Record<LayerType, LayerConfig> = {
  Input: { shape: [1, 28, 28] },
  Dense: { rows: 4, cols: 6, units: 24, activation: 'linear' },
  Conv2D: { filters: 32, kernel_size: 3, activation: 'relu', padding: 1 },
  MaxPool2D: { kernel_size: 2 },
  LSTM: { units: 64 },
  GRU: { units: 64 },
  Dropout: { rate: 0.5 },
  BatchNorm: {},
  Flatten: {},
  Reshape: { shape: [-1] },
  Output: { num_classes: 10, activation: 'softmax' },
}

// ── Store ──────────────────────────────────────────────

interface GraphState {
  nodes: Record<string, LayerNode>
  edges: Record<string, Edge>
  selectedNodeId: string | null
  selectedEdgeId: string | null
  draggingNodeId: string | null
  highlightSelectionActive: boolean
  highlightSelectionStart: [number, number] | null
  highlightSelectionEnd: [number, number] | null
  highlightedNodeIds: string[]
  connectionSource: string | null
  connectionStart: [number, number, number] | null
  connectionCursor: [number, number, number] | null

  addNode: (type: LayerType, position: [number, number, number]) => string
  removeNode: (id: string) => void
  removeNodes: (ids: string[]) => void
  updateNode: (id: string, patch: Partial<LayerNode>) => void
  updateNodeConfig: (id: string, config: Partial<LayerConfig>) => void
  setNodePosition: (
    id: string,
    position: [number, number, number],
    recordHistory?: boolean
  ) => void
  setNodesPosition: (
    positions: Record<string, [number, number, number]>,
    recordHistory?: boolean
  ) => void

  addEdge: (sourceId: string, targetId: string) => string
  removeEdge: (id: string) => void
  updateEdgeWeights: (id: string, stats: Edge['weightStats']) => void

  selectNode: (id: string | null) => void
  selectEdge: (id: string | null) => void
  setDraggingNodeId: (id: string | null) => void
  startHighlightSelection: (start: [number, number]) => void
  updateHighlightSelection: (end: [number, number]) => void
  endHighlightSelection: () => void
  setHighlightedNodes: (ids: string[]) => void
  clearHighlightedNodes: () => void
  setConnectionSource: (id: string | null) => void
  startConnectionDrag: (
    sourceId: string,
    startPoint: [number, number, number]
  ) => void
  updateConnectionCursor: (point: [number, number, number]) => void
  completeConnectionDrag: (targetId: string | null) => void
  cancelConnectionDrag: () => void
  undo: () => boolean

  toJSON: () => GraphJSON
  fromJSON: (json: GraphJSON) => void
  clear: () => void
}

let nextNodeId = 1
let nextEdgeId = 1
const MAX_UNDO_HISTORY = 100
const undoPast: GraphSnapshot[] = []
const undoFuture: GraphSnapshot[] = []

function cloneNodes(
  nodes: Record<string, LayerNode>
): Record<string, LayerNode> {
  const cloned: Record<string, LayerNode> = {}
  for (const [nodeId, node] of Object.entries(nodes)) {
    cloned[nodeId] = {
      ...node,
      position: [...node.position] as [number, number, number],
      rotation: [...node.rotation] as [number, number, number],
      config: { ...node.config },
      shape: {
        input: node.shape.input ? [...node.shape.input] : null,
        output: node.shape.output ? [...node.shape.output] : null,
      },
    }
  }
  return cloned
}

function cloneEdges(edges: Record<string, Edge>): Record<string, Edge> {
  const cloned: Record<string, Edge> = {}
  for (const [edgeId, edge] of Object.entries(edges)) {
    cloned[edgeId] = {
      ...edge,
      weightStats: edge.weightStats ? { ...edge.weightStats } : undefined,
    }
  }
  return cloned
}

function takeSnapshot(state: GraphState): GraphSnapshot {
  return {
    nodes: cloneNodes(state.nodes),
    edges: cloneEdges(state.edges),
    nextNodeId,
    nextEdgeId,
  }
}

function pushUndoSnapshot(state: GraphState) {
  undoPast.push(takeSnapshot(state))
  if (undoPast.length > MAX_UNDO_HISTORY) {
    undoPast.shift()
  }
  undoFuture.length = 0
}

function hasNodePatchChanges(
  node: LayerNode,
  patch: Partial<LayerNode>
): boolean {
  const nodeRecord = node as unknown as Record<string, unknown>

  for (const [key, value] of Object.entries(patch)) {
    if (key === 'position' || key === 'rotation') {
      if (!Array.isArray(value)) return true
      const current = nodeRecord[key]
      if (!Array.isArray(current)) return true
      if (current.length !== value.length) return true
      if (current.some((entry, index) => entry !== value[index])) return true
      continue
    }

    if (nodeRecord[key] !== value) return true
  }
  return false
}

function hasConfigPatchChanges(
  config: LayerConfig,
  patch: Partial<LayerConfig>
): boolean {
  for (const [key, value] of Object.entries(patch)) {
    if (config[key] !== value) return true
  }
  return false
}

function isSamePosition(
  current: [number, number, number],
  next: [number, number, number]
): boolean {
  return (
    current[0] === next[0] &&
    current[1] === next[1] &&
    current[2] === next[2]
  )
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: {},
  edges: {},
  selectedNodeId: null,
  selectedEdgeId: null,
  draggingNodeId: null,
  highlightSelectionActive: false,
  highlightSelectionStart: null,
  highlightSelectionEnd: null,
  highlightedNodeIds: [],
  connectionSource: null,
  connectionStart: null,
  connectionCursor: null,

  addNode: (type, position) => {
    pushUndoSnapshot(get())
    const id = `node_${nextNodeId++}`
    const node: LayerNode = {
      id,
      type,
      position,
      rotation: [0, 0, 0],
      config: { ...DEFAULT_CONFIGS[type] },
      shape: { input: null, output: null },
    }
    set((s) => ({ nodes: { ...s.nodes, [id]: node } }))
    return id
  },

  removeNode: (id) => {
    if (!get().nodes[id]) return
    pushUndoSnapshot(get())
    set((s) => {
      const remainingNodes = { ...s.nodes }
      delete remainingNodes[id]
      // Also remove connected edges
      const remainingEdges: Record<string, Edge> = {}
      let selectedEdgeStillExists = false
      for (const [eid, edge] of Object.entries(s.edges)) {
        if (edge.source !== id && edge.target !== id) {
          remainingEdges[eid] = edge
          if (eid === s.selectedEdgeId) {
            selectedEdgeStillExists = true
          }
        }
      }
      return {
        nodes: remainingNodes,
        edges: remainingEdges,
        selectedNodeId: s.selectedNodeId === id ? null : s.selectedNodeId,
        selectedEdgeId: selectedEdgeStillExists ? s.selectedEdgeId : null,
        draggingNodeId: s.draggingNodeId === id ? null : s.draggingNodeId,
        highlightedNodeIds: s.highlightedNodeIds.filter((nodeId) => nodeId !== id),
        connectionSource: s.connectionSource === id ? null : s.connectionSource,
        connectionStart: s.connectionSource === id ? null : s.connectionStart,
        connectionCursor: s.connectionSource === id ? null : s.connectionCursor,
      }
    })
  },

  removeNodes: (ids) => {
    if (ids.length === 0) return
    const idsToRemove = new Set(ids)
    const hasAtLeastOneExisting = ids.some((id) => Boolean(get().nodes[id]))
    if (!hasAtLeastOneExisting) return
    pushUndoSnapshot(get())
    set((s) => {
      const remainingNodes: Record<string, LayerNode> = {}
      for (const [nodeId, node] of Object.entries(s.nodes)) {
        if (!idsToRemove.has(nodeId)) {
          remainingNodes[nodeId] = node
        }
      }

      const remainingEdges: Record<string, Edge> = {}
      let selectedEdgeStillExists = false
      for (const [edgeId, edge] of Object.entries(s.edges)) {
        if (!idsToRemove.has(edge.source) && !idsToRemove.has(edge.target)) {
          remainingEdges[edgeId] = edge
          if (edgeId === s.selectedEdgeId) {
            selectedEdgeStillExists = true
          }
        }
      }

      return {
        nodes: remainingNodes,
        edges: remainingEdges,
        selectedNodeId:
          s.selectedNodeId && idsToRemove.has(s.selectedNodeId)
            ? null
            : s.selectedNodeId,
        selectedEdgeId: selectedEdgeStillExists ? s.selectedEdgeId : null,
        draggingNodeId:
          s.draggingNodeId && idsToRemove.has(s.draggingNodeId)
            ? null
            : s.draggingNodeId,
        highlightedNodeIds: s.highlightedNodeIds.filter(
          (nodeId) => !idsToRemove.has(nodeId)
        ),
        connectionSource:
          s.connectionSource && idsToRemove.has(s.connectionSource)
            ? null
            : s.connectionSource,
        connectionStart:
          s.connectionSource && idsToRemove.has(s.connectionSource)
            ? null
            : s.connectionStart,
        connectionCursor:
          s.connectionSource && idsToRemove.has(s.connectionSource)
            ? null
            : s.connectionCursor,
      }
    })
  },

  updateNode: (id, patch) => {
    const current = get().nodes[id]
    if (!current) return
    if (!hasNodePatchChanges(current, patch)) return
    pushUndoSnapshot(get())
    set((s) => ({
      nodes: {
        ...s.nodes,
        [id]: { ...s.nodes[id], ...patch },
      },
    }))
  },

  updateNodeConfig: (id, config) => {
    const currentNode = get().nodes[id]
    if (!currentNode) return
    if (!hasConfigPatchChanges(currentNode.config, config)) return
    pushUndoSnapshot(get())
    set((s) => ({
      nodes: {
        ...s.nodes,
        [id]: {
          ...s.nodes[id],
          config: { ...s.nodes[id].config, ...config },
        },
      },
    }))
  },

  setNodePosition: (id, position, recordHistory = true) => {
    const currentNode = get().nodes[id]
    if (!currentNode) return
    if (isSamePosition(currentNode.position, position)) return
    if (recordHistory) {
      pushUndoSnapshot(get())
    }
    set((s) => ({
      nodes: {
        ...s.nodes,
        [id]: { ...s.nodes[id], position },
      },
    }))
  },

  setNodesPosition: (positions, recordHistory = true) => {
    const state = get()
    const changedPositions: Record<string, [number, number, number]> = {}
    for (const [nodeId, position] of Object.entries(positions)) {
      const existingNode = state.nodes[nodeId]
      if (!existingNode) continue
      if (isSamePosition(existingNode.position, position)) continue
      changedPositions[nodeId] = position
    }
    if (Object.keys(changedPositions).length === 0) return
    if (recordHistory) {
      pushUndoSnapshot(state)
    }

    set((s) => {
      const nextNodes = { ...s.nodes }

      for (const [nodeId, position] of Object.entries(changedPositions)) {
        const existingNode = nextNodes[nodeId]
        if (!existingNode) continue
        nextNodes[nodeId] = { ...existingNode, position }
      }

      return { nodes: nextNodes }
    })
  },

  addEdge: (sourceId, targetId) => {
    // Prevent duplicate edges
    const existing = Object.values(get().edges).find(
      (e) => e.source === sourceId && e.target === targetId
    )
    if (existing) return existing.id

    pushUndoSnapshot(get())
    const id = `edge_${nextEdgeId++}`
    const edge: Edge = { id, source: sourceId, target: targetId }
    set((s) => ({ edges: { ...s.edges, [id]: edge } }))
    return id
  },

  removeEdge: (id) => {
    if (!get().edges[id]) return
    pushUndoSnapshot(get())
    set((s) => {
      const remaining = { ...s.edges }
      delete remaining[id]
      return {
        edges: remaining,
        selectedEdgeId: s.selectedEdgeId === id ? null : s.selectedEdgeId,
      }
    })
  },

  updateEdgeWeights: (id, stats) => {
    set((s) => ({
      edges: {
        ...s.edges,
        [id]: { ...s.edges[id], weightStats: stats },
      },
    }))
  },

  selectNode: (id) => set({ selectedNodeId: id, selectedEdgeId: null }),
  selectEdge: (id) => set({ selectedEdgeId: id, selectedNodeId: null }),
  setDraggingNodeId: (id) => set({ draggingNodeId: id }),
  startHighlightSelection: (start) =>
    set({
      highlightSelectionActive: true,
      highlightSelectionStart: start,
      highlightSelectionEnd: start,
      highlightedNodeIds: [],
    }),
  updateHighlightSelection: (end) => {
    if (!get().highlightSelectionActive) return
    set({ highlightSelectionEnd: end })
  },
  endHighlightSelection: () =>
    set({
      highlightSelectionActive: false,
      highlightSelectionStart: null,
      highlightSelectionEnd: null,
    }),
  setHighlightedNodes: (ids) => set({ highlightedNodeIds: ids }),
  clearHighlightedNodes: () => set({ highlightedNodeIds: [] }),
  setConnectionSource: (id) =>
    set({
      connectionSource: id,
      connectionStart: null,
      connectionCursor: null,
    }),

  startConnectionDrag: (sourceId, startPoint) =>
    set({
      connectionSource: sourceId,
      connectionStart: startPoint,
      connectionCursor: startPoint,
    }),

  updateConnectionCursor: (point) => {
    if (!get().connectionSource) return
    set({ connectionCursor: point })
  },

  completeConnectionDrag: (targetId) => {
    const sourceId = get().connectionSource
    if (sourceId && targetId && sourceId !== targetId) {
      get().addEdge(sourceId, targetId)
    }
    set({
      connectionSource: null,
      connectionStart: null,
      connectionCursor: null,
    })
  },

  cancelConnectionDrag: () =>
    set({
      connectionSource: null,
      connectionStart: null,
      connectionCursor: null,
    }),

  toJSON: () => {
    const { nodes, edges } = get()
    return {
      nodes: Object.values(nodes).map((n) => ({
        id: n.id,
        type: n.type,
        config: n.config,
      })),
      edges: Object.values(edges).map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
      })),
    }
  },

  fromJSON: (json) => {
    const nodes: Record<string, LayerNode> = {}
    let maxNodeNum = 0
    json.nodes.forEach((n, i) => {
      nodes[n.id] = {
        id: n.id,
        type: n.type,
        position: [i * 3, 0, 0],
        rotation: [0, 0, 0],
        config: n.config,
        shape: { input: null, output: null },
      }

      const match = n.id.match(/_(\d+)$/)
      if (match) {
        const numeric = Number(match[1])
        if (Number.isFinite(numeric)) {
          maxNodeNum = Math.max(maxNodeNum, numeric)
        }
      }
    })

    const edges: Record<string, Edge> = {}
    let maxEdgeNum = 0
    json.edges.forEach((e) => {
      edges[e.id] = { id: e.id, source: e.source, target: e.target }

      const match = e.id.match(/_(\d+)$/)
      if (match) {
        const numeric = Number(match[1])
        if (Number.isFinite(numeric)) {
          maxEdgeNum = Math.max(maxEdgeNum, numeric)
        }
      }
    })

    nextNodeId = maxNodeNum > 0 ? maxNodeNum + 1 : nextNodeId
    nextEdgeId = maxEdgeNum > 0 ? maxEdgeNum + 1 : nextEdgeId
    undoPast.length = 0
    undoFuture.length = 0
    set({
      nodes,
      edges,
      selectedNodeId: null,
      selectedEdgeId: null,
      draggingNodeId: null,
      highlightSelectionActive: false,
      highlightSelectionStart: null,
      highlightSelectionEnd: null,
      highlightedNodeIds: [],
      connectionSource: null,
      connectionStart: null,
      connectionCursor: null,
    })
  },

  clear: () =>
    set((s) => {
      if (
        Object.keys(s.nodes).length > 0 ||
        Object.keys(s.edges).length > 0
      ) {
        pushUndoSnapshot(s)
      }
      undoFuture.length = 0
      return {
        nodes: {},
        edges: {},
        selectedNodeId: null,
        selectedEdgeId: null,
        draggingNodeId: null,
        highlightSelectionActive: false,
        highlightSelectionStart: null,
        highlightSelectionEnd: null,
        highlightedNodeIds: [],
        connectionSource: null,
        connectionStart: null,
        connectionCursor: null,
      }
    }),

  undo: () => {
    const snapshot = undoPast.pop()
    if (!snapshot) return false

    const current = get()
    undoFuture.push(takeSnapshot(current))
    nextNodeId = snapshot.nextNodeId
    nextEdgeId = snapshot.nextEdgeId

    set({
      nodes: cloneNodes(snapshot.nodes),
      edges: cloneEdges(snapshot.edges),
      selectedNodeId: null,
      selectedEdgeId: null,
      draggingNodeId: null,
      highlightSelectionActive: false,
      highlightSelectionStart: null,
      highlightSelectionEnd: null,
      highlightedNodeIds: [],
      connectionSource: null,
      connectionStart: null,
      connectionCursor: null,
    })
    return true
  },
}))
