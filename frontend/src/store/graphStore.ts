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

// ── Default configs per layer type ─────────────────────

const DEFAULT_CONFIGS: Record<LayerType, LayerConfig> = {
  Input: { shape: [1, 28, 28] },
  Dense: { units: 128, activation: 'relu' },
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
  connectionSource: string | null

  addNode: (type: LayerType, position: [number, number, number]) => string
  removeNode: (id: string) => void
  updateNode: (id: string, patch: Partial<LayerNode>) => void
  updateNodeConfig: (id: string, config: Partial<LayerConfig>) => void
  setNodePosition: (id: string, position: [number, number, number]) => void

  addEdge: (sourceId: string, targetId: string) => string
  removeEdge: (id: string) => void
  updateEdgeWeights: (id: string, stats: Edge['weightStats']) => void

  selectNode: (id: string | null) => void
  selectEdge: (id: string | null) => void
  setConnectionSource: (id: string | null) => void

  toJSON: () => GraphJSON
  fromJSON: (json: GraphJSON) => void
  clear: () => void
}

let nextNodeId = 1
let nextEdgeId = 1

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: {},
  edges: {},
  selectedNodeId: null,
  selectedEdgeId: null,
  connectionSource: null,

  addNode: (type, position) => {
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
    set((s) => {
      const { [id]: _, ...remainingNodes } = s.nodes
      // Also remove connected edges
      const remainingEdges: Record<string, Edge> = {}
      for (const [eid, edge] of Object.entries(s.edges)) {
        if (edge.source !== id && edge.target !== id) {
          remainingEdges[eid] = edge
        }
      }
      return {
        nodes: remainingNodes,
        edges: remainingEdges,
        selectedNodeId: s.selectedNodeId === id ? null : s.selectedNodeId,
      }
    })
  },

  updateNode: (id, patch) => {
    set((s) => ({
      nodes: {
        ...s.nodes,
        [id]: { ...s.nodes[id], ...patch },
      },
    }))
  },

  updateNodeConfig: (id, config) => {
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

  setNodePosition: (id, position) => {
    set((s) => ({
      nodes: {
        ...s.nodes,
        [id]: { ...s.nodes[id], position },
      },
    }))
  },

  addEdge: (sourceId, targetId) => {
    // Prevent duplicate edges
    const existing = Object.values(get().edges).find(
      (e) => e.source === sourceId && e.target === targetId
    )
    if (existing) return existing.id

    const id = `edge_${nextEdgeId++}`
    const edge: Edge = { id, source: sourceId, target: targetId }
    set((s) => ({ edges: { ...s.edges, [id]: edge } }))
    return id
  },

  removeEdge: (id) => {
    set((s) => {
      const { [id]: _, ...remaining } = s.edges
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
  setConnectionSource: (id) => set({ connectionSource: id }),

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
    set({ nodes, edges, selectedNodeId: null, selectedEdgeId: null })
  },

  clear: () =>
    set({
      nodes: {},
      edges: {},
      selectedNodeId: null,
      selectedEdgeId: null,
      connectionSource: null,
    }),
}))
