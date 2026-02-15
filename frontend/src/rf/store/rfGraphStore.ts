import { create } from 'zustand'
import type { RFEdge, RFGraphPayload, RFNode, RFNodeType, RFTrainingConfig } from '../types'

interface DatasetDefaults {
  shape: number[]
  numClasses: number
}

const DATASET_DEFAULTS: Record<string, DatasetDefaults> = {
  iris: { shape: [4], numClasses: 3 },
  wine: { shape: [11], numClasses: 6 },
  breast_cancer: { shape: [30], numClasses: 2 },
}

const DEFAULT_TRAINING: RFTrainingConfig = {
  dataset: 'iris',
  testSize: 0.2,
  randomState: 42,
  stratify: true,
  logEveryTrees: 5,
}

function createPresetGraph(dataset: string): { nodes: RFNode[]; edges: RFEdge[] } {
  const defaults = DATASET_DEFAULTS[dataset] ?? DATASET_DEFAULTS.iris
  const nodes: RFNode[] = [
    { id: 'rf_node_1', type: 'RFInput', position: [-4.5, 0, 0], config: { shape: defaults.shape } },
    { id: 'rf_node_2', type: 'RFFlatten', position: [-1.5, 0, 0], config: {} },
    {
      id: 'rf_node_3',
      type: 'RandomForestClassifier',
      position: [1.5, 0, 0],
      config: {
        n_estimators: 100,
        max_depth: null,
        criterion: 'gini',
        max_features: 'sqrt',
        min_samples_split: 2,
        min_samples_leaf: 1,
        bootstrap: true,
        random_state: 42,
      },
    },
    { id: 'rf_node_4', type: 'RFOutput', position: [4.5, 0, 0], config: { num_classes: defaults.numClasses } },
  ]

  const edges: RFEdge[] = [
    { id: 'rf_edge_1', source: 'rf_node_1', target: 'rf_node_2' },
    { id: 'rf_edge_2', source: 'rf_node_2', target: 'rf_node_3' },
    { id: 'rf_edge_3', source: 'rf_node_3', target: 'rf_node_4' },
  ]

  return { nodes, edges }
}

function indexById<T extends { id: string }>(items: T[]): Record<string, T> {
  const indexed: Record<string, T> = {}
  items.forEach((item) => {
    indexed[item.id] = item
  })
  return indexed
}

function resetCounters(nodes: Record<string, RFNode>, edges: Record<string, RFEdge>): void {
  let maxNodeNum = 0
  let maxEdgeNum = 0

  Object.keys(nodes).forEach((id) => {
    const match = id.match(/_(\d+)$/)
    if (match) maxNodeNum = Math.max(maxNodeNum, Number(match[1]) || 0)
  })
  Object.keys(edges).forEach((id) => {
    const match = id.match(/_(\d+)$/)
    if (match) maxEdgeNum = Math.max(maxEdgeNum, Number(match[1]) || 0)
  })

  nextNodeId = Math.max(nextNodeId, maxNodeNum + 1)
  nextEdgeId = Math.max(nextEdgeId, maxEdgeNum + 1)
}

interface RFGraphState {
  nodes: Record<string, RFNode>
  edges: Record<string, RFEdge>
  selectedNodeId: string | null
  selectedEdgeId: string | null
  connectionSource: string | null
  training: RFTrainingConfig
  addNode: (type: RFNodeType, position?: [number, number, number]) => string
  removeNode: (nodeId: string) => void
  addEdge: (sourceId: string, targetId: string) => string
  removeEdge: (edgeId: string) => void
  setNodePosition: (nodeId: string, position: [number, number, number]) => void
  selectNode: (nodeId: string | null) => void
  selectEdge: (edgeId: string | null) => void
  setConnectionSource: (nodeId: string | null) => void
  autoConnectByX: () => void
  clearGraph: () => void
  setDataset: (dataset: string) => void
  setTraining: (patch: Partial<RFTrainingConfig>) => void
  setNodeConfig: (nodeId: string, patch: Record<string, unknown>) => void
  resetPreset: () => void
  toPayload: () => RFGraphPayload
}

let nextNodeId = 1
let nextEdgeId = 1

export const useRFGraphStore = create<RFGraphState>((set, get) => {
  const preset = createPresetGraph(DEFAULT_TRAINING.dataset)
  const presetNodes = indexById(preset.nodes)
  const presetEdges = indexById(preset.edges)
  resetCounters(presetNodes, presetEdges)

  return {
    nodes: presetNodes,
    edges: presetEdges,
    selectedNodeId: null,
    selectedEdgeId: null,
    connectionSource: null,
    training: { ...DEFAULT_TRAINING },

    addNode: (type, position) => {
      const id = `rf_node_${nextNodeId++}`
      const defaults = datasetDefaults(get().training.dataset)
      const node: RFNode = {
        id,
        type,
        position: position ?? [0, 0, 0],
        config:
          type === 'RFInput'
            ? { shape: defaults.shape }
            : type === 'RFOutput'
              ? { num_classes: defaults.numClasses }
              : type === 'RandomForestClassifier'
                ? {
                    n_estimators: 100,
                    max_depth: null,
                    criterion: 'gini',
                    max_features: 'sqrt',
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    bootstrap: true,
                    random_state: get().training.randomState,
                  }
                : {},
      }
      set((state) => ({ nodes: { ...state.nodes, [id]: node } }))
      return id
    },

    removeNode: (nodeId) =>
      set((state) => {
        const { [nodeId]: _removed, ...nodes } = state.nodes
        const edges: Record<string, RFEdge> = {}
        Object.values(state.edges).forEach((edge) => {
          if (edge.source !== nodeId && edge.target !== nodeId) edges[edge.id] = edge
        })
        return {
          nodes,
          edges,
          selectedNodeId: state.selectedNodeId === nodeId ? null : state.selectedNodeId,
          connectionSource: state.connectionSource === nodeId ? null : state.connectionSource,
        }
      }),

    addEdge: (sourceId, targetId) => {
      const existing = Object.values(get().edges).find(
        (edge) => edge.source === sourceId && edge.target === targetId
      )
      if (existing) return existing.id
      const id = `rf_edge_${nextEdgeId++}`
      const edge: RFEdge = { id, source: sourceId, target: targetId }
      set((state) => ({ edges: { ...state.edges, [id]: edge } }))
      return id
    },

    removeEdge: (edgeId) =>
      set((state) => {
        const { [edgeId]: _removed, ...edges } = state.edges
        return {
          edges,
          selectedEdgeId: state.selectedEdgeId === edgeId ? null : state.selectedEdgeId,
        }
      }),

    setNodePosition: (nodeId, position) =>
      set((state) => ({
        nodes: {
          ...state.nodes,
          [nodeId]: { ...state.nodes[nodeId], position },
        },
      })),

    selectNode: (nodeId) => set({ selectedNodeId: nodeId, selectedEdgeId: null }),
    selectEdge: (edgeId) => set({ selectedEdgeId: edgeId, selectedNodeId: null }),
    setConnectionSource: (nodeId) => set({ connectionSource: nodeId }),

    autoConnectByX: () =>
      set((state) => {
        const ordered = Object.values(state.nodes).sort((a, b) => a.position[0] - b.position[0])
        const edges: Record<string, RFEdge> = {}
        for (let index = 0; index < ordered.length - 1; index += 1) {
          const edgeId = `rf_edge_${nextEdgeId++}`
          edges[edgeId] = {
            id: edgeId,
            source: ordered[index].id,
            target: ordered[index + 1].id,
          }
        }
        return { edges }
      }),

    clearGraph: () =>
      set({
        nodes: {},
        edges: {},
        selectedNodeId: null,
        selectedEdgeId: null,
        connectionSource: null,
      }),

    setDataset: (dataset) =>
      set((state) => {
        const graph = createPresetGraph(dataset)
        const nodes = indexById(graph.nodes)
        const edges = indexById(graph.edges)
        resetCounters(nodes, edges)
        return {
          training: { ...state.training, dataset },
          nodes,
          edges,
          selectedNodeId: null,
          selectedEdgeId: null,
          connectionSource: null,
        }
      }),

    setTraining: (patch) =>
      set((state) => ({
        training: { ...state.training, ...patch },
      })),

    setNodeConfig: (nodeId, patch) =>
      set((state) => ({
        nodes: {
          ...state.nodes,
          [nodeId]: {
            ...state.nodes[nodeId],
            config: { ...state.nodes[nodeId].config, ...patch },
          },
        },
      })),

    resetPreset: () =>
      set((state) => {
        const graph = createPresetGraph(state.training.dataset)
        const nodes = indexById(graph.nodes)
        const edges = indexById(graph.edges)
        resetCounters(nodes, edges)
        return {
          nodes,
          edges,
          selectedNodeId: null,
          selectedEdgeId: null,
          connectionSource: null,
        }
      }),

    toPayload: () => {
      const state = get()
      const nodes = Object.values(state.nodes).sort((a, b) => a.id.localeCompare(b.id))
      const edges = Object.values(state.edges).sort((a, b) => a.id.localeCompare(b.id))
      return {
        nodes: nodes.map((node) => ({
          id: node.id,
          type: node.type,
          config: node.config,
        })),
        edges,
        training: {
          dataset: state.training.dataset,
          test_size: state.training.testSize,
          random_state: state.training.randomState,
          stratify: state.training.stratify,
          log_every_trees: state.training.logEveryTrees,
        },
      }
    },
  }
})

export function datasetDefaults(dataset: string): DatasetDefaults {
  return DATASET_DEFAULTS[dataset] ?? DATASET_DEFAULTS.iris
}
