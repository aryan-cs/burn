import { useGraphStore, type LayerNode, type LayerType } from '../store/graphStore'
import { useTrainingStore } from '../store/trainingStore'
import { getNeuralNetworkOrder } from './graphOrder'

const DEFAULT_INPUT_SHAPE: [number, number, number] = [1, 28, 28]
const DEFAULT_OUTPUT_CLASSES = 10
const DEFAULT_DENSE_UNITS = 24
const DEFAULT_HIDDEN_ACTIVATION = 'relu'
const DEFAULT_OUTPUT_ACTIVATION = 'softmax'

interface BackendNode {
  id: string
  type: LayerType
  config: Record<string, unknown>
}

interface BackendEdge {
  id: string
  source: string
  target: string
}

/**
 * Serializes the full graph + training config into the JSON format
 * expected by the backend API. Node/edge topology is preserved exactly
 * from the current graph state (no synthetic nodes/edges are injected).
 */
export function serializeForBackend() {
  const graphState = useGraphStore.getState()
  const orderedNodeIds = getNeuralNetworkOrder(graphState.nodes, graphState.edges)
  const { nodes, edges } = toBackendGraph(graphState.nodes, graphState.edges, orderedNodeIds)
  const config = useTrainingStore.getState().config

  return {
    nodes,
    edges,
    training: {
      dataset: config.dataset,
      epochs: config.epochs,
      batch_size: config.batchSize,
      optimizer: config.optimizer,
      learning_rate: config.learningRate,
      loss: config.loss,
    },
  }
}

/**
 * Exports the graph as a downloadable JSON file.
 */
export function exportGraphJSON() {
  const data = serializeForBackend()
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'mlcanvas-model.json'
  a.click()
  URL.revokeObjectURL(url)
}

function toBackendGraph(
  sourceNodes: ReturnType<typeof useGraphStore.getState>['nodes'],
  sourceEdges: ReturnType<typeof useGraphStore.getState>['edges'],
  orderedNodeIds: string[]
): { nodes: BackendNode[]; edges: BackendEdge[] } {
  const orderIndex = new Map<string, number>()
  orderedNodeIds.forEach((nodeId, index) => {
    orderIndex.set(nodeId, index)
  })

  const backendNodes: BackendNode[] = orderedNodeIds
    .map((nodeId) => sourceNodes[nodeId])
    .filter((node): node is LayerNode => Boolean(node))
    .map((node) => toBackendNode(node))

  const backendEdges: BackendEdge[] = Object.values(sourceEdges)
    .filter((edge) => Boolean(sourceNodes[edge.source]) && Boolean(sourceNodes[edge.target]))
    .sort((left, right) => compareEdgesByNodeOrder(left, right, orderIndex))
    .map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
    }))

  return {
    nodes: backendNodes,
    edges: backendEdges,
  }
}

function toPositiveInt(value: unknown, fallback: number): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback
  }
  return Math.floor(parsed)
}

function getNodeUnitCount(
  node: LayerNode | null | undefined,
  fallback = DEFAULT_OUTPUT_CLASSES
): number {
  if (!node) return fallback

  const rows = toPositiveInt(node.config.rows, 0)
  const cols = toPositiveInt(node.config.cols, 0)
  if (rows > 0 && cols > 0) {
    return rows * cols
  }

  const units = toPositiveInt(node.config.units, 0)
  if (units > 0) return units

  return fallback
}

function toDropoutRate(value: unknown, fallback = 0.5): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  if (parsed < 0 || parsed >= 1) return fallback
  return parsed
}

function toBackendNode(node: LayerNode): BackendNode {
  const config: Record<string, unknown> = { ...node.config }

  if (node.type === 'Input') {
    config.shape = toInputShape(node.config.shape)
  }

  if (node.type === 'Dense') {
    config.units = getNodeUnitCount(node, DEFAULT_DENSE_UNITS)
    config.activation = toActivationOrFallback(
      node.config.activation,
      DEFAULT_HIDDEN_ACTIVATION
    )
  }

  if (node.type === 'Dropout') {
    config.rate = toDropoutRate(node.config.rate)
  }

  if (node.type === 'Output') {
    config.num_classes = toPositiveInt(node.config.num_classes, DEFAULT_OUTPUT_CLASSES)
    config.activation = toActivationOrFallback(
      node.config.activation,
      DEFAULT_OUTPUT_ACTIVATION
    )
  }

  return {
    id: node.id,
    type: node.type,
    config,
  }
}

function compareEdgesByNodeOrder(
  left: BackendEdge,
  right: BackendEdge,
  orderIndex: Map<string, number>
): number {
  const leftSource = orderIndex.get(left.source) ?? Number.MAX_SAFE_INTEGER
  const rightSource = orderIndex.get(right.source) ?? Number.MAX_SAFE_INTEGER
  if (leftSource !== rightSource) return leftSource - rightSource

  const leftTarget = orderIndex.get(left.target) ?? Number.MAX_SAFE_INTEGER
  const rightTarget = orderIndex.get(right.target) ?? Number.MAX_SAFE_INTEGER
  if (leftTarget !== rightTarget) return leftTarget - rightTarget

  return left.id.localeCompare(right.id)
}

function toActivationOrFallback(value: unknown, fallback: string): string {
  if (typeof value !== 'string') return fallback
  const normalized = value.trim()
  return normalized.length > 0 ? normalized : fallback
}

function toInputShape(rawShape: unknown): [number, number, number] {
  if (!Array.isArray(rawShape) || rawShape.length !== 3) {
    return [...DEFAULT_INPUT_SHAPE]
  }

  const parsed = rawShape.map((entry) => toPositiveInt(entry, 0))
  if (parsed.some((entry) => entry <= 0)) {
    return [...DEFAULT_INPUT_SHAPE]
  }

  return [parsed[0], parsed[1], parsed[2]]
}
