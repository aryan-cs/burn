import { useGraphStore } from '../store/graphStore'
import { useTrainingStore } from '../store/trainingStore'
import { getNeuralNetworkOrder } from './graphOrder'

const DEFAULT_INPUT_SHAPE: [number, number, number] = [1, 28, 28]
const DEFAULT_OUTPUT_CLASSES = 10
const DEFAULT_HIDDEN_ACTIVATION = 'relu'
const DEFAULT_OUTPUT_ACTIVATION = 'softmax'

interface BackendNode {
  id: string
  type: 'Input' | 'Flatten' | 'Dense' | 'Output'
  config: Record<string, unknown>
}

interface BackendEdge {
  id: string
  source: string
  target: string
}

/**
 * Serializes the full graph + training config into the JSON format
 * expected by the backend API.
 */
export function serializeForBackend() {
  const graphState = useGraphStore.getState()
  const orderedNodeIds = getNeuralNetworkOrder(graphState.nodes, graphState.edges)
  const { nodes, edges } = toBackendSequentialGraph(graphState.nodes, orderedNodeIds)
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

function toBackendSequentialGraph(
  sourceNodes: ReturnType<typeof useGraphStore.getState>['nodes'],
  orderedNodeIds: string[]
): { nodes: BackendNode[]; edges: BackendEdge[] } {
  if (orderedNodeIds.length === 0) {
    return { nodes: [], edges: [] }
  }

  const backendNodes: BackendNode[] = []
  const backendEdges: BackendEdge[] = []

  const inputSourceId = orderedNodeIds[0]
  const inputSourceNode = sourceNodes[inputSourceId]
  const inputShape = toInputShape(inputSourceNode?.config.shape)
  backendNodes.push({
    id: inputSourceId,
    type: 'Input',
    config: { shape: inputShape },
  })

  const flattenId = `${inputSourceId}__flatten`
  backendNodes.push({
    id: flattenId,
    type: 'Flatten',
    config: {},
  })
  backendEdges.push({
    id: `edge_auto_${backendEdges.length + 1}`,
    source: inputSourceId,
    target: flattenId,
  })

  let previousNodeId = flattenId
  const hiddenNodeIds = orderedNodeIds.slice(1, -1)
  hiddenNodeIds.forEach((nodeId) => {
    const sourceNode = sourceNodes[nodeId]
    backendNodes.push({
      id: nodeId,
      type: 'Dense',
      config: {
        units: getNodeUnitCount(sourceNode),
        activation: toActivationOrFallback(
          sourceNode?.config.activation,
          DEFAULT_HIDDEN_ACTIVATION
        ),
      },
    })
    backendEdges.push({
      id: `edge_auto_${backendEdges.length + 1}`,
      source: previousNodeId,
      target: nodeId,
    })
    previousNodeId = nodeId
  })

  const outputSourceId =
    orderedNodeIds.length > 1 ? orderedNodeIds[orderedNodeIds.length - 1] : null
  const outputId = outputSourceId ?? `${inputSourceId}__output`
  const outputSourceNode = outputSourceId ? sourceNodes[outputSourceId] : null
  backendNodes.push({
    id: outputId,
    type: 'Output',
    config: {
      num_classes: getNodeUnitCount(outputSourceNode, DEFAULT_OUTPUT_CLASSES),
      activation: toActivationOrFallback(
        outputSourceNode?.config.activation,
        DEFAULT_OUTPUT_ACTIVATION
      ),
    },
  })
  backendEdges.push({
    id: `edge_auto_${backendEdges.length + 1}`,
    source: previousNodeId,
    target: outputId,
  })

  return { nodes: backendNodes, edges: backendEdges }
}

function toPositiveInt(value: unknown, fallback: number): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback
  }
  return Math.floor(parsed)
}

function getNodeUnitCount(
  node: ReturnType<typeof useGraphStore.getState>['nodes'][string] | null | undefined,
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
