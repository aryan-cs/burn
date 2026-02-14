import { useEffect, useState } from 'react'
import { Viewport } from './canvas/Viewport'
import { useConnectionDraw } from './hooks/useConnectionDraw'
import { useWebSocket } from './hooks/useWebSocket'
import { useGraphStore, type Edge, type LayerNode } from './store/graphStore'
import { useTrainingStore } from './store/trainingStore'
import { serializeForBackend } from './utils/graphSerializer'
import {
  getLayerRolesForColoring,
  getNeuralNetworkOrder,
  type LayerRole,
} from './utils/graphOrder'
import {
  createEmptyInferenceGrid,
  inferenceGridToPayload,
} from './ui/InferencePixelPad'
import { BuildTab } from './ui/tabs/BuildTab'
import { TestTab } from './ui/tabs/TestTab'
import { TrainTab } from './ui/tabs/TrainTab'

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

interface ValidationResponse {
  valid: boolean
  errors: Array<{ message?: string }>
}

interface TrainResponse {
  job_id: string
  status: string
}

interface InferResponse {
  predictions?: number[]
  probabilities?: number[][]
}

type DashboardTab = 'validate' | 'train' | 'infer'

async function requestJson<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...(init?.headers ?? {}),
    },
  })

  const text = await response.text()
  if (!response.ok) {
    throw new Error(getErrorMessageFromBody(text, response.status))
  }
  return JSON.parse(text) as T
}

function getErrorMessageFromBody(body: string, status: number): string {
  if (body.length === 0) return `HTTP ${status}`

  try {
    const parsed = JSON.parse(body) as {
      detail?: { message?: string; errors?: Array<{ message?: string }> } | string
      message?: string
      errors?: Array<{ message?: string }>
    }

    if (typeof parsed.detail === 'string') return parsed.detail
    if (parsed.detail?.message) return parsed.detail.message
    if (parsed.detail?.errors?.[0]?.message) {
      return parsed.detail.errors[0].message ?? `HTTP ${status}`
    }
    if (parsed.errors?.[0]?.message) return parsed.errors[0].message ?? `HTTP ${status}`
    if (parsed.message) return parsed.message
    return body
  } catch {
    return body
  }
}

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
  const normalizeSequentialEdges = useGraphStore((s) => s.normalizeSequentialEdges)
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
  const edgeCount = Object.keys(edges).length
  const orderedNodeIds = getNeuralNetworkOrder(nodes, edges)
  const buildLayerItems = orderedNodeIds.flatMap((nodeId) => {
    const node = nodes[nodeId]
    if (!node) return []

    const role = nodeRoles.get(nodeId) ?? 'hidden'
    return [
      {
        id: nodeId,
        name: getLayerDisplayName(
          nodeId,
          role,
          toStringOrFallback(node.config.name, '')
        ),
        role,
        rows: toPositiveInt(node.config.rows, DEFAULT_LAYER_ROWS),
        cols: toPositiveInt(node.config.cols, DEFAULT_LAYER_COLS),
      },
    ]
  })
  const weightCount = getWeightCount(nodes, edges)
  const biasCount = getBiasCount(nodes, edges)
  const layerTypeSummary = getLayerTypeSummary(nodes)
  const neuronCount = Object.values(nodes).reduce((total, node) => {
    return total + getLayerNeuronCount(node)
  }, 0)

  const trainingStatus = useTrainingStore((s) => s.status)
  const trainingJobId = useTrainingStore((s) => s.jobId)
  const trainingConfig = useTrainingStore((s) => s.config)
  const setTrainingConfig = useTrainingStore((s) => s.setConfig)
  const startTraining = useTrainingStore((s) => s.startTraining)
  const setTrainingError = useTrainingStore((s) => s.setError)
  const trainingMetrics = useTrainingStore((s) => s.metrics)
  const currentEpoch = useTrainingStore((s) => s.currentEpoch)
  const totalEpochs = useTrainingStore((s) => s.totalEpochs)
  const trainingErrorMessage = useTrainingStore((s) => s.errorMessage)
  const { sendStop } = useWebSocket()

  const [editingNodeId, setEditingNodeId] = useState<string | null>(null)
  const [draftName, setDraftName] = useState('')
  const [activeTab, setActiveTab] = useState<DashboardTab>('validate')
  const [hasValidatedModel, setHasValidatedModel] = useState(false)
  const [backendBusyAction, setBackendBusyAction] = useState<string | null>(null)
  const [backendMessage, setBackendMessage] = useState('Ready to validate/train.')
  const [inferenceGrid, setInferenceGrid] = useState<number[][]>(() =>
    createEmptyInferenceGrid()
  )
  const [inferenceOutput, setInferenceOutput] = useState('No inference output yet.')
  const [inferenceTopPrediction, setInferenceTopPrediction] = useState<number | null>(null)
  const isEditingName = Boolean(editingNodeId && editingNodeId === selectedNodeId)
  const isBackendBusy = backendBusyAction !== null
  const latestTrainingMetric = trainingMetrics[trainingMetrics.length - 1]
  const latestTrainLoss = latestTrainingMetric
    ? latestTrainingMetric.trainLoss ?? latestTrainingMetric.loss
    : null
  const latestTrainAccuracy = latestTrainingMetric
    ? latestTrainingMetric.trainAccuracy ?? latestTrainingMetric.accuracy
    : null
  const latestTestLoss = latestTrainingMetric
    ? latestTrainingMetric.testLoss ?? latestTrainingMetric.loss
    : null
  const latestTestAccuracy = latestTrainingMetric
    ? latestTrainingMetric.testAccuracy ?? latestTrainingMetric.accuracy
    : null
  const trainLossSeries = trainingMetrics.map((metric) => metric.trainLoss ?? metric.loss)
  const testLossSeries = trainingMetrics.map((metric) => metric.testLoss ?? metric.loss)
  const trainAccuracySeries = trainingMetrics.map(
    (metric) => metric.trainAccuracy ?? metric.accuracy
  )
  const testAccuracySeries = trainingMetrics.map(
    (metric) => metric.testAccuracy ?? metric.accuracy
  )
  const canOpenTrainTab =
    hasValidatedModel || trainingStatus !== 'idle' || Boolean(trainingJobId)
  const canOpenInferTab =
    (trainingStatus === 'complete' || inferenceTopPrediction !== null) &&
    Boolean(trainingJobId)
  const activeTabIndex = activeTab === 'validate' ? 0 : activeTab === 'train' ? 1 : 2

  const runBackendAction = async (
    actionName: string,
    action: () => Promise<void>
  ) => {
    if (isBackendBusy) return
    setBackendBusyAction(actionName)
    try {
      await action()
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setBackendMessage(message)
      if (actionName === 'train') {
        setTrainingError(message)
      }
    } finally {
      setBackendBusyAction(null)
    }
  }

  useEffect(() => {
    if (trainingStatus === 'complete') {
      setActiveTab('infer')
    }
  }, [trainingStatus])

  useEffect(() => {
    setHasValidatedModel(false)
  }, [layerCount, edgeCount])

  const handleAddLayer = () => {
    const nodeCount = Object.keys(useGraphStore.getState().nodes).length
    const position: [number, number, number] = [0, ALIGN_Y, -nodeCount * LAYER_SPACING]
    const nodeId = addNode('Dense', position)

    updateNodeConfig(nodeId, {
      rows: DEFAULT_LAYER_ROWS,
      cols: DEFAULT_LAYER_COLS,
      units: DEFAULT_LAYER_ROWS * DEFAULT_LAYER_COLS,
      activation: DEFAULT_ACTIVATION,
    })
    selectNode(nodeId)
    setHasValidatedModel(false)
  }

  const handleAlign = () => {
    if (layerCount === 0) return

    cancelConnectionDrag()
    setDraggingNodeId(null)
    normalizeSequentialEdges()

    const { nodes: alignedNodes, edges: alignedEdges } = useGraphStore.getState()
    const alignedNodeIds = getNeuralNetworkOrder(alignedNodes, alignedEdges)
    if (alignedNodeIds.length === 0) return

    const nextPositions: Record<string, [number, number, number]> = {}
    const center = (alignedNodeIds.length - 1) / 2
    alignedNodeIds.forEach((nodeId, index) => {
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
    setHasValidatedModel(false)
  }

  const handleColsChange = (nextColsValue: string) => {
    if (!selectedNodeId) return
    const nextCols = Number(nextColsValue)
    if (!Number.isInteger(nextCols) || nextCols <= 0) return
    updateNodeConfig(selectedNodeId, {
      cols: nextCols,
      units: selectedRows * nextCols,
    })
    setHasValidatedModel(false)
  }

  const handleActivationChange = (nextActivation: string) => {
    if (!selectedNodeId) return

    if (selectedRole === 'output') {
      updateNodeConfig(selectedNodeId, { activation: nextActivation })
      setHasValidatedModel(false)
      return
    }

    Object.keys(nodes).forEach((nodeId) => {
      const role = nodeRoles.get(nodeId) ?? 'hidden'
      if (role !== 'output') {
        updateNodeConfig(nodeId, { activation: nextActivation })
      }
    })
    setHasValidatedModel(false)
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

  const handleValidateModel = () =>
    runBackendAction('validate', async () => {
      const payload = serializeForBackend()
      if (payload.nodes.length === 0) {
        throw new Error('Add at least one layer before validating.')
      }

      const response = await requestJson<ValidationResponse>('/api/model/validate', {
        method: 'POST',
        body: JSON.stringify(payload),
      })

      if (!response.valid) {
        const firstError = response.errors[0]?.message ?? 'Graph validation failed.'
        setBackendMessage(firstError)
        setHasValidatedModel(false)
        return
      }

      setHasValidatedModel(true)
      setActiveTab('train')
      setBackendMessage(`Validation passed (${payload.nodes.length} backend nodes).`)
    })

  const handleTrainModel = () =>
    runBackendAction('train', async () => {
      if (!hasValidatedModel) {
        setActiveTab('validate')
        throw new Error('Validate the model first.')
      }
      const payload = serializeForBackend()
      if (payload.nodes.length === 0) {
        throw new Error('Add at least one layer before training.')
      }

      const response = await requestJson<TrainResponse>('/api/model/train', {
        method: 'POST',
        body: JSON.stringify(payload),
      })

      startTraining(response.job_id, trainingConfig.epochs)
      setActiveTab('train')
      setBackendMessage(`Training started (job_id=${response.job_id}).`)
    })

  const handleStopModel = () =>
    runBackendAction('stop', async () => {
      if (!trainingJobId) {
        throw new Error('No active job to stop.')
      }

      sendStop()
      await requestJson<{ job_id: string; status: string }>('/api/model/stop', {
        method: 'POST',
        body: JSON.stringify({ job_id: trainingJobId }),
      })

      setBackendMessage(`Stop requested for ${trainingJobId}.`)
    })

  const handleInferModel = () =>
    runBackendAction('infer', async () => {
      if (!trainingJobId) {
        throw new Error('Train a model before running inference.')
      }
      if (trainingStatus !== 'complete') {
        throw new Error('Wait for training to complete before inferencing.')
      }

      const response = await requestJson<InferResponse>('/api/model/infer', {
        method: 'POST',
        body: JSON.stringify({
          job_id: trainingJobId,
          inputs: inferenceGridToPayload(inferenceGrid),
          return_probabilities: true,
        }),
      })

      setInferenceOutput(JSON.stringify(response, null, 2))
      const nextPrediction = response.predictions?.[0]
      setInferenceTopPrediction(nextPrediction ?? null)
      setActiveTab('infer')
      setBackendMessage(
        nextPrediction !== undefined
          ? `Inference complete. Predicted class ${nextPrediction}.`
          : 'Inference complete.'
      )
    })

  return (
    <div className="app-shell">
      <section className="app-sidebar">
        <div className="app-sidebar-inner">
          <div className="app-tab-strip">
            <div
              aria-hidden
              className="app-tab-indicator"
              style={{ transform: `translateX(${activeTabIndex * 100}%)` }}
            >
              <div className="app-tab-indicator-inner">
                <div className="app-tab-indicator-glow" />
                <div className="app-tab-indicator-line" />
              </div>
            </div>
            <button
              type="button"
              onClick={() => setActiveTab('validate')}
              className={`app-tab-button ${
                activeTab === 'validate' ? 'app-tab-button-active' : 'app-tab-button-inactive'
              }`}
            >
              BUILD
            </button>
            <button
              type="button"
              disabled={!canOpenTrainTab}
              onClick={() => {
                if (!canOpenTrainTab) return
                setActiveTab('train')
              }}
              className={`app-tab-button ${
                activeTab === 'train' ? 'app-tab-button-active' : 'app-tab-button-inactive'
              }`}
            >
              Train
            </button>
            <button
              type="button"
              disabled={!canOpenInferTab}
              onClick={() => {
                if (!canOpenInferTab) return
                setActiveTab('infer')
              }}
              className={`app-tab-button ${
                activeTab === 'infer' ? 'app-tab-button-active' : 'app-tab-button-inactive'
              }`}
            >
              Test
            </button>
          </div>

          {activeTab === 'validate' ? (
            <BuildTab
              layerItems={buildLayerItems}
              hasSelectedNode={selectedNode !== null}
              isEditingName={isEditingName}
              draftName={draftName}
              selectedDisplayName={selectedDisplayName}
              selectedRows={selectedRows}
              selectedCols={selectedCols}
              selectedActivation={selectedActivation}
              activationOptions={ACTIVATION_OPTIONS}
              layerCount={layerCount}
              neuronCount={neuronCount}
              weightCount={weightCount}
              biasCount={biasCount}
              layerTypeSummary={layerTypeSummary}
              sharedNonOutputActivation={sharedNonOutputActivation}
              onAddLayer={handleAddLayer}
              onBeginNameEdit={beginNameEdit}
              onDraftNameChange={(value) => setDraftName(value)}
              onNameCommit={commitNameEdit}
              onNameCancel={cancelNameEdit}
              onRowsChange={handleRowsChange}
              onColsChange={handleColsChange}
              onActivationChange={handleActivationChange}
              onValidate={handleValidateModel}
              validateDisabled={isBackendBusy || layerCount === 0}
              validateLabel={backendBusyAction === 'validate' ? 'Building...' : 'Build'}
            />
          ) : null}

          {activeTab === 'train' ? (
            <TrainTab
              trainingStatus={trainingStatus}
              trainingStatusClass={getTrainingStatusClass(trainingStatus)}
              trainingErrorMessage={trainingErrorMessage}
              backendMessage={backendMessage}
              trainingConfig={trainingConfig}
              isBackendBusy={isBackendBusy}
              onDatasetChange={(value) => setTrainingConfig({ dataset: value })}
              onEpochsChange={(value) => setTrainingConfig({ epochs: value })}
              onBatchSizeChange={(value) => setTrainingConfig({ batchSize: value })}
              onLearningRateChange={(value) => setTrainingConfig({ learningRate: value })}
              trainingJobId={trainingJobId}
              currentEpoch={currentEpoch}
              totalEpochs={totalEpochs}
              latestTrainLoss={latestTrainLoss}
              latestTrainAccuracy={latestTrainAccuracy}
              latestTestLoss={latestTestLoss}
              latestTestAccuracy={latestTestAccuracy}
              trainLossSeries={trainLossSeries}
              testLossSeries={testLossSeries}
              trainAccuracySeries={trainAccuracySeries}
              testAccuracySeries={testAccuracySeries}
              isTraining={trainingStatus === 'training'}
              onStopModel={handleStopModel}
              stopDisabled={isBackendBusy || !trainingJobId}
              stopLabel={backendBusyAction === 'stop' ? 'Stopping...' : 'Stop Training'}
              onTrainModel={handleTrainModel}
              trainDisabled={isBackendBusy || !hasValidatedModel || layerCount === 0}
              trainLabel={backendBusyAction === 'train' ? 'Training...' : 'Start Training'}
              onGoToTest={() => setActiveTab('infer')}
              goToTestDisabled={!canOpenInferTab}
            />
          ) : null}

          {activeTab === 'infer' ? (
            <TestTab
              trainingStatus={trainingStatus}
              trainingStatusClass={getTrainingStatusClass(trainingStatus)}
              trainingJobId={trainingJobId}
              backendMessage={backendMessage}
              inferenceGrid={inferenceGrid}
              setInferenceGrid={setInferenceGrid}
              padDisabled={isBackendBusy || !trainingJobId || trainingStatus !== 'complete'}
              inferenceTopPrediction={inferenceTopPrediction}
              inferenceOutput={inferenceOutput}
              onInferModel={handleInferModel}
              inferDisabled={isBackendBusy || !trainingJobId || trainingStatus !== 'complete'}
              inferLabel={backendBusyAction === 'infer' ? 'Inferencing...' : 'Run Inference'}
              onBackToBuild={() => setActiveTab('validate')}
            />
          ) : null}
        </div>
      </section>

      <section className="app-viewport-panel">
        <Viewport />
        <button
          onClick={handleAlign}
          disabled={layerCount === 0}
          aria-label="Align"
          title="Align"
          className="align-button"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            height="24px"
            viewBox="0 -960 960 960"
            width="24px"
            fill="#e3e3e3"
          >
            <path d="M180-120q-24 0-42-18t-18-42v-172h60v172h172v60H180Zm428 0v-60h172v-172h60v172q0 24-18 42t-42 18H608ZM120-608v-172q0-24 18-42t42-18h172v60H180v172h-60Zm660 0v-172H608v-60h172q24 0 42 18t18 42v172h-60ZM347.5-347.5Q293-402 293-480t54.5-132.5Q402-667 480-667t132.5 54.5Q667-558 667-480t-54.5 132.5Q558-293 480-293t-132.5-54.5Zm223-42Q607-426 607-480t-36.5-90.5Q534-607 480-607t-90.5 36.5Q353-534 353-480t36.5 90.5Q426-353 480-353t90.5-36.5ZM480-480Z" />
          </svg>
        </button>
      </section>
    </div>
  )
}

export default App

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

function getTrainingStatusClass(status: string): string {
  if (status === 'training') {
    return 'status-pill status-pill-training'
  }
  if (status === 'complete') {
    return 'status-pill status-pill-complete'
  }
  if (status === 'error') {
    return 'status-pill status-pill-error'
  }
  return 'status-pill status-pill-idle'
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
  if (role === 'input') return 'Input Layer'
  if (role === 'output') return 'Output Layer'

  const match = nodeId.match(/_(\d+)$/)
  if (!match) return 'Layer'
  return `Layer ${match[1]}`
}

function getWeightCount(
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

function getBiasCount(
  nodes: Record<string, LayerNode>,
  edges: Record<string, Edge>
): number {
  const uniqueTargetIds = new Set<string>()
  Object.values(edges).forEach((edge) => {
    if (!nodes[edge.source] || !nodes[edge.target] || edge.source === edge.target) return
    uniqueTargetIds.add(edge.target)
  })

  let totalBiases = 0
  uniqueTargetIds.forEach((targetId) => {
    const targetNode = nodes[targetId]
    if (!targetNode) return
    totalBiases += getLayerNeuronCount(targetNode)
  })
  return totalBiases
}

function getLayerTypeSummary(nodes: Record<string, LayerNode>): string {
  const nodeTypes = new Set(Object.values(nodes).map((node) => node.type))
  if (nodeTypes.size === 0) return 'None'
  if (nodeTypes.size === 1) return Array.from(nodeTypes)[0]
  return 'Mixed'
}
