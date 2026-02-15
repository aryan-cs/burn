import { useEffect, useMemo, useState } from 'react'
import { Viewport } from './canvas/Viewport'
import { useConnectionDraw } from './hooks/useConnectionDraw'
import { useWebSocket } from './hooks/useWebSocket'
import { useGraphStore, type LayerNode } from './store/graphStore'
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
import { DeployTab } from './ui/tabs/DeployTab'
import { TestTab } from './ui/tabs/TestTab'
import { TrainTab } from './ui/tabs/TrainTab'

const DEFAULT_LAYER_ROWS = 4
const DEFAULT_LAYER_COLS = 6
const DEFAULT_INPUT_SHAPE: [number, number, number] = [1, 28, 28]
const DEFAULT_OUTPUT_CLASSES = 10
const LAYER_SPACING = 4
const ALIGN_LAYER_SPACING = 2.25
const ALIGN_Y = 0.8
const TRAIN_TO_TEST_SWITCH_DELAY_MS = 500
const DEFAULT_ACTIVATION = 'linear'
const DEFAULT_INFERENCE_ROWS = 28
const DEFAULT_INFERENCE_COLS = 28
const LOW_DETAIL_STORAGE_KEY = 'mlcanvas.nn.low_detail_mode'
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
  errors: ValidationIssue[]
  warnings: string[]
}

interface ValidationIssue {
  message?: string
  node_id?: string | null
  expected?: unknown
  got?: unknown
}

interface TrainResponse {
  job_id: string
  status: string
}

interface InferResponse {
  predictions?: number[]
  probabilities?: number[][]
}

interface DeploymentResponse {
  deployment_id: string
  job_id: string
  status: string
  target: string
  endpoint_path: string
  created_at: string
  last_used_at?: string | null
  request_count: number
  name?: string | null
}

interface DeploymentInferResponse {
  deployment_id: string
  job_id: string
  predictions?: number[]
  probabilities?: number[][]
  logits?: number[][]
}

type DashboardTab = 'validate' | 'train' | 'infer' | 'deploy'
type BuildStatus = 'idle' | 'success' | 'error'

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
  const addEdge = useGraphStore((s) => s.addEdge)
  const removeEdge = useGraphStore((s) => s.removeEdge)
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
  const selectedInputShape = selectedNode?.type === 'Input'
    ? toInputShapeOrDefault(selectedNode.config.shape)
    : null
  const selectedRows = selectedNode?.type === 'Input'
    ? (selectedInputShape?.[1] ?? DEFAULT_INPUT_SHAPE[1])
    : toPositiveInt(selectedNode?.config.rows, DEFAULT_LAYER_ROWS)
  const selectedCols = selectedNode?.type === 'Input'
    ? (selectedInputShape?.[2] ?? DEFAULT_INPUT_SHAPE[2])
    : toPositiveInt(selectedNode?.config.cols, DEFAULT_LAYER_COLS)
  const selectedChannels = selectedNode?.type === 'Input'
    ? (selectedInputShape?.[0] ?? DEFAULT_INPUT_SHAPE[0])
    : DEFAULT_INPUT_SHAPE[0]
  const selectedUnits = selectedNode?.type === 'Dense'
    ? toPositiveInt(nodeUnits(selectedNode), DEFAULT_LAYER_ROWS * DEFAULT_LAYER_COLS)
    : DEFAULT_LAYER_ROWS * DEFAULT_LAYER_COLS
  const selectedDropoutRate = selectedNode?.type === 'Dropout'
    ? clampDropoutRate(selectedNode.config.rate)
    : 0.5
  const selectedOutputClasses = selectedNode?.type === 'Output'
    ? toPositiveInt(selectedNode.config.num_classes, DEFAULT_OUTPUT_CLASSES)
    : DEFAULT_OUTPUT_CLASSES
  const sharedNonOutputActivation = getSharedNonOutputActivation(
    nodes,
    nodeRoles,
    DEFAULT_ACTIVATION
  )
  const selectedActivation = toStringOrFallback(
    selectedNode?.config.activation,
    DEFAULT_ACTIVATION
  )
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
        sizeLabel: getLayerSizeLabel(node),
      },
    ]
  })
  const stats = useMemo(
    () => computeGraphStats(nodes, orderedNodeIds),
    [nodes, orderedNodeIds]
  )
  const weightCount = stats.weightCount
  const biasCount = stats.biasCount
  const layerTypeSummary = getLayerTypeSummary(nodes)
  const neuronCount = stats.neuronCount
  const inferenceGridSize = getInferenceGridSizeFromInputLayer(
    nodes,
    orderedNodeIds,
    DEFAULT_INFERENCE_ROWS,
    DEFAULT_INFERENCE_COLS
  )

  const trainingStatus = useTrainingStore((s) => s.status)
  const trainingJobId = useTrainingStore((s) => s.jobId)
  const trainingConfig = useTrainingStore((s) => s.config)
  const setTrainingConfig = useTrainingStore((s) => s.setConfig)
  const startTraining = useTrainingStore((s) => s.startTraining)
  const setTrainingError = useTrainingStore((s) => s.setError)
  const trainingMetrics = useTrainingStore((s) => s.metrics)
  const currentEpoch = useTrainingStore((s) => s.currentEpoch)
  const { sendStop } = useWebSocket()

  const [editingNodeId, setEditingNodeId] = useState<string | null>(null)
  const [draftName, setDraftName] = useState('')
  const [activeTab, setActiveTab] = useState<DashboardTab>('validate')
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [isLowDetailMode, setIsLowDetailMode] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem(LOW_DETAIL_STORAGE_KEY) === '1'
  })
  const [hasValidatedModel, setHasValidatedModel] = useState(false)
  const [buildStatus, setBuildStatus] = useState<BuildStatus>('idle')
  const [buildIssues, setBuildIssues] = useState<string[]>([])
  const [buildWarnings, setBuildWarnings] = useState<string[]>([])
  const [backendBusyAction, setBackendBusyAction] = useState<string | null>(null)
  const [, setBackendMessage] = useState('Ready to validate/train.')
  const [inferenceGrid, setInferenceGrid] = useState<number[][]>(() =>
    createEmptyInferenceGrid(inferenceGridSize.rows, inferenceGridSize.cols)
  )
  const [inferenceTopPrediction, setInferenceTopPrediction] = useState<number | null>(null)
  const [deployment, setDeployment] = useState<DeploymentResponse | null>(null)
  const [deployTopPrediction, setDeployTopPrediction] = useState<number | null>(null)
  const [deployOutput, setDeployOutput] = useState('No deployed inference output yet.')
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
  const canOpenDeployTab =
    Boolean(deployment) || (Boolean(trainingJobId) && trainingStatus === 'complete')
  const activeTabIndex =
    activeTab === 'validate' ? 0 : activeTab === 'train' ? 1 : activeTab === 'infer' ? 2 : 3

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
      if (actionName === 'validate') {
        setBuildStatus('error')
        setBuildWarnings([])
        setBuildIssues([message])
        setHasValidatedModel(false)
      }
      if (actionName === 'train') {
        setTrainingError(message)
      }
    } finally {
      setBackendBusyAction(null)
    }
  }

  useEffect(() => {
    if (trainingStatus !== 'complete') return

    const timer = window.setTimeout(() => {
      setActiveTab('infer')
    }, TRAIN_TO_TEST_SWITCH_DELAY_MS)

    return () => {
      window.clearTimeout(timer)
    }
  }, [trainingStatus])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(LOW_DETAIL_STORAGE_KEY, isLowDetailMode ? '1' : '0')
  }, [isLowDetailMode])

  useEffect(() => {
    setHasValidatedModel(false)
    setBuildStatus('idle')
    setBuildIssues([])
    setBuildWarnings([])
  }, [layerCount, edgeCount])

  useEffect(() => {
    setInferenceGrid((prev) => {
      const prevRows = prev.length
      const prevCols = prev.reduce((max, row) => Math.max(max, row.length), 0)
      if (prevRows === inferenceGridSize.rows && prevCols === inferenceGridSize.cols) {
        return prev
      }
      return createEmptyInferenceGrid(inferenceGridSize.rows, inferenceGridSize.cols)
    })
  }, [inferenceGridSize.rows, inferenceGridSize.cols])

  const handleAddLayer = () => {
    const state = useGraphStore.getState()
    const nodeIds = Object.keys(state.nodes)
    const nodeCount = nodeIds.length
    const orderedNodeIds = getNeuralNetworkOrder(state.nodes, state.edges)
    const outputCounts = new Map<string, number>()
    Object.values(state.edges).forEach((edge) => {
      outputCounts.set(edge.source, (outputCounts.get(edge.source) ?? 0) + 1)
    })

    const inputNodeId = orderedNodeIds.find((nodeId) => state.nodes[nodeId]?.type === 'Input') ?? null
    const flattenNodeId =
      orderedNodeIds.find((nodeId) => state.nodes[nodeId]?.type === 'Flatten') ?? null
    const outputNodeId = orderedNodeIds.find((nodeId) => state.nodes[nodeId]?.type === 'Output')
    const edgeIntoOutput = outputNodeId
      ? Object.values(state.edges).find((edge) => edge.target === outputNodeId)
      : null
    const tailNodeId =
      [...orderedNodeIds]
        .reverse()
        .find((nodeId) => (outputCounts.get(nodeId) ?? 0) === 0) ?? null

    const fallbackPosition: [number, number, number] = [0, ALIGN_Y, -nodeCount * LAYER_SPACING]

    const positionAfterNode = (sourceId: string | null): [number, number, number] => {
      if (!sourceId) return fallbackPosition
      const sourcePosition = state.nodes[sourceId]?.position
      if (!sourcePosition) return fallbackPosition
      return [sourcePosition[0], sourcePosition[1], sourcePosition[2] - ALIGN_LAYER_SPACING]
    }

    const positionBetweenNodes = (
      sourceId: string | null,
      targetId: string | null
    ): [number, number, number] => {
      if (!sourceId || !targetId) return fallbackPosition
      const sourcePosition = state.nodes[sourceId]?.position
      const targetPosition = state.nodes[targetId]?.position
      if (sourcePosition && targetPosition) {
        return [
          (sourcePosition[0] + targetPosition[0]) / 2,
          (sourcePosition[1] + targetPosition[1]) / 2,
          (sourcePosition[2] + targetPosition[2]) / 2,
        ]
      }
      return fallbackPosition
    }

    const addDefaultDense = (position: [number, number, number]): string => {
      const nodeId = addNode('Dense', position)
      updateNodeConfig(nodeId, {
        rows: DEFAULT_LAYER_ROWS,
        cols: DEFAULT_LAYER_COLS,
        units: DEFAULT_LAYER_ROWS * DEFAULT_LAYER_COLS,
        activation: DEFAULT_ACTIVATION,
      })
      return nodeId
    }

    if (!inputNodeId) {
      const nodeId = addNode('Input', [0, ALIGN_Y, 3.4])
      selectNode(nodeId)
      setHasValidatedModel(false)
      return
    }

    if (!flattenNodeId) {
      const anchorSource = edgeIntoOutput?.source ?? tailNodeId ?? inputNodeId
      const nodeId = addNode(
        'Flatten',
        edgeIntoOutput
          ? positionBetweenNodes(anchorSource, edgeIntoOutput.target)
          : positionAfterNode(anchorSource)
      )
      if (edgeIntoOutput) {
        removeEdge(edgeIntoOutput.id)
        addEdge(anchorSource, nodeId)
        addEdge(nodeId, edgeIntoOutput.target)
      } else {
        addEdge(anchorSource, nodeId)
      }
      selectNode(nodeId)
      setHasValidatedModel(false)
      return
    }

    const denseCount = orderedNodeIds.filter((nodeId) => state.nodes[nodeId]?.type === 'Dense').length
    if (denseCount === 0) {
      const anchorSource = edgeIntoOutput?.source ?? tailNodeId ?? flattenNodeId
      const nodeId = addDefaultDense(
        edgeIntoOutput
          ? positionBetweenNodes(anchorSource, edgeIntoOutput.target)
          : positionAfterNode(anchorSource)
      )

      if (edgeIntoOutput) {
        removeEdge(edgeIntoOutput.id)
        addEdge(anchorSource, nodeId)
        addEdge(nodeId, edgeIntoOutput.target)
      } else {
        addEdge(anchorSource, nodeId)
      }
      selectNode(nodeId)
      setHasValidatedModel(false)
      return
    }

    if (!outputNodeId) {
      const anchorSource = tailNodeId ?? orderedNodeIds[orderedNodeIds.length - 1] ?? null
      const nodeId = addNode('Output', positionAfterNode(anchorSource))
      if (anchorSource) {
        addEdge(anchorSource, nodeId)
      }
      selectNode(nodeId)
      setHasValidatedModel(false)
      return
    }

    const insertSource = edgeIntoOutput?.source ?? tailNodeId
    const nodeId = addDefaultDense(
      edgeIntoOutput
        ? positionBetweenNodes(insertSource, outputNodeId)
        : positionAfterNode(insertSource)
    )

    if (outputNodeId && edgeIntoOutput) {
      removeEdge(edgeIntoOutput.id)
      addEdge(edgeIntoOutput.source, nodeId)
      addEdge(nodeId, outputNodeId)
    } else {
      if (insertSource && insertSource !== outputNodeId) {
        addEdge(insertSource, nodeId)
      }
      addEdge(nodeId, outputNodeId)
    }

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
    if (!selectedNodeId || !selectedNode) return
    const nextRows = Number(nextRowsValue)
    if (!Number.isInteger(nextRows) || nextRows <= 0) return

    if (selectedNode.type === 'Input') {
      const [channels, _height, width] = toInputShapeOrDefault(selectedNode.config.shape)
      updateNodeConfig(selectedNodeId, {
        shape: [channels, nextRows, width],
      })
      setHasValidatedModel(false)
      return
    }

    if (selectedNode.type !== 'Dense') return
    updateNodeConfig(selectedNodeId, {
      rows: nextRows,
      units: nextRows * selectedCols,
    })
    setHasValidatedModel(false)
  }

  const handleColsChange = (nextColsValue: string) => {
    if (!selectedNodeId || !selectedNode) return
    const nextCols = Number(nextColsValue)
    if (!Number.isInteger(nextCols) || nextCols <= 0) return

    if (selectedNode.type === 'Input') {
      const [channels, height] = toInputShapeOrDefault(selectedNode.config.shape)
      updateNodeConfig(selectedNodeId, {
        shape: [channels, height, nextCols],
      })
      setHasValidatedModel(false)
      return
    }

    if (selectedNode.type !== 'Dense') return
    updateNodeConfig(selectedNodeId, {
      cols: nextCols,
      units: selectedRows * nextCols,
    })
    setHasValidatedModel(false)
  }

  const handleActivationChange = (nextActivation: string) => {
    if (!selectedNodeId || !selectedNode) return
    if (selectedNode.type !== 'Dense' && selectedNode.type !== 'Output') return
    updateNodeConfig(selectedNodeId, { activation: nextActivation })
    setHasValidatedModel(false)
  }

  const handleChannelsChange = (nextValue: string) => {
    if (!selectedNodeId || selectedNode?.type !== 'Input') return
    const channels = Number(nextValue)
    if (!Number.isInteger(channels) || channels <= 0) return
    const [, height, width] = toInputShapeOrDefault(selectedNode.config.shape)
    updateNodeConfig(selectedNodeId, { shape: [channels, height, width] })
    setHasValidatedModel(false)
  }

  const handleUnitsChange = (nextValue: string) => {
    if (!selectedNodeId || selectedNode?.type !== 'Dense') return
    const units = Number(nextValue)
    if (!Number.isInteger(units) || units <= 0) return
    const { rows, cols } = denseGridFromUnits(units)
    updateNodeConfig(selectedNodeId, { units, rows, cols })
    setHasValidatedModel(false)
  }

  const handleDropoutRateChange = (nextValue: string) => {
    if (!selectedNodeId || selectedNode?.type !== 'Dropout') return
    const rate = clampDropoutRate(nextValue)
    updateNodeConfig(selectedNodeId, { rate })
    setHasValidatedModel(false)
  }

  const handleOutputClassesChange = (nextValue: string) => {
    if (!selectedNodeId || selectedNode?.type !== 'Output') return
    const classes = Number(nextValue)
    if (!Number.isInteger(classes) || classes <= 0) return
    updateNodeConfig(selectedNodeId, { num_classes: classes })
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

      setBuildWarnings(response.warnings ?? [])
      if (!response.valid) {
        const issues =
          response.errors.length > 0
            ? response.errors.map((issue) => formatValidationIssue(issue))
            : ['Graph validation failed.']
        const firstError = issues[0] ?? 'Graph validation failed.'
        setBuildStatus('error')
        setBuildIssues(issues)
        setBackendMessage(firstError)
        setHasValidatedModel(false)
        return
      }

      setBuildStatus('success')
      setBuildIssues([])
      setHasValidatedModel(true)
      setActiveTab('train')
      if (response.warnings.length > 0) {
        setBackendMessage(
          `Validation passed with ${response.warnings.length} warning(s).`
        )
      } else {
        setBackendMessage(`Validation passed (${payload.nodes.length} backend nodes).`)
      }
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
      setDeployment(null)
      setDeployTopPrediction(null)
      setDeployOutput('No deployed inference output yet.')
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

      const nextPrediction = response.predictions?.[0]
      setInferenceTopPrediction(nextPrediction ?? null)
      setActiveTab('infer')
      setBackendMessage(
        nextPrediction !== undefined
          ? `Inference complete. Predicted class ${nextPrediction}.`
          : 'Inference complete.'
      )
    })

  const handleDeployModel = () =>
    runBackendAction('deploy', async () => {
      if (!trainingJobId) {
        throw new Error('Train a model before deploying.')
      }
      if (trainingStatus !== 'complete') {
        throw new Error('Wait for training to complete before deploying.')
      }

      const response = await requestJson<DeploymentResponse>('/api/deploy', {
        method: 'POST',
        body: JSON.stringify({
          job_id: trainingJobId,
          target: 'local',
          name: `${trainingConfig.dataset}-${trainingJobId.slice(0, 8)}`,
        }),
      })

      setDeployment(response)
      setActiveTab('deploy')
    })

  const handleRefreshDeployment = () =>
    runBackendAction('deploy_status', async () => {
      if (!deployment) {
        throw new Error('No deployment to refresh.')
      }

      const response = await requestJson<DeploymentResponse>(
        `/api/deploy/status?deployment_id=${deployment.deployment_id}`
      )
      setDeployment(response)
    })

  const handleStopDeployment = () =>
    runBackendAction('deploy_stop', async () => {
      if (!deployment) {
        throw new Error('No deployment to stop.')
      }

      const response = await requestJson<DeploymentResponse>(
        `/api/deploy/${deployment.deployment_id}`,
        {
          method: 'DELETE',
        }
      )
      setDeployment(response)
    })

  const handleInferViaDeployment = () =>
    runBackendAction('deploy_infer', async () => {
      if (!deployment) {
        throw new Error('Create a deployment first.')
      }
      if (deployment.status !== 'running') {
        throw new Error('Deployment is not running.')
      }

      const response = await requestJson<DeploymentInferResponse>(
        `/api/deploy/${deployment.deployment_id}/infer`,
        {
          method: 'POST',
          body: JSON.stringify({
            inputs: inferenceGridToPayload(inferenceGrid),
            return_probabilities: true,
          }),
        }
      )

      const nextPrediction = response.predictions?.[0]
      setDeployTopPrediction(nextPrediction ?? null)
      setDeployOutput(JSON.stringify(response, null, 2))

      const status = await requestJson<DeploymentResponse>(
        `/api/deploy/status?deployment_id=${deployment.deployment_id}`
      )
      setDeployment(status)
      setActiveTab('deploy')
    })

  return (
    <div className={`app-shell ${isSidebarCollapsed ? 'app-shell-collapsed' : ''}`}>
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
              Build
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
            <button
              type="button"
              disabled={!canOpenDeployTab}
              onClick={() => {
                if (!canOpenDeployTab) return
                setActiveTab('deploy')
              }}
              className={`app-tab-button ${
                activeTab === 'deploy' ? 'app-tab-button-active' : 'app-tab-button-inactive'
              }`}
            >
              Deploy
            </button>
          </div>

          {activeTab === 'validate' ? (
            <BuildTab
              layerItems={buildLayerItems}
              hasSelectedNode={selectedNode !== null}
              selectedNodeId={selectedNodeId}
              selectedNodeType={selectedNode?.type ?? null}
              isEditingName={isEditingName}
              draftName={draftName}
              selectedDisplayName={selectedDisplayName}
              selectedRows={selectedRows}
              selectedCols={selectedCols}
              selectedChannels={selectedChannels}
              selectedUnits={selectedUnits}
              selectedDropoutRate={selectedDropoutRate}
              selectedOutputClasses={selectedOutputClasses}
              selectedActivation={selectedActivation}
              selectedShapeLabel={selectedNode ? getLayerSizeLabel(selectedNode) : '—'}
              canEditSize={
                selectedNode?.type === 'Dense' || selectedNode?.type === 'Input'
              }
              sizeFieldLabel={selectedNode?.type === 'Input' ? 'Image Size' : 'Size'}
              canEditActivation={
                selectedNode?.type === 'Dense' || selectedNode?.type === 'Output'
              }
              canEditChannels={selectedNode?.type === 'Input'}
              canEditUnits={selectedNode?.type === 'Dense'}
              canEditDropoutRate={selectedNode?.type === 'Dropout'}
              canEditOutputClasses={selectedNode?.type === 'Output'}
              activationOptions={ACTIVATION_OPTIONS}
              layerCount={layerCount}
              neuronCount={neuronCount}
              weightCount={weightCount}
              biasCount={biasCount}
              layerTypeSummary={layerTypeSummary}
              sharedNonOutputActivation={sharedNonOutputActivation}
              onAddLayer={handleAddLayer}
              onSelectLayer={(nodeId) => selectNode(nodeId)}
              onBeginNameEdit={beginNameEdit}
              onDraftNameChange={(value) => setDraftName(value)}
              onNameCommit={commitNameEdit}
              onNameCancel={cancelNameEdit}
              onRowsChange={handleRowsChange}
              onColsChange={handleColsChange}
              onChannelsChange={handleChannelsChange}
              onUnitsChange={handleUnitsChange}
              onDropoutRateChange={handleDropoutRateChange}
              onOutputClassesChange={handleOutputClassesChange}
              onActivationChange={handleActivationChange}
              onValidate={handleValidateModel}
              buildStatus={buildStatus}
              buildStatusMessage={getBuildStatusMessage(buildStatus, buildIssues.length)}
              buildIssues={buildIssues}
              buildWarnings={buildWarnings}
              validateDisabled={isBackendBusy || layerCount === 0}
              validateLabel={backendBusyAction === 'validate' ? 'Building...' : 'Build'}
            />
          ) : null}

          {activeTab === 'train' ? (
            <TrainTab
              trainingConfig={trainingConfig}
              isBackendBusy={isBackendBusy}
              onDatasetChange={(value) => setTrainingConfig({ dataset: value })}
              onEpochsChange={(value) => setTrainingConfig({ epochs: value })}
              onBatchSizeChange={(value) => setTrainingConfig({ batchSize: value })}
              onOptimizerChange={(value) => setTrainingConfig({ optimizer: value })}
              onLearningRateChange={(value) => setTrainingConfig({ learningRate: value })}
              onLossChange={(value) => setTrainingConfig({ loss: value })}
              currentEpoch={currentEpoch}
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
              trainLabel={backendBusyAction === 'train' ? 'Training...' : 'Train'}
            />
          ) : null}

          {activeTab === 'infer' ? (
            <TestTab
              inferenceGrid={inferenceGrid}
              setInferenceGrid={setInferenceGrid}
              padDisabled={isBackendBusy || !trainingJobId || trainingStatus !== 'complete'}
              inferenceTopPrediction={inferenceTopPrediction}
              onInferModel={handleInferModel}
              inferDisabled={isBackendBusy || !trainingJobId || trainingStatus !== 'complete'}
              inferLabel={backendBusyAction === 'infer' ? 'Inferencing...' : 'Run Inference'}
            />
          ) : null}

          {activeTab === 'deploy' ? (
            <DeployTab
              trainingJobId={trainingJobId}
              trainingStatus={trainingStatus}
              deployment={deployment}
              deployTopPrediction={deployTopPrediction}
              deployOutput={deployOutput}
              onDeployModel={handleDeployModel}
              onRefreshDeployment={handleRefreshDeployment}
              onStopDeployment={handleStopDeployment}
              onInferDeployment={handleInferViaDeployment}
              deployDisabled={
                isBackendBusy ||
                !trainingJobId ||
                trainingStatus !== 'complete' ||
                deployment?.status === 'running'
              }
              refreshDisabled={isBackendBusy || !deployment}
              stopDisabled={isBackendBusy || !deployment || deployment.status !== 'running'}
              inferDisabled={isBackendBusy || !deployment || deployment.status !== 'running'}
              deployLabel={backendBusyAction === 'deploy' ? 'Deploying...' : 'Deploy Locally'}
              refreshLabel={backendBusyAction === 'deploy_status' ? 'Refreshing...' : 'Refresh'}
              stopLabel={backendBusyAction === 'deploy_stop' ? 'Stopping...' : 'Stop Deploy'}
              inferLabel={
                backendBusyAction === 'deploy_infer'
                  ? 'Running Endpoint...'
                  : 'Run Endpoint Inference'
              }
            />
          ) : null}
        </div>
      </section>

      <section className="app-viewport-panel">
        <button
          type="button"
          onClick={() => setIsSidebarCollapsed((prev) => !prev)}
          className="sidebar-toggle-button"
          aria-label={isSidebarCollapsed ? 'Expand left panel' : 'Collapse left panel'}
          title={isSidebarCollapsed ? 'Expand' : 'Collapse'}
        >
          {isSidebarCollapsed ? (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="16px"
              viewBox="0 -960 960 960"
              width="16px"
              fill="#e3e3e3"
            >
              <path d="m321-80-71-71 329-329-329-329 71-71 400 400L321-80Z" />
            </svg>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="16px"
              viewBox="0 -960 960 960"
              width="16px"
              fill="#e3e3e3"
            >
              <path d="M560-80 160-480l400-400 71 71-329 329 329 329-71 71Z" />
            </svg>
          )}
        </button>
        <Viewport lowDetailMode={isLowDetailMode} />
        <button
          type="button"
          onClick={() => setIsLowDetailMode((prev) => !prev)}
          className={`detail-mode-button ${
            isLowDetailMode ? 'detail-mode-button-active' : ''
          }`}
          aria-pressed={isLowDetailMode}
          aria-label="Toggle low detail mode"
          title="Reduce connection line detail for better performance"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            height="16px"
            viewBox="0 -960 960 960"
            width="16px"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M120-200v-80h240v80H120Zm0-200v-80h480v80H120Zm0-200v-80h720v80H120Z" />
          </svg>
          <span>{isLowDetailMode ? 'Low Detail' : 'Full Detail'}</span>
        </button>
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

function getInferenceGridSizeFromInputLayer(
  nodes: Record<string, LayerNode>,
  orderedNodeIds: string[],
  defaultRows: number,
  defaultCols: number
): { rows: number; cols: number } {
  const inputNodeId = orderedNodeIds[0]
  const inputNode = inputNodeId ? nodes[inputNodeId] : null
  if (!inputNode) {
    return { rows: defaultRows, cols: defaultCols }
  }

  const shape = inputNode.config.shape
  if (Array.isArray(shape) && shape.length >= 3) {
    const shapeRows = toPositiveInt(shape[1], 0)
    const shapeCols = toPositiveInt(shape[2], 0)
    if (shapeRows > 0 && shapeCols > 0) {
      return { rows: shapeRows, cols: shapeCols }
    }
  }

  return {
    rows: toPositiveInt(inputNode.config.rows, defaultRows),
    cols: toPositiveInt(inputNode.config.cols, defaultCols),
  }
}

function getBuildStatusMessage(status: BuildStatus, issueCount: number): string {
  if (status === 'success') {
    return 'Build passed. Graph is valid for backend compilation.'
  }
  if (status === 'error') {
    return `Build failed (${issueCount} issue${issueCount === 1 ? '' : 's'}).`
  }
  return 'Run Build to validate topology, layer order, and shape compatibility.'
}

function formatValidationIssue(issue: ValidationIssue): string {
  const baseParts: string[] = []
  if (issue.node_id) {
    baseParts.push(`[${issue.node_id}]`)
  }
  if (issue.message) {
    baseParts.push(issue.message)
  }

  let message = baseParts.join(' ').trim()
  if (message.length === 0) {
    message = 'Graph validation failed.'
  }

  const expected = formatCompactValue(issue.expected)
  const got = formatCompactValue(issue.got)
  if (expected || got) {
    message = `${message} | expected: ${expected ?? 'n/a'} | got: ${got ?? 'n/a'}`
  }
  return message
}

function formatCompactValue(value: unknown): string | null {
  if (value === null || value === undefined) return null
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)

  try {
    const serialized = JSON.stringify(value)
    if (!serialized) return null
    return serialized.length > 180 ? `${serialized.slice(0, 177)}...` : serialized
  } catch {
    return String(value)
  }
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

function getLayerTypeSummary(nodes: Record<string, LayerNode>): string {
  const nodeTypes = new Set(Object.values(nodes).map((node) => node.type))
  if (nodeTypes.size === 0) return 'None'
  if (nodeTypes.size === 1) return Array.from(nodeTypes)[0]
  return 'Mixed'
}

function getLayerSizeLabel(node: LayerNode): string {
  if (node.type === 'Input') {
    const shape = toShapeArray(node.config.shape)
    if (shape) {
      return shape.join(' x ')
    }
    return DEFAULT_INPUT_SHAPE.join(' x ')
  }

  if (node.type === 'Output') {
    const classes = toPositiveInt(node.config.num_classes, 0)
    if (classes > 0) return `${classes} classes`
    return `${DEFAULT_OUTPUT_CLASSES} classes`
  }

  if (node.type === 'Dropout') {
    const rate = Number(node.config.rate)
    if (Number.isFinite(rate)) {
      return `p=${rate.toFixed(2)}`
    }
    return 'p=0.50'
  }

  if (node.type === 'Flatten') {
    return 'Flatten'
  }

  const rows = toPositiveInt(node.config.rows, DEFAULT_LAYER_ROWS)
  const cols = toPositiveInt(node.config.cols, DEFAULT_LAYER_COLS)
  return `${rows} x ${cols}`
}

function toShapeArray(rawShape: unknown): number[] | null {
  if (!Array.isArray(rawShape) || rawShape.length === 0) return null
  const values = rawShape.map((value) => toPositiveInt(value, 0))
  if (values.some((value) => value <= 0)) return null
  return values
}

function toInputShapeOrDefault(rawShape: unknown): [number, number, number] {
  if (!Array.isArray(rawShape) || rawShape.length !== 3) {
    return [...DEFAULT_INPUT_SHAPE]
  }
  const parsed = rawShape.map((value) => toPositiveInt(value, 0))
  if (parsed.some((value) => value <= 0)) {
    return [...DEFAULT_INPUT_SHAPE]
  }
  return [parsed[0], parsed[1], parsed[2]]
}

function nodeUnits(node: LayerNode): number {
  const units = toPositiveInt(node.config.units, 0)
  if (units > 0) return units
  const rows = toPositiveInt(node.config.rows, DEFAULT_LAYER_ROWS)
  const cols = toPositiveInt(node.config.cols, DEFAULT_LAYER_COLS)
  return rows * cols
}

function denseGridFromUnits(units: number): { rows: number; cols: number } {
  if (units <= 1) return { rows: 1, cols: 1 }

  let bestRows = 1
  let bestCols = units
  let bestGap = bestCols - bestRows

  for (let rows = 1; rows <= Math.floor(Math.sqrt(units)); rows += 1) {
    if (units % rows !== 0) continue
    const cols = units / rows
    const gap = Math.abs(cols - rows)
    if (gap < bestGap) {
      bestGap = gap
      bestRows = rows
      bestCols = cols
    }
  }

  return { rows: bestRows, cols: bestCols }
}

function clampDropoutRate(value: unknown): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return 0.5
  if (parsed < 0) return 0
  if (parsed >= 1) return 0.99
  return Number(parsed.toFixed(4))
}

function computeGraphStats(
  nodes: Record<string, LayerNode>,
  orderedNodeIds: string[]
): {
  neuronCount: number
  weightCount: number
  biasCount: number
} {
  let neuronCount = 0
  let weightCount = 0
  let biasCount = 0
  let previousOutputShape: number[] | null = null

  for (const nodeId of orderedNodeIds) {
    const node = nodes[nodeId]
    if (!node) continue

    const inputShape = node.type === 'Input' ? null : previousOutputShape
    const outputShape = inferNodeOutputShape(node, inputShape)
    if (outputShape !== null) {
      neuronCount += product(outputShape)
    }

    if ((node.type === 'Dense' || node.type === 'Output') && inputShape && outputShape) {
      if (inputShape.length === 1 && outputShape.length === 1) {
        const inFeatures = inputShape[0]
        const outFeatures = outputShape[0]
        weightCount += inFeatures * outFeatures
        biasCount += outFeatures
      }
    }

    previousOutputShape = outputShape
  }

  return { neuronCount, weightCount, biasCount }
}

function inferNodeOutputShape(
  node: LayerNode,
  inputShape: number[] | null
): number[] | null {
  if (node.type === 'Input') {
    return toInputShapeOrDefault(node.config.shape)
  }

  if (!inputShape) return null

  if (node.type === 'Flatten') {
    return [product(inputShape)]
  }

  if (node.type === 'Dense') {
    if (inputShape.length !== 1) return null
    const units = toPositiveInt(node.config.units, 0)
    if (units > 0) return [units]

    const rows = toPositiveInt(node.config.rows, DEFAULT_LAYER_ROWS)
    const cols = toPositiveInt(node.config.cols, DEFAULT_LAYER_COLS)
    return [rows * cols]
  }

  if (node.type === 'Dropout') {
    return [...inputShape]
  }

  if (node.type === 'Output') {
    if (inputShape.length !== 1) return null
    return [toPositiveInt(node.config.num_classes, DEFAULT_OUTPUT_CLASSES)]
  }

  return null
}

function product(values: number[]): number {
  return values.reduce((acc, value) => acc * value, 1)
}
