import { useEffect, useMemo, useRef, useState } from 'react'
import { MetricLineChart, MetricTile } from '../ui/tabs/Metrics'
import {
  getLinearRegressionDataset,
  type LinearRegressionDataset,
} from './datasets'
import './linreg.css'

type DashboardTab = 'build' | 'train' | 'test' | 'deploy'
type BuildStatus = 'idle' | 'success' | 'error'
type TrainingStatus = 'idle' | 'training' | 'complete' | 'stopped' | 'error'
type LayerRole = 'input' | 'hidden' | 'output'
type LinRegNodeType = 'Input' | 'Normalize' | 'LinearRegressor' | 'Output'

interface BuilderNode {
  id: string
  type: LinRegNodeType
  role: LayerRole
  name: string
  sizeLabel: string
  subtitle: string
}

interface TrainingMetric {
  epoch: number
  trainLoss: number
  testLoss: number
  trainMae: number
  testMae: number
  trainR2: number
  testR2: number
}

interface TrainedModel {
  datasetId: string
  featureNames: string[]
  targetName: string
  weights: number[]
  bias: number
  includeNormalization: boolean
  means: number[]
  stds: number[]
  fitIntercept: boolean
  l2Penalty: number
  trainSplitCount: number
  testSplitCount: number
}

interface DeploymentState {
  deployment_id: string
  job_id: string
  status: string
  target: string
  endpoint_path: string
  created_at: string
  last_used_at?: string | null
  request_count: number
  name?: string | null
  model_family?: string
}

interface DatasetSplit {
  trainX: number[][]
  trainY: number[]
  testX: number[][]
  testY: number[]
}

interface NormalizationStats {
  means: number[]
  stds: number[]
}

export interface LinearRegressionInitialConfig {
  datasetId: string
  includeNormalization: boolean
  fitIntercept: boolean
  l2Penalty: number
  epochs: number
  learningRate: number
  testSplit: number
  randomSeed: number
}

const DEFAULT_INITIAL_CONFIG: LinearRegressionInitialConfig = {
  datasetId: 'study_hours',
  includeNormalization: true,
  fitIntercept: true,
  l2Penalty: 0,
  epochs: 420,
  learningRate: 0.03,
  testSplit: 0.2,
  randomSeed: 42,
}

const COPY_FEEDBACK_MS = 1800
const MIN_TEST_SPLIT = 0.05
const MAX_TEST_SPLIT = 0.45

const NODE_IDS = {
  input: 'linreg_input',
  normalize: 'linreg_normalize',
  linear: 'linreg_linear',
  output: 'linreg_output',
} as const

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
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
    throw new Error(extractErrorMessage(text, response.status))
  }
  return JSON.parse(text) as T
}

function extractErrorMessage(raw: string, status: number): string {
  if (!raw) return `HTTP ${status}`
  try {
    const parsed = JSON.parse(raw) as {
      detail?: { message?: string } | string
      message?: string
    }
    if (typeof parsed.detail === 'string') return parsed.detail
    if (parsed.detail?.message) return parsed.detail.message
    if (parsed.message) return parsed.message
    return raw
  } catch {
    return raw
  }
}

interface LinearRegressionPageProps {
  initialConfig?: Partial<LinearRegressionInitialConfig>
}

export default function LinearRegressionPage({ initialConfig }: LinearRegressionPageProps) {
  const resolvedInitial = {
    ...DEFAULT_INITIAL_CONFIG,
    ...(initialConfig ?? {}),
  }

  const [datasetId, setDatasetId] = useState(resolvedInitial.datasetId)
  const [includeNormalization, setIncludeNormalization] = useState(resolvedInitial.includeNormalization)
  const [fitIntercept, setFitIntercept] = useState(resolvedInitial.fitIntercept)
  const [l2Penalty, setL2Penalty] = useState(resolvedInitial.l2Penalty)
  const [epochs, setEpochs] = useState(resolvedInitial.epochs)
  const [learningRate, setLearningRate] = useState(resolvedInitial.learningRate)
  const [testSplit, setTestSplit] = useState(resolvedInitial.testSplit)
  const [randomSeed, setRandomSeed] = useState(resolvedInitial.randomSeed)

  const [nodeNameOverrides, setNodeNameOverrides] = useState<Record<string, string>>({})
  const [selectedNodeId, setSelectedNodeId] = useState<string>(NODE_IDS.input)
  const [editingNodeId, setEditingNodeId] = useState<string | null>(null)
  const [draftName, setDraftName] = useState('')

  const [activeTab, setActiveTab] = useState<DashboardTab>('build')
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)

  const [buildStatus, setBuildStatus] = useState<BuildStatus>('idle')
  const [buildIssues, setBuildIssues] = useState<string[]>([])
  const [buildWarnings, setBuildWarnings] = useState<string[]>([])
  const [hasBuiltModel, setHasBuiltModel] = useState(false)

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>('idle')
  const [trainingMessage, setTrainingMessage] = useState('Build the pipeline, then start training.')
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([])
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [trainedModel, setTrainedModel] = useState<TrainedModel | null>(null)

  const [inferenceInputs, setInferenceInputs] = useState<number[]>(() => {
    const initialDataset = getLinearRegressionDataset(resolvedInitial.datasetId)
    const first = initialDataset.samples[0]
    if (first) return first.slice()
    return new Array(initialDataset.featureNames.length).fill(0)
  })
  const [inferenceActualInput, setInferenceActualInput] = useState('')
  const [selectedSampleIndex, setSelectedSampleIndex] = useState(0)
  const [inferenceTopPrediction, setInferenceTopPrediction] = useState<number | null>(null)
  const [inferenceOutput, setInferenceOutput] = useState('No inference output yet.')

  const [deployment, setDeployment] = useState<DeploymentState | null>(null)
  const [deployTopPrediction, setDeployTopPrediction] = useState<number | null>(null)
  const [deployOutput, setDeployOutput] = useState('No deployed inference output yet.')
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')

  const stopTrainingRef = useRef(false)

  const dataset = useMemo(
    () => getLinearRegressionDataset(datasetId),
    [datasetId]
  )

  const nodes = useMemo<BuilderNode[]>(() => {
    const next: BuilderNode[] = [
      {
        id: NODE_IDS.input,
        type: 'Input',
        role: 'input',
        name: nodeNameOverrides[NODE_IDS.input] ?? 'Input Layer',
        sizeLabel: `${dataset.featureNames.length} feature${dataset.featureNames.length === 1 ? '' : 's'}`,
        subtitle: dataset.featureNames.join(', '),
      },
    ]

    if (includeNormalization) {
      next.push({
        id: NODE_IDS.normalize,
        type: 'Normalize',
        role: 'hidden',
        name: nodeNameOverrides[NODE_IDS.normalize] ?? 'Normalize (Z-Score)',
        sizeLabel: `${dataset.featureNames.length}x mean/std`,
        subtitle: 'Train-set mean/std scaling',
      })
    }

    next.push({
      id: NODE_IDS.linear,
      type: 'LinearRegressor',
      role: 'hidden',
      name: nodeNameOverrides[NODE_IDS.linear] ?? 'Linear Regressor',
      sizeLabel: `${dataset.featureNames.length} weight${dataset.featureNames.length === 1 ? '' : 's'}${fitIntercept ? ' + bias' : ''}`,
      subtitle: `MSE + L2(${trimNumber(l2Penalty, 4)})`,
    })

    next.push({
      id: NODE_IDS.output,
      type: 'Output',
      role: 'output',
      name: nodeNameOverrides[NODE_IDS.output] ?? 'Output',
      sizeLabel: '1 value',
      subtitle: dataset.targetName,
    })

    return next
  }, [dataset.featureNames, dataset.targetName, fitIntercept, includeNormalization, l2Penalty, nodeNameOverrides])

  const selectedNode = useMemo(
    () => nodes.find((node) => node.id === selectedNodeId) ?? null,
    [nodes, selectedNodeId]
  )

  const isEditingName = Boolean(selectedNode && editingNodeId === selectedNode.id)

  useEffect(() => {
    return () => {
      stopTrainingRef.current = true
    }
  }, [])

  useEffect(() => {
    if (copyStatus === 'idle') return
    const timer = window.setTimeout(() => setCopyStatus('idle'), COPY_FEEDBACK_MS)
    return () => window.clearTimeout(timer)
  }, [copyStatus])

  const layerCount = nodes.length
  const featureCount = dataset.featureNames.length
  const parameterCount = featureCount + (fitIntercept ? 1 : 0)
  const sampleCount = dataset.samples.length

  const latestMetric = trainingMetrics[trainingMetrics.length - 1]
  const trainLossSeries = trainingMetrics.map((metric) => metric.trainLoss)
  const testLossSeries = trainingMetrics.map((metric) => metric.testLoss)
  const trainR2Series = trainingMetrics.map((metric) => metric.trainR2)
  const testR2Series = trainingMetrics.map((metric) => metric.testR2)

  const lossBounds = getBoundsFromSeries(trainLossSeries, testLossSeries, 0)
  const r2Bounds = getBoundsFromSeries(trainR2Series, testR2Series, -1)

  const canOpenTrainTab = hasBuiltModel || trainingStatus !== 'idle' || trainingMetrics.length > 0
  const canOpenTestTab = trainedModel !== null
  const canOpenDeployTab = trainedModel !== null || deployment !== null
  const isTraining = trainingStatus === 'training'

  const activeTabIndex =
    activeTab === 'build' ? 0 : activeTab === 'train' ? 1 : activeTab === 'test' ? 2 : 3

  const endpointUrl = deployment
    ? `${resolveBackendBaseUrl()}${deployment.endpoint_path}`
    : 'http://127.0.0.1:8000/api/deploy/<deployment_id>/infer'
  const endpointLiteral = toPythonSingleQuotedString(endpointUrl)
  const pythonSnippet = useMemo(() => {
    const payloadInputs =
      inferenceInputs.length > 0
        ? `[${inferenceInputs.map((value) => trimNumber(value, 5)).join(', ')}]`
        : `[${dataset.featureNames.map(() => '0.0').join(', ')}]`

    return [
      '# pip install requests',
      'import requests',
      '',
      `endpoint = ${endpointLiteral}`,
      'payload = {',
      `    'inputs': ${payloadInputs},`,
      "    'return_probabilities': False,",
      '}',
      '',
      'response = requests.post(endpoint, json=payload, timeout=30)',
      'response.raise_for_status()',
      'result = response.json()',
      "print('prediction:', result['predictions'][0])",
      '',
    ].join('\n')
  }, [dataset.featureNames, endpointLiteral, inferenceInputs])

  const invalidatePipelineState = (message: string) => {
    setHasBuiltModel(false)
    setBuildStatus('idle')
    setBuildIssues([])
    setBuildWarnings([])
    setTrainingStatus('idle')
    setCurrentEpoch(0)
    setTrainingMetrics([])
    setTrainedModel(null)
    setDeployment(null)
    setTrainingMessage(message)
  }

  const updateDataset = (nextDatasetId: string) => {
    const nextDataset = getLinearRegressionDataset(nextDatasetId)
    setDatasetId(nextDataset.id)
    const firstSample = nextDataset.samples[0]?.slice() ?? new Array(nextDataset.featureNames.length).fill(0)
    setInferenceInputs(firstSample)
    setSelectedSampleIndex(0)
    setInferenceActualInput('')
    setInferenceTopPrediction(null)
    setInferenceOutput('No inference output yet.')
    setDeployTopPrediction(null)
    setDeployOutput('No deployed inference output yet.')
    invalidatePipelineState('Dataset changed. Build again before training.')
  }

  const updateNormalizationEnabled = (value: boolean) => {
    setIncludeNormalization(value)
    if (!value && selectedNodeId === NODE_IDS.normalize) {
      setSelectedNodeId(NODE_IDS.linear)
    }
    invalidatePipelineState('Pipeline changed. Build again before training.')
  }

  const updateFitIntercept = (value: boolean) => {
    setFitIntercept(value)
    invalidatePipelineState('Pipeline changed. Build again before training.')
  }

  const updateL2Penalty = (value: number) => {
    setL2Penalty(value)
    invalidatePipelineState('Pipeline changed. Build again before training.')
  }

  const handleAddLayer = () => {
    if (includeNormalization) {
      setBuildStatus('error')
      setBuildIssues(['Only one Normalize layer is supported for Linear Regression in this MVP.'])
      setBuildWarnings([])
      return
    }

    invalidatePipelineState('Pipeline changed. Build again before training.')
    setIncludeNormalization(true)
    setSelectedNodeId(NODE_IDS.normalize)
    setBuildStatus('idle')
    setBuildIssues([])
    setBuildWarnings(['Normalize layer inserted before the Linear Regressor.'])
  }

  const beginNameEdit = () => {
    if (!selectedNode) return
    setEditingNodeId(selectedNode.id)
    setDraftName(selectedNode.name)
  }

  const commitNameEdit = () => {
    if (!selectedNode || editingNodeId !== selectedNode.id) {
      setEditingNodeId(null)
      setDraftName('')
      return
    }

    const trimmed = draftName.trim()
    setNodeNameOverrides((current) => {
      const next = { ...current }
      if (trimmed.length === 0) {
        delete next[selectedNode.id]
      } else {
        next[selectedNode.id] = trimmed
      }
      return next
    })
    setEditingNodeId(null)
    setDraftName('')
  }

  const cancelNameEdit = () => {
    setEditingNodeId(null)
    setDraftName('')
  }

  const runBuild = () => {
    const errors: string[] = []
    const warnings: string[] = []

    if (dataset.samples.length < 8) {
      errors.push('Dataset needs at least 8 samples to split train/test reliably.')
    }
    if (dataset.featureNames.length === 0) {
      errors.push('Input layer must expose at least one feature.')
    }
    if (dataset.targets.length !== dataset.samples.length) {
      errors.push('Dataset features/targets length mismatch.')
    }
    if (l2Penalty < 0) {
      errors.push('L2 penalty must be >= 0.')
    }

    const boundedTestSplit = clamp(testSplit, MIN_TEST_SPLIT, MAX_TEST_SPLIT)
    if (Math.abs(testSplit - boundedTestSplit) > 1e-9) {
      warnings.push(
        `Test split was clamped to ${trimNumber(boundedTestSplit, 2)} (allowed ${MIN_TEST_SPLIT} - ${MAX_TEST_SPLIT}).`
      )
      setTestSplit(boundedTestSplit)
    }

    if (!includeNormalization) {
      warnings.push('Normalize layer is disabled. Convergence can be slower for wide feature ranges.')
    }
    if (!fitIntercept) {
      warnings.push('Intercept disabled. Model will be forced through origin.')
    }

    if (errors.length > 0) {
      setBuildStatus('error')
      setBuildIssues(errors)
      setBuildWarnings(warnings)
      setHasBuiltModel(false)
      return
    }

    setBuildStatus('success')
    setBuildIssues([])
    setBuildWarnings(warnings)
    setHasBuiltModel(true)
    setTrainingMessage('Build succeeded. Start training to fit coefficients.')
    setActiveTab('train')
  }

  const runTraining = async () => {
    if (isTraining) return
    if (!hasBuiltModel) {
      setTrainingStatus('error')
      setTrainingMessage('Run Build before training.')
      return
    }

    const boundedEpochs = clampInt(epochs, 1, 10000)
    const boundedLearningRate = clamp(learningRate, 0.000001, 5)
    const boundedSplit = clamp(testSplit, MIN_TEST_SPLIT, MAX_TEST_SPLIT)
    const boundedSeed = clampInt(randomSeed, -999999, 999999)

    setEpochs(boundedEpochs)
    setLearningRate(boundedLearningRate)
    setTestSplit(boundedSplit)
    setRandomSeed(boundedSeed)

    const split = splitDataset(dataset.samples, dataset.targets, boundedSplit, boundedSeed)
    const normalization = includeNormalization
      ? computeNormalization(split.trainX)
      : createNoopNormalization(featureCount)

    const trainX = applyNormalization(split.trainX, normalization)
    const testX = applyNormalization(split.testX, normalization)

    let weights = new Array(featureCount).fill(0)
    let bias = 0

    stopTrainingRef.current = false
    setTrainingStatus('training')
    setTrainingMessage(`Training for ${boundedEpochs} epochs...`)
    setCurrentEpoch(0)
    setTrainingMetrics([])
    setTrainedModel(null)
    setDeployment(null)
    setDeployTopPrediction(null)
    setDeployOutput('No deployed inference output yet.')

    try {
      for (let epoch = 1; epoch <= boundedEpochs; epoch += 1) {
        if (stopTrainingRef.current) {
          setTrainingStatus('stopped')
          setTrainingMessage(`Training stopped at epoch ${epoch - 1}.`)
          return
        }

        const step = gradientStep({
          x: trainX,
          y: split.trainY,
          weights,
          bias,
          learningRate: boundedLearningRate,
          fitIntercept,
          l2Penalty,
        })
        weights = step.weights
        bias = step.bias

        const trainPredictions = predictBatch(trainX, weights, bias)
        const testPredictions = predictBatch(testX, weights, bias)

        const metric: TrainingMetric = {
          epoch,
          trainLoss: meanSquaredError(split.trainY, trainPredictions),
          testLoss: meanSquaredError(split.testY, testPredictions),
          trainMae: meanAbsoluteError(split.trainY, trainPredictions),
          testMae: meanAbsoluteError(split.testY, testPredictions),
          trainR2: r2Score(split.trainY, trainPredictions),
          testR2: r2Score(split.testY, testPredictions),
        }

        setTrainingMetrics((current) => [...current, metric])
        setCurrentEpoch(epoch)

        if (epoch % 3 === 0 || epoch === boundedEpochs) {
          setTrainingMessage(
            `Epoch ${epoch}/${boundedEpochs} · train_mse=${trimNumber(metric.trainLoss, 4)} · test_mse=${trimNumber(metric.testLoss, 4)}`
          )
          await nextFrame()
        }
      }

      const finalModel: TrainedModel = {
        datasetId: dataset.id,
        featureNames: dataset.featureNames,
        targetName: dataset.targetName,
        weights,
        bias,
        includeNormalization,
        means: normalization.means,
        stds: normalization.stds,
        fitIntercept,
        l2Penalty,
        trainSplitCount: split.trainX.length,
        testSplitCount: split.testX.length,
      }

      setTrainedModel(finalModel)
      setTrainingStatus('complete')
      setTrainingMessage(
        `Training complete. Final test MSE ${trimNumber(
          meanSquaredError(split.testY, predictBatch(testX, weights, bias)),
          5
        )}.`
      )
      setActiveTab('test')
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      setTrainingStatus('error')
      setTrainingMessage(`Training failed: ${text}`)
    }
  }

  const stopTraining = () => {
    if (!isTraining) return
    stopTrainingRef.current = true
    setTrainingMessage('Stopping training...')
  }

  const runInference = (source: 'test' | 'deploy'): boolean => {
    if (!trainedModel) {
      const message = 'Train the model before running inference.'
      if (source === 'deploy') {
        setDeployOutput(message)
      } else {
        setInferenceOutput(message)
      }
      return false
    }

    if (inferenceInputs.length !== trainedModel.featureNames.length) {
      const message = `Input size mismatch. Expected ${trainedModel.featureNames.length} values.`
      if (source === 'deploy') {
        setDeployOutput(message)
      } else {
        setInferenceOutput(message)
      }
      return false
    }

    const prediction = predictWithModel(inferenceInputs, trainedModel)
    const maybeActual = Number(inferenceActualInput)
    const hasActual = Number.isFinite(maybeActual)
    const residual = hasActual ? maybeActual - prediction : null

    const payload = {
      dataset: trainedModel.datasetId,
      feature_names: trainedModel.featureNames,
      inputs: inferenceInputs,
      prediction,
      actual: hasActual ? maybeActual : null,
      residual,
      output_name: trainedModel.targetName,
      model: {
        weights: trainedModel.weights,
        bias: trainedModel.bias,
        normalization: trainedModel.includeNormalization,
      },
    }

    if (source === 'deploy') {
      setDeployTopPrediction(prediction)
      setDeployOutput(JSON.stringify(payload, null, 2))
      return true
    }

    setInferenceTopPrediction(prediction)
    setInferenceOutput(JSON.stringify(payload, null, 2))
    return true
  }

  const handleLoadSample = () => {
    if (dataset.samples.length === 0) return
    const next = (selectedSampleIndex + 1) % dataset.samples.length
    setSelectedSampleIndex(next)
    setInferenceInputs(dataset.samples[next].slice())
    const target = dataset.targets[next]
    setInferenceActualInput(String(target))
  }

  const deployModel = async () => {
    if (!trainedModel) {
      setDeployOutput('Train the model before deploying.')
      return
    }

    const fallbackJobId = createLinregJobId(dataset.id)
    try {
      const response = await requestJson<DeploymentState>('/api/deploy/external', {
        method: 'POST',
        body: JSON.stringify({
          model_family: 'linreg',
          target: 'local',
          name: `${dataset.name} LinReg Endpoint`,
          job_id: fallbackJobId,
          runtime_config: {
            weights: trainedModel.weights,
            bias: trainedModel.bias,
            means: trainedModel.means,
            stds: trainedModel.stds,
            feature_names: trainedModel.featureNames,
            target_name: trainedModel.targetName,
          },
        }),
      })
      setDeployment(response)
      setDeployOutput('Deployment started locally and registered in Deployment Manager.')
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      setDeployOutput(`Failed to deploy: ${text}`)
    }
  }

  const refreshDeployment = async () => {
    if (!deployment) {
      setDeployOutput('No deployment found. Deploy model first.')
      return
    }

    try {
      const status = await requestJson<DeploymentState>(
        `/api/deploy/status?deployment_id=${encodeURIComponent(deployment.deployment_id)}`
      )
      setDeployment(status)
      setDeployOutput(
        `Deployment ${status.deployment_id} is ${status.status}. Requests served: ${status.request_count}.`
      )
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      setDeployOutput(`Failed to refresh deployment: ${text}`)
    }
  }

  const stopDeployment = async () => {
    if (!deployment) {
      setDeployOutput('No deployment found.')
      return
    }

    try {
      const stopped = await requestJson<DeploymentState>(
        `/api/deploy/${encodeURIComponent(deployment.deployment_id)}`,
        { method: 'DELETE' }
      )
      setDeployment(stopped)
      setDeployOutput(`Deployment ${stopped.deployment_id} stopped.`)
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      setDeployOutput(`Failed to stop deployment: ${text}`)
    }
  }

  const runDeploymentInference = async () => {
    if (!deployment) {
      setDeployOutput('No deployment found. Deploy model first.')
      return
    }
    if (deployment.status !== 'running') {
      setDeployOutput('Deployment is stopped. Start a new deployment first.')
      return
    }

    const success = runInference('deploy')
    if (!success) return

    try {
      const inferResponse = await requestJson<{
        predictions?: number[]
      }>(
        `/api/deploy/${encodeURIComponent(deployment.deployment_id)}/infer`,
        {
          method: 'POST',
          body: JSON.stringify({
            inputs: inferenceInputs,
            return_probabilities: false,
          }),
        }
      )
      if (Array.isArray(inferResponse.predictions) && inferResponse.predictions.length > 0) {
        setDeployTopPrediction(Number(inferResponse.predictions[0]))
      }
      setDeployOutput(JSON.stringify(inferResponse, null, 2))

      const latest = await requestJson<DeploymentState>(
        `/api/deploy/status?deployment_id=${encodeURIComponent(deployment.deployment_id)}`
      )
      setDeployment(latest)
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      setDeployOutput(`Deployment inference failed: ${text}`)
    }
  }

  const copyPythonSnippet = async () => {
    const success = await copyToClipboard(pythonSnippet)
    setCopyStatus(success ? 'copied' : 'failed')
  }

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
              onClick={() => setActiveTab('build')}
              className={`app-tab-button ${
                activeTab === 'build' ? 'app-tab-button-active' : 'app-tab-button-inactive'
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
              disabled={!canOpenTestTab}
              onClick={() => {
                if (!canOpenTestTab) return
                setActiveTab('test')
              }}
              className={`app-tab-button ${
                activeTab === 'test' ? 'app-tab-button-active' : 'app-tab-button-inactive'
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

          {activeTab === 'build' ? (
            <div className="tab-panel">
              <section className="panel-card panel-card-layers">
                <div className="layer-list-shell">
                  <ol className="layer-list">
                    {nodes.map((node) => {
                      const isSelected = node.id === selectedNodeId
                      return (
                        <li
                          key={node.id}
                          className={`layer-list-item ${isSelected ? 'layer-list-item-active' : ''}`}
                          role="button"
                          tabIndex={0}
                          onClick={() => setSelectedNodeId(node.id)}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter' || event.key === ' ') {
                              event.preventDefault()
                              setSelectedNodeId(node.id)
                            }
                          }}
                        >
                          <div className="layer-list-item-left">
                            <span className={getRoleDotClass(node.role)} />
                            <span className="layer-list-item-name">{node.name}</span>
                          </div>
                          <span className="layer-list-item-size">{node.sizeLabel}</span>
                        </li>
                      )
                    })}

                    <li className="layer-list-add-item">
                      <button
                        onClick={handleAddLayer}
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

              {selectedNode ? (
                <section className="panel-card layer-editor-card">
                  <div className="layer-editor-header">
                    {isEditingName ? (
                      <input
                        value={draftName}
                        autoFocus
                        onChange={(event) => setDraftName(event.target.value)}
                        onBlur={commitNameEdit}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter') {
                            event.preventDefault()
                            commitNameEdit()
                          } else if (event.key === 'Escape') {
                            event.preventDefault()
                            cancelNameEdit()
                          }
                        }}
                        className="layer-name-input"
                      />
                    ) : (
                      <button
                        type="button"
                        onClick={beginNameEdit}
                        className="layer-name-button"
                      >
                        {selectedNode.name}
                      </button>
                    )}
                  </div>
                  <p className="panel-muted-text panel-muted-text-tight">Type: {selectedNode.type}</p>

                  {selectedNode.type === 'Input' ? (
                    <div className="linreg-editor-grid">
                      <label className="field-group field-group-inline">
                        <span className="field-label">Dataset</span>
                        <select
                          value={datasetId}
                          onChange={(event) => updateDataset(event.target.value)}
                          className="activation-select"
                          disabled={isTraining}
                        >
                          <option value="study_hours">Study Hours</option>
                          <option value="home_value_tiny">Home Value Tiny</option>
                        </select>
                      </label>

                      <div className="field-group field-group-inline">
                        <span className="field-label">Feature Shape</span>
                        <div className="field-readonly">[{dataset.featureNames.length}]</div>
                      </div>
                    </div>
                  ) : null}

                  {selectedNode.type === 'Normalize' ? (
                    <div className="linreg-editor-grid">
                      <label className="field-group field-group-inline field-group-checkbox">
                        <span className="field-label">Enabled</span>
                        <input
                          type="checkbox"
                          checked={includeNormalization}
                          onChange={(event) => updateNormalizationEnabled(event.target.checked)}
                        />
                      </label>

                      <div className="field-group field-group-inline">
                        <span className="field-label">Mode</span>
                        <div className="field-readonly">z-score</div>
                      </div>
                    </div>
                  ) : null}

                  {selectedNode.type === 'LinearRegressor' ? (
                    <div className="linreg-editor-grid">
                      <label className="field-group field-group-inline field-group-checkbox">
                        <span className="field-label">Fit Intercept</span>
                        <input
                          type="checkbox"
                          checked={fitIntercept}
                          onChange={(event) => updateFitIntercept(event.target.checked)}
                          disabled={isTraining}
                        />
                      </label>

                      <label className="field-group field-group-inline">
                        <span className="field-label">L2 Penalty</span>
                        <input
                          type="number"
                          min={0}
                          step={0.0001}
                          value={l2Penalty}
                          onChange={(event) => updateL2Penalty(Math.max(0, Number(event.target.value) || 0))}
                          className="size-input"
                          disabled={isTraining}
                        />
                      </label>
                    </div>
                  ) : null}

                  {selectedNode.type === 'Output' ? (
                    <div className="linreg-editor-grid">
                      <div className="field-group field-group-inline">
                        <span className="field-label">Target</span>
                        <div className="field-readonly">{dataset.targetName}</div>
                      </div>

                      <div className="field-group field-group-inline">
                        <span className="field-label">Value Type</span>
                        <div className="field-readonly">Continuous scalar</div>
                      </div>
                    </div>
                  ) : null}
                </section>
              ) : null}

              <section className="panel-card build-summary-card">
                <div className="summary-grid">
                  <MetricTile label="Layers" value={String(layerCount)} />
                  <MetricTile label="Features" value={String(featureCount)} />
                  <MetricTile label="Parameters" value={String(parameterCount)} />
                  <MetricTile label="Samples" value={String(sampleCount)} />
                  <MetricTile label="Model Type" value="Linear Regression" />
                  <MetricTile
                    label="Preprocessing"
                    value={includeNormalization ? 'Normalize' : 'None'}
                  />
                </div>
              </section>

              <section className="panel-card build-feedback-card">
                <div className={`build-feedback-status build-feedback-status-${buildStatus}`}>
                  {getBuildStatusMessage(buildStatus, buildIssues.length)}
                </div>

                {buildWarnings.length > 0 ? (
                  <ul className="build-feedback-list">
                    {buildWarnings.map((warning, index) => (
                      <li
                        key={`build-warning-${index}`}
                        className="build-feedback-item build-feedback-item-warning"
                      >
                        {warning}
                      </li>
                    ))}
                  </ul>
                ) : null}

                {buildIssues.length > 0 ? (
                  <ul className="build-feedback-list">
                    {buildIssues.map((issue, index) => (
                      <li
                        key={`build-issue-${index}`}
                        className="build-feedback-item build-feedback-item-error"
                      >
                        {issue}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </section>

              <div className="panel-actions">
                <button
                  onClick={runBuild}
                  className="btn btn-validate"
                  disabled={isTraining}
                >
                  Build
                </button>
              </div>
            </div>
          ) : null}

          {activeTab === 'train' ? (
            <div className="tab-panel">
              <section className="panel-card panel-card-fill">
                <div className="linreg-train-graphs">
                  <div className="linreg-train-chart">
                    <h3 className="panel-subtitle">Loss (MSE)</h3>
                    <div className="panel-chart">
                      <MetricLineChart
                        primaryLabel="Train MSE"
                        secondaryLabel="Test MSE"
                        primaryValues={trainLossSeries}
                        secondaryValues={testLossSeries}
                        minValue={lossBounds.min}
                        maxValue={lossBounds.max}
                        primaryColor="#ffb429"
                        secondaryColor="#ffd89c"
                        xAxisLabel="Epoch"
                        yAxisLabel="MSE"
                      />
                    </div>
                  </div>

                  <div className="linreg-train-chart">
                    <h3 className="panel-subtitle">R² Score</h3>
                    <div className="panel-chart">
                      <MetricLineChart
                        primaryLabel="Train R²"
                        secondaryLabel="Test R²"
                        primaryValues={trainR2Series}
                        secondaryValues={testR2Series}
                        minValue={r2Bounds.min}
                        maxValue={r2Bounds.max}
                        primaryColor="#79d7ff"
                        secondaryColor="#6fffc8"
                        xAxisLabel="Epoch"
                        yAxisLabel="R²"
                      />
                    </div>
                  </div>
                </div>

                <div className="linreg-train-metrics">
                  <MetricTile label="Epoch" value={currentEpoch > 0 ? String(currentEpoch) : '—'} compact />
                  <MetricTile
                    label="Train MAE"
                    value={latestMetric ? trimNumber(latestMetric.trainMae, 4) : '—'}
                    compact
                  />
                  <MetricTile
                    label="Test MAE"
                    value={latestMetric ? trimNumber(latestMetric.testMae, 4) : '—'}
                    compact
                  />
                  <MetricTile
                    label="Status"
                    value={trainingStatus}
                    compact
                  />
                </div>

                <p className="linreg-train-status-line">{trainingMessage}</p>
              </section>

              <section className="panel-card train-settings-card">
                <div className="config-grid train-config-grid">
                  <label className="config-row">
                    <span className="config-label">Dataset</span>
                    <select
                      value={datasetId}
                      onChange={(event) => updateDataset(event.target.value)}
                      className="config-control"
                      disabled={isTraining}
                    >
                      <option value="study_hours">Study Hours</option>
                      <option value="home_value_tiny">Home Value Tiny</option>
                    </select>
                  </label>

                  <label className="config-row">
                    <span className="config-label">Epochs</span>
                    <input
                      type="number"
                      min={1}
                      max={10000}
                      value={epochs}
                      onChange={(event) => setEpochs(clampInt(Number(event.target.value) || 1, 1, 10000))}
                      className="config-control config-control-numeric"
                      disabled={isTraining}
                    />
                  </label>

                  <label className="config-row">
                    <span className="config-label">Learning Rate</span>
                    <input
                      type="number"
                      min={0.000001}
                      max={5}
                      step={0.0001}
                      value={learningRate}
                      onChange={(event) => setLearningRate(clamp(Number(event.target.value) || 0, 0.000001, 5))}
                      className="config-control config-control-numeric"
                      disabled={isTraining}
                    />
                  </label>

                  <label className="config-row">
                    <span className="config-label">Test Split</span>
                    <input
                      type="number"
                      min={MIN_TEST_SPLIT}
                      max={MAX_TEST_SPLIT}
                      step={0.01}
                      value={testSplit}
                      onChange={(event) => setTestSplit(clamp(Number(event.target.value) || 0, MIN_TEST_SPLIT, MAX_TEST_SPLIT))}
                      className="config-control config-control-numeric"
                      disabled={isTraining}
                    />
                  </label>

                  <label className="config-row">
                    <span className="config-label">Random Seed</span>
                    <input
                      type="number"
                      value={randomSeed}
                      onChange={(event) => setRandomSeed(clampInt(Number(event.target.value) || 0, -999999, 999999))}
                      className="config-control config-control-numeric"
                      disabled={isTraining}
                    />
                  </label>

                  <label className="config-row">
                    <span className="config-label">Normalize</span>
                    <select
                      value={includeNormalization ? 'true' : 'false'}
                      onChange={(event) => updateNormalizationEnabled(event.target.value === 'true')}
                      className="config-control"
                      disabled={isTraining}
                    >
                      <option value="true">Enabled</option>
                      <option value="false">Disabled</option>
                    </select>
                  </label>
                </div>
              </section>

              <div className="panel-actions">
                {isTraining ? (
                  <button
                    onClick={stopTraining}
                    className="btn btn-validate btn-danger"
                  >
                    Stop Training
                  </button>
                ) : (
                  <button
                    onClick={() => void runTraining()}
                    className="btn btn-validate"
                    disabled={!hasBuiltModel}
                  >
                    Train
                  </button>
                )}
              </div>
            </div>
          ) : null}

          {activeTab === 'test' ? (
            <div className="tab-panel">
              <section className="panel-card panel-card-fill">
                <div className="linreg-inference-grid">
                  {dataset.featureNames.map((name, index) => (
                    <label key={name} className="config-row">
                      <span className="config-label">{name}</span>
                      <input
                        type="number"
                        value={inferenceInputs[index] ?? 0}
                        onChange={(event) => {
                          const nextValue = Number(event.target.value)
                          setInferenceInputs((current) => {
                            const next = current.slice()
                            next[index] = Number.isFinite(nextValue) ? nextValue : 0
                            return next
                          })
                        }}
                        className="config-control config-control-numeric"
                        disabled={!trainedModel}
                      />
                    </label>
                  ))}

                  <label className="config-row">
                    <span className="config-label">Actual (optional)</span>
                    <input
                      type="number"
                      value={inferenceActualInput}
                      onChange={(event) => setInferenceActualInput(event.target.value)}
                      className="config-control config-control-numeric"
                      disabled={!trainedModel}
                    />
                  </label>
                </div>

                <div className="linreg-test-actions-row">
                  <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={handleLoadSample}
                    disabled={dataset.samples.length === 0}
                  >
                    Load Next Dataset Sample
                  </button>
                  <span className="linreg-sample-index">
                    Sample {selectedSampleIndex + 1} / {dataset.samples.length}
                  </span>
                </div>

                <div className="inference-top-prediction">
                  Prediction: {inferenceTopPrediction !== null ? trimNumber(inferenceTopPrediction, 6) : 'none'}
                </div>
                <pre className="deploy-output linreg-inference-output">{inferenceOutput}</pre>
              </section>

              <div className="panel-actions">
                <button
                  onClick={() => runInference('test')}
                  className="btn btn-validate"
                  disabled={!trainedModel}
                >
                  Run Inference
                </button>
              </div>
            </div>
          ) : null}

          {activeTab === 'deploy' ? (
            <div className="tab-panel deploy-tab-panel">
              <section className="panel-card deploy-panel-card">
                <div className="deploy-summary-grid">
                  <div className="deploy-summary-item">
                    <span className="deploy-summary-label">Dataset</span>
                    <span className="deploy-summary-value">{dataset.name}</span>
                  </div>
                  <div className="deploy-summary-item">
                    <span className="deploy-summary-label">Train Status</span>
                    <span className="deploy-summary-value">{trainingStatus}</span>
                  </div>
                  <div className="deploy-summary-item">
                    <span className="deploy-summary-label">Deploy Status</span>
                    <span
                      className={`deploy-status-pill ${
                        deployment?.status === 'running' ? 'deploy-status-running' : 'deploy-status-stopped'
                      }`}
                    >
                      {deployment?.status ?? 'not deployed'}
                    </span>
                  </div>
                  <div className="deploy-summary-item">
                    <span className="deploy-summary-label">Requests</span>
                    <span className="deploy-summary-value">{deployment?.request_count ?? 0}</span>
                  </div>
                </div>

                <div className="deploy-endpoint-card">
                  <p className="deploy-endpoint-title">Endpoint</p>
                  <code className="deploy-endpoint-value">
                    {deployment ? endpointUrl : 'Deploy model to generate endpoint.'}
                  </code>
                </div>

                <div className="deploy-code-shell">
                  <div className="deploy-code-head">
                    <p className="deploy-code-title">Python Client (requests)</p>
                    <button
                      type="button"
                      onClick={copyPythonSnippet}
                      className={`deploy-code-copy-button ${
                        copyStatus === 'copied'
                          ? 'deploy-code-copy-button-copied'
                          : copyStatus === 'failed'
                            ? 'deploy-code-copy-button-failed'
                            : ''
                      }`.trim()}
                      aria-label="Copy Python client code"
                      title="Copy Python client code"
                    >
                      <svg
                        className="deploy-code-copy-icon"
                        xmlns="http://www.w3.org/2000/svg"
                        height="16px"
                        viewBox="0 -960 960 960"
                        width="16px"
                        fill="currentColor"
                        aria-hidden="true"
                      >
                        <path d="M360-240q-33 0-56.5-23.5T280-320v-480q0-33 23.5-56.5T360-880h360q33 0 56.5 23.5T800-800v480q0 33-23.5 56.5T720-240H360Zm0-80h360v-480H360v480ZM240-80q-33 0-56.5-23.5T160-160v-560h80v560h440v80H240Zm120-240v-480 480Z" />
                      </svg>
                      <span className="deploy-code-copy-text">
                        {copyStatus === 'copied' ? 'Copied' : copyStatus === 'failed' ? 'Failed' : 'Copy'}
                      </span>
                    </button>
                  </div>
                  <pre className="deploy-code">
                    <code>{pythonSnippet}</code>
                  </pre>
                </div>

                <div className="deploy-output-shell">
                  <p className="deploy-output-title">Deployed Inference</p>
                  <p className="deploy-top-prediction">
                    Prediction: {deployTopPrediction !== null ? trimNumber(deployTopPrediction, 6) : 'none'}
                  </p>
                  <pre className="deploy-output">{deployOutput}</pre>
                </div>
              </section>

              <div className="panel-actions deploy-actions">
                <button
                  onClick={() => {
                    void deployModel()
                  }}
                  disabled={!trainedModel || deployment?.status === 'running'}
                  className="btn btn-validate"
                >
                  Deploy Locally
                </button>
                <button
                  onClick={() => {
                    void refreshDeployment()
                  }}
                  disabled={!deployment}
                  className="btn btn-ghost"
                >
                  Refresh
                </button>
                <button
                  onClick={() => {
                    void runDeploymentInference()
                  }}
                  disabled={!deployment || deployment.status !== 'running'}
                  className="btn btn-ghost"
                >
                  Run Endpoint Inference
                </button>
                <button
                  onClick={() => {
                    void stopDeployment()
                  }}
                  disabled={!deployment || deployment.status !== 'running'}
                  className="btn btn-validate btn-danger"
                >
                  Stop Deploy
                </button>
              </div>
            </div>
          ) : null}
        </div>
      </section>

      <section className="app-viewport-panel">
        <button
          type="button"
          onClick={() => setIsSidebarCollapsed((current) => !current)}
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
              <path d="M400-80 0-480l400-400 71 71-329 329 329 329-71 71Z" />
            </svg>
          )}
        </button>
        <a
          href="/"
          className="builder-home-button"
          aria-label="Go to home"
          title="Home"
        >
          <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e3e3e3">
            <path d="M240-200h120v-240h240v240h120v-360L480-740 240-560v360Zm-80 80v-480l320-240 320 240v480H520v-240h-80v240H160Zm320-350Z" />
          </svg>
        </a>

        <div className="linreg-viewport-shell">
          <header className="linreg-viewport-header">
            <div>
              <p className="linreg-kicker">Linear Regression Builder</p>
              <h2>{dataset.name}</h2>
              <p>{dataset.description}</p>
            </div>
            <div className="linreg-view-stats">
              <div>
                <span>Samples</span>
                <strong>{dataset.samples.length}</strong>
              </div>
              <div>
                <span>Features</span>
                <strong>{dataset.featureNames.length}</strong>
              </div>
              <div>
                <span>Trained</span>
                <strong>{trainedModel ? 'yes' : 'no'}</strong>
              </div>
            </div>
          </header>

          <section className="linreg-pipeline-shell">
            {nodes.map((node, index) => {
              const isSelected = selectedNodeId === node.id
              return (
                <div key={node.id} className="linreg-pipeline-node-wrap">
                  <button
                    type="button"
                    onClick={() => setSelectedNodeId(node.id)}
                    className={`linreg-pipeline-node ${isSelected ? 'linreg-pipeline-node-active' : ''}`}
                  >
                    <span className={`linreg-pipeline-node-role linreg-pipeline-node-role-${node.role}`}>
                      {node.type}
                    </span>
                    <h3>{node.name}</h3>
                    <p>{node.subtitle}</p>
                    <span className="linreg-node-size-chip">{node.sizeLabel}</span>
                  </button>
                  {index < nodes.length - 1 ? (
                    <span className="linreg-pipeline-arrow" aria-hidden>
                      →
                    </span>
                  ) : null}
                </div>
              )
            })}
          </section>

          <section className="linreg-viewport-grid">
            <article className="linreg-viewport-card">
              <h3>Dataset + Regression Fit</h3>
              <RegressionScatterPlot
                dataset={dataset}
                trainedModel={trainedModel}
                inferenceInputs={inferenceInputs}
                inferencePrediction={inferenceTopPrediction}
              />
            </article>

            <article className="linreg-viewport-card">
              <h3>Model Equation</h3>
              <p className="linreg-equation-line">
                {trainedModel ? formatEquation(trainedModel) : 'Train model to populate equation.'}
              </p>

              {trainedModel ? (
                <ul className="linreg-coeff-list">
                  {trainedModel.weights.map((weight, index) => (
                    <li key={`${trainedModel.featureNames[index]}-${index}`}>
                      <span>{trainedModel.featureNames[index]}</span>
                      <span>{trimNumber(weight, 6)}</span>
                    </li>
                  ))}
                  <li>
                    <span>bias</span>
                    <span>{trimNumber(trainedModel.bias, 6)}</span>
                  </li>
                </ul>
              ) : (
                <p className="linreg-equation-placeholder">
                  Weights and bias appear here after training.
                </p>
              )}
            </article>
          </section>
        </div>
      </section>
    </div>
  )
}

interface RegressionScatterPlotProps {
  dataset: LinearRegressionDataset
  trainedModel: TrainedModel | null
  inferenceInputs: number[]
  inferencePrediction: number | null
}

function RegressionScatterPlot({
  dataset,
  trainedModel,
  inferenceInputs,
  inferencePrediction,
}: RegressionScatterPlotProps) {
  if (dataset.featureNames.length !== 1) {
    return (
      <div className="linreg-scatter-empty">
        Plot preview is available for single-feature datasets.
      </div>
    )
  }

  const width = 720
  const height = 280
  const padding = { top: 14, right: 18, bottom: 34, left: 40 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom

  const xValues = dataset.samples.map((row) => row[0])
  const yValues = dataset.targets

  const xMin = Math.min(...xValues)
  const xMax = Math.max(...xValues)
  const yMin = Math.min(...yValues)
  const yMax = Math.max(...yValues)

  const safeXSpan = Math.max(xMax - xMin, 1e-6)
  const safeYSpan = Math.max(yMax - yMin, 1e-6)

  const toX = (value: number) => padding.left + ((value - xMin) / safeXSpan) * plotWidth
  const toY = (value: number) => padding.top + (1 - (value - yMin) / safeYSpan) * plotHeight

  const linePoints = trainedModel
    ? [
        { x: xMin, y: predictWithModel([xMin], trainedModel) },
        { x: xMax, y: predictWithModel([xMax], trainedModel) },
      ]
    : null

  const inferenceX = inferenceInputs[0]

  return (
    <div className="linreg-scatter-shell">
      <svg viewBox={`0 0 ${width} ${height}`} className="linreg-scatter-svg">
        <rect
          x={padding.left}
          y={padding.top}
          width={plotWidth}
          height={plotHeight}
          rx={10}
          className="linreg-scatter-bg"
        />

        <line
          x1={padding.left}
          y1={padding.top + plotHeight}
          x2={padding.left + plotWidth}
          y2={padding.top + plotHeight}
          className="linreg-scatter-axis"
        />
        <line
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={padding.top + plotHeight}
          className="linreg-scatter-axis"
        />

        {xValues.map((xValue, index) => (
          <circle
            key={`point-${index}`}
            cx={toX(xValue)}
            cy={toY(yValues[index])}
            r={3.2}
            className="linreg-scatter-point"
          />
        ))}

        {linePoints ? (
          <line
            x1={toX(linePoints[0].x)}
            y1={toY(linePoints[0].y)}
            x2={toX(linePoints[1].x)}
            y2={toY(linePoints[1].y)}
            className="linreg-scatter-fit-line"
          />
        ) : null}

        {Number.isFinite(inferenceX) ? (
          <line
            x1={toX(inferenceX)}
            y1={padding.top}
            x2={toX(inferenceX)}
            y2={padding.top + plotHeight}
            className="linreg-scatter-cursor"
          />
        ) : null}

        {inferencePrediction !== null && Number.isFinite(inferenceX) ? (
          <circle
            cx={toX(inferenceX)}
            cy={toY(inferencePrediction)}
            r={5}
            className="linreg-scatter-prediction"
          />
        ) : null}

        <text x={padding.left + plotWidth / 2} y={height - 8} className="linreg-scatter-label">
          {dataset.featureNames[0]}
        </text>
        <text
          x={12}
          y={padding.top + plotHeight / 2}
          className="linreg-scatter-label"
          transform={`rotate(-90 12 ${padding.top + plotHeight / 2})`}
        >
          {dataset.targetName}
        </text>
      </svg>
    </div>
  )
}

function getRoleDotClass(role: LayerRole): string {
  if (role === 'input') return 'layer-dot layer-dot-input'
  if (role === 'output') return 'layer-dot layer-dot-output'
  return 'layer-dot layer-dot-hidden'
}

function getBuildStatusMessage(status: BuildStatus, issueCount: number): string {
  if (status === 'success') return 'Build successful. Pipeline is valid.'
  if (status === 'error') return issueCount === 1 ? 'Build failed with 1 issue.' : `Build failed with ${issueCount} issues.`
  return 'Ready to build.'
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(Math.max(value, min), max)
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(Math.max(Math.round(value), min), max)
}

function trimNumber(value: number, digits = 4): string {
  if (!Number.isFinite(value)) return '0'
  return Number(value.toFixed(digits)).toString()
}

function splitDataset(
  x: number[][],
  y: number[],
  testSplit: number,
  seed: number
): DatasetSplit {
  const indices = x.map((_, index) => index)
  const random = createSeededRandom(seed)

  for (let index = indices.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1))
    const current = indices[index]
    indices[index] = indices[swapIndex]
    indices[swapIndex] = current
  }

  const rawTestCount = Math.round(indices.length * testSplit)
  const testCount = clampInt(rawTestCount, 1, Math.max(indices.length - 2, 1))
  const splitAt = indices.length - testCount

  const trainIndices = indices.slice(0, splitAt)
  const testIndices = indices.slice(splitAt)

  return {
    trainX: trainIndices.map((index) => x[index].slice()),
    trainY: trainIndices.map((index) => y[index]),
    testX: testIndices.map((index) => x[index].slice()),
    testY: testIndices.map((index) => y[index]),
  }
}

function computeNormalization(x: number[][]): NormalizationStats {
  const featureCount = x[0]?.length ?? 0
  const means = new Array(featureCount).fill(0)
  const stds = new Array(featureCount).fill(1)

  if (featureCount === 0 || x.length === 0) {
    return { means, stds }
  }

  for (let feature = 0; feature < featureCount; feature += 1) {
    let sum = 0
    for (let row = 0; row < x.length; row += 1) {
      sum += x[row][feature]
    }
    const mean = sum / x.length
    means[feature] = mean

    let variance = 0
    for (let row = 0; row < x.length; row += 1) {
      const delta = x[row][feature] - mean
      variance += delta * delta
    }
    const std = Math.sqrt(variance / x.length)
    stds[feature] = std > 1e-9 ? std : 1
  }

  return { means, stds }
}

function createNoopNormalization(featureCount: number): NormalizationStats {
  return {
    means: new Array(featureCount).fill(0),
    stds: new Array(featureCount).fill(1),
  }
}

function applyNormalization(x: number[][], stats: NormalizationStats): number[][] {
  return x.map((row) =>
    row.map((value, feature) => (value - stats.means[feature]) / stats.stds[feature])
  )
}

interface GradientStepInput {
  x: number[][]
  y: number[]
  weights: number[]
  bias: number
  learningRate: number
  fitIntercept: boolean
  l2Penalty: number
}

function gradientStep({
  x,
  y,
  weights,
  bias,
  learningRate,
  fitIntercept,
  l2Penalty,
}: GradientStepInput): { weights: number[]; bias: number } {
  const sampleCount = x.length
  if (sampleCount === 0) {
    return { weights: weights.slice(), bias }
  }

  const featureCount = weights.length
  const weightGradients = new Array(featureCount).fill(0)
  let biasGradient = 0

  for (let sample = 0; sample < sampleCount; sample += 1) {
    const prediction = dot(x[sample], weights) + (fitIntercept ? bias : 0)
    const error = prediction - y[sample]

    for (let feature = 0; feature < featureCount; feature += 1) {
      weightGradients[feature] += (2 / sampleCount) * x[sample][feature] * error
    }

    if (fitIntercept) {
      biasGradient += (2 / sampleCount) * error
    }
  }

  for (let feature = 0; feature < featureCount; feature += 1) {
    if (l2Penalty > 0) {
      weightGradients[feature] += 2 * l2Penalty * weights[feature]
    }
  }

  const nextWeights = weights.map((weight, feature) => weight - learningRate * weightGradients[feature])
  const nextBias = fitIntercept ? bias - learningRate * biasGradient : 0

  return { weights: nextWeights, bias: nextBias }
}

function dot(a: number[], b: number[]): number {
  let sum = 0
  for (let index = 0; index < a.length; index += 1) {
    sum += a[index] * b[index]
  }
  return sum
}

function predictBatch(x: number[][], weights: number[], bias: number): number[] {
  return x.map((row) => dot(row, weights) + bias)
}

function meanSquaredError(yTrue: number[], yPred: number[]): number {
  if (yTrue.length === 0) return 0
  let total = 0
  for (let index = 0; index < yTrue.length; index += 1) {
    const error = yTrue[index] - yPred[index]
    total += error * error
  }
  return total / yTrue.length
}

function meanAbsoluteError(yTrue: number[], yPred: number[]): number {
  if (yTrue.length === 0) return 0
  let total = 0
  for (let index = 0; index < yTrue.length; index += 1) {
    total += Math.abs(yTrue[index] - yPred[index])
  }
  return total / yTrue.length
}

function r2Score(yTrue: number[], yPred: number[]): number {
  if (yTrue.length === 0) return 0
  const mean = yTrue.reduce((sum, value) => sum + value, 0) / yTrue.length
  let ssRes = 0
  let ssTot = 0
  for (let index = 0; index < yTrue.length; index += 1) {
    const residual = yTrue[index] - yPred[index]
    ssRes += residual * residual
    const centered = yTrue[index] - mean
    ssTot += centered * centered
  }
  if (ssTot < 1e-12) return 0
  return 1 - ssRes / ssTot
}

function getBoundsFromSeries(
  primary: number[],
  secondary: number[],
  fallbackMin: number
): { min: number; max: number } {
  const values = [...primary, ...secondary]
  if (values.length === 0) {
    return { min: fallbackMin, max: fallbackMin + 1 }
  }

  let min = values[0]
  let max = values[0]
  for (let index = 1; index < values.length; index += 1) {
    min = Math.min(min, values[index])
    max = Math.max(max, values[index])
  }

  if (Math.abs(max - min) < 1e-9) {
    return { min: min - 0.5, max: max + 0.5 }
  }
  return { min, max }
}

function predictWithModel(input: number[], model: TrainedModel): number {
  const normalized = input.map(
    (value, feature) => (value - model.means[feature]) / model.stds[feature]
  )
  return dot(normalized, model.weights) + model.bias
}

function formatEquation(model: TrainedModel): string {
  const terms = model.featureNames.map((feature, index) => {
    const coeff = trimNumber(model.weights[index], 4)
    return `${coeff}*${feature}`
  })

  const biasTerm = model.fitIntercept ? ` + ${trimNumber(model.bias, 4)}` : ''
  return `${model.targetName} = ${terms.join(' + ')}${biasTerm}`
}

function createSeededRandom(seed: number): () => number {
  let value = (seed | 0) ^ 0x9e3779b9

  return () => {
    value |= 0
    value = (value + 0x6d2b79f5) | 0
    let t = Math.imul(value ^ (value >>> 15), 1 | value)
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

async function nextFrame(): Promise<void> {
  await new Promise<void>((resolve) => {
    window.setTimeout(resolve, 0)
  })
}

function resolveBackendBaseUrl(): string {
  const envBase = import.meta.env.VITE_BACKEND_HTTP_URL?.trim()
  if (envBase) {
    return trimTrailingSlash(envBase)
  }
  if (typeof window === 'undefined') {
    return 'http://127.0.0.1:8000'
  }
  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:'
  const hostname = window.location.hostname || '127.0.0.1'
  return `${protocol}//${hostname}:8000`
}

function trimTrailingSlash(value: string): string {
  return value.endsWith('/') ? value.slice(0, -1) : value
}

function toPythonSingleQuotedString(value: string): string {
  return `'${value.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`
}

function createLinregJobId(datasetId: string): string {
  const safeDataset = datasetId.replace(/[^a-z0-9]+/gi, "_").toLowerCase()
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `linreg_${safeDataset}_${crypto.randomUUID().replaceAll('-', '').slice(0, 10)}`
  }
  return `linreg_${safeDataset}_${Math.random().toString(16).slice(2, 12)}`
}

async function copyToClipboard(value: string): Promise<boolean> {
  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value)
      return true
    }
  } catch {
    // fallback path below
  }

  try {
    if (typeof document === 'undefined') return false
    const textarea = document.createElement('textarea')
    textarea.value = value
    textarea.setAttribute('readonly', 'true')
    textarea.style.position = 'fixed'
    textarea.style.left = '-9999px'
    document.body.appendChild(textarea)
    textarea.focus()
    textarea.select()
    const copied = document.execCommand('copy')
    document.body.removeChild(textarea)
    return copied
  } catch {
    return false
  }
}
