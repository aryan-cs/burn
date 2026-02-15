import type { MutableRefObject } from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { VLMArchitectureViewport } from './VLMArchitectureViewport'
import { getVlmArchitecture, type VLMArchitectureSpec } from './architecture'
import './vlm.css'

export interface VLMTrainingConfig {
  dataset: string
  modelId: string
  epochs: number
  batchSize: number
  stepsPerEpoch: number
  learningRate: number
}

interface VLMTrainResponse {
  job_id: string
  status: string
}

interface VLMStatusResponse {
  job_id: string
  status: string
  terminal: boolean
  error?: string | null
  dataset: string
  model_id: string
  epochs: number
  current_epoch: number
  latest_loss?: number | null
  final_metrics?: Record<string, unknown> | null
  has_artifact: boolean
}

interface VLMDetection {
  label: string
  label_id: number
  score: number
  box: [number, number, number, number]
}

interface SmoothedDetection extends VLMDetection {
  confidence: number
  lastSeenMs: number
}

interface VLMInferResponse {
  job_id_used?: string | null
  runtime_backend: string
  runtime_model_id: string
  image_width: number
  image_height: number
  detections: VLMDetection[]
  warning?: string | null
}

interface VLMProgressEvent {
  type: 'vlm_progress'
  phase?: 'step' | 'epoch'
  epoch: number
  epochs: number
  step?: number
  steps_per_epoch?: number
  loss: number
  status?: string
}

interface VLMDoneEvent {
  type: 'vlm_done'
  status?: string
  final_loss?: number
  epochs_ran?: number
  model_path?: string
}

interface VLMErrorEvent {
  type: 'vlm_error'
  message: string
}

type VLMSocketEvent = VLMProgressEvent | VLMDoneEvent | VLMErrorEvent

type DashboardTab = 'build' | 'train' | 'test'

interface VLMDatasetMeta {
  id: string
  name: string
  task: string
  description: string
}

interface VLMModelMeta {
  id: string
  name: string
  provider: string
  task: string
}

interface VLMSettingsResponse {
  datasets: VLMDatasetMeta[]
  models: VLMModelMeta[]
}

type VLMArchitectureResponse = VLMArchitectureSpec

interface VLMPageProps {
  initialConfig?: Partial<VLMTrainingConfig>
}

const DEFAULT_CONFIG: VLMTrainingConfig = {
  dataset: 'synthetic_boxes_tiny',
  modelId: 'hustvl/yolos-tiny',
  epochs: 1,
  batchSize: 1,
  stepsPerEpoch: 1,
  learningRate: 0.00001,
}

const LIVE_INFER_INTERVAL_MS = 250
const LIVE_INFER_MAX_SIDE = 320
const LIVE_INFER_JPEG_QUALITY = 0.6
const LIVE_INFER_MAX_DETECTIONS = 10
const LOW_DETAIL_STORAGE_KEY = 'mlcanvas.vlm.low_detail_mode'

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

export default function VLMPage({ initialConfig }: VLMPageProps) {
  const [config, setConfig] = useState<VLMTrainingConfig>({
    ...DEFAULT_CONFIG,
    ...(initialConfig ?? {}),
  })
  const [datasets, setDatasets] = useState<VLMDatasetMeta[]>([])
  const [models, setModels] = useState<VLMModelMeta[]>([])
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState('idle')
  const [terminal, setTerminal] = useState(false)
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [latestLoss, setLatestLoss] = useState<number | null>(null)
  const [runtimeWarning, setRuntimeWarning] = useState<string | null>(null)
  const [message, setMessage] = useState('Select a model architecture and start building.')
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [wsState, setWsState] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')
  const [trainingLog, setTrainingLog] = useState<string[]>([])

  const [activeTab, setActiveTab] = useState<DashboardTab>('build')
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [isLowDetailMode, setIsLowDetailMode] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem(LOW_DETAIL_STORAGE_KEY) === '1'
  })

  const [cameraReady, setCameraReady] = useState(false)
  const [liveModeEnabled, setLiveModeEnabled] = useState(true)
  const [liveInferenceRunning, setLiveInferenceRunning] = useState(false)
  const [detections, setDetections] = useState<VLMDetection[]>([])
  const [isDetectionModalOpen, setIsDetectionModalOpen] = useState(false)
  const [scoreThreshold, setScoreThreshold] = useState(0.45)
  const [lastImageSize, setLastImageSize] = useState<[number, number] | null>(null)
  const [viewportResetKey, setViewportResetKey] = useState(0)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const fullscreenCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const liveInferenceBusyRef = useRef(false)
  const liveUiTickRef = useRef(0)
  const smoothedDetectionsRef = useRef<SmoothedDetection[]>([])

  const [architecture, setArchitecture] = useState<VLMArchitectureSpec>(
    () => getVlmArchitecture(config.modelId)
  )
  const [selectedStageId, setSelectedStageId] = useState<string | null>(
    architecture.stages[0]?.id ?? null
  )

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(LOW_DETAIL_STORAGE_KEY, isLowDetailMode ? '1' : '0')
  }, [isLowDetailMode])

  useEffect(() => {
    setSelectedStageId((current) => {
      if (current && architecture.stages.some((stage) => stage.id === current)) {
        return current
      }
      return architecture.stages[0]?.id ?? null
    })
  }, [architecture])

  useEffect(() => {
    let cancelled = false

    const loadArchitecture = async () => {
      try {
        const spec = await requestJson<VLMArchitectureResponse>(
          `/api/vlm/architecture?model_id=${encodeURIComponent(config.modelId)}`
        )
        if (cancelled) return
        setArchitecture(spec)
        if (spec.warning) {
          setMessage(spec.warning)
          pushTrainingLogRef.current?.(`Architecture warning: ${spec.warning}`)
        }
      } catch (error) {
        if (cancelled) return
        const fallback = getVlmArchitecture(config.modelId)
        setArchitecture(fallback)
        const text = error instanceof Error ? error.message : String(error)
        pushTrainingLogRef.current?.(`Architecture fallback used: ${text}`)
      }
    }

    void loadArchitecture()
    return () => {
      cancelled = true
    }
  }, [config.modelId])

  const selectedStage = useMemo(
    () => architecture.stages.find((stage) => stage.id === selectedStageId) ?? architecture.stages[0] ?? null,
    [architecture, selectedStageId]
  )

  const isBusy = busyAction !== null
  const handleAlignViewport = useCallback(() => {
    setSelectedStageId(architecture.stages[0]?.id ?? null)
    setViewportResetKey((current) => current + 1)
  }, [architecture.stages])
  const syncFullscreenCanvas = useCallback((sourceCanvas: HTMLCanvasElement) => {
    const targetCanvas = fullscreenCanvasRef.current
    if (!targetCanvas || !isDetectionModalOpen) return
    if (targetCanvas.width !== sourceCanvas.width || targetCanvas.height !== sourceCanvas.height) {
      targetCanvas.width = sourceCanvas.width
      targetCanvas.height = sourceCanvas.height
    }
    const targetContext = targetCanvas.getContext('2d')
    if (!targetContext) return
    targetContext.clearRect(0, 0, targetCanvas.width, targetCanvas.height)
    targetContext.drawImage(sourceCanvas, 0, 0, targetCanvas.width, targetCanvas.height)
  }, [isDetectionModalOpen])

  useEffect(() => {
    if (!isDetectionModalOpen) return
    const sourceCanvas = canvasRef.current
    if (sourceCanvas) {
      syncFullscreenCanvas(sourceCanvas)
    }
  }, [isDetectionModalOpen, syncFullscreenCanvas])

  useEffect(() => {
    if (!isDetectionModalOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsDetectionModalOpen(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isDetectionModalOpen])

  const runAction = useCallback(async (name: string, fn: () => Promise<void>) => {
    if (isBusy) return
    setBusyAction(name)
    try {
      await fn()
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      setMessage(text)
      pushTrainingLogRef.current?.(`Error: ${text}`)
    } finally {
      setBusyAction(null)
    }
  }, [isBusy])

  const pushTrainingLog = useCallback((line: string) => {
    setTrainingLog((current) => {
      const stamped = `${new Date().toLocaleTimeString()} · ${line}`
      return [...current.slice(-119), stamped]
    })
  }, [])

  const pushTrainingLogRef = useRef<((line: string) => void) | null>(null)
  useEffect(() => {
    pushTrainingLogRef.current = pushTrainingLog
  }, [pushTrainingLog])

  const refreshStatus = useCallback(async (targetJobId: string) => {
    const next = await requestJson<VLMStatusResponse>(`/api/vlm/status?job_id=${encodeURIComponent(targetJobId)}`)
    setStatus(next.status)
    setTerminal(next.terminal)
    setCurrentEpoch(next.current_epoch)
    setLatestLoss(
      typeof next.latest_loss === 'number'
        ? next.latest_loss
        : (typeof next.final_metrics?.final_loss === 'number' ? Number(next.final_metrics.final_loss) : null)
    )
    if (next.error) {
      setMessage(next.error)
      pushTrainingLog(`Status error: ${next.error}`)
    }
  }, [pushTrainingLog])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const settings = await requestJson<VLMSettingsResponse>('/api/vlm/datasets')
        if (cancelled) return
        setDatasets(settings.datasets)
        setModels(settings.models)

        setConfig((current) => {
          const nextDataset = settings.datasets.some((item) => item.id === current.dataset)
            ? current.dataset
            : (settings.datasets[0]?.id ?? current.dataset)
          const nextModel = settings.models.some((item) => item.id === current.modelId)
            ? current.modelId
            : (settings.models[0]?.id ?? current.modelId)
          return {
            ...current,
            dataset: nextDataset,
            modelId: nextModel,
          }
        })
      } catch (error) {
        if (cancelled) return
        const text = error instanceof Error ? error.message : String(error)
        setMessage(`Failed to load VLM settings: ${text}`)
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!jobId || terminal || wsState === 'connected') return
    const timer = window.setInterval(() => {
      void refreshStatus(jobId)
    }, 2500)
    return () => window.clearInterval(timer)
  }, [jobId, terminal, refreshStatus, wsState])

  useEffect(() => {
    if (!jobId) return
    if (terminal) return
    if (status !== 'queued' && status !== 'running' && status !== 'stopping') return

    setWsState('connecting')
    const socket = new WebSocket(buildVlmWsUrl(jobId))
    wsRef.current = socket

    socket.onopen = () => {
      setWsState('connected')
      pushTrainingLog(`WebSocket connected for VLM job ${jobId}`)
    }

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data) as VLMSocketEvent
      if (data.type === 'vlm_progress') {
        setStatus(data.status ?? 'running')
        setTerminal(false)
        setCurrentEpoch(data.epoch)
        setLatestLoss(data.loss)
        const line = data.phase === 'step' && typeof data.step === 'number' && typeof data.steps_per_epoch === 'number'
          ? `Epoch ${data.epoch}/${data.epochs} · step ${data.step}/${data.steps_per_epoch} · avg_loss=${data.loss.toFixed(4)}`
          : `Epoch ${data.epoch}/${data.epochs} · loss=${data.loss.toFixed(4)}`
        setMessage(line)
        pushTrainingLog(line)
        return
      }
      if (data.type === 'vlm_done') {
        const terminalStatus = data.status ?? 'completed'
        setStatus(terminalStatus)
        setTerminal(true)
        if (typeof data.epochs_ran === 'number') {
          setCurrentEpoch(data.epochs_ran)
        }
        if (typeof data.final_loss === 'number') {
          setLatestLoss(data.final_loss)
        }
        const line = `Training ${terminalStatus}${typeof data.final_loss === 'number' ? ` · final_loss=${data.final_loss.toFixed(4)}` : ''}`
        setMessage(line)
        pushTrainingLog(line)
        return
      }
      if (data.type === 'vlm_error') {
        setStatus('failed')
        setTerminal(true)
        setMessage(data.message)
        pushTrainingLog(`Error: ${data.message}`)
      }
    }

    socket.onerror = () => {
      setWsState('disconnected')
      pushTrainingLog('WebSocket error. Falling back to status polling.')
    }

    socket.onclose = (event) => {
      wsRef.current = null
      setWsState('disconnected')
      if (!terminal) {
        const reason = event.reason?.trim() ? ` reason=${event.reason}` : ''
        pushTrainingLog(`WebSocket closed (code=${event.code}${reason}). Falling back to status polling.`)
      }
    }

    return () => {
      socket.close()
      wsRef.current = null
      setWsState('disconnected')
    }
  }, [jobId, pushTrainingLog, status, terminal])

  useEffect(() => {
    return () => {
      stopCameraStream(streamRef)
      wsRef.current?.close()
      wsRef.current = null
      liveInferenceBusyRef.current = false
    }
  }, [])

  const runInferenceFrame = useCallback(
    async (silent = false) => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas || !cameraReady) {
        throw new Error('Start the camera before running detection.')
      }

      const rawWidth = video.videoWidth || 640
      const rawHeight = video.videoHeight || 480
      const scale = Math.min(1, LIVE_INFER_MAX_SIDE / Math.max(rawWidth, rawHeight))
      const width = Math.max(1, Math.round(rawWidth * scale))
      const height = Math.max(1, Math.round(rawHeight * scale))
      canvas.width = width
      canvas.height = height
      const context = canvas.getContext('2d')
      if (!context) {
        throw new Error('Canvas context could not be created.')
      }

      context.drawImage(video, 0, 0, width, height)
      const imageBase64 = canvas.toDataURL('image/jpeg', LIVE_INFER_JPEG_QUALITY)

      const response = await requestJson<VLMInferResponse>('/api/vlm/infer', {
        method: 'POST',
        body: JSON.stringify({
          job_id: jobId,
          image_base64: imageBase64,
          score_threshold: scoreThreshold,
          max_detections: LIVE_INFER_MAX_DETECTIONS,
        }),
      })

      const nowMs = performance.now()
      const smoothedDetections = smoothDetections(
        smoothedDetectionsRef.current,
        response.detections,
        nowMs
      )
      smoothedDetectionsRef.current = smoothedDetections

      drawDetections(context, smoothedDetections)
      syncFullscreenCanvas(canvas)
      const shouldSyncUi = !silent || liveUiTickRef.current % 3 === 0
      liveUiTickRef.current += 1
      if (shouldSyncUi) {
        setDetections(smoothedDetections)
        setLastImageSize([response.image_width, response.image_height])
        setRuntimeWarning(response.warning ?? null)
      }

      if (!silent) {
        const runtimeLabel = `${response.runtime_backend} · ${response.runtime_model_id}`
        setMessage(
          smoothedDetections.length > 0
            ? `Detected ${smoothedDetections.length} object(s) using ${runtimeLabel}.`
            : `No objects detected above threshold (${runtimeLabel}).`
        )
      }
    },
    [cameraReady, jobId, scoreThreshold, syncFullscreenCanvas]
  )

  useEffect(() => {
    if (!cameraReady || !liveModeEnabled) {
      setLiveInferenceRunning(false)
      return
    }

    let disposed = false
    let timer: number | null = null

    const tick = async () => {
      if (disposed || document.hidden || liveInferenceBusyRef.current) {
        if (!disposed) {
          timer = window.setTimeout(() => {
            void tick()
          }, LIVE_INFER_INTERVAL_MS)
        }
        return
      }

      liveInferenceBusyRef.current = true
      try {
        await runInferenceFrame(true)
        if (!disposed) {
          setLiveInferenceRunning(true)
        }
      } catch (error) {
        if (!disposed) {
          const text = error instanceof Error ? error.message : String(error)
          setMessage(`Live detection error: ${text}`)
          setLiveInferenceRunning(false)
        }
      } finally {
        liveInferenceBusyRef.current = false
        if (!disposed) {
          timer = window.setTimeout(() => {
            void tick()
          }, LIVE_INFER_INTERVAL_MS)
        }
      }
    }

    void tick()

    return () => {
      disposed = true
      if (timer !== null) {
        window.clearTimeout(timer)
      }
      liveInferenceBusyRef.current = false
      setLiveInferenceRunning(false)
    }
  }, [cameraReady, liveModeEnabled, runInferenceFrame])

  const startCamera = useCallback(async () => {
    await runAction('camera_start', async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera API is not available in this browser.')
      }

      stopCameraStream(streamRef)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      })
      const video = videoRef.current
      if (!video) {
        stopCameraStreamRef(stream)
        throw new Error('Camera element is unavailable.')
      }
      streamRef.current = stream
      video.srcObject = stream
      await video.play()
      setCameraReady(true)
      setLiveModeEnabled(true)
      setActiveTab('test')
      setMessage('Camera ready. Live detection started.')
    })
  }, [runAction])

  const stopCamera = useCallback(() => {
    stopCameraStream(streamRef)
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    smoothedDetectionsRef.current = []
    setCameraReady(false)
    setLiveInferenceRunning(false)
    setMessage('Camera stopped.')
  }, [])

  const runTrain = useCallback(() => {
    void runAction('train', async () => {
      const response = await requestJson<VLMTrainResponse>('/api/vlm/train', {
        method: 'POST',
        body: JSON.stringify({
          training: {
            dataset: config.dataset,
            model_id: config.modelId,
            epochs: config.epochs,
            batch_size: config.batchSize,
            steps_per_epoch: config.stepsPerEpoch,
            learning_rate: config.learningRate,
          },
        }),
      })
      setJobId(response.job_id)
      setStatus(response.status)
      setTerminal(false)
      setCurrentEpoch(0)
      setLatestLoss(null)
      setTrainingLog([])
      setWsState('disconnected')
      setActiveTab('train')
      setMessage(`VLM training started (job_id=${response.job_id}).`)
      pushTrainingLog(`Started VLM training job ${response.job_id}`)
    })
  }, [config, pushTrainingLog, runAction])

  const runStop = useCallback(() => {
    void runAction('stop', async () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ command: 'stop' }))
      }
      const response = await requestJson<{ job_id: string; status: string }>('/api/vlm/stop', {
        method: 'POST',
        body: JSON.stringify(jobId ? { job_id: jobId } : {}),
      })
      setJobId(response.job_id)
      setStatus(response.status)
      setMessage(`Stop requested for ${response.job_id}.`)
      pushTrainingLog(`Stop requested for ${response.job_id}`)
    })
  }, [jobId, pushTrainingLog, runAction])

  const runStatus = useCallback(() => {
    void runAction('status', async () => {
      if (!jobId) {
        throw new Error('No VLM job selected.')
      }
      await refreshStatus(jobId)
      setMessage(`Status refreshed for ${jobId}.`)
    })
  }, [jobId, refreshStatus, runAction])

  const runDetect = useCallback(() => {
    void runAction('detect', async () => {
      await runInferenceFrame(false)
    })
  }, [runAction, runInferenceFrame])

  const clearDetections = useCallback(() => {
    const canvas = canvasRef.current
    if (canvas) {
      const context = canvas.getContext('2d')
      if (context) {
        context.clearRect(0, 0, canvas.width, canvas.height)
      }
    }
    const fullscreenCanvas = fullscreenCanvasRef.current
    if (fullscreenCanvas) {
      const fullscreenContext = fullscreenCanvas.getContext('2d')
      if (fullscreenContext) {
        fullscreenContext.clearRect(0, 0, fullscreenCanvas.width, fullscreenCanvas.height)
      }
    }
    smoothedDetectionsRef.current = []
    setDetections([])
    setLastImageSize(null)
    setMessage('Cleared last detection result.')
  }, [])

  const datasetName = useMemo(
    () => datasets.find((item) => item.id === config.dataset)?.name ?? config.dataset,
    [config.dataset, datasets]
  )

  const activeTabIndex = activeTab === 'build' ? 0 : activeTab === 'train' ? 1 : 2
  const activityPulse = status === 'running'
    ? 1
    : (liveInferenceRunning ? 0.5 : (detections.length > 0 ? 0.28 : 0.08))

  return (
    <div className={`app-shell ${isSidebarCollapsed ? 'app-shell-collapsed' : ''}`}>
      <section className="app-sidebar">
        <div className="app-sidebar-inner">
          <div className="app-tab-strip app-tab-strip-three">
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
              className={`app-tab-button ${activeTab === 'build' ? 'app-tab-button-active' : 'app-tab-button-inactive'}`}
            >
              Build
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('train')}
              className={`app-tab-button ${activeTab === 'train' ? 'app-tab-button-active' : 'app-tab-button-inactive'}`}
            >
              Train
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('test')}
              className={`app-tab-button ${activeTab === 'test' ? 'app-tab-button-active' : 'app-tab-button-inactive'}`}
            >
              Test
            </button>
          </div>

          {activeTab === 'build' ? (
            <div className="tab-panel">
              <section className="panel-card panel-card-layers">
                <div className="layer-list-shell">
                  <ol className="layer-list">
                    {architecture.stages.map((stage, index) => {
                      const isSelected = stage.id === selectedStage?.id
                      return (
                        <li
                          key={stage.id}
                          className={`layer-list-item ${isSelected ? 'layer-list-item-active' : ''}`}
                          role="button"
                          tabIndex={0}
                          onClick={() => setSelectedStageId(stage.id)}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter' || event.key === ' ') {
                              event.preventDefault()
                              setSelectedStageId(stage.id)
                            }
                          }}
                        >
                          <div className="layer-list-item-left">
                            <span className={getStageDotClass(index, architecture.stages.length)} />
                            <span className="layer-list-item-name">{stage.label}</span>
                          </div>
                          <span className="layer-list-item-size">{stage.detail}</span>
                        </li>
                      )
                    })}
                  </ol>
                </div>
              </section>

              <section className="panel-card layer-editor-card">
                <div className="config-grid vlm-test-status-grid">
                  <label className="config-row">
                    <span className="config-label">Dataset</span>
                    <select
                      className="config-control vlm-config-control"
                      value={config.dataset}
                      disabled={isBusy}
                      onChange={(event) => setConfig((current) => ({ ...current, dataset: event.target.value }))}
                    >
                      {datasets.map((dataset) => (
                        <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
                      ))}
                    </select>
                  </label>

                  <label className="config-row">
                    <span className="config-label">Model</span>
                    <select
                      className="config-control vlm-config-control"
                      value={config.modelId}
                      disabled={isBusy}
                      onChange={(event) => setConfig((current) => ({ ...current, modelId: event.target.value }))}
                    >
                      {models.map((model) => (
                        <option key={model.id} value={model.id}>{model.name}</option>
                      ))}
                    </select>
                  </label>
                </div>

                {selectedStage ? (
                  <div className="vlm-stage-card">
                    <div className="vlm-stage-card-head">
                      <p className="panel-subtitle">{selectedStage.label}</p>
                      <span className="layer-list-item-size">{selectedStage.detail}</span>
                    </div>
                    <p className="panel-muted-text panel-muted-text-tight">{selectedStage.description}</p>
                  </div>
                ) : null}

                <div className="summary-grid">
                  <div className="metric-tile metric-tile-compact">
                    <p className="metric-tile-label">Architecture</p>
                    <p className="metric-tile-value metric-tile-value-compact">{architecture.name}</p>
                  </div>
                  <div className="metric-tile metric-tile-compact">
                    <p className="metric-tile-label">Stages</p>
                    <p className="metric-tile-value metric-tile-value-compact">{architecture.stages.length}</p>
                  </div>
                  <div className="metric-tile metric-tile-compact">
                    <p className="metric-tile-label">Dataset</p>
                    <p className="metric-tile-value metric-tile-value-compact">{datasetName}</p>
                  </div>
                  <div className="metric-tile metric-tile-compact">
                    <p className="metric-tile-label">Current Status</p>
                    <p className="metric-tile-value metric-tile-value-compact">{status}</p>
                  </div>
                </div>

              </section>
            </div>
          ) : null}

          {activeTab === 'train' ? (
            <div className="tab-panel">
              <section className="panel-card train-settings-card">
                <div className="config-grid train-config-grid">
                  <label className="config-row">
                    <span className="config-label">Epochs</span>
                    <input
                      className="config-control config-control-numeric"
                      type="number"
                      min={1}
                      value={config.epochs}
                      disabled={isBusy}
                      onChange={(event) =>
                        setConfig((current) => ({ ...current, epochs: Math.max(1, Number(event.target.value) || 1) }))
                      }
                    />
                  </label>
                  <label className="config-row">
                    <span className="config-label">Batch</span>
                    <input
                      className="config-control config-control-numeric"
                      type="number"
                      min={1}
                      value={config.batchSize}
                      disabled={isBusy}
                      onChange={(event) =>
                        setConfig((current) => ({ ...current, batchSize: Math.max(1, Number(event.target.value) || 1) }))
                      }
                    />
                  </label>
                  <label className="config-row">
                    <span className="config-label">Steps/Epoch</span>
                    <input
                      className="config-control config-control-numeric"
                      type="number"
                      min={1}
                      value={config.stepsPerEpoch}
                      disabled={isBusy}
                      onChange={(event) =>
                        setConfig((current) => ({ ...current, stepsPerEpoch: Math.max(1, Number(event.target.value) || 1) }))
                      }
                    />
                  </label>
                  <label className="config-row">
                    <span className="config-label">Learning Rate</span>
                    <input
                      className="config-control config-control-numeric"
                      type="number"
                      min={0.000001}
                      step={0.000001}
                      value={config.learningRate}
                      disabled={isBusy}
                      onChange={(event) =>
                        setConfig((current) => ({
                          ...current,
                          learningRate: Math.max(0.000001, Number(event.target.value) || 0.000001),
                        }))
                      }
                    />
                  </label>
                </div>
              </section>

              <section className="panel-card build-summary-card">
                <div className="summary-grid">
                  <div className="metric-tile metric-tile-compact">
                    <p className="metric-tile-label">Job ID</p>
                    <p className="metric-tile-value metric-tile-value-compact">{jobId ?? 'none'}</p>
                  </div>
                  <div className="metric-tile metric-tile-compact">
                    <p className="metric-tile-label">Status</p>
                    <p className="metric-tile-value metric-tile-value-compact">{status}</p>
                  </div>
                  <div className="metric-tile metric-tile-compact">
                    <p className="metric-tile-label">Epoch</p>
                    <p className="metric-tile-value metric-tile-value-compact">{currentEpoch}/{config.epochs}</p>
                  </div>
                  <div className="metric-tile metric-tile-compact">
                    <p className="metric-tile-label">Latest Loss</p>
                    <p className="metric-tile-value metric-tile-value-compact">{latestLoss === null ? 'n/a' : latestLoss.toFixed(4)}</p>
                  </div>
                </div>
              </section>

              <section className="panel-card panel-card-fill build-feedback-card">
                <div className={getVlmStatusFeedbackClass(status)}>
                  websocket: {wsState} · terminal: {terminal ? 'yes' : 'no'} · status: {status}
                </div>
                <ul className="build-feedback-list vlm-log-list">
                  {trainingLog.length === 0 ? (
                    <li className="build-feedback-item vlm-log-item">No live events yet.</li>
                  ) : (
                    trainingLog.map((line, index) => (
                      <li
                        key={`${line}-${index}`}
                        className={`build-feedback-item vlm-log-item ${line.includes('Error:') ? 'build-feedback-item-error' : ''}`}
                      >
                        {line}
                      </li>
                    ))
                  )}
                </ul>
              </section>

              <div className="panel-actions vlm-inline-actions">
                <button type="button" className="btn btn-success" onClick={runTrain} disabled={isBusy}>
                  {busyAction === 'train' ? 'Training...' : 'Train'}
                </button>
                <button type="button" className="btn btn-ghost" onClick={runStatus} disabled={isBusy || !jobId}>
                  {busyAction === 'status' ? 'Refreshing...' : 'Status'}
                </button>
                <button type="button" className="btn btn-danger" onClick={runStop} disabled={isBusy || !jobId}>
                  {busyAction === 'stop' ? 'Stopping...' : 'Stop'}
                </button>
              </div>
            </div>
          ) : null}

          {activeTab === 'test' ? (
            <div className="tab-panel">
              <section className="panel-card train-settings-card">
                <div className="config-grid">
                  <div className="config-row">
                    <span className="config-label">Camera</span>
                    <span className={`status-pill ${cameraReady ? 'status-pill-complete' : 'status-pill-idle'}`}>
                      {cameraReady ? 'ready' : 'off'}
                    </span>
                  </div>
                  <div className="config-row">
                    <span className="config-label">Live Mode</span>
                    <span className={`status-pill ${liveModeEnabled ? 'status-pill-training' : 'status-pill-idle'}`}>
                      {liveModeEnabled ? (liveInferenceRunning ? 'running' : 'starting') : 'paused'}
                    </span>
                  </div>
                  <div className="config-row">
                    <span className="config-label">Interval</span>
                    <span className="status-pill status-pill-idle">{LIVE_INFER_INTERVAL_MS}ms</span>
                  </div>
                </div>
                <div className="vlm-slider-wrap">
                  <div className="vlm-slider-head">
                    <span className="config-label">Detection Threshold</span>
                    <span className="status-pill status-pill-idle">{scoreThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    className="vlm-score-slider"
                    type="range"
                    min={0.05}
                    max={0.95}
                    step={0.01}
                    value={scoreThreshold}
                    onChange={(event) => setScoreThreshold(Number(event.target.value))}
                  />
                  <p className="vlm-slider-help">Higher values show fewer, higher-confidence detections.</p>
                </div>
                {runtimeWarning ? <p className="build-feedback-item build-feedback-item-warning">{runtimeWarning}</p> : null}

                <div className="panel-actions vlm-inline-actions vlm-test-actions">
                  <button type="button" className="btn btn-ghost" onClick={startCamera} disabled={isBusy || cameraReady}>
                    {busyAction === 'camera_start' ? 'Starting...' : 'Start Camera'}
                  </button>
                  <button type="button" className="btn btn-danger" onClick={stopCamera} disabled={!cameraReady}>
                    Stop Camera
                  </button>
                  <button type="button" className="btn btn-ghost" onClick={runDetect} disabled={isBusy || !cameraReady}>
                    {busyAction === 'detect' ? 'Detecting...' : 'Detect Once'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={() => setLiveModeEnabled((current) => !current)}
                    disabled={!cameraReady}
                  >
                    {liveModeEnabled ? 'Pause Live' : 'Resume Live'}
                  </button>
                  <button type="button" className="btn btn-ghost" onClick={clearDetections}>Clear</button>
                </div>
              </section>

              <section className="panel-card panel-card-fill vlm-test-panel">
                <div className="vlm-canvas-grid">
                  <div className="vlm-feed-card">
                    <p>Live Feed</p>
                    <video ref={videoRef} className="vlm-video" autoPlay muted playsInline />
                  </div>
                  <div className="vlm-feed-card">
                    <div className="vlm-feed-card-head">
                      <p>Detection Frame</p>
                      <button
                        type="button"
                        className="vlm-frame-expand-button"
                        aria-label="Open detection frame fullscreen"
                        title="Open fullscreen"
                        onClick={() => setIsDetectionModalOpen(true)}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                          <path d="M220-220h140v-80h-60v-60h-80v140Zm380 0h140v-140h-80v60h-60v80ZM220-600h80v-60h60v-80H220v140Zm440 0h80v-140H600v80h60v60ZM120-120v-300h80v220h220v80H120Zm420 0v-80h220v-220h80v300H540ZM120-540v-300h300v80H200v220h-80Zm640 0v-220H540v-80h300v300h-80Z"/>
                        </svg>
                      </button>
                    </div>
                    <canvas ref={canvasRef} className="vlm-canvas" />
                  </div>
                </div>
                <div className="vlm-detection-list">
                  {detections.length === 0 ? (
                    <p>No detection results yet.</p>
                  ) : (
                    detections.map((detection, index) => (
                      <div key={`${detection.label}-${index}`} className="vlm-detection-item">
                        <span>{detection.label}</span>
                        <span>{(detection.score * 100).toFixed(1)}%</span>
                      </div>
                    ))
                  )}
                </div>
              </section>
            </div>
          ) : null}

          <section className="panel-card build-feedback-card vlm-global-feedback">
            <p className="build-feedback-status build-feedback-status-idle">{message}</p>
            {lastImageSize ? (
              <p className="panel-muted-text panel-muted-text-tight">Last image size: {lastImageSize[0]} x {lastImageSize[1]}</p>
            ) : null}
          </section>
        </div>
      </section>

      <section className="app-viewport-panel vlm-viewport-panel">
        <button
          type="button"
          onClick={() => setIsSidebarCollapsed((prev) => !prev)}
          className="sidebar-toggle-button"
          aria-label={isSidebarCollapsed ? 'Expand left panel' : 'Collapse left panel'}
          title={isSidebarCollapsed ? 'Expand' : 'Collapse'}
        >
          {isSidebarCollapsed ? (
            <svg xmlns="http://www.w3.org/2000/svg" height="16px" viewBox="0 -960 960 960" width="16px" fill="#e3e3e3">
              <path d="m321-80-71-71 329-329-329-329 71-71 400 400L321-80Z" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" height="16px" viewBox="0 -960 960 960" width="16px" fill="#e3e3e3">
              <path d="M560-80 160-480l400-400 71 71-329 329 329 329-71 71Z" />
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

        <VLMArchitectureViewport
          key={viewportResetKey}
          architecture={architecture}
          selectedStageId={selectedStage?.id ?? null}
          onSelectStage={setSelectedStageId}
          lowDetailMode={isLowDetailMode}
          activityPulse={activityPulse}
        />

        <button
          type="button"
          onClick={handleAlignViewport}
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

        <button
          type="button"
          onClick={() => setIsLowDetailMode((prev) => !prev)}
          className={`detail-mode-button ${isLowDetailMode ? 'detail-mode-button-active' : ''}`}
          aria-pressed={isLowDetailMode}
          aria-label="Toggle low detail mode"
          title="Reduce 3D rendering detail"
        >
          <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e3e3e3" aria-hidden="true">
            <path d="M480-40q-50 0-85-35t-35-85q0-14 2.5-26.5T371-211L211-372q-12 5-25 8.5t-27 3.5q-50 0-84.5-35T40-480q0-50 34.5-85t84.5-35q39 0 70 22.5t43 57.5h95q9-26 28-44.5t45-27.5v-95q-35-12-57.5-43T360-800q0-50 35-85t85-35q50 0 85 35t35 85q0 14-3 27t-9 25l160 160q12-6 25-9t27-3q50 0 85 35t35 85q0 50-35 85t-85 35q-39 0-70-22.5T687-440h-95q-9 26-27.5 45T520-367v94q35 12 57.5 43t22.5 70q0 50-35 85t-85 35Zm-40-233v-94q-13-5-24-12t-20.5-16.5Q386-405 379-416t-12-24h-95q0 1-.5 2.5l-1 3q-.5 1.5-1 2.5l-1.5 3 29 29q24 24 51 51.5t51 51.5l29 29q2-1 3.5-1.5t3-1.5 3-1.5q1.5-.5 2.5-.5Zm152-247h95q0-2 .5-3.5t1.5-3 1.5-3q.5-1.5 1.5-2.5l-29-29-51-51-51-51-29-29q-1 1-2.5 1.5t-3 1.5-3 1.5q-1.5.5-3.5.5v95q12 4 23.5 11.5T564-564q9 9 16.5 20.5T592-520Zm208 80q17 0 28.5-11.5T840-480q0-17-11.5-28.5T800-520q-17 0-28.5 11.5T760-480q0 17 11.5 28.5T800-440Zm-320 0q17 0 28.5-11.5T520-480q0-17-11.5-28.5T480-520q-17 0-28.5 11.5T440-480q0 17 11.5 28.5T480-440Zm0 320q17 0 28.5-11.5T520-160q0-17-11.5-28.5T480-200q-17 0-28.5 11.5T440-160q0 17 11.5 28.5T480-120ZM160-440q17 0 28.5-11.5T200-480q0-17-11.5-28.5T160-520q-17 0-28.5 11.5T120-480q0 17 11.5 28.5T160-440Zm320-320q17 0 28.5-11.5T520-800q0-17-11.5-28.5T480-840q-17 0-28.5 11.5T440-800q0 17 11.5 28.5T480-760Z" />
          </svg>
        </button>
      </section>

      {isDetectionModalOpen ? (
        <div
          className="vlm-frame-modal-overlay"
          role="presentation"
          onClick={() => setIsDetectionModalOpen(false)}
        >
          <div
            className="vlm-frame-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Detection frame fullscreen"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              className="vlm-frame-modal-close"
              aria-label="Close fullscreen detection frame"
              title="Close"
              onClick={() => setIsDetectionModalOpen(false)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" height="22px" viewBox="0 -960 960 960" width="22px" fill="#e3e3e3">
                <path d="m251.33-203.33-48-48L432-480 203.33-708.67l48-48L480-528l228.67-228.67 48 48L528-480l228.67 228.67-48 48L480-432 251.33-203.33Z"/>
              </svg>
            </button>
            <canvas ref={fullscreenCanvasRef} className="vlm-frame-modal-canvas" />
          </div>
        </div>
      ) : null}
    </div>
  )
}

function smoothDetections(
  previous: SmoothedDetection[],
  incoming: VLMDetection[],
  nowMs: number
): SmoothedDetection[] {
  const matchedPrevious = new Set<number>()
  const next: SmoothedDetection[] = []

  for (const detection of incoming) {
    let bestIndex = -1
    let bestDistance = Number.POSITIVE_INFINITY

    for (let index = 0; index < previous.length; index += 1) {
      if (matchedPrevious.has(index)) continue
      const candidate = previous[index]
      if (candidate.label_id !== detection.label_id) continue

      const distance = detectionCenterDistance(candidate.box, detection.box)
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = index
      }
    }

    if (bestIndex >= 0 && bestDistance <= 140) {
      const prior = previous[bestIndex]
      matchedPrevious.add(bestIndex)
      const confidence = Math.min(1, prior.confidence * 0.6 + 0.5)
      next.push({
        ...detection,
        box: lerpBox(prior.box, detection.box, 0.45),
        score: prior.score * 0.35 + detection.score * 0.65,
        confidence,
        lastSeenMs: nowMs,
      })
      continue
    }

    next.push({
      ...detection,
      confidence: 0.6,
      lastSeenMs: nowMs,
    })
  }

  for (let index = 0; index < previous.length; index += 1) {
    if (matchedPrevious.has(index)) continue
    const prior = previous[index]
    const ageMs = nowMs - prior.lastSeenMs
    if (ageMs > 650) continue
    const fade = Math.max(0, 1 - ageMs / 650)
    const confidence = prior.confidence * 0.72 * fade
    if (confidence <= 0.08) continue
    next.push({
      ...prior,
      confidence,
    })
  }

  return next
    .sort((a, b) => (b.confidence - a.confidence) || (b.score - a.score))
    .slice(0, LIVE_INFER_MAX_DETECTIONS)
}

function detectionCenterDistance(
  boxA: [number, number, number, number],
  boxB: [number, number, number, number]
): number {
  const centerAX = (boxA[0] + boxA[2]) * 0.5
  const centerAY = (boxA[1] + boxA[3]) * 0.5
  const centerBX = (boxB[0] + boxB[2]) * 0.5
  const centerBY = (boxB[1] + boxB[3]) * 0.5
  return Math.hypot(centerAX - centerBX, centerAY - centerBY)
}

function lerpBox(
  from: [number, number, number, number],
  to: [number, number, number, number],
  t: number
): [number, number, number, number] {
  const clampedT = Math.min(1, Math.max(0, t))
  return [
    from[0] + (to[0] - from[0]) * clampedT,
    from[1] + (to[1] - from[1]) * clampedT,
    from[2] + (to[2] - from[2]) * clampedT,
    from[3] + (to[3] - from[3]) * clampedT,
  ]
}

function drawDetections(
  context: CanvasRenderingContext2D,
  detections: ReadonlyArray<VLMDetection | SmoothedDetection>
): void {
  context.lineWidth = 2
  context.font = '15px Space Grotesk'

  detections.forEach((detection) => {
    const confidence = 'confidence' in detection
      ? Math.min(1, Math.max(0.08, detection.confidence))
      : 1
    const [x1, y1, x2, y2] = detection.box
    const width = Math.max(1, x2 - x1)
    const height = Math.max(1, y2 - y1)

    context.lineWidth = 1.2 + confidence * 1.3
    context.strokeStyle = `rgba(255, 180, 41, ${0.35 + confidence * 0.6})`
    context.fillStyle = `rgba(255, 180, 41, ${0.08 + confidence * 0.16})`
    context.strokeRect(x1, y1, width, height)
    context.fillRect(x1, y1, width, height)

    const label = `${detection.label} ${(detection.score * 100).toFixed(1)}%`
    context.fillStyle = `rgba(0, 0, 0, ${0.68 + confidence * 0.22})`
    const textWidth = context.measureText(label).width
    context.fillRect(x1, Math.max(0, y1 - 22), textWidth + 12, 20)
    context.fillStyle = `rgba(255, 210, 122, ${0.72 + confidence * 0.28})`
    context.fillText(label, x1 + 6, Math.max(14, y1 - 7))
  })
}

function stopCameraStream(streamRef: MutableRefObject<MediaStream | null>): void {
  const stream = streamRef.current
  if (!stream) return
  stream.getTracks().forEach((track) => track.stop())
  streamRef.current = null
}

function stopCameraStreamRef(stream: MediaStream): void {
  stream.getTracks().forEach((track) => track.stop())
}

function buildVlmWsUrl(jobId: string): string {
  const envWsBase = import.meta.env.VITE_BACKEND_WS_URL?.trim()
  if (envWsBase) {
    return `${trimTrailingSlash(envWsBase)}/ws/vlm/training/${jobId}`
  }

  const envHttpBase = import.meta.env.VITE_BACKEND_HTTP_URL?.trim()
  if (envHttpBase) {
    const wsBase = envHttpBase
      .replace(/^https:\/\//, 'wss://')
      .replace(/^http:\/\//, 'ws://')
    return `${trimTrailingSlash(wsBase)}/ws/vlm/training/${jobId}`
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  if (window.location.port === '5173' && isLocalHostname(window.location.hostname)) {
    return `${protocol}//${window.location.hostname}:8000/ws/vlm/training/${jobId}`
  }
  return `${protocol}//${window.location.host}/ws/vlm/training/${jobId}`
}

function trimTrailingSlash(value: string): string {
  return value.endsWith('/') ? value.slice(0, -1) : value
}

function isLocalHostname(hostname: string): boolean {
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '0.0.0.0'
}

function getStageDotClass(index: number, total: number): string {
  if (index === 0) return 'layer-dot layer-dot-input'
  if (index === total - 1) return 'layer-dot layer-dot-output'
  return 'layer-dot layer-dot-hidden'
}

function getVlmStatusFeedbackClass(status: string): string {
  if (status === 'completed') {
    return 'build-feedback-status build-feedback-status-success'
  }
  if (status === 'failed' || status === 'error') {
    return 'build-feedback-status build-feedback-status-error'
  }
  return 'build-feedback-status build-feedback-status-idle'
}
