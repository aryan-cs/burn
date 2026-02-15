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
  const [scoreThreshold, setScoreThreshold] = useState(0.45)
  const [lastImageSize, setLastImageSize] = useState<[number, number] | null>(null)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const liveInferenceBusyRef = useRef(false)
  const liveUiTickRef = useRef(0)

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

      drawDetections(context, response.detections)
      const shouldSyncUi = !silent || liveUiTickRef.current % 3 === 0
      liveUiTickRef.current += 1
      if (shouldSyncUi) {
        setDetections(response.detections)
        setLastImageSize([response.image_width, response.image_height])
        setRuntimeWarning(response.warning ?? null)
      }

      if (!silent) {
        const runtimeLabel = `${response.runtime_backend} · ${response.runtime_model_id}`
        setMessage(
          response.detections.length > 0
            ? `Detected ${response.detections.length} object(s) using ${runtimeLabel}.`
            : `No objects detected above threshold (${runtimeLabel}).`
        )
      }
    },
    [cameraReady, jobId, scoreThreshold]
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
                <div className="config-grid">
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

                <div className="panel-actions-split">
                  <button type="button" className="btn btn-ghost vlm-action-button" onClick={() => setActiveTab('train')}>
                    Go To Train
                  </button>
                  <button type="button" className="btn btn-ghost vlm-action-button" onClick={() => setActiveTab('test')}>
                    Go To Test
                  </button>
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
                  <div className="config-row">
                    <span className="config-label">Threshold</span>
                    <span className="status-pill status-pill-idle">{scoreThreshold.toFixed(2)}</span>
                  </div>
                </div>
                <div className="vlm-slider-wrap">
                  <input
                    className="vlm-score-slider"
                    type="range"
                    min={0.05}
                    max={0.95}
                    step={0.01}
                    value={scoreThreshold}
                    onChange={(event) => setScoreThreshold(Number(event.target.value))}
                  />
                </div>
                {runtimeWarning ? <p className="build-feedback-item build-feedback-item-warning">{runtimeWarning}</p> : null}

                <div className="panel-actions vlm-inline-actions">
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
                    <p>Detection Frame</p>
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

        <VLMArchitectureViewport
          architecture={architecture}
          selectedStageId={selectedStage?.id ?? null}
          onSelectStage={setSelectedStageId}
          lowDetailMode={isLowDetailMode}
          activityPulse={activityPulse}
        />

        <div className="vlm-viewport-topbar">
          <div className="vlm-viewport-heading">{architecture.name}</div>
          <div className="vlm-viewport-chips">
            <span className="vlm-chip">dataset: {datasetName}</span>
            <span className="vlm-chip">status: {status}</span>
            <span className="vlm-chip">ws: {wsState}</span>
          </div>
        </div>

        {selectedStage ? (
          <div className="vlm-stage-focus">
            <div className="vlm-stage-focus-title">{selectedStage.label}</div>
            <div className="vlm-stage-focus-detail">{selectedStage.detail}</div>
            <p>{selectedStage.description}</p>
          </div>
        ) : null}

        <button
          type="button"
          onClick={() => setIsLowDetailMode((prev) => !prev)}
          className={`detail-mode-button ${isLowDetailMode ? 'detail-mode-button-active' : ''}`}
          aria-pressed={isLowDetailMode}
          aria-label="Toggle low detail mode"
          title="Reduce 3D rendering detail"
        >
          <svg xmlns="http://www.w3.org/2000/svg" height="16px" viewBox="0 -960 960 960" width="16px" fill="currentColor" aria-hidden="true">
            <path d="M120-200v-80h240v80H120Zm0-200v-80h480v80H120Zm0-200v-80h720v80H120Z" />
          </svg>
          <span>{isLowDetailMode ? 'Low Detail' : 'Full Detail'}</span>
        </button>
      </section>
    </div>
  )
}

function drawDetections(context: CanvasRenderingContext2D, detections: VLMDetection[]): void {
  context.lineWidth = 2
  context.font = '15px Space Grotesk'

  detections.forEach((detection) => {
    const [x1, y1, x2, y2] = detection.box
    const width = Math.max(1, x2 - x1)
    const height = Math.max(1, y2 - y1)

    context.strokeStyle = '#ffb429'
    context.fillStyle = 'rgba(255, 180, 41, 0.15)'
    context.strokeRect(x1, y1, width, height)
    context.fillRect(x1, y1, width, height)

    const label = `${detection.label} ${(detection.score * 100).toFixed(1)}%`
    context.fillStyle = 'rgba(0, 0, 0, 0.78)'
    const textWidth = context.measureText(label).width
    context.fillRect(x1, Math.max(0, y1 - 22), textWidth + 12, 20)
    context.fillStyle = '#ffd27a'
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
