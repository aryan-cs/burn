import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTrainingStore, type EpochMetric } from '../store/trainingStore'
import { serializeForBackend } from '../utils/graphSerializer'
import {
  InferencePixelPad,
  countActivePixels,
  createEmptyInferenceGrid,
  inferenceGridToPayload,
} from './InferencePixelPad'

type LogLevel = 'info' | 'success' | 'warn' | 'error'

interface LogEntry {
  id: number
  at: string
  level: LogLevel
  message: string
}

interface ValidationResponse {
  valid: boolean
  shapes: Record<string, { input: number[] | null; output: number[] | null }>
  errors: Array<{ message: string; node_id?: string }>
  execution_order: string[]
  warnings: string[]
}

interface CompileResponse {
  valid: boolean
  summary: {
    param_count: number
    layers: Array<Record<string, unknown>>
    resolved_training: Record<string, unknown>
  }
  python_source: string
  warnings: string[]
}

interface TrainResponse {
  job_id: string
  status: string
}

interface StatusResponse {
  job_id: string
  status: string
  terminal: boolean
  error: string | null
  final_metrics: Record<string, number> | null
  has_python_source: boolean
  has_artifact: boolean
}

interface InferResponse {
  job_id: string
  input_shape: number[]
  output_shape: number[]
  logits: number[][]
  predictions?: number[]
  probabilities?: number[][]
}

function levelClass(level: LogLevel): string {
  switch (level) {
    case 'success':
      return 'text-emerald-300'
    case 'warn':
      return 'text-amber-300'
    case 'error':
      return 'text-rose-300'
    default:
      return 'text-sky-200'
  }
}

function statusClass(status: string): string {
  switch (status) {
    case 'training':
      return 'text-amber-200 border-amber-300/40 bg-amber-500/20'
    case 'complete':
      return 'text-emerald-200 border-emerald-300/40 bg-emerald-500/20'
    case 'error':
      return 'text-rose-200 border-rose-300/40 bg-rose-500/20'
    case 'idle':
    default:
      return 'text-slate-200 border-slate-300/30 bg-slate-500/20'
  }
}

function formatMetric(metric: EpochMetric): string {
  const trainLoss = metric.trainLoss ?? metric.loss
  const trainAcc = metric.trainAccuracy ?? metric.accuracy
  const testLoss = metric.testLoss ?? metric.loss
  const testAcc = metric.testAccuracy ?? metric.accuracy
  return `epoch ${metric.epoch} train_loss=${trainLoss.toFixed(4)} train_acc=${(
    trainAcc * 100
  ).toFixed(2)}% test_loss=${testLoss.toFixed(4)} test_acc=${(testAcc * 100).toFixed(2)}%`
}

function triggerTextDownload(filename: string, text: string): void {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

function triggerBlobDownload(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

function errorTextFromBody(body: string): string {
  try {
    const parsed = JSON.parse(body) as {
      detail?: { message?: string; errors?: Array<{ message?: string }> } | string
      message?: string
    }

    if (typeof parsed.detail === 'string') return parsed.detail
    if (parsed.detail?.message) return parsed.detail.message
    if (parsed.detail?.errors?.[0]?.message) return parsed.detail.errors[0].message ?? body
    if (parsed.message) return parsed.message
    return body
  } catch {
    return body
  }
}

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
    throw new Error(errorTextFromBody(text || `HTTP ${response.status}`))
  }

  return JSON.parse(text) as T
}

async function requestText(path: string): Promise<string> {
  const response = await fetch(path, { headers: { Accept: '*/*' } })
  const text = await response.text()
  if (!response.ok) {
    throw new Error(errorTextFromBody(text || `HTTP ${response.status}`))
  }
  return text
}

async function requestBlob(path: string): Promise<Blob> {
  const response = await fetch(path, { headers: { Accept: '*/*' } })
  const body = await response.blob()
  if (!response.ok) {
    const text = await body.text()
    throw new Error(errorTextFromBody(text || `HTTP ${response.status}`))
  }
  return body
}

function nowTime(): string {
  return new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

export function BackendWorkbench() {
  const status = useTrainingStore((s) => s.status)
  const jobId = useTrainingStore((s) => s.jobId)
  const metrics = useTrainingStore((s) => s.metrics)
  const config = useTrainingStore((s) => s.config)
  const setConfig = useTrainingStore((s) => s.setConfig)
  const startTraining = useTrainingStore((s) => s.startTraining)
  const setStatus = useTrainingStore((s) => s.setStatus)
  const setError = useTrainingStore((s) => s.setError)

  const [collapsed, setCollapsed] = useState(false)
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [validationData, setValidationData] = useState<ValidationResponse | null>(null)
  const [compileData, setCompileData] = useState<CompileResponse | null>(null)
  const [statusData, setStatusData] = useState<StatusResponse | null>(null)
  const [inferenceGrid, setInferenceGrid] = useState<number[][]>(() =>
    createEmptyInferenceGrid()
  )
  const [inferenceOutput, setInferenceOutput] = useState<string>('')
  const [logs, setLogs] = useState<LogEntry[]>([])

  const logCounter = useRef(1)
  const seenMetricsCount = useRef(0)
  const logsContainerRef = useRef<HTMLDivElement | null>(null)

  const effectiveJobId = activeJobId ?? jobId
  const isBusy = busyAction !== null

  const addLog = useCallback((level: LogLevel, message: string) => {
    setLogs((prev) => {
      const nextEntry: LogEntry = {
        id: logCounter.current++,
        at: nowTime(),
        level,
        message,
      }
      return [...prev.slice(-299), nextEntry]
    })
  }, [])

  useEffect(() => {
    if (jobId) {
      setActiveJobId(jobId)
    }
  }, [jobId])

  useEffect(() => {
    const container = logsContainerRef.current
    if (container) {
      container.scrollTop = container.scrollHeight
    }
  }, [logs])

  useEffect(() => {
    if (metrics.length <= seenMetricsCount.current) return

    const newMetrics = metrics.slice(seenMetricsCount.current)
    newMetrics.forEach((metric) => {
      addLog('info', formatMetric(metric))
    })
    seenMetricsCount.current = metrics.length
  }, [metrics, addLog])

  const statusLabel = useMemo(() => status.toUpperCase(), [status])
  const activePixelCount = useMemo(() => countActivePixels(inferenceGrid), [inferenceGrid])
  const payloadPreview = useMemo(
    () => JSON.stringify(inferenceGridToPayload(inferenceGrid), null, 1),
    [inferenceGrid]
  )

  const buildPayload = useCallback(() => serializeForBackend(), [])

  const validateGraph = useCallback(async (): Promise<ValidationResponse> => {
    addLog('info', 'POST /api/model/validate')
    const payload = buildPayload()
    const response = await requestJson<ValidationResponse>('/api/model/validate', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
    setValidationData(response)

    if (!response.valid) {
      addLog('error', `Validation failed (${response.errors.length} errors)`)
    } else {
      addLog('success', `Validation passed (${response.execution_order.length} nodes)`)
    }

    return response
  }, [addLog, buildPayload])

  const compileGraph = useCallback(async (): Promise<CompileResponse> => {
    addLog('info', 'POST /api/model/compile')
    const payload = buildPayload()
    const response = await requestJson<CompileResponse>('/api/model/compile', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
    setCompileData(response)

    const warningCount = response.warnings.length
    addLog(
      warningCount > 0 ? 'warn' : 'success',
      `Compile success (params=${response.summary.param_count}, warnings=${warningCount})`
    )

    return response
  }, [addLog, buildPayload])

  const trainGraph = useCallback(async (): Promise<string> => {
    addLog('info', 'POST /api/model/train')
    const payload = buildPayload()
    const response = await requestJson<TrainResponse>('/api/model/train', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
    setActiveJobId(response.job_id)
    startTraining(response.job_id, config.epochs)
    addLog('success', `Training started (job_id=${response.job_id})`)
    return response.job_id
  }, [addLog, buildPayload, config.epochs, startTraining])

  const fetchJobStatus = useCallback(
    async (targetJobId: string): Promise<StatusResponse> => {
      const response = await requestJson<StatusResponse>(
        `/api/model/status?job_id=${encodeURIComponent(targetJobId)}`
      )
      setStatusData(response)
      return response
    },
    []
  )

  const waitForTerminal = useCallback(
    async (targetJobId: string, timeoutMs = 15 * 60 * 1000): Promise<StatusResponse> => {
      const started = Date.now()
      while (Date.now() - started < timeoutMs) {
        const statusResponse = await fetchJobStatus(targetJobId)
        if (statusResponse.terminal) {
          return statusResponse
        }
        await sleep(1000)
      }

      throw new Error(`Timed out waiting for job ${targetJobId} to become terminal`)
    },
    [fetchJobStatus]
  )

  const exportPython = useCallback(
    async (targetJobId: string): Promise<void> => {
      const text = await requestText(
        `/api/model/export?job_id=${encodeURIComponent(targetJobId)}&format=py`
      )
      triggerTextDownload(`${targetJobId}.py`, text)
      addLog('success', `Downloaded ${targetJobId}.py`)
    },
    [addLog]
  )

  const exportPt = useCallback(
    async (targetJobId: string): Promise<void> => {
      const blob = await requestBlob(
        `/api/model/export?job_id=${encodeURIComponent(targetJobId)}&format=pt`
      )
      triggerBlobDownload(`${targetJobId}.pt`, blob)
      addLog('success', `Downloaded ${targetJobId}.pt`)
    },
    [addLog]
  )

  const infer = useCallback(
    async (targetJobId: string): Promise<InferResponse> => {
      const parsedInput = inferenceGridToPayload(inferenceGrid)

      addLog('info', 'POST /api/model/infer')
      const response = await requestJson<InferResponse>('/api/model/infer', {
        method: 'POST',
        body: JSON.stringify({
          job_id: targetJobId,
          inputs: parsedInput,
          return_probabilities: true,
        }),
      })

      setInferenceOutput(JSON.stringify(response, null, 2))
      const topPrediction = response.predictions?.[0]
      addLog('success', `Inference complete${topPrediction !== undefined ? ` (pred=${topPrediction})` : ''}`)

      return response
    },
    [addLog, inferenceGrid]
  )

  const runAction = useCallback(
    async (name: string, fn: () => Promise<void>) => {
      if (isBusy) return
      setBusyAction(name)
      try {
        await fn()
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        addLog('error', message)
      } finally {
        setBusyAction(null)
      }
    },
    [addLog, isBusy]
  )

  const handleValidate = () =>
    runAction('validate', async () => {
      await validateGraph()
    })

  const handleCompile = () =>
    runAction('compile', async () => {
      await compileGraph()
    })

  const handleTrain = () =>
    runAction('train', async () => {
      await trainGraph()
    })

  const handleStop = () =>
    runAction('stop', async () => {
      if (!effectiveJobId) {
        throw new Error('No active job id to stop')
      }
      await requestJson<{ job_id: string; status: string }>('/api/model/stop', {
        method: 'POST',
        body: JSON.stringify({ job_id: effectiveJobId }),
      })
      setStatus('idle')
      addLog('warn', `Stop requested for ${effectiveJobId}`)
    })

  const handleRefreshStatus = () =>
    runAction('status', async () => {
      if (!effectiveJobId) {
        throw new Error('No active job id to query')
      }
      const result = await fetchJobStatus(effectiveJobId)
      addLog('info', `Status: ${result.status} (terminal=${result.terminal})`)
    })

  const handleExportPy = () =>
    runAction('export-py', async () => {
      if (effectiveJobId) {
        await exportPython(effectiveJobId)
        return
      }

      if (!compileData) {
        throw new Error('No active job and no compile output to export')
      }

      triggerTextDownload('generated_model.py', compileData.python_source)
      addLog('warn', 'Downloaded generated_model.py from compile output (no job id)')
    })

  const handleExportPt = () =>
    runAction('export-pt', async () => {
      if (!effectiveJobId) {
        throw new Error('No active job id for .pt export')
      }
      await exportPt(effectiveJobId)
    })

  const handleInfer = () =>
    runAction('infer', async () => {
      if (!effectiveJobId) {
        throw new Error('No active job id for inference')
      }
      await infer(effectiveJobId)
    })

  const handlePipeline = () =>
    runAction('pipeline', async () => {
      const validation = await validateGraph()
      if (!validation.valid) {
        throw new Error('Pipeline stopped: validation failed')
      }

      await compileGraph()
      const createdJobId = await trainGraph()
      const terminal = await waitForTerminal(createdJobId)
      addLog('info', `Terminal status: ${terminal.status}`)

      if (terminal.status === 'failed') {
        setError(terminal.error ?? 'Training failed')
        throw new Error(terminal.error ?? 'Pipeline stopped: training failed')
      }

      await exportPython(createdJobId)
      await exportPt(createdJobId)
      await infer(createdJobId)
      addLog('success', 'Pipeline complete')
    })

  const handleClearLogs = () => {
    setLogs([])
    addLog('info', 'Log buffer cleared')
  }

  return (
    <div className="absolute left-4 right-4 top-4 z-20 md:left-52 md:right-auto md:w-[38rem]">
      <div className="rounded-2xl border border-sky-300/20 bg-[#0f1720]/92 p-3 shadow-[0_24px_80px_-32px_rgba(0,200,255,0.55)] backdrop-blur-xl">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-sky-200/70">
              Backend Interface
            </div>
            <h2 className="font-mono text-lg text-sky-100">Model Orchestrator</h2>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`rounded-md border px-2 py-1 font-mono text-[10px] ${statusClass(
                status
              )}`}
            >
              {statusLabel}
            </span>
            <button
              onClick={() => setCollapsed((v) => !v)}
              className="rounded-md border border-white/20 bg-white/5 px-2 py-1 font-mono text-[11px] text-white/80 transition hover:bg-white/10"
            >
              {collapsed ? 'Expand' : 'Collapse'}
            </button>
          </div>
        </div>

        {!collapsed && (
          <>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
              <button
                onClick={handlePipeline}
                disabled={isBusy}
                className="col-span-2 rounded-lg border border-cyan-300/40 bg-cyan-500/20 px-3 py-2 font-mono text-xs text-cyan-100 transition hover:bg-cyan-500/30 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {busyAction === 'pipeline' ? 'Running Pipeline...' : 'Run Full Pipeline'}
              </button>
              <button
                onClick={handleValidate}
                disabled={isBusy}
                className="rounded-lg border border-white/20 bg-white/5 px-2 py-2 font-mono text-xs text-white/90 transition hover:bg-white/10 disabled:opacity-50"
              >
                Validate
              </button>
              <button
                onClick={handleCompile}
                disabled={isBusy}
                className="rounded-lg border border-white/20 bg-white/5 px-2 py-2 font-mono text-xs text-white/90 transition hover:bg-white/10 disabled:opacity-50"
              >
                Compile
              </button>
              <button
                onClick={handleTrain}
                disabled={isBusy}
                className="rounded-lg border border-emerald-300/30 bg-emerald-500/20 px-2 py-2 font-mono text-xs text-emerald-100 transition hover:bg-emerald-500/30 disabled:opacity-50"
              >
                Train
              </button>
              <button
                onClick={handleStop}
                disabled={isBusy}
                className="rounded-lg border border-rose-300/30 bg-rose-500/20 px-2 py-2 font-mono text-xs text-rose-100 transition hover:bg-rose-500/30 disabled:opacity-50"
              >
                Stop
              </button>
              <button
                onClick={handleRefreshStatus}
                disabled={isBusy}
                className="rounded-lg border border-white/20 bg-white/5 px-2 py-2 font-mono text-xs text-white/90 transition hover:bg-white/10 disabled:opacity-50"
              >
                Status
              </button>
              <button
                onClick={handleExportPy}
                disabled={isBusy}
                className="rounded-lg border border-white/20 bg-white/5 px-2 py-2 font-mono text-xs text-white/90 transition hover:bg-white/10 disabled:opacity-50"
              >
                Export .py
              </button>
              <button
                onClick={handleExportPt}
                disabled={isBusy}
                className="rounded-lg border border-white/20 bg-white/5 px-2 py-2 font-mono text-xs text-white/90 transition hover:bg-white/10 disabled:opacity-50"
              >
                Export .pt
              </button>
              <button
                onClick={handleInfer}
                disabled={isBusy}
                className="rounded-lg border border-indigo-300/30 bg-indigo-500/20 px-2 py-2 font-mono text-xs text-indigo-100 transition hover:bg-indigo-500/30 disabled:opacity-50"
              >
                Infer
              </button>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2 font-mono text-xs text-white/80">
              <label className="flex items-center justify-between rounded-lg border border-white/10 bg-black/20 px-2 py-1.5">
                <span className="text-white/55">Dataset</span>
                <select
                  value={config.dataset}
                  onChange={(e) => setConfig({ dataset: e.target.value })}
                  disabled={isBusy}
                  className="w-24 rounded border border-white/20 bg-black/40 px-1 py-0.5 text-right text-white"
                >
                  <option value="mnist">mnist</option>
                </select>
              </label>
              <label className="flex items-center justify-between rounded-lg border border-white/10 bg-black/20 px-2 py-1.5">
                <span className="text-white/55">Epochs</span>
                <input
                  type="number"
                  value={config.epochs}
                  min={1}
                  onChange={(e) => setConfig({ epochs: Number(e.target.value) })}
                  disabled={isBusy}
                  className="w-24 rounded border border-white/20 bg-black/40 px-1 py-0.5 text-right text-white"
                />
              </label>
              <label className="flex items-center justify-between rounded-lg border border-white/10 bg-black/20 px-2 py-1.5">
                <span className="text-white/55">Batch</span>
                <input
                  type="number"
                  value={config.batchSize}
                  min={1}
                  onChange={(e) => setConfig({ batchSize: Number(e.target.value) })}
                  disabled={isBusy}
                  className="w-24 rounded border border-white/20 bg-black/40 px-1 py-0.5 text-right text-white"
                />
              </label>
              <label className="flex items-center justify-between rounded-lg border border-white/10 bg-black/20 px-2 py-1.5">
                <span className="text-white/55">LR</span>
                <input
                  type="number"
                  step={0.0001}
                  value={config.learningRate}
                  min={0.000001}
                  onChange={(e) => setConfig({ learningRate: Number(e.target.value) })}
                  disabled={isBusy}
                  className="w-24 rounded border border-white/20 bg-black/40 px-1 py-0.5 text-right text-white"
                />
              </label>
            </div>

            <div className="mt-3 grid gap-2 md:grid-cols-[2fr_1fr]">
              <InferencePixelPad
                grid={inferenceGrid}
                setGrid={setInferenceGrid}
                disabled={isBusy}
              />
              <div className="min-w-0 rounded-lg border border-white/10 bg-black/25 p-2">
                <div className="mb-1 font-mono text-[11px] text-white/60">Inference Input</div>
                <div className="space-y-1 font-mono text-[11px] text-white/75">
                  <div className="rounded border border-white/10 bg-[#04070d] px-2 py-1">
                    shape: [1, 28, 28]
                  </div>
                  <div className="rounded border border-white/10 bg-[#04070d] px-2 py-1">
                    active pixels: {activePixelCount} / 784
                  </div>
                </div>
                <div className="mt-2 text-[10px] text-white/45">
                  payload preview
                </div>
                <pre className="mt-1 max-h-28 overflow-auto whitespace-pre-wrap break-words rounded border border-white/10 bg-[#04070d] p-2 font-mono text-[9px] text-cyan-100">
                  {payloadPreview}
                </pre>
              </div>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2 font-mono text-[11px] text-white/70">
              <div className="rounded-md border border-white/10 bg-black/20 px-2 py-1">
                <div className="text-white/50">Job ID</div>
                <div className="truncate text-cyan-200">{effectiveJobId ?? '—'}</div>
              </div>
              <div className="rounded-md border border-white/10 bg-black/20 px-2 py-1">
                <div className="text-white/50">Compile Params</div>
                <div className="text-cyan-100">{compileData?.summary.param_count ?? '—'}</div>
              </div>
              <div className="rounded-md border border-white/10 bg-black/20 px-2 py-1">
                <div className="text-white/50">Validation</div>
                <div className={validationData?.valid ? 'text-emerald-200' : 'text-rose-200'}>
                  {validationData ? (validationData.valid ? 'valid' : 'invalid') : '—'}
                </div>
              </div>
              <div className="rounded-md border border-white/10 bg-black/20 px-2 py-1">
                <div className="text-white/50">Terminal</div>
                <div className="text-cyan-100">{statusData ? String(statusData.terminal) : '—'}</div>
              </div>
            </div>

            <div className="mt-3 rounded-lg border border-white/10 bg-[#04070d]">
              <div className="flex items-center justify-between border-b border-white/10 px-2 py-1.5">
                <span className="font-mono text-[11px] uppercase tracking-[0.08em] text-white/60">
                  Operation Log
                </span>
                <button
                  onClick={handleClearLogs}
                  className="rounded border border-white/15 bg-white/5 px-2 py-0.5 font-mono text-[10px] text-white/70 transition hover:bg-white/10"
                >
                  Clear
                </button>
              </div>
              <div
                ref={logsContainerRef}
                className="h-40 overflow-y-auto px-2 py-1.5 font-mono text-[11px] leading-5"
              >
                {logs.length === 0 ? (
                  <div className="text-white/35">No logs yet. Run a step to begin.</div>
                ) : (
                  logs.map((entry) => (
                    <div key={entry.id} className={levelClass(entry.level)}>
                      <span className="mr-2 text-white/35">[{entry.at}]</span>
                      {entry.message}
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="mt-3 rounded-lg border border-white/10 bg-black/25 p-2">
              <div className="mb-1 font-mono text-[11px] text-white/60">Inference Output</div>
              <pre className="max-h-36 overflow-auto whitespace-pre-wrap break-words rounded-md border border-white/10 bg-[#04070d] p-2 font-mono text-[10px] text-cyan-100">
                {inferenceOutput || 'No inference output yet.'}
              </pre>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
