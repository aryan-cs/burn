import { useCallback, useEffect, useMemo, useState } from 'react'
import './rf.css'
import { RfChartsPanel } from './components/RfChartsPanel'
import { RfDatasetPanel } from './components/RfDatasetPanel'
import { RfGraphView } from './components/RfGraphView'
import { RfInferencePanel } from './components/RfInferencePanel'
import { RfLogsPanel } from './components/RfLogsPanel'
import { RfTrainingPanel } from './components/RfTrainingPanel'
import { useRfWebSocket } from './hooks/useRfWebSocket'
import { datasetDefaults, useRFGraphStore } from './store/rfGraphStore'
import { useRFRunStore } from './store/rfRunStore'
import { serializeRFGraph } from './utils/rfSerializer'
import type {
  RFCompileResponse,
  RFDatasetMeta,
  RFInferResponse,
  RFStatusResponse,
  RFTrainResponse,
  RFValidationResponse,
} from './types'

function errorTextFromBody(body: string): string {
  try {
    const parsed = JSON.parse(body) as {
      detail?: { message?: string; errors?: Array<{ message?: string }> } | string
      message?: string
      errors?: Array<{ message?: string }>
    }
    if (typeof parsed.detail === 'string') return parsed.detail
    if (parsed.detail?.message) return parsed.detail.message
    if (parsed.errors?.[0]?.message) return parsed.errors[0].message ?? body
    if (parsed.detail?.errors?.[0]?.message) return parsed.detail.errors[0].message ?? body
    if (parsed.message) return parsed.message
    return body
  } catch {
    return body
  }
}

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
    throw new Error(errorTextFromBody(text || `HTTP ${response.status}`))
  }
  return JSON.parse(text) as T
}

async function requestBlob(path: string): Promise<Blob> {
  const response = await fetch(path)
  const blob = await response.blob()
  if (!response.ok) {
    const text = await blob.text()
    throw new Error(errorTextFromBody(text || `HTTP ${response.status}`))
  }
  return blob
}

function triggerDownload(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

export default function RandomForestPage() {
  useRfWebSocket()

  const nodesMap = useRFGraphStore((state) => state.nodes)
  const training = useRFGraphStore((state) => state.training)
  const setDataset = useRFGraphStore((state) => state.setDataset)
  const setTraining = useRFGraphStore((state) => state.setTraining)

  const status = useRFRunStore((state) => state.status)
  const jobId = useRFRunStore((state) => state.jobId)
  const compileData = useRFRunStore((state) => state.compileData)
  const inferenceData = useRFRunStore((state) => state.inferenceData)
  const progress = useRFRunStore((state) => state.progress)
  const finalResult = useRFRunStore((state) => state.finalResult)
  const logs = useRFRunStore((state) => state.logs)
  const addLog = useRFRunStore((state) => state.addLog)
  const clearLogs = useRFRunStore((state) => state.clearLogs)
  const setStatus = useRFRunStore((state) => state.setStatus)
  const setJobId = useRFRunStore((state) => state.setJobId)
  const setValidation = useRFRunStore((state) => state.setValidation)
  const setCompileData = useRFRunStore((state) => state.setCompileData)
  const setStatusData = useRFRunStore((state) => state.setStatusData)
  const setInferenceData = useRFRunStore((state) => state.setInferenceData)
  const setFinalResult = useRFRunStore((state) => state.setFinalResult)
  const setError = useRFRunStore((state) => state.setError)
  const resetRun = useRFRunStore((state) => state.resetRun)

  const [datasets, setDatasets] = useState<RFDatasetMeta[]>([])
  const [inferenceValues, setInferenceValues] = useState<number[]>(
    () => new Array(datasetDefaults(training.dataset).shape[0]).fill(0)
  )
  const [leftCollapsed, setLeftCollapsed] = useState(false)
  const [rightCollapsed, setRightCollapsed] = useState(false)

  const nodes = useMemo(() => Object.values(nodesMap), [nodesMap])
  const inputNode = useMemo(() => nodes.find((node) => node.type === 'RFInput') ?? null, [nodes])
  const featureCount = useMemo(() => {
    if (compileData?.summary?.expected_feature_count) return compileData.summary.expected_feature_count
    if (Array.isArray(inputNode?.config.shape) && typeof inputNode.config.shape[0] === 'number') {
      return Number(inputNode.config.shape[0])
    }
    return datasetDefaults(training.dataset).shape[0]
  }, [compileData?.summary?.expected_feature_count, inputNode, training.dataset])
  const featureNames = useMemo(() => finalResult?.feature_names ?? [], [finalResult?.feature_names])

  useEffect(() => {
    setInferenceValues((current) => {
      if (current.length === featureCount) return current
      return new Array(featureCount).fill(0)
    })
  }, [featureCount])

  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const response = await requestJson<{ datasets: RFDatasetMeta[] }>('/api/rf/datasets')
        setDatasets(response.datasets)
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        addLog('error', `Failed to fetch RF datasets: ${message}`)
      }
    }
    void loadDatasets()
  }, [addLog])

  const withApiError = useCallback(
    (message: string, error: unknown) => {
      const text = error instanceof Error ? error.message : String(error)
      setError(text)
      addLog('error', `${message}: ${text}`)
    },
    [addLog, setError]
  )

  const runValidate = useCallback(async () => {
    try {
      setError(null)
      addLog('info', 'POST /api/rf/validate')
      const response = await requestJson<RFValidationResponse>('/api/rf/validate', {
        method: 'POST',
        body: JSON.stringify(serializeRFGraph()),
      })
      setValidation(response)
      if (response.valid) {
        addLog('success', `RF validation passed (${response.execution_order.length} nodes)`)
      } else {
        addLog('warn', `RF validation failed (${response.errors.length} errors)`)
      }
    } catch (error) {
      withApiError('Validate failed', error)
    }
  }, [addLog, setError, setValidation, withApiError])

  const runCompile = useCallback(async () => {
    try {
      setError(null)
      addLog('info', 'POST /api/rf/compile')
      const response = await requestJson<RFCompileResponse>('/api/rf/compile', {
        method: 'POST',
        body: JSON.stringify(serializeRFGraph()),
      })
      setCompileData(response)
      addLog(
        response.warnings.length > 0 ? 'warn' : 'success',
        `RF compile success (features=${response.summary.expected_feature_count}, warnings=${response.warnings.length})`
      )
    } catch (error) {
      withApiError('Compile failed', error)
    }
  }, [addLog, setCompileData, setError, withApiError])

  const runTrain = useCallback(async () => {
    try {
      setError(null)
      setFinalResult(null)
      setInferenceData(null)
      addLog('info', 'POST /api/rf/train')
      const response = await requestJson<RFTrainResponse>('/api/rf/train', {
        method: 'POST',
        body: JSON.stringify(serializeRFGraph()),
      })
      setJobId(response.job_id)
      setStatus('training')
      addLog('success', `RF training started (job_id=${response.job_id})`)
    } catch (error) {
      withApiError('Train failed', error)
    }
  }, [addLog, setError, setFinalResult, setInferenceData, setJobId, setStatus, withApiError])

  const runStop = useCallback(async () => {
    try {
      if (!jobId) {
        addLog('warn', 'No RF job to stop')
        return
      }
      await requestJson<{ job_id: string; status: string }>('/api/rf/stop', {
        method: 'POST',
        body: JSON.stringify({ job_id: jobId }),
      })
      setStatus('idle')
      addLog('warn', `RF stop requested (${jobId})`)
    } catch (error) {
      withApiError('Stop failed', error)
    }
  }, [addLog, jobId, setStatus, withApiError])

  const runStatus = useCallback(async () => {
    try {
      if (!jobId) {
        addLog('warn', 'No RF job selected for status check')
        return
      }
      const response = await requestJson<RFStatusResponse>(`/api/rf/status?job_id=${encodeURIComponent(jobId)}`)
      setStatusData(response)
      addLog('info', `RF status ${response.status} (terminal=${response.terminal})`)
    } catch (error) {
      withApiError('Status failed', error)
    }
  }, [addLog, jobId, setStatusData, withApiError])

  const runExportPy = useCallback(async () => {
    try {
      if (!jobId) {
        addLog('warn', 'No RF job selected for .py export')
        return
      }
      const blob = await requestBlob(`/api/rf/export?job_id=${encodeURIComponent(jobId)}&format=py`)
      triggerDownload(`${jobId}.py`, blob)
      addLog('success', `Exported ${jobId}.py`)
    } catch (error) {
      withApiError('Export .py failed', error)
    }
  }, [addLog, jobId, withApiError])

  const runExportPkl = useCallback(async () => {
    try {
      if (!jobId) {
        addLog('warn', 'No RF job selected for .pkl export')
        return
      }
      const blob = await requestBlob(`/api/rf/export?job_id=${encodeURIComponent(jobId)}&format=pkl`)
      triggerDownload(`${jobId}.pkl`, blob)
      addLog('success', `Exported ${jobId}.pkl`)
    } catch (error) {
      withApiError('Export .pkl failed', error)
    }
  }, [addLog, jobId, withApiError])

  const runInfer = useCallback(async () => {
    try {
      if (!jobId) {
        addLog('warn', 'No RF job selected for inference')
        return
      }
      setError(null)
      const response = await requestJson<RFInferResponse>('/api/rf/infer', {
        method: 'POST',
        body: JSON.stringify({
          job_id: jobId,
          inputs: inferenceValues.slice(0, featureCount),
          return_probabilities: true,
        }),
      })
      setInferenceData(response)
      addLog('success', 'Inference completed')
    } catch (error) {
      withApiError('Inference failed', error)
    }
  }, [addLog, featureCount, inferenceValues, jobId, setError, setInferenceData, withApiError])

  const imageToFeatureVector = useCallback(async (file: File): Promise<number[]> => {
    const objectUrl = URL.createObjectURL(file)
    try {
      const image = await new Promise<HTMLImageElement>((resolve, reject) => {
        const next = new Image()
        next.onload = () => resolve(next)
        next.onerror = () => reject(new Error('Failed to decode image'))
        next.src = objectUrl
      })

      const side = Math.max(2, Math.ceil(Math.sqrt(featureCount)))
      const canvas = document.createElement('canvas')
      canvas.width = side
      canvas.height = side
      const context = canvas.getContext('2d')
      if (!context) throw new Error('Could not create image processing context')
      context.drawImage(image, 0, 0, side, side)

      const imageData = context.getImageData(0, 0, side, side).data
      const vector: number[] = []
      for (let pixel = 0; pixel < side * side && vector.length < featureCount; pixel += 1) {
        const offset = pixel * 4
        const red = imageData[offset]
        const green = imageData[offset + 1]
        const blue = imageData[offset + 2]
        const gray = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
        vector.push(Number(gray.toFixed(6)))
      }
      while (vector.length < featureCount) vector.push(0)
      return vector
    } finally {
      URL.revokeObjectURL(objectUrl)
    }
  }, [featureCount])

  const handleInferenceImage = useCallback(async (file: File) => {
    try {
      const vector = await imageToFeatureVector(file)
      setInferenceValues(vector)
      addLog('success', `Loaded image "${file.name}" into ${vector.length} inference features`)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      addLog('error', `Image upload failed: ${message}`)
    }
  }, [addLog, imageToFeatureVector])

  const handleDatasetChange = useCallback(
    (dataset: string) => {
      setDataset(dataset)
      resetRun()
      const nextCount = datasetDefaults(dataset).shape[0]
      setInferenceValues(new Array(nextCount).fill(0))
      addLog('info', `Switched dataset to ${dataset}`)
    },
    [addLog, resetRun, setDataset]
  )

  return (
    <div className="rf-page">
      <div className="rf-builder-layer">
        <RfGraphView />
      </div>

      <div className="rf-overlay-root">
        {!leftCollapsed ? (
          <aside className="rf-side-panel rf-side-left">
            <section className="rf-card">
              <div className="rf-card-title-row">
                <span className="rf-card-title">RF Builder</span>
                <button className="rf-btn rf-btn-sm" onClick={() => setLeftCollapsed(true)}>
                  Collapse
                </button>
              </div>
              <div className="rf-hint">
                Scratch-like ML workflow: place nodes in 3D, connect them, then validate/compile/train.
              </div>
            </section>
            <RfDatasetPanel datasets={datasets} datasetId={training.dataset} onChangeDataset={handleDatasetChange} />
            <RfTrainingPanel
              training={training}
              status={status}
              activeJobId={jobId}
              onPatchTraining={setTraining}
              onValidate={runValidate}
              onCompile={runCompile}
              onTrain={runTrain}
              onStop={runStop}
              onStatus={runStatus}
              onExportPy={runExportPy}
              onExportPkl={runExportPkl}
            />
            <RfInferencePanel
              featureCount={featureCount}
              featureNames={featureNames}
              values={inferenceValues}
              onChangeValues={setInferenceValues}
              onLoadImage={handleInferenceImage}
              onInfer={runInfer}
              inferenceData={inferenceData}
            />
          </aside>
        ) : (
          <button className="rf-dock-toggle rf-dock-toggle-left" onClick={() => setLeftCollapsed(false)}>
            Open Left
          </button>
        )}

        {!rightCollapsed ? (
          <aside className="rf-side-panel rf-side-right">
            <section className="rf-card">
              <div className="rf-card-title-row">
                <span className="rf-card-title">Outputs</span>
                <button className="rf-btn rf-btn-sm" onClick={() => setRightCollapsed(true)}>
                  Collapse
                </button>
              </div>
              <div className="rf-hint">Live metrics, artifacts, schema payloads, and debug views.</div>
            </section>
            <RfChartsPanel progress={progress} finalResult={finalResult} />
            <section className="rf-card">
              <div className="rf-card-title">Compile Source Preview</div>
              <pre className="rf-json">
                {compileData?.python_source
                  ? compileData.python_source.slice(0, 4000)
                  : 'Compile output will appear here.'}
              </pre>
            </section>
            <RfLogsPanel logs={logs} onClear={clearLogs} />
          </aside>
        ) : (
          <button className="rf-dock-toggle rf-dock-toggle-right" onClick={() => setRightCollapsed(false)}>
            Open Right
          </button>
        )}
      </div>
    </div>
  )
}
