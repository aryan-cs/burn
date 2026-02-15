import { useCallback, useEffect, useMemo, useState } from 'react'
import './rf.css'
import { RfChartsPanel } from './components/RfChartsPanel'
import { RfDatasetPanel } from './components/RfDatasetPanel'
import { RfGraphView } from './components/RfGraphView'
import { RfInferencePanel } from './components/RfInferencePanel'
import { RfLogsPanel } from './components/RfLogsPanel'
import { RfForestConfigPanel, RfTrainingPanel } from './components/RfTrainingPanel'
import { useRfWebSocket } from './hooks/useRfWebSocket'
import { datasetDefaults, useRFGraphStore } from './store/rfGraphStore'
import { useRFRunStore } from './store/rfRunStore'
import { serializeRFGraph } from './utils/rfSerializer'
import type {
  RFCompileResponse,
  RFDatasetMeta,
  RFInferResponse,
  RFNode,
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

function toPositiveInt(value: unknown, fallback: number): number {
  const asNumber = Number(value)
  if (!Number.isFinite(asNumber)) return fallback
  const rounded = Math.round(asNumber)
  return rounded > 0 ? rounded : fallback
}

const LOW_DETAIL_STORAGE_KEY = 'mlcanvas.rf.low_detail_mode'

export default function RandomForestPage() {
  useRfWebSocket()

  const nodesMap = useRFGraphStore((state) => state.nodes)
  const training = useRFGraphStore((state) => state.training)
  const visualization = useRFGraphStore((state) => state.visualization)
  const setDataset = useRFGraphStore((state) => state.setDataset)
  const setTraining = useRFGraphStore((state) => state.setTraining)
  const setVisualization = useRFGraphStore((state) => state.setVisualization)
  const setNodeConfig = useRFGraphStore((state) => state.setNodeConfig)
  const resetPreset = useRFGraphStore((state) => state.resetPreset)

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
  const [activeTab, setActiveTab] = useState<'build' | 'train' | 'test'>('build')
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [isLowDetailMode, setIsLowDetailMode] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem(LOW_DETAIL_STORAGE_KEY) === '1'
  })

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
  const modelNode = useMemo(
    () => nodes.find((node) => node.type === 'RandomForestClassifier') ?? null,
    [nodes]
  )
  const totalTrees = useMemo(() => toPositiveInt(modelNode?.config.n_estimators, 100), [modelNode])
  const latestProgress = progress[progress.length - 1]
  const builtTrees = useMemo(
    () =>
      status === 'complete'
        ? totalTrees
        : Math.max(
            0,
            Math.min(
              totalTrees,
              Number.isFinite(latestProgress?.trees_built) ? latestProgress.trees_built : 0
            )
          ),
    [latestProgress?.trees_built, status, totalTrees]
  )
  const completionPercent = useMemo(
    () => (totalTrees > 0 ? Math.round((builtTrees / totalTrees) * 100) : 0),
    [builtTrees, totalTrees]
  )
  const activeTabIndex = activeTab === 'build' ? 0 : activeTab === 'train' ? 1 : 2

  useEffect(() => {
    setInferenceValues((current) => {
      if (current.length === featureCount) return current
      return new Array(featureCount).fill(0)
    })
  }, [featureCount])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(LOW_DETAIL_STORAGE_KEY, isLowDetailMode ? '1' : '0')
  }, [isLowDetailMode])

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

  const handleAlign = useCallback(() => {
    const state = useRFGraphStore.getState()
    const orderedNodes = Object.values(state.nodes).sort(compareRFNodesForAlignment)
    if (orderedNodes.length === 0) return

    const spacing = 3.2
    const centerOffset = ((orderedNodes.length - 1) * spacing) / 2
    orderedNodes.forEach((node, index) => {
      state.setNodePosition(node.id, [index * spacing - centerOffset, 0, 0])
    })
    state.autoConnectByX()
  }, [])

  return (
    <div className={`app-shell rf-page ${isSidebarCollapsed ? 'app-shell-collapsed' : ''}`}>
      <section className="app-sidebar rf-sidebar">
        <div className="app-sidebar-inner">
          <div className="app-tab-strip rf-tab-strip">
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
              onClick={() => setActiveTab('train')}
              className={`app-tab-button ${
                activeTab === 'train' ? 'app-tab-button-active' : 'app-tab-button-inactive'
              }`}
            >
              Train
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('test')}
              className={`app-tab-button ${
                activeTab === 'test' ? 'app-tab-button-active' : 'app-tab-button-inactive'
              }`}
            >
              Test
            </button>
          </div>

          {activeTab === 'build' ? (
            <div className="tab-panel rf-tab-panel">
              <section className="rf-card">
                <div className="rf-card-title">Random Forest Builder</div>
                <div className="rf-hint">
                  Place nodes in 3D, connect the flow, and configure dataset + model settings.
                </div>
              </section>
              <RfDatasetPanel
                datasets={datasets}
                datasetId={training.dataset}
                onChangeDataset={handleDatasetChange}
              />
              <RfLogsPanel logs={logs} onClear={clearLogs} />
            </div>
          ) : null}

          {activeTab === 'train' ? (
            <div className="tab-panel rf-tab-panel">
              <RfTrainingPanel
                training={training}
                visualization={visualization}
                status={status}
                activeJobId={jobId}
                builtTrees={builtTrees}
                totalTrees={totalTrees}
                completionPercent={completionPercent}
                onPatchTraining={setTraining}
                onPatchVisualization={setVisualization}
                onValidate={runValidate}
                onCompile={runCompile}
                onTrain={runTrain}
                onStop={runStop}
                onStatus={runStatus}
                onExportPy={runExportPy}
                onExportPkl={runExportPkl}
              />
              <RfForestConfigPanel
                modelNode={modelNode}
                onPatchNodeConfig={setNodeConfig}
                onResetPreset={resetPreset}
              />
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
            </div>
          ) : null}

          {activeTab === 'test' ? (
            <div className="tab-panel rf-tab-panel">
              <RfInferencePanel
                featureCount={featureCount}
                featureNames={featureNames}
                values={inferenceValues}
                onChangeValues={setInferenceValues}
                onLoadImage={handleInferenceImage}
                onInfer={runInfer}
                inferenceData={inferenceData}
              />
              <RfLogsPanel logs={logs} onClear={clearLogs} />
            </div>
          ) : null}
        </div>
      </section>

      <section className="app-viewport-panel rf-viewport-panel">
        <button
          type="button"
          className="sidebar-toggle-button rf-sidebar-toggle"
          onClick={() => setIsSidebarCollapsed((current) => !current)}
          aria-label={isSidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={isSidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isSidebarCollapsed ? (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="16px"
              viewBox="0 -960 960 960"
              width="16px"
              fill="#e3e3e3"
              aria-hidden="true"
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
              aria-hidden="true"
            >
              <path d="M400-80 0-480l400-400 71 71-329 329 329 329-71 71Z" />
            </svg>
          )}
        </button>
        <div className="builder-nav-links">
          <a
            href="/"
            className="builder-home-button"
            aria-label="Go to home"
            title="Home"
          >
            Home
          </a>
          <a
            href="/deployments"
            className="builder-deployments-button"
            aria-label="Go to deployments"
            title="Deployments"
          >
            Deployments
          </a>
        </div>
        <button
          type="button"
          onClick={() => setIsLowDetailMode((prev) => !prev)}
          className={`detail-mode-button ${
            isLowDetailMode ? 'detail-mode-button-active' : ''
          }`}
          aria-pressed={isLowDetailMode}
          aria-label="Toggle low detail mode"
          title="Reduce render detail"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            height="24px"
            viewBox="0 -960 960 960"
            width="24px"
            fill="#e3e3e3"
            aria-hidden="true"
          >
            <path d="M480-40q-50 0-85-35t-35-85q0-14 2.5-26.5T371-211L211-372q-12 5-25 8.5t-27 3.5q-50 0-84.5-35T40-480q0-50 34.5-85t84.5-35q39 0 70 22.5t43 57.5h95q9-26 28-44.5t45-27.5v-95q-35-12-57.5-43T360-800q0-50 35-85t85-35q50 0 85 35t35 85q0 14-3 27t-9 25l160 160q12-6 25-9t27-3q50 0 85 35t35 85q0 50-35 85t-85 35q-39 0-70-22.5T687-440h-95q-9 26-27.5 45T520-367v94q35 12 57.5 43t22.5 70q0 50-35 85t-85 35Zm-40-233v-94q-13-5-24-12t-20.5-16.5Q386-405 379-416t-12-24h-95q0 1-.5 2.5l-1 3q-.5 1.5-1 2.5l-1.5 3 29 29q24 24 51 51.5t51 51.5l29 29q2-1 3.5-1.5t3-1.5 3-1.5q1.5-.5 2.5-.5Zm152-247h95q0-2 .5-3.5t1.5-3 1.5-3q.5-1.5 1.5-2.5l-29-29-51-51-51-51-29-29q-1 1-2.5 1.5t-3 1.5-3 1.5q-1.5.5-3.5.5v95q12 4 23.5 11.5T564-564q9 9 16.5 20.5T592-520Zm208 80q17 0 28.5-11.5T840-480q0-17-11.5-28.5T800-520q-17 0-28.5 11.5T760-480q0 17 11.5 28.5T800-440Zm-320 0q17 0 28.5-11.5T520-480q0-17-11.5-28.5T480-520q-17 0-28.5 11.5T440-480q0 17 11.5 28.5T480-440Zm0 320q17 0 28.5-11.5T520-160q0-17-11.5-28.5T480-200q-17 0-28.5 11.5T440-160q0 17 11.5 28.5T480-120ZM160-440q17 0 28.5-11.5T200-480q0-17-11.5-28.5T160-520q-17 0-28.5 11.5T120-480q0 17 11.5 28.5T160-440Zm320-320q17 0 28.5-11.5T520-800q0-17-11.5-28.5T480-840q-17 0-28.5 11.5T440-800q0 17 11.5 28.5T480-760Z" />
          </svg>
        </button>
        <button
          type="button"
          onClick={handleAlign}
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
        <div className="rf-builder-layer">
          <RfGraphView lowDetailMode={isLowDetailMode} />
        </div>
      </section>
    </div>
  )
}

function compareRFNodesForAlignment(left: RFNode, right: RFNode): number {
  const typeOrderDelta = getRFNodeTypeOrder(left.type) - getRFNodeTypeOrder(right.type)
  if (typeOrderDelta !== 0) return typeOrderDelta
  return left.id.localeCompare(right.id)
}

function getRFNodeTypeOrder(type: RFNode['type']): number {
  if (type === 'RFInput') return 0
  if (type === 'RFFlatten') return 1
  if (type === 'RandomForestClassifier') return 2
  if (type === 'RFOutput') return 3
  return 4
}
