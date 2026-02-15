import { useEffect, useMemo, useState } from 'react'
import './deployments.css'

interface DeploymentInfo {
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

interface DeploymentLog {
  timestamp: string
  level: string
  event: string
  message: string
  details?: Record<string, unknown> | null
}

interface DeploymentInferResponse {
  deployment_id?: string
  job_id?: string
  input_shape?: number[]
  output_shape?: number[]
  predictions?: number[]
  probabilities?: number[][]
  logits?: number[][]
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

export default function DeploymentsPage() {
  const [deployments, setDeployments] = useState<DeploymentInfo[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selected, setSelected] = useState<DeploymentInfo | null>(null)
  const [logs, setLogs] = useState<DeploymentLog[]>([])
  const [message, setMessage] = useState('Loading deployments...')
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [isLoadingSelected, setIsLoadingSelected] = useState(false)
  const [inferResult, setInferResult] = useState<DeploymentInferResponse | null>(null)

  const activeCount = useMemo(
    () => deployments.filter((deployment) => deployment.status === 'running').length,
    [deployments]
  )
  const selectedFromList = useMemo(
    () => (selectedId ? deployments.find((deployment) => deployment.deployment_id === selectedId) ?? null : null),
    [deployments, selectedId]
  )
  const showEmptyState = deployments.length === 0

  const runAction = async (actionName: string, action: () => Promise<void>) => {
    if (busyAction) return
    setBusyAction(actionName)
    try {
      await action()
    } catch (error) {
      const nextMessage = error instanceof Error ? error.message : String(error)
      setMessage(nextMessage)
    } finally {
      setBusyAction(null)
    }
  }

  const fetchDeployments = async (silent = false) => {
    const response = await requestJson<{ deployments: DeploymentInfo[] }>('/api/deploy/list')
    setDeployments(response.deployments)
    if (response.deployments.length === 0) {
      setSelectedId(null)
      setSelected(null)
      setLogs([])
      if (!silent) {
        setMessage('No deployments found. Train a model and deploy it from the NN Deploy tab.')
      }
      return
    }

    if (!silent) {
      setMessage(
        `Loaded ${response.deployments.length} deployment(s), ${response.deployments.filter((item) => item.status === 'running').length} running.`
      )
    }

    const hasCurrentSelection = selectedId
      ? response.deployments.some((item) => item.deployment_id === selectedId)
      : false

    if (!hasCurrentSelection) {
      const preferred = response.deployments.find((item) => item.status === 'running') ?? response.deployments[0]
      setSelectedId(preferred.deployment_id)
    }
  }

  const fetchSelected = async (deploymentId: string) => {
    setIsLoadingSelected(true)
    try {
      const [statusResponse, logsResponse] = await Promise.all([
        requestJson<DeploymentInfo>(`/api/deploy/status?deployment_id=${deploymentId}`),
        requestJson<{ deployment_id: string; logs: DeploymentLog[] }>(
          `/api/deploy/logs?deployment_id=${deploymentId}&limit=250`
        ),
      ])
      setSelected(statusResponse)
      setLogs(logsResponse.logs)
    } finally {
      setIsLoadingSelected(false)
    }
  }

  useEffect(() => {
    void runAction('init', async () => {
      await fetchDeployments()
    })
    const timer = window.setInterval(() => {
      void fetchDeployments(true)
    }, 5000)
    return () => {
      window.clearInterval(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!selectedId) return
    let cancelled = false

    const run = async () => {
      setIsLoadingSelected(true)
      try {
        const [statusResponse, logsResponse] = await Promise.all([
          requestJson<DeploymentInfo>(`/api/deploy/status?deployment_id=${selectedId}`),
          requestJson<{ deployment_id: string; logs: DeploymentLog[] }>(
            `/api/deploy/logs?deployment_id=${selectedId}&limit=250`
          ),
        ])
        if (cancelled) return
        setSelected(statusResponse)
        setLogs(logsResponse.logs)
      } catch (error) {
        if (cancelled) return
        const nextMessage = error instanceof Error ? error.message : String(error)
        setMessage(nextMessage)
      } finally {
        if (!cancelled) setIsLoadingSelected(false)
      }
    }

    void run()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId])

  const toggleSelected = () =>
    runAction('toggle', async () => {
      if (!selectedId || !selectedFromList) throw new Error('Select a deployment first.')
      const wasRunning = selectedFromList.status === 'running'
      if (wasRunning) {
        await requestJson<DeploymentInfo>(`/api/deploy/${selectedId}`, { method: 'DELETE' })
      } else {
        await requestJson<DeploymentInfo>(`/api/deploy/${selectedId}/start`, { method: 'POST' })
      }
      await fetchDeployments(true)
      await fetchSelected(selectedId)
      setMessage(`${wasRunning ? 'Stopped' : 'Started'} ${getDeploymentDisplayName(selectedFromList)}.`)
    })

  const inferSelected = () =>
    runAction('infer', async () => {
      if (!selectedId) throw new Error('Select a deployment first.')
      const response = await requestJson<DeploymentInferResponse>(`/api/deploy/${selectedId}/infer`, {
        method: 'POST',
        body: JSON.stringify({
          inputs: [createZeroMnistSample()],
          return_probabilities: true,
        }),
      })
      setInferResult(response)
      await fetchSelected(selectedId)
      setMessage('Endpoint inference request completed.')
    })

  return (
    <div className="deployments-root">
      <header className="deployments-header">
        <div>
          <p className="deployments-kicker">View Your Models.</p>
          <h1 className="deployments-title">Local Deployment Manager</h1>
        </div>
        <div className="deployments-summary">
          <a href="/" className="deployments-link deployments-icon-link" aria-label="Back to Hub" title="Back to Hub">
            <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e3e3e3"><path d="M240-200h120v-240h240v240h120v-360L480-740 240-560v360Zm-80 80v-480l320-240 320 240v480H520v-240h-80v240H160Zm320-350Z"/></svg>
          </a>
        </div>
      </header>

      <main className="deployments-main">
        <section className="deployments-list-panel">
          <div className="deployments-toolbar">
            <button
              type="button"
              onClick={() => void runAction('refresh_list', fetchDeployments)}
              disabled={busyAction !== null}
              className="deployments-btn deployments-icon-btn"
              aria-label="Refresh deployment list"
              title={busyAction === 'refresh_list' ? 'Refreshing...' : 'Refresh List'}
            >
              <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e3e3e3"><path d="M480-160q-134 0-227-93t-93-227q0-134 93-227t227-93q69 0 132 28.5T720-690v-110h80v280H520v-80h168q-32-56-87.5-88T480-720q-100 0-170 70t-70 170q0 100 70 170t170 70q77 0 139-44t87-116h84q-28 106-114 173t-196 67Z"/></svg>
            </button>
            <div className="deployments-toolbar-summary" aria-live="polite">
              <span>Running: {activeCount}</span>
              <span>Total: {deployments.length}</span>
            </div>
          </div>
          <div className="deployments-list">
            {showEmptyState ? (
              <p className="deployments-empty">No deployments.</p>
            ) : (
              deployments.map((deployment) => (
                <button
                  key={deployment.deployment_id}
                  type="button"
                  onClick={() => setSelectedId(deployment.deployment_id)}
                  className={`deployments-item ${
                    selectedId === deployment.deployment_id ? 'deployments-item-active' : ''
                  }`}
                >
                  <div className="deployments-item-head">
                    <span className="deployments-item-name">{getDeploymentDisplayName(deployment)}</span>
                    <span
                      className={`deployments-item-status ${
                        deployment.status === 'running' ? 'deployments-item-status-running' : ''
                      }`}
                    >
                      {deployment.status}
                    </span>
                  </div>
                  <div className="deployments-item-sub">
                    id: {deployment.deployment_id.slice(0, 10)} · job: {deployment.job_id.slice(0, 10)} · req: {deployment.request_count}
                  </div>
                </button>
              ))
            )}
          </div>
        </section>

        <section className="deployments-detail-panel">
          {showEmptyState ? (
            <section className="deployments-empty-stage deployments-empty-stage-panel" aria-live="polite">
              <p>It's pretty empty here...</p>
            </section>
          ) : (
            <>
              <div className="deployments-detail-head">
                <h2>Deployment Details</h2>
                <div className="deployments-detail-actions">
                  <button
                    type="button"
                    onClick={() => {
                      if (!selectedId) return
                      void runAction('refresh_selected', async () => {
                        await fetchSelected(selectedId)
                        setMessage(`Refreshed ${getDeploymentDisplayName(selectedFromList)}.`)
                      })
                    }}
                    disabled={busyAction !== null || !selectedId}
                    className="deployments-btn deployments-icon-btn"
                    aria-label="Refresh selected deployment"
                    title={busyAction === 'refresh_selected' ? 'Refreshing...' : 'Refresh'}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e3e3e3"><path d="M480-160q-134 0-227-93t-93-227q0-134 93-227t227-93q69 0 132 28.5T720-690v-110h80v280H520v-80h168q-32-56-87.5-88T480-720q-100 0-170 70t-70 170q0 100 70 170t170 70q77 0 139-44t87-116h84q-28 106-114 173t-196 67Z"/></svg>
                  </button>
                  <button
                    type="button"
                    onClick={inferSelected}
                    disabled={busyAction !== null || !selectedFromList || selectedFromList.status !== 'running'}
                    className="deployments-btn deployments-icon-btn"
                    aria-label={busyAction === 'infer' ? 'Inferencing...' : 'Run Test Inference'}
                    title={busyAction === 'infer' ? 'Inferencing...' : 'Run Test Inference'}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e3e3e3"><path d="M200-120q-51 0-72.5-45.5T138-250l222-270v-240h-40q-17 0-28.5-11.5T280-800q0-17 11.5-28.5T320-840h320q17 0 28.5 11.5T680-800q0 17-11.5 28.5T640-760h-40v240l222 270q32 39 10.5 84.5T760-120H200Zm0-80h560L520-492v-268h-80v268L200-200Zm280-280Z"/></svg>
                  </button>
                  <button
                    type="button"
                    onClick={toggleSelected}
                    disabled={busyAction !== null || !selectedFromList}
                    className="deployments-btn deployments-icon-btn"
                    aria-label={busyAction === 'toggle' ? 'Toggling...' : 'Toggle Instance'}
                    title={busyAction === 'toggle' ? 'Toggling...' : 'Toggle Instance'}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e3e3e3"><path d="M200-312v-336l240 168-240 168Zm320-8v-320h80v320h-80Zm160 0v-320h80v320h-80Z"/></svg>
                  </button>
                </div>
              </div>

              <p className="deployments-message">{message}</p>

              {selected ? (
                <>
                  <div className="deployments-meta-grid">
                    <div><span>Name</span><code>{getDeploymentDisplayName(selected)}</code></div>
                    <div><span>Deployment ID</span><code>{selected.deployment_id}</code></div>
                    <div><span>Job ID</span><code>{selected.job_id}</code></div>
                    <div><span>Status</span><code>{selected.status}</code></div>
                    <div><span>Target</span><code>{selected.target}</code></div>
                    <div><span>Endpoint</span><code>{selected.endpoint_path}</code></div>
                    <div><span>Requests</span><code>{selected.request_count}</code></div>
                    <div><span>Created</span><code>{formatDateTime(selected.created_at)}</code></div>
                  </div>

                  <div className="deployments-block">
                    <h3>Logs</h3>
                    <div className="deployments-logs">
                      {logs.length === 0 ? (
                        <p className="deployments-empty">No logs yet.</p>
                      ) : (
                        logs.map((log, index) => (
                          <div key={`${log.timestamp}-${index}`} className="deployments-log-line">
                            <span className="deployments-log-time">{formatTime(log.timestamp)}</span>
                            <span className={`deployments-log-level deployments-log-level-${log.level}`}>
                              {log.level}
                            </span>
                            <span className="deployments-log-event">{log.event}</span>
                            <span className="deployments-log-message">{log.message}</span>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  <div className="deployments-block">
                    <h3>Last Inference Result</h3>
                    <div className="deployments-infer">
                      {inferResult ? (
                        <FormattedInferenceResult result={inferResult} />
                      ) : (
                        <p className="deployments-empty">No endpoint inference run yet.</p>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <p className="deployments-empty">
                  {isLoadingSelected ? 'Loading deployment details...' : 'Select a deployment from the left panel.'}
                </p>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  )
}

function formatTime(value: string): string {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleTimeString()
}

function formatDateTime(value: string): string {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
}

function getDeploymentDisplayName(deployment: DeploymentInfo | null): string {
  if (!deployment) return 'Deployment'
  if (deployment.name && deployment.name.trim().length > 0) {
    return deployment.name.trim()
  }
  return `Local NN Endpoint (${deployment.job_id.slice(0, 8)})`
}

function createZeroMnistSample(): number[][] {
  return Array.from({ length: 28 }, () => Array.from({ length: 28 }, () => 0))
}

function FormattedInferenceResult({ result }: { result: DeploymentInferResponse }) {
  const predictionCount = result.predictions?.length ?? 0
  const topPrediction = predictionCount > 0 ? result.predictions?.[0] : null
  const firstProbabilities = result.probabilities?.[0] ?? []
  const topConfidence =
    firstProbabilities.length > 0 ? Math.max(...firstProbabilities) : null

  const probabilityRows = firstProbabilities
    .map((value, index) => ({ index, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10)

  return (
    <div className="deployments-infer-content">
      <div className="deployments-infer-summary">
        <div>
          <span>Top Prediction</span>
          <strong>{topPrediction !== null && topPrediction !== undefined ? topPrediction : 'N/A'}</strong>
        </div>
        <div>
          <span>Top Confidence</span>
          <strong>{topConfidence !== null ? `${(topConfidence * 100).toFixed(2)}%` : 'N/A'}</strong>
        </div>
        <div>
          <span>Predictions</span>
          <strong>{predictionCount}</strong>
        </div>
        <div>
          <span>Input Shape</span>
          <strong>{formatShape(result.input_shape)}</strong>
        </div>
        <div>
          <span>Output Shape</span>
          <strong>{formatShape(result.output_shape)}</strong>
        </div>
      </div>

      {result.predictions && result.predictions.length > 0 ? (
        <div className="deployments-infer-pills">
          {result.predictions.map((prediction, index) => (
            <span key={`prediction-${index}`} className="deployments-infer-pill">
              sample {index + 1}: {prediction}
            </span>
          ))}
        </div>
      ) : null}

      {probabilityRows.length > 0 ? (
        <div className="deployments-probability-list">
          {probabilityRows.map((row) => (
            <div key={`prob-${row.index}`} className="deployments-probability-row">
              <span className="deployments-probability-label">Class {row.index}</span>
              <div className="deployments-probability-track">
                <div
                  className="deployments-probability-fill"
                  style={{ width: `${(row.value * 100).toFixed(2)}%` }}
                />
              </div>
              <span className="deployments-probability-value">{(row.value * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function formatShape(shape: number[] | undefined): string {
  if (!shape || shape.length === 0) return 'N/A'
  const compact = [...shape]
  while (compact.length > 2 && compact[0] === 1) {
    compact.shift()
  }
  return compact.join(' × ')
}
