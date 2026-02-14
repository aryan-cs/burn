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
  const [inferResult, setInferResult] = useState('No endpoint inference run yet.')

  const activeCount = useMemo(
    () => deployments.filter((deployment) => deployment.status === 'running').length,
    [deployments]
  )
  const selectedFromList = useMemo(
    () => (selectedId ? deployments.find((deployment) => deployment.deployment_id === selectedId) ?? null : null),
    [deployments, selectedId]
  )

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

  const stopSelected = () =>
    runAction('stop', async () => {
      if (!selectedId) throw new Error('Select a deployment first.')
      await requestJson<DeploymentInfo>(`/api/deploy/${selectedId}`, { method: 'DELETE' })
      await fetchDeployments(true)
      await fetchSelected(selectedId)
      setMessage(`Stopped ${getDeploymentDisplayName(selectedFromList)}.`)
    })

  const startSelected = () =>
    runAction('start', async () => {
      if (!selectedId) throw new Error('Select a deployment first.')
      await requestJson<DeploymentInfo>(`/api/deploy/${selectedId}/start`, { method: 'POST' })
      await fetchDeployments(true)
      await fetchSelected(selectedId)
      setMessage(`Started ${getDeploymentDisplayName(selectedFromList)}.`)
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
      setInferResult(JSON.stringify(response, null, 2))
      await fetchSelected(selectedId)
      setMessage('Endpoint inference request completed.')
    })

  return (
    <div className="deployments-root">
      <header className="deployments-header">
        <div>
          <p className="deployments-kicker">Deployment Control</p>
          <h1 className="deployments-title">Local Deployment Manager</h1>
        </div>
        <div className="deployments-summary">
          <span>Total: {deployments.length}</span>
          <span>Running: {activeCount}</span>
          <a href="/" className="deployments-link">Back To Hub</a>
        </div>
      </header>

      <main className="deployments-main">
        <section className="deployments-list-panel">
          <div className="deployments-toolbar">
            <button
              type="button"
              onClick={() => void runAction('refresh_list', fetchDeployments)}
              disabled={busyAction !== null}
              className="deployments-btn"
            >
              {busyAction === 'refresh_list' ? 'Refreshing...' : 'Refresh List'}
            </button>
          </div>
          <div className="deployments-list">
            {deployments.length === 0 ? (
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
                className="deployments-btn"
              >
                {busyAction === 'refresh_selected' ? 'Refreshing...' : 'Refresh'}
              </button>
              <button
                type="button"
                onClick={inferSelected}
                disabled={busyAction !== null || !selectedFromList || selectedFromList.status !== 'running'}
                className="deployments-btn"
              >
                {busyAction === 'infer' ? 'Inferencing...' : 'Run Test Inference'}
              </button>
              <button
                type="button"
                onClick={startSelected}
                disabled={busyAction !== null || !selectedFromList || selectedFromList.status === 'running'}
                className="deployments-btn deployments-btn-success"
              >
                {busyAction === 'start' ? 'Starting...' : 'Start'}
              </button>
              <button
                type="button"
                onClick={stopSelected}
                disabled={busyAction !== null || !selectedFromList || selectedFromList.status !== 'running'}
                className="deployments-btn deployments-btn-danger"
              >
                {busyAction === 'stop' ? 'Stopping...' : 'Stop'}
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
                <pre className="deployments-infer">{inferResult}</pre>
              </div>
            </>
          ) : (
            <p className="deployments-empty">
              {isLoadingSelected ? 'Loading deployment details...' : 'Select a deployment from the left panel.'}
            </p>
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
