import { useEffect, useMemo, useState } from 'react'

interface DeploymentView {
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

interface DeployTabProps {
  trainingJobId: string | null
  trainingStatus: string
  deployment: DeploymentView | null
  deployTarget: 'local' | 'modal' | 'sandbox'
  onDeployTargetChange: (target: 'local' | 'modal' | 'sandbox') => void
  deployTopPrediction: number | null
  deployOutput: string
  onDeployModel: () => void
  onRefreshDeployment: () => void
  onStopDeployment: () => void
  onInferDeployment: () => void
  deployDisabled: boolean
  refreshDisabled: boolean
  stopDisabled: boolean
  inferDisabled: boolean
  deployLabel: string
  refreshLabel: string
  stopLabel: string
  inferLabel: string
}

export function DeployTab({
  trainingJobId,
  trainingStatus,
  deployment,
  deployTarget,
  onDeployTargetChange,
  deployTopPrediction,
  deployOutput,
  onDeployModel,
  onRefreshDeployment,
  onStopDeployment,
  onInferDeployment,
  deployDisabled,
  refreshDisabled,
  stopDisabled,
  inferDisabled,
  deployLabel,
  refreshLabel,
  stopLabel,
  inferLabel,
}: DeployTabProps) {
  const endpointUrl =
    deployment
      ? resolveDeploymentEndpointUrl(deployment.target, deployment.endpoint_path)
      : null
  const endpointLiteral = endpointUrl
    ? toPythonSingleQuotedString(endpointUrl)
    : "'http://127.0.0.1:8000/api/deploy/<deployment_id>/infer'"
  const needsDeploymentIdInPayload =
    deployment?.target === 'modal' || deployment?.target === 'sandbox'
  const deploymentIdLiteral = deployment?.deployment_id
    ? toPythonSingleQuotedString(deployment.deployment_id)
    : "'<deployment_id>'"
  const pythonSnippet = useMemo(
    () => buildPythonClientSnippet(endpointLiteral, needsDeploymentIdInPayload, deploymentIdLiteral),
    [endpointLiteral, needsDeploymentIdInPayload, deploymentIdLiteral]
  )
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')

  useEffect(() => {
    if (copyStatus === 'idle') return
    const timer = window.setTimeout(() => setCopyStatus('idle'), 1800)
    return () => window.clearTimeout(timer)
  }, [copyStatus])

  const handleCopySnippet = async () => {
    const success = await copyToClipboard(pythonSnippet)
    setCopyStatus(success ? 'copied' : 'failed')
  }
  const copyButtonStateClass =
    copyStatus === 'copied'
      ? 'deploy-code-copy-button-copied'
      : copyStatus === 'failed'
        ? 'deploy-code-copy-button-failed'
        : ''

  return (
    <div className="tab-panel deploy-tab-panel">
      <section className="panel-card deploy-panel-card">
        <div className="deploy-summary-grid">
          <div className="deploy-summary-item">
            <span className="deploy-summary-label">Training Job</span>
            <span className="deploy-summary-value">{trainingJobId ?? 'none'}</span>
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
          <div className="deploy-summary-item">
            <span className="deploy-summary-label">Selected Target</span>
            <span className="deploy-summary-value">{deployTarget}</span>
          </div>
        </div>

        <div className="deploy-endpoint-card">
          <p className="deploy-endpoint-title">Deploy Target</p>
          <div className="deploy-target-toggle" role="group" aria-label="Deployment target">
            <button
              type="button"
              onClick={() => onDeployTargetChange('local')}
              className={`btn btn-ghost ${deployTarget === 'local' ? 'deploy-target-active' : ''}`}
            >
              Local
            </button>
            <button
              type="button"
              onClick={() => onDeployTargetChange('modal')}
              className={`btn btn-ghost ${deployTarget === 'modal' ? 'deploy-target-active' : ''}`}
            >
              Modal
            </button>
            <button
              type="button"
              onClick={() => onDeployTargetChange('sandbox')}
              className={`btn btn-ghost ${deployTarget === 'sandbox' ? 'deploy-target-active' : ''}`}
            >
              Modal Sandbox
            </button>
          </div>
        </div>

        <div className="deploy-endpoint-card">
          <p className="deploy-endpoint-title">Endpoint</p>
          <code className="deploy-endpoint-value">{endpointUrl ?? 'Deploy model to generate endpoint.'}</code>
        </div>

        <div className="deploy-code-shell">
          <div className="deploy-code-head">
            <p className="deploy-code-title">Python Client (requests)</p>
            <button
              type="button"
              onClick={handleCopySnippet}
              className={`deploy-code-copy-button ${copyButtonStateClass}`.trim()}
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
            <code>
              <span className="py-comment"># pip install requests</span>
              {'\n'}
              <span className="py-keyword">import</span> requests
              {'\n\n'}
              endpoint = <span className="py-string">{endpointLiteral}</span>
              {'\n'}
              payload = {'{'}
              {'\n'}
              {needsDeploymentIdInPayload ? (
                <>
                  {'    '}
                  <span className="py-string">'deployment_id'</span>: <span className="py-string">{deploymentIdLiteral}</span>,
                  {'\n'}
                </>
              ) : null}
              {'    '}
              <span className="py-string">'inputs'</span>: [[[0.0 <span className="py-keyword">for</span> _ <span className="py-keyword">in</span> <span className="py-number">range</span>(<span className="py-number">28</span>)] <span className="py-keyword">for</span> _ <span className="py-keyword">in</span> <span className="py-number">range</span>(<span className="py-number">28</span>)]],
              {'\n'}
              {'    '}
              <span className="py-string">'return_probabilities'</span>: <span className="py-const">True</span>,
              {'\n'}
              {'}'}
              {'\n\n'}
              response = requests.<span className="py-func">post</span>(endpoint, json=payload, timeout=<span className="py-number">30</span>)
              {'\n'}
              response.<span className="py-func">raise_for_status</span>()
              {'\n'}
              result = response.<span className="py-func">json</span>()
              {'\n'}
              <span className="py-func">print</span>(<span className="py-string">'prediction:'</span>, result[<span className="py-string">'predictions'</span>][<span className="py-number">0</span>])
            </code>
          </pre>
        </div>

        <div className="deploy-output-shell">
          <p className="deploy-output-title">Deployed Inference</p>
          <p className="deploy-top-prediction">
            Top Prediction: {deployTopPrediction ?? 'none'}
          </p>
          <pre className="deploy-output">{deployOutput}</pre>
        </div>
      </section>

      <div className="panel-actions deploy-actions">
        <button
          onClick={onDeployModel}
          disabled={deployDisabled}
          className="btn btn-validate"
        >
          {deployLabel}
        </button>
        <button
          onClick={onRefreshDeployment}
          disabled={refreshDisabled}
          className="btn btn-ghost"
        >
          {refreshLabel}
        </button>
        <button
          onClick={onInferDeployment}
          disabled={inferDisabled}
          className="btn btn-ghost"
        >
          {inferLabel}
        </button>
        <button
          onClick={onStopDeployment}
          disabled={stopDisabled}
          className="btn btn-validate btn-danger"
        >
          {stopLabel}
        </button>
      </div>
    </div>
  )
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

function resolveDeploymentEndpointUrl(target: string, endpointPath: string): string {
  const trimmed = endpointPath.trim()
  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed
  }
  if (target === 'local') {
    return `http://127.0.0.1:8000${trimmed}`
  }
  return `${resolveBackendBaseUrl()}${trimmed}`
}

function trimTrailingSlash(value: string): string {
  return value.endsWith('/') ? value.slice(0, -1) : value
}

function toPythonSingleQuotedString(value: string): string {
  return `'${value.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`
}

function buildPythonClientSnippet(
  endpointLiteral: string,
  includeDeploymentId: boolean,
  deploymentIdLiteral: string
): string {
  const deploymentLine = includeDeploymentId
    ? `    'deployment_id': ${deploymentIdLiteral},`
    : null
  return [
    '# pip install requests',
    'import requests',
    '',
    `endpoint = ${endpointLiteral}`,
    'payload = {',
    ...(deploymentLine ? [deploymentLine] : []),
    "    'inputs': [[[0.0 for _ in range(28)] for _ in range(28)]],",
    "    'return_probabilities': True,",
    '}',
    '',
    'response = requests.post(endpoint, json=payload, timeout=30)',
    'response.raise_for_status()',
    'result = response.json()',
    "print('prediction:', result['predictions'][0])",
    '',
  ].join('\n')
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
