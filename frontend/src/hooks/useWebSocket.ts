import { useEffect, useRef, useCallback } from 'react'
import { useTrainingStore } from '../store/trainingStore'

const WS_RETRY_BASE_DELAY_MS = 400
const WS_RETRY_MAX_DELAY_MS = 2500
const WS_MAX_RETRIES = 20

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<number | null>(null)
  const jobId = useTrainingStore((s) => s.jobId)
  const status = useTrainingStore((s) => s.status)
  const addMetric = useTrainingStore((s) => s.addMetric)
  const updateWeights = useTrainingStore((s) => s.updateWeights)
  const updateLossLandscape = useTrainingStore((s) => s.updateLossLandscape)
  const setStatus = useTrainingStore((s) => s.setStatus)
  const setError = useTrainingStore((s) => s.setError)

  useEffect(() => {
    if (status !== 'training' || !jobId) return

    let disposed = false
    let retries = 0

    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
    }

    const scheduleReconnect = () => {
      if (disposed) return
      const store = useTrainingStore.getState()
      if (store.status !== 'training' || store.jobId !== jobId) return
      if (retries >= WS_MAX_RETRIES) {
        setError('WebSocket connection failed')
        return
      }
      const delay = Math.min(
        WS_RETRY_BASE_DELAY_MS * Math.max(1, retries + 1),
        WS_RETRY_MAX_DELAY_MS
      )
      retries += 1
      clearReconnectTimer()
      reconnectTimerRef.current = window.setTimeout(connect, delay)
    }

    const connect = () => {
      if (disposed) return
      const ws = new WebSocket(buildTrainingWsUrl(jobId))
      wsRef.current = ws

      ws.onopen = () => {
        retries = 0
      }

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)

        switch (data.type) {
          case 'ws_connected':
          case 'training_backend':
            // Informational events; no state update required.
            break
          case 'epoch_update':
            {
              const epochValue = toNumber(data.epoch)
              if (epochValue === null || epochValue <= 0) break
            addMetric({
              epoch: epochValue,
              loss: data.loss,
              accuracy: data.accuracy,
              trainLoss: data.train_loss,
              trainAccuracy: data.train_accuracy,
              testLoss: data.test_loss,
              testAccuracy: data.test_accuracy,
            })
            if (data.weights) {
              updateWeights(data.weights)
            }
            if (data.loss_landscape) {
              const parsedLandscape = parseLossLandscapePayload(data.loss_landscape)
              if (parsedLandscape) {
                updateLossLandscape(parsedLandscape)
              }
            }
            break
            }
          case 'training_done':
            setStatus('complete')
            break
          case 'error':
            setError(data.message)
            break
        }
      }

      ws.onerror = () => {
        // Let onclose handle retry policy to avoid failing fast on transient disconnects.
      }

      ws.onclose = () => {
        wsRef.current = null
        scheduleReconnect()
      }
    }

    connect()

    return () => {
      disposed = true
      clearReconnectTimer()
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [status, jobId, addMetric, updateWeights, updateLossLandscape, setStatus, setError])

  const sendStop = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'stop' }))
    }
  }, [])

  return { sendStop }
}

function buildTrainingWsUrl(jobId: string): string {
  const envWsBase = import.meta.env.VITE_BACKEND_WS_URL?.trim()
  if (envWsBase) {
    return `${trimTrailingSlash(envWsBase)}/ws/training/${jobId}`
  }

  const envHttpBase = import.meta.env.VITE_BACKEND_HTTP_URL?.trim()
  if (envHttpBase) {
    const wsBase = envHttpBase
      .replace(/^https:\/\//, 'wss://')
      .replace(/^http:\/\//, 'ws://')
    return `${trimTrailingSlash(wsBase)}/ws/training/${jobId}`
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws/training/${jobId}`
}

function trimTrailingSlash(value: string): string {
  return value.endsWith('/') ? value.slice(0, -1) : value
}

function parseLossLandscapePayload(raw: unknown) {
  if (!raw || typeof raw !== 'object') return null
  const payload = raw as Record<string, unknown>

  const xAxis = toNumericArray(payload.x_axis)
  const zAxis = toNumericArray(payload.z_axis)
  if (xAxis.length === 0 || zAxis.length === 0) return null

  const gridLoss = toNumericGrid(payload.grid_loss)
  const rawPath = Array.isArray(payload.path) ? payload.path : []
  const path = rawPath
    .map((entry) => parseLandscapePoint(entry))
    .filter((entry): entry is { epoch: number; x: number; z: number; loss: number } => entry !== null)

  return {
    objective: String(payload.objective ?? 'reference_loss'),
    datasetSplit: String(payload.dataset_split ?? 'test'),
    gridSize: toPositiveInt(payload.grid_size, Math.max(xAxis.length, zAxis.length)),
    radius: toPositiveNumber(payload.radius, 1),
    xAxis,
    zAxis,
    gridLoss,
    path,
    point: parseLandscapePoint(payload.point),
    sampleCount: toPositiveInt(payload.sample_count, 0),
  }
}

function parseLandscapePoint(raw: unknown): { epoch: number; x: number; z: number; loss: number } | null {
  if (!raw || typeof raw !== 'object') return null
  const value = raw as Record<string, unknown>
  const epoch = toNumber(value.epoch)
  const x = toNumber(value.x)
  const z = toNumber(value.z)
  const loss = toNumber(value.loss)
  if (epoch === null || epoch <= 0 || x === null || z === null || loss === null) return null
  return { epoch, x, z, loss }
}

function toNumericArray(raw: unknown): number[] {
  if (!Array.isArray(raw)) return []
  return raw
    .map((entry) => toNumber(entry))
    .filter((entry): entry is number => entry !== null)
}

function toNumericGrid(raw: unknown): number[][] | null {
  if (!Array.isArray(raw)) return null
  const rows: number[][] = []
  for (const row of raw) {
    const parsed = toNumericArray(row)
    if (parsed.length > 0) rows.push(parsed)
  }
  return rows.length > 0 ? rows : null
}

function toNumber(value: unknown): number | null {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function toPositiveInt(value: unknown, fallback: number): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.max(0, Math.round(parsed))
}

function toPositiveNumber(value: unknown, fallback: number): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.max(0, parsed)
}
