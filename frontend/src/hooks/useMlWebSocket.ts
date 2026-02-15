import { useEffect, useRef } from 'react'
import { useMlStore } from '../store/mlStore'

/**
 * Connects to the ML training WebSocket and dispatches progress
 * updates to the ML store. Returns a `sendStop` callback.
 */
export function useMlWebSocket() {
  const jobId = useMlStore((s) => s.jobId)
  const status = useMlStore((s) => s.status)
  const setProgress = useMlStore((s) => s.setProgress)
  const setEvaluation = useMlStore((s) => s.setEvaluation)
  const setDataLoaded = useMlStore((s) => s.setDataLoaded)
  const setStatus = useMlStore((s) => s.setStatus)
  const setError = useMlStore((s) => s.setError)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!jobId) return
    if (status === 'idle' || status === 'complete' || status === 'error' || status === 'stopped') {
      return
    }

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host = window.location.host
    const url = `${protocol}://${host}/ws/ml/training/${jobId}`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)

        if (msg.type === 'status') {
          if (msg.status === 'training') {
            setStatus('training')
          }
        }

        if (msg.type === 'data_loaded') {
          setDataLoaded(
            msg.feature_names ?? [],
            msg.target_names ?? [],
            msg.task ?? 'classification',
          )
        }

        if (msg.type === 'training_progress') {
          setProgress({
            progress: msg.progress,
            step: msg.step,
            totalSteps: msg.total_steps,
            trainAccuracy: msg.train_accuracy,
            testAccuracy: msg.test_accuracy,
            trainR2: msg.train_r2,
            testR2: msg.test_r2,
          })
        }

        if (msg.type === 'evaluation') {
          setEvaluation(
            msg.train_metrics ?? {},
            msg.test_metrics ?? {},
            msg.feature_importances ?? null,
            msg.elapsed ?? 0,
          )
        }

        if (msg.type === 'training_done') {
          setStatus('complete')
        }

        if (msg.type === 'error') {
          setError(msg.error ?? 'Unknown training error')
        }
      } catch {
        // ignore malformed messages
      }
    }

    ws.onerror = () => {
      setError('WebSocket connection error')
    }

    ws.onclose = () => {
      wsRef.current = null
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, status])

  const sendStop = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'stop' }))
    }
  }

  return { sendStop }
}
