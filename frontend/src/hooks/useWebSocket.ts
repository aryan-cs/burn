import { useEffect, useRef, useCallback } from 'react'
import { useTrainingStore } from '../store/trainingStore'

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const jobId = useTrainingStore((s) => s.jobId)
  const status = useTrainingStore((s) => s.status)
  const addMetric = useTrainingStore((s) => s.addMetric)
  const updateWeights = useTrainingStore((s) => s.updateWeights)
  const setStatus = useTrainingStore((s) => s.setStatus)
  const setError = useTrainingStore((s) => s.setError)

  useEffect(() => {
    if (status !== 'training' || !jobId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/training/${jobId}`)
    wsRef.current = ws

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      switch (data.type) {
        case 'epoch_update':
          addMetric({
            epoch: data.epoch,
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
          break
        case 'training_done':
          setStatus('complete')
          break
        case 'error':
          setError(data.message)
          break
      }
    }

    ws.onerror = () => {
      setError('WebSocket connection failed')
    }

    ws.onclose = () => {
      wsRef.current = null
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [status, jobId, addMetric, updateWeights, setStatus, setError])

  const sendStop = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ command: 'stop' }))
  }, [])

  return { sendStop }
}
