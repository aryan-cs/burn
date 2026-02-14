import { useEffect, useRef } from 'react'
import { useRFRunStore } from '../store/rfRunStore'
import type { RFDoneMessage, RFErrorMessage, RFProgressMessage } from '../types'

type RFSocketMessage = RFProgressMessage | RFDoneMessage | RFErrorMessage

export function useRfWebSocket(): void {
  const wsRef = useRef<WebSocket | null>(null)
  const jobId = useRFRunStore((state) => state.jobId)
  const status = useRFRunStore((state) => state.status)
  const addProgress = useRFRunStore((state) => state.addProgress)
  const setFinalResult = useRFRunStore((state) => state.setFinalResult)
  const setStatus = useRFRunStore((state) => state.setStatus)
  const setError = useRFRunStore((state) => state.setError)
  const addLog = useRFRunStore((state) => state.addLog)

  useEffect(() => {
    if (status !== 'training' || !jobId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/rf/training/${jobId}`)
    wsRef.current = socket

    socket.onopen = () => {
      addLog('info', `WebSocket connected for RF job ${jobId}`)
    }

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data) as RFSocketMessage
      if (data.type === 'rf_progress') {
        addProgress(data)
        addLog(
          'info',
          `progress trees=${data.trees_built}/${data.total_trees} test_acc=${(data.test_accuracy * 100).toFixed(2)}%`
        )
        return
      }
      if (data.type === 'rf_done') {
        setFinalResult(data)
        setStatus('complete')
        addLog('success', 'RF training completed')
        return
      }
      if (data.type === 'rf_error') {
        setError(data.message)
        addLog('error', data.message)
      }
    }

    socket.onerror = () => {
      setError('RF websocket connection failed')
      addLog('error', 'RF websocket connection failed')
    }

    socket.onclose = () => {
      wsRef.current = null
    }

    return () => {
      socket.close()
      wsRef.current = null
    }
  }, [addLog, addProgress, jobId, setError, setFinalResult, setStatus, status])
}
