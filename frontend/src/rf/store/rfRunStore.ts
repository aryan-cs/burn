import { create } from 'zustand'
import type {
  RFCompileResponse,
  RFDoneMessage,
  RFInferResponse,
  RFProgressMessage,
  RFStatusResponse,
  RFValidationResponse,
} from '../types'

export type RFRunStatus = 'idle' | 'training' | 'complete' | 'error'
export type RFLogLevel = 'info' | 'success' | 'warn' | 'error'

export interface RFLogEntry {
  id: number
  at: string
  level: RFLogLevel
  message: string
}

interface RFRunState {
  status: RFRunStatus
  jobId: string | null
  validation: RFValidationResponse | null
  compileData: RFCompileResponse | null
  statusData: RFStatusResponse | null
  inferenceData: RFInferResponse | null
  progress: RFProgressMessage[]
  finalResult: RFDoneMessage | null
  errorMessage: string | null
  logs: RFLogEntry[]
  logCounter: number
  setStatus: (status: RFRunStatus) => void
  setJobId: (jobId: string | null) => void
  setValidation: (data: RFValidationResponse | null) => void
  setCompileData: (data: RFCompileResponse | null) => void
  setStatusData: (data: RFStatusResponse | null) => void
  setInferenceData: (data: RFInferResponse | null) => void
  addProgress: (data: RFProgressMessage) => void
  setFinalResult: (data: RFDoneMessage | null) => void
  setError: (message: string | null) => void
  addLog: (level: RFLogLevel, message: string) => void
  clearLogs: () => void
  resetRun: () => void
}

function nowTime(): string {
  return new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export const useRFRunStore = create<RFRunState>((set) => ({
  status: 'idle',
  jobId: null,
  validation: null,
  compileData: null,
  statusData: null,
  inferenceData: null,
  progress: [],
  finalResult: null,
  errorMessage: null,
  logs: [],
  logCounter: 1,

  setStatus: (status) => set({ status }),
  setJobId: (jobId) => set({ jobId }),
  setValidation: (validation) => set({ validation }),
  setCompileData: (compileData) => set({ compileData }),
  setStatusData: (statusData) => set({ statusData }),
  setInferenceData: (inferenceData) => set({ inferenceData }),
  addProgress: (data) => set((state) => ({ progress: [...state.progress, data] })),
  setFinalResult: (finalResult) => set({ finalResult }),
  setError: (errorMessage) => set({ errorMessage, status: errorMessage ? 'error' : 'idle' }),
  addLog: (level, message) =>
    set((state) => ({
      logCounter: state.logCounter + 1,
      logs: [
        ...state.logs.slice(-299),
        {
          id: state.logCounter,
          at: nowTime(),
          level,
          message,
        },
      ],
    })),
  clearLogs: () => set({ logs: [] }),
  resetRun: () =>
    set({
      status: 'idle',
      jobId: null,
      statusData: null,
      inferenceData: null,
      progress: [],
      finalResult: null,
      errorMessage: null,
    }),
}))

export function levelClass(level: RFLogLevel): string {
  switch (level) {
    case 'success':
      return 'rf-log-success'
    case 'warn':
      return 'rf-log-warn'
    case 'error':
      return 'rf-log-error'
    default:
      return 'rf-log-info'
  }
}
