import { create } from 'zustand'

export type TrainingStatus = 'idle' | 'training' | 'paused' | 'complete' | 'error'

export interface EpochMetric {
  epoch: number
  loss: number
  accuracy: number
}

export interface WeightSnapshot {
  [paramName: string]: {
    mean: number
    std: number
    min: number
    max: number
    histogram?: number[]
  }
}

interface TrainingConfig {
  dataset: string
  epochs: number
  batchSize: number
  optimizer: string
  learningRate: number
  loss: string
}

interface TrainingState {
  status: TrainingStatus
  jobId: string | null
  currentEpoch: number
  totalEpochs: number
  metrics: EpochMetric[]
  latestWeights: WeightSnapshot | null
  config: TrainingConfig
  errorMessage: string | null

  setConfig: (patch: Partial<TrainingConfig>) => void
  startTraining: (jobId: string, totalEpochs: number) => void
  addMetric: (metric: EpochMetric) => void
  updateWeights: (snapshot: WeightSnapshot) => void
  setStatus: (status: TrainingStatus) => void
  setError: (message: string) => void
  reset: () => void
}

const DEFAULT_CONFIG: TrainingConfig = {
  dataset: 'mnist',
  epochs: 20,
  batchSize: 64,
  optimizer: 'adam',
  learningRate: 0.001,
  loss: 'cross_entropy',
}

export const useTrainingStore = create<TrainingState>((set) => ({
  status: 'idle',
  jobId: null,
  currentEpoch: 0,
  totalEpochs: 0,
  metrics: [],
  latestWeights: null,
  config: { ...DEFAULT_CONFIG },
  errorMessage: null,

  setConfig: (patch) =>
    set((s) => ({ config: { ...s.config, ...patch } })),

  startTraining: (jobId, totalEpochs) =>
    set({
      status: 'training',
      jobId,
      totalEpochs,
      currentEpoch: 0,
      metrics: [],
      latestWeights: null,
      errorMessage: null,
    }),

  addMetric: (metric) =>
    set((s) => ({
      metrics: [...s.metrics, metric],
      currentEpoch: metric.epoch,
    })),

  updateWeights: (snapshot) =>
    set({ latestWeights: snapshot }),

  setStatus: (status) => set({ status }),

  setError: (message) =>
    set({ status: 'error', errorMessage: message }),

  reset: () =>
    set({
      status: 'idle',
      jobId: null,
      currentEpoch: 0,
      totalEpochs: 0,
      metrics: [],
      latestWeights: null,
      errorMessage: null,
    }),
}))
