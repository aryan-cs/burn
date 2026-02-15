import { create } from 'zustand'

// ── Types ──────────────────────────────────────────

export type MlModelType = 'linear_regression' | 'logistic_regression' | 'random_forest'
export type MlTask = 'classification' | 'regression'
export type MlTrainingStatus = 'idle' | 'loading_data' | 'training' | 'complete' | 'error' | 'stopped'

export interface MlDatasetInfo {
  id: string
  name: string
  task: MlTask
  n_features: number
  n_classes: number
  n_samples: number
  feature_names: string[]
  target_names: string[]
  description: string
}

export interface MlProgressUpdate {
  progress: number
  step: number
  totalSteps: number
  trainAccuracy?: number
  testAccuracy?: number
  trainR2?: number
  testR2?: number
}

export interface MlMetrics {
  [key: string]: number
}

export interface MlHyperparameters {
  // Linear Regression
  fit_intercept?: boolean
  // Logistic Regression
  C?: number
  max_iter?: number
  penalty?: string
  solver?: string
  // Random Forest
  n_estimators?: number
  max_depth?: number | null
  min_samples_split?: number
  min_samples_leaf?: number
  criterion?: string
  max_features?: string
}

// ── Default hyperparams per model type ─────────────

const DEFAULT_HYPERPARAMS: Record<MlModelType, MlHyperparameters> = {
  linear_regression: {
    fit_intercept: true,
  },
  logistic_regression: {
    C: 1.0,
    max_iter: 200,
    penalty: 'l2',
    solver: 'lbfgs',
  },
  random_forest: {
    n_estimators: 100,
    max_depth: null,
    min_samples_split: 2,
    min_samples_leaf: 1,
    criterion: 'gini',
    max_features: 'sqrt',
  },
}

// ── Store state ────────────────────────────────────

interface MlState {
  modelType: MlModelType
  dataset: string
  testSize: number
  hyperparams: MlHyperparameters
  status: MlTrainingStatus
  jobId: string | null
  progress: MlProgressUpdate | null
  progressHistory: MlProgressUpdate[]
  trainMetrics: MlMetrics | null
  testMetrics: MlMetrics | null
  featureImportances: Record<string, number> | null
  featureNames: string[]
  targetNames: string[]
  task: MlTask | null
  errorMessage: string | null
  trainingTime: number | null
  datasets: MlDatasetInfo[]

  setModelType: (type: MlModelType) => void
  setDataset: (dataset: string) => void
  setTestSize: (size: number) => void
  setHyperparams: (patch: Partial<MlHyperparameters>) => void
  setDatasets: (datasets: MlDatasetInfo[]) => void
  startTraining: (jobId: string) => void
  setProgress: (update: MlProgressUpdate) => void
  setEvaluation: (
    trainMetrics: MlMetrics,
    testMetrics: MlMetrics,
    featureImportances: Record<string, number> | null,
    elapsed: number,
  ) => void
  setDataLoaded: (featureNames: string[], targetNames: string[], task: MlTask) => void
  setStatus: (status: MlTrainingStatus) => void
  setError: (message: string) => void
  reset: () => void
}

export const useMlStore = create<MlState>((set) => ({
  modelType: 'logistic_regression',
  dataset: 'iris',
  testSize: 0.2,
  hyperparams: { ...DEFAULT_HYPERPARAMS.logistic_regression },
  status: 'idle',
  jobId: null,
  progress: null,
  progressHistory: [],
  trainMetrics: null,
  testMetrics: null,
  featureImportances: null,
  featureNames: [],
  targetNames: [],
  task: null,
  errorMessage: null,
  trainingTime: null,
  datasets: [],

  setModelType: (type) =>
    set({
      modelType: type,
      hyperparams: { ...DEFAULT_HYPERPARAMS[type] },
    }),

  setDataset: (dataset) => set({ dataset }),

  setTestSize: (size) => set({ testSize: size }),

  setHyperparams: (patch) =>
    set((s) => ({ hyperparams: { ...s.hyperparams, ...patch } })),

  setDatasets: (datasets) => set({ datasets }),

  startTraining: (jobId) =>
    set({
      status: 'loading_data',
      jobId,
      progress: null,
      progressHistory: [],
      trainMetrics: null,
      testMetrics: null,
      featureImportances: null,
      errorMessage: null,
      trainingTime: null,
    }),

  setProgress: (update) =>
    set((s) => ({
      progress: update,
      progressHistory: [...s.progressHistory, update],
    })),

  setDataLoaded: (featureNames, targetNames, task) =>
    set({ featureNames, targetNames, task }),

  setEvaluation: (trainMetrics, testMetrics, featureImportances, elapsed) =>
    set({
      trainMetrics,
      testMetrics,
      featureImportances,
      trainingTime: elapsed,
    }),

  setStatus: (status) => set({ status }),

  setError: (message) =>
    set({ status: 'error', errorMessage: message }),

  reset: () =>
    set({
      status: 'idle',
      jobId: null,
      progress: null,
      progressHistory: [],
      trainMetrics: null,
      testMetrics: null,
      featureImportances: null,
      featureNames: [],
      targetNames: [],
      task: null,
      errorMessage: null,
      trainingTime: null,
    }),
}))
