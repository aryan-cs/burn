import { create } from 'zustand'

export type TrainingStatus = 'idle' | 'training' | 'paused' | 'complete' | 'error'

export interface EpochMetric {
  epoch: number
  loss: number
  accuracy: number
  trainLoss?: number
  trainAccuracy?: number
  testLoss?: number
  testAccuracy?: number
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

export interface LossLandscapePoint {
  epoch: number
  x: number
  z: number
  loss: number
}

export interface LossLandscapeData {
  objective: string
  datasetSplit: string
  gridSize: number
  radius: number
  xAxis: number[]
  zAxis: number[]
  gridLoss: number[][] | null
  path: LossLandscapePoint[]
  point: LossLandscapePoint | null
  sampleCount: number
}

interface TrainingConfig {
  dataset: string
  location: 'local' | 'cloud'
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
  lossLandscape: LossLandscapeData | null
  config: TrainingConfig
  errorMessage: string | null

  setConfig: (patch: Partial<TrainingConfig>) => void
  startTraining: (jobId: string, totalEpochs: number) => void
  addMetric: (metric: EpochMetric) => void
  updateWeights: (snapshot: WeightSnapshot) => void
  updateLossLandscape: (snapshot: LossLandscapeData) => void
  setStatus: (status: TrainingStatus) => void
  setError: (message: string) => void
  reset: () => void
}

const DEFAULT_CONFIG: TrainingConfig = {
  dataset: 'mnist',
  location: 'local',
  epochs: 20,
  batchSize: 64,
  optimizer: 'adam',
  learningRate: 0.001,
  loss: 'cross_entropy',
}

const SYNTHETIC_GRID_SIZE = 17
const SYNTHETIC_RADIUS = 1.4
const SYNTHETIC_PATH_LIMIT = 256
const MAX_METRIC_POINTS = 420

export const useTrainingStore = create<TrainingState>((set) => ({
  status: 'idle',
  jobId: null,
  currentEpoch: 0,
  totalEpochs: 0,
  metrics: [],
  latestWeights: null,
  lossLandscape: createSyntheticLandscape(DEFAULT_CONFIG, `boot:${Date.now()}`),
  config: { ...DEFAULT_CONFIG },
  errorMessage: null,

  setConfig: (patch) =>
    set((s) => {
      const nextConfig = { ...s.config, ...patch }
      const shouldRefreshPreview =
        s.status !== 'training' &&
        (s.status === 'idle' || s.lossLandscape?.objective.startsWith('synthetic_') === true)
      return {
        config: nextConfig,
        lossLandscape: shouldRefreshPreview
          ? createSyntheticLandscape(nextConfig, `cfg:${Date.now()}`)
          : s.lossLandscape,
      }
    }),

  startTraining: (jobId, totalEpochs) =>
    set((state) => ({
      status: 'training',
      jobId,
      totalEpochs,
      currentEpoch: 0,
      metrics: [],
      latestWeights: null,
      lossLandscape: createSyntheticLandscape(
        state.config,
        `job:${jobId}:${totalEpochs}:${Date.now()}`
      ),
      errorMessage: null,
    })),

  addMetric: (metric) =>
    set((s) => {
      const nextMetrics = [...s.metrics]
      const existingIdx = nextMetrics.findIndex((entry) => entry.epoch === metric.epoch)
      if (existingIdx >= 0) {
        nextMetrics[existingIdx] = metric
      } else {
        nextMetrics.push(metric)
      }
      nextMetrics.sort((left, right) => left.epoch - right.epoch)
      if (nextMetrics.length > MAX_METRIC_POINTS) {
        nextMetrics.splice(0, nextMetrics.length - MAX_METRIC_POINTS)
      }
      const nextLandscape =
        s.lossLandscape && s.lossLandscape.objective.startsWith('synthetic_')
          ? appendSyntheticLandscapePoint(
              s.lossLandscape,
              metric.epoch,
              metric.testLoss ?? metric.loss,
              s.totalEpochs
            )
          : s.lossLandscape
      return {
        metrics: nextMetrics,
        currentEpoch: metric.epoch,
        lossLandscape: nextLandscape,
      }
    }),

  updateWeights: (snapshot) =>
    set({ latestWeights: snapshot }),

  updateLossLandscape: (snapshot) =>
    set((state) => {
      const previous = state.lossLandscape
      return {
        lossLandscape: {
          objective: snapshot.objective,
          datasetSplit: snapshot.datasetSplit,
          gridSize: snapshot.gridSize,
          radius: snapshot.radius,
          xAxis: snapshot.xAxis,
          zAxis: snapshot.zAxis,
          gridLoss:
            snapshot.gridLoss && snapshot.gridLoss.length > 0
              ? snapshot.gridLoss
              : previous?.gridLoss ?? null,
          path:
            snapshot.path.length > 0
              ? snapshot.path
              : previous?.path ?? [],
          point: snapshot.point ?? previous?.point ?? null,
          sampleCount: snapshot.sampleCount,
        },
      }
    }),

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
      lossLandscape: createSyntheticLandscape(DEFAULT_CONFIG, `reset:${Date.now()}`),
      errorMessage: null,
    }),
}))

function createSyntheticLandscape(
  config: TrainingConfig,
  seedText: string
): LossLandscapeData {
  const seed = hashString(
    `${config.dataset}|${config.optimizer}|${config.loss}|${config.learningRate}|${seedText}`
  )
  const random = mulberry32(seed)
  const radius = SYNTHETIC_RADIUS + random() * 0.4
  const axis = createAxis(SYNTHETIC_GRID_SIZE, radius)
  const phase = random() * Math.PI * 2
  const bowl = 0.72 + random() * 0.38
  const wave = 0.16 + random() * 0.2
  const cross = 0.12 + random() * 0.14
  const offset = 0.24 + random() * 0.36
  const gridLoss = axis.map((z) =>
    axis.map((x) => syntheticSurfaceLoss(x, z, phase, bowl, wave, cross, offset))
  )
  const center = Math.floor(axis.length / 2)
  const centerLoss = gridLoss[center]?.[center] ?? 0.5
  const initialPoint: LossLandscapePoint = {
    epoch: 0,
    x: 0,
    z: 0,
    loss: Number(centerLoss.toFixed(6)),
  }

  return {
    objective: `synthetic_${normalizeLossName(config.loss)}`,
    datasetSplit: 'preview',
    gridSize: SYNTHETIC_GRID_SIZE,
    radius,
    xAxis: axis,
    zAxis: axis,
    gridLoss,
    path: [initialPoint],
    point: initialPoint,
    sampleCount: 640 + Math.floor(random() * 2048),
  }
}

function appendSyntheticLandscapePoint(
  landscape: LossLandscapeData,
  epoch: number,
  observedLoss: number,
  totalEpochs: number
): LossLandscapeData {
  const normalizedEpoch = Math.max(1, Math.floor(epoch))
  const pathSeed = hashString(
    `${landscape.objective}|${landscape.sampleCount}|${normalizedEpoch}`
  )
  const random = mulberry32(pathSeed)
  const lastPoint = landscape.point ?? landscape.path[landscape.path.length - 1] ?? {
    epoch: 0,
    x: 0,
    z: 0,
    loss: 0.5,
  }
  const safeTotalEpochs = Math.max(6, Math.floor(totalEpochs) || 0)
  const step = landscape.radius / safeTotalEpochs
  const driftX =
    (random() - 0.5) * step * 1.5 +
    Math.cos(normalizedEpoch * 0.63 + random() * Math.PI) * step * 0.55
  const driftZ =
    (random() - 0.5) * step * 1.5 +
    Math.sin(normalizedEpoch * 0.51 + random() * Math.PI) * step * 0.55
  const x = clamp(lastPoint.x + driftX, -landscape.radius, landscape.radius)
  const z = clamp(lastPoint.z + driftZ, -landscape.radius, landscape.radius)
  const terrainLoss = sampleSurfaceGrid(landscape, x, z)
  const observed =
    Number.isFinite(observedLoss) && observedLoss > 0 ? observedLoss : lastPoint.loss
  const nextLoss = Math.max(
    0.001,
    terrainLoss * 0.58 + observed * 0.42 + (random() - 0.5) * 0.05
  )
  const nextPoint: LossLandscapePoint = {
    epoch: normalizedEpoch,
    x: Number(x.toFixed(6)),
    z: Number(z.toFixed(6)),
    loss: Number(nextLoss.toFixed(6)),
  }
  const nextPath = [...landscape.path.filter((point) => point.epoch !== normalizedEpoch), nextPoint]
    .sort((left, right) => left.epoch - right.epoch)
    .slice(-SYNTHETIC_PATH_LIMIT)

  return {
    ...landscape,
    path: nextPath,
    point: nextPoint,
  }
}

function sampleSurfaceGrid(landscape: LossLandscapeData, x: number, z: number): number {
  const grid = landscape.gridLoss
  if (!grid || grid.length === 0 || !grid[0] || grid[0].length === 0) return 0.5

  const rows = Math.min(grid.length, landscape.zAxis.length)
  const cols = Math.min(grid[0].length, landscape.xAxis.length)
  if (rows <= 1 || cols <= 1) return grid[0]?.[0] ?? 0.5

  const xNorm = clamp((x + landscape.radius) / (2 * Math.max(landscape.radius, 1e-6)), 0, 1)
  const zNorm = clamp((z + landscape.radius) / (2 * Math.max(landscape.radius, 1e-6)), 0, 1)
  const xIndex = xNorm * (cols - 1)
  const zIndex = zNorm * (rows - 1)

  const x0 = Math.floor(xIndex)
  const z0 = Math.floor(zIndex)
  const x1 = Math.min(cols - 1, x0 + 1)
  const z1 = Math.min(rows - 1, z0 + 1)
  const tx = xIndex - x0
  const tz = zIndex - z0

  const v00 = grid[z0]?.[x0] ?? 0.5
  const v10 = grid[z0]?.[x1] ?? v00
  const v01 = grid[z1]?.[x0] ?? v00
  const v11 = grid[z1]?.[x1] ?? v00
  const top = v00 * (1 - tx) + v10 * tx
  const bottom = v01 * (1 - tx) + v11 * tx
  return top * (1 - tz) + bottom * tz
}

function syntheticSurfaceLoss(
  x: number,
  z: number,
  phase: number,
  bowl: number,
  wave: number,
  cross: number,
  offset: number
): number {
  const bowlTerm = bowl * (x * x + 0.84 * z * z)
  const waveTerm =
    wave * Math.sin(x * 2.24 + phase) +
    cross * Math.cos(z * 2.06 - phase * 0.62) +
    0.08 * Math.sin((x + z) * 1.66 + phase * 0.28)
  return Number(Math.max(0.02, offset + bowlTerm + waveTerm).toFixed(6))
}

function createAxis(size: number, radius: number): number[] {
  if (size <= 1) return [0]
  const step = (radius * 2) / (size - 1)
  return Array.from({ length: size }, (_, index) =>
    Number((-radius + step * index).toFixed(6))
  )
}

function normalizeLossName(lossName: string): string {
  const normalized = lossName.trim().toLowerCase().replace(/[\s-]+/g, '_')
  return normalized.length > 0 ? normalized : 'loss'
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function hashString(input: string): number {
  let hash = 2166136261 >>> 0
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

function mulberry32(seed: number): () => number {
  let current = seed >>> 0
  return () => {
    current = (current + 0x6d2b79f5) >>> 0
    let value = Math.imul(current ^ (current >>> 15), 1 | current)
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}
