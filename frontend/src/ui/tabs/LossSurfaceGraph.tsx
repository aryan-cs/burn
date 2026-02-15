import { Line, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useMemo } from 'react'
import { Vector3 } from 'three'

interface LossLandscapeTopology {
  layerCount: number
  neuronCount: number
  weightCount: number
  biasCount: number
}

interface LossSurfaceGraphProps {
  lossValues: number[]
  optimizer: string
  currentEpoch: number
  totalEpochs: number
  topology: LossLandscapeTopology
}

interface SurfaceProfile {
  bowlX: number
  bowlZ: number
  cross: number
  tiltX: number
  tiltZ: number
  ridgeAmpA: number
  ridgeAmpB: number
  ridgeFreqAX: number
  ridgeFreqAZ: number
  ridgeFreqBX: number
  ridgeFreqBZ: number
  ridgePhaseA: number
  ridgePhaseB: number
  startX: number
  startZ: number
  stepScale: number
  sgdMomentum: number
  lossLiftGain: number
}

const SURFACE_MIN = -2.6
const SURFACE_MAX = 2.6
const GRID_STEPS = 18
const FIXED_POLAR_ANGLE = 1.02

export function LossSurfaceGraph({
  lossValues,
  optimizer,
  currentEpoch,
  totalEpochs,
  topology,
}: LossSurfaceGraphProps) {
  const profile = useMemo(
    () => buildSurfaceProfile(topology, optimizer),
    [
      optimizer,
      topology.layerCount,
      topology.neuronCount,
      topology.weightCount,
      topology.biasCount,
    ]
  )
  const gridLines = useMemo(() => buildGridLines(profile), [profile])
  const pathPoints = useMemo(
    () => buildOptimizerPath(lossValues, optimizer, profile),
    [lossValues, optimizer, profile]
  )
  const markerPoint = useMemo(
    () =>
      pathPoints[pathPoints.length - 1] ??
      new Vector3(
        profile.startX,
        surfaceHeight(profile.startX, profile.startZ, profile),
        profile.startZ
      ),
    [pathPoints, profile]
  )
  const safeTotalEpochs = Math.max(1, Math.floor(totalEpochs))
  const safeCurrentEpoch = clamp(Math.floor(currentEpoch), 0, safeTotalEpochs)

  return (
    <div className="loss-surface-shell">
      <Canvas className="loss-surface-canvas" camera={{ position: [4.7, 3.5, 4.6], fov: 44 }}>
        <color attach="background" args={['#121212']} />
        <ambientLight intensity={0.72} />
        <directionalLight position={[4, 7, 3]} intensity={0.72} />
        <directionalLight position={[-3, 5, -4]} intensity={0.36} color="#ffd27a" />

        {gridLines.map((points, index) => (
          <Line
            key={`surface-grid-${index}`}
            points={points}
            color="#9d7a3f"
            transparent
            opacity={0.35}
            lineWidth={1}
          />
        ))}

        {pathPoints.length > 1 ? <Line points={pathPoints} color="#ffb429" lineWidth={2.2} /> : null}

        <mesh position={markerPoint.toArray()}>
          <sphereGeometry args={[0.09, 20, 20]} />
          <meshStandardMaterial color="#ffd27a" emissive="#8a5a12" emissiveIntensity={0.95} />
        </mesh>

        <OrbitControls
          enablePan={false}
          enableRotate
          enableZoom
          minPolarAngle={FIXED_POLAR_ANGLE}
          maxPolarAngle={FIXED_POLAR_ANGLE}
          minDistance={3.2}
          maxDistance={10}
          rotateSpeed={0.75}
          zoomSpeed={0.85}
        />
      </Canvas>

      <div className="loss-surface-caption">{formatOptimizerLabel(optimizer)} optimizer path</div>
      <div className="loss-surface-epoch">
        Epoch {safeCurrentEpoch} of {safeTotalEpochs}
      </div>
    </div>
  )
}

function buildGridLines(profile: SurfaceProfile): Vector3[][] {
  const lines: Vector3[][] = []
  const step = (SURFACE_MAX - SURFACE_MIN) / GRID_STEPS

  for (let row = 0; row <= GRID_STEPS; row += 1) {
    const z = SURFACE_MIN + row * step
    const points: Vector3[] = []
    for (let col = 0; col <= GRID_STEPS; col += 1) {
      const x = SURFACE_MIN + col * step
      points.push(new Vector3(x, surfaceHeight(x, z, profile), z))
    }
    lines.push(points)
  }

  for (let col = 0; col <= GRID_STEPS; col += 1) {
    const x = SURFACE_MIN + col * step
    const points: Vector3[] = []
    for (let row = 0; row <= GRID_STEPS; row += 1) {
      const z = SURFACE_MIN + row * step
      points.push(new Vector3(x, surfaceHeight(x, z, profile), z))
    }
    lines.push(points)
  }

  return lines
}

function buildOptimizerPath(
  lossValues: number[],
  optimizer: string,
  profile: SurfaceProfile
): Vector3[] {
  const values = normalizeLossSeries(lossValues)
  const points: Vector3[] = []

  let x = profile.startX
  let z = profile.startZ
  let velX = 0
  let velZ = 0
  let meanX = 0
  let meanZ = 0
  let varX = 0
  let varZ = 0

  values.forEach((lossNorm, index) => {
    const { gx, gz } = surfaceGradient(x, z, profile)
    const algorithm = optimizer.trim().toLowerCase()

    if (algorithm === 'sgd') {
      const step = 0.18 * profile.stepScale
      const momentum = profile.sgdMomentum
      velX = momentum * velX - step * gx
      velZ = momentum * velZ - step * gz
    } else {
      const step = 0.12 * profile.stepScale
      const beta1 = 0.9
      const beta2 = 0.999
      const eps = 1e-8

      meanX = beta1 * meanX + (1 - beta1) * gx
      meanZ = beta1 * meanZ + (1 - beta1) * gz
      varX = beta2 * varX + (1 - beta2) * gx * gx
      varZ = beta2 * varZ + (1 - beta2) * gz * gz

      const corr1 = 1 - Math.pow(beta1, index + 1)
      const corr2 = 1 - Math.pow(beta2, index + 1)
      const mHatX = meanX / corr1
      const mHatZ = meanZ / corr1
      const vHatX = varX / corr2
      const vHatZ = varZ / corr2

      velX = -step * (mHatX / (Math.sqrt(vHatX) + eps))
      velZ = -step * (mHatZ / (Math.sqrt(vHatZ) + eps))
    }

    x = clamp(x + velX, SURFACE_MIN + 0.08, SURFACE_MAX - 0.08)
    z = clamp(z + velZ, SURFACE_MIN + 0.08, SURFACE_MAX - 0.08)

    const liftedLoss = 0.26 + lossNorm * (1.05 + profile.lossLiftGain)
    const y = Math.max(surfaceHeight(x, z, profile) + 0.04, liftedLoss)
    points.push(new Vector3(x, y, z))
  })

  return points
}
function normalizeLossSeries(values: number[]): number[] {
  const finiteValues = values.filter((value) => Number.isFinite(value) && value >= 0)
  if (finiteValues.length === 0) {
    return [1]
  }

  const min = Math.min(...finiteValues)
  const max = Math.max(...finiteValues)
  const range = Math.max(max - min, 1e-6)
  return finiteValues.map((value) => (value - min) / range)
}

function buildSurfaceProfile(
  topology: LossLandscapeTopology,
  optimizer: string
): SurfaceProfile {
  const layerCount = Math.max(1, Math.floor(topology.layerCount) || 1)
  const neuronCount = Math.max(1, Math.floor(topology.neuronCount) || 1)
  const weightCount = Math.max(1, Math.floor(topology.weightCount) || 1)
  const biasCount = Math.max(1, Math.floor(topology.biasCount) || 1)

  const depthFactor = clamp01((layerCount - 2) / 12)
  const complexityFactor = normalizeLog(
    neuronCount + weightCount + biasCount,
    3,
    400000
  )
  const densityFactor = clamp01(
    weightCount / Math.max(1, neuronCount * layerCount * 24)
  )

  const seed = hashNumbers([
    layerCount,
    neuronCount,
    weightCount,
    biasCount,
    hashString(optimizer.trim().toLowerCase()),
  ])
  const rand = seededRandom(seed)
  const roughness = clamp(
    0.65 + complexityFactor * 0.34 + (rand() - 0.5) * 0.16,
    0.45,
    1.12
  )

  const bowlBase = 0.075 + complexityFactor * 0.03 + depthFactor * 0.018
  const startRadius = 1.86 + depthFactor * 0.48 + (1 - densityFactor) * 0.18

  const startAngle = Math.PI * (0.2 + rand() * 0.45)
  const startX = clamp(
    Math.cos(startAngle) * startRadius,
    SURFACE_MIN + 0.45,
    SURFACE_MAX - 0.45
  )
  const startZ = clamp(
    Math.sin(startAngle) * startRadius,
    SURFACE_MIN + 0.45,
    SURFACE_MAX - 0.45
  )

  return {
    bowlX: bowlBase * (0.92 + rand() * 0.22),
    bowlZ: bowlBase * (0.92 + rand() * 0.22),
    cross: (rand() - 0.5) * (0.014 + depthFactor * 0.012),
    tiltX: (rand() - 0.5) * 0.062,
    tiltZ: (rand() - 0.5) * 0.062,
    ridgeAmpA: (0.07 + complexityFactor * 0.11) * roughness,
    ridgeAmpB: (0.04 + densityFactor * 0.085) * roughness,
    ridgeFreqAX: 0.95 + rand() * 0.82 + depthFactor * 0.42,
    ridgeFreqAZ: 0.98 + rand() * 0.76 + complexityFactor * 0.36,
    ridgeFreqBX: 1.08 + rand() * 0.88 + depthFactor * 0.35,
    ridgeFreqBZ: 1.04 + rand() * 0.72 + complexityFactor * 0.34,
    ridgePhaseA: rand() * Math.PI * 2,
    ridgePhaseB: rand() * Math.PI * 2,
    startX,
    startZ,
    stepScale: clamp(
      0.97 + depthFactor * 0.2 - complexityFactor * 0.12 + (rand() - 0.5) * 0.1,
      0.74,
      1.24
    ),
    sgdMomentum: clamp(
      0.74 + depthFactor * 0.1 - densityFactor * 0.07,
      0.68,
      0.9
    ),
    lossLiftGain: 0.16 + complexityFactor * 0.3,
  }
}

function surfaceHeight(x: number, z: number, profile: SurfaceProfile): number {
  const bowl =
    profile.bowlX * x * x +
    profile.bowlZ * z * z +
    profile.cross * x * z +
    profile.tiltX * x +
    profile.tiltZ * z
  const ridgeA =
    profile.ridgeAmpA *
    Math.sin(profile.ridgeFreqAX * x + profile.ridgePhaseA) *
    Math.cos(profile.ridgeFreqAZ * z + profile.ridgePhaseB)
  const ridgeB =
    profile.ridgeAmpB *
    Math.cos(profile.ridgeFreqBX * x + profile.ridgePhaseB) *
    Math.sin(profile.ridgeFreqBZ * z + profile.ridgePhaseA)
  return 0.2 + bowl + ridgeA + ridgeB
}

function surfaceGradient(
  x: number,
  z: number,
  profile: SurfaceProfile
): { gx: number; gz: number } {
  const angleAX = profile.ridgeFreqAX * x + profile.ridgePhaseA
  const angleAZ = profile.ridgeFreqAZ * z + profile.ridgePhaseB
  const angleBX = profile.ridgeFreqBX * x + profile.ridgePhaseB
  const angleBZ = profile.ridgeFreqBZ * z + profile.ridgePhaseA

  const gx =
    2 * profile.bowlX * x +
    profile.cross * z +
    profile.tiltX +
    profile.ridgeAmpA * profile.ridgeFreqAX * Math.cos(angleAX) * Math.cos(angleAZ) -
    profile.ridgeAmpB * profile.ridgeFreqBX * Math.sin(angleBX) * Math.sin(angleBZ)

  const gz =
    2 * profile.bowlZ * z +
    profile.cross * x +
    profile.tiltZ -
    profile.ridgeAmpA * profile.ridgeFreqAZ * Math.sin(angleAX) * Math.sin(angleAZ) +
    profile.ridgeAmpB * profile.ridgeFreqBZ * Math.cos(angleBX) * Math.cos(angleBZ)

  return { gx, gz }
}

function normalizeLog(value: number, min: number, max: number): number {
  const bounded = clamp(value, min, max)
  const minLog = Math.log10(min)
  const maxLog = Math.log10(max)
  return clamp01((Math.log10(bounded) - minLog) / Math.max(maxLog - minLog, 1e-6))
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function hashNumbers(values: number[]): number {
  let hash = 2166136261
  values.forEach((value, index) => {
    const normalized = (Math.floor(Math.abs(value)) + 1 + index * 97) >>> 0
    hash ^= normalized + 0x9e3779b9 + ((hash << 6) >>> 0) + (hash >>> 2)
    hash = Math.imul(hash >>> 0, 16777619)
  })
  return hash >>> 0
}

function hashString(value: string): number {
  let hash = 0
  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(31, hash) + value.charCodeAt(index)
    hash |= 0
  }
  return hash >>> 0
}

function clamp01(value: number): number {
  return clamp(value, 0, 1)
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function formatOptimizerLabel(value: string): string {
  if (!value) return ''
  return value
    .split('_')
    .map((part) => {
      if (part.length === 0) return part
      return part.charAt(0).toUpperCase() + part.slice(1)
    })
    .join(' ')
}
