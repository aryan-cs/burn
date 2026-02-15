import { Line, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useMemo } from 'react'
import { Vector3 } from 'three'

interface LossSurfaceGraphProps {
  lossValues: number[]
  optimizer: string
  currentEpoch: number
  totalEpochs: number
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
}: LossSurfaceGraphProps) {
  const { gridLines, pathPoints, markerPoint } = useMemo(() => {
    const lines = buildGridLines()
    const points = buildOptimizerPath(lossValues, optimizer)
    const marker = points[points.length - 1] ?? new Vector3(2.2, surfaceHeight(2.2, 2.2), 2.2)
    return { gridLines: lines, pathPoints: points, markerPoint: marker }
  }, [lossValues, optimizer])
  const safeTotalEpochs = Math.max(1, Math.floor(totalEpochs))
  const safeCurrentEpoch = clamp(Math.floor(currentEpoch), 0, safeTotalEpochs)

  return (
    <div className="loss-surface-shell">
      <Canvas
        className="loss-surface-canvas"
        camera={{ position: [4.7, 3.5, 4.6], fov: 44 }}
      >
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

        {pathPoints.length > 1 ? (
          <Line points={pathPoints} color="#ffb429" lineWidth={2.2} />
        ) : null}

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

      <div className="loss-surface-caption">
        {formatOptimizerLabel(optimizer)} optimizer path
      </div>
      <div className="loss-surface-epoch">
        Epoch {safeCurrentEpoch} of {safeTotalEpochs}
      </div>
    </div>
  )
}

function buildGridLines(): Vector3[][] {
  const lines: Vector3[][] = []
  const step = (SURFACE_MAX - SURFACE_MIN) / GRID_STEPS

  for (let row = 0; row <= GRID_STEPS; row += 1) {
    const z = SURFACE_MIN + row * step
    const points: Vector3[] = []
    for (let col = 0; col <= GRID_STEPS; col += 1) {
      const x = SURFACE_MIN + col * step
      points.push(new Vector3(x, surfaceHeight(x, z), z))
    }
    lines.push(points)
  }

  for (let col = 0; col <= GRID_STEPS; col += 1) {
    const x = SURFACE_MIN + col * step
    const points: Vector3[] = []
    for (let row = 0; row <= GRID_STEPS; row += 1) {
      const z = SURFACE_MIN + row * step
      points.push(new Vector3(x, surfaceHeight(x, z), z))
    }
    lines.push(points)
  }

  return lines
}

function buildOptimizerPath(lossValues: number[], optimizer: string): Vector3[] {
  const values = normalizeLossSeries(lossValues)
  const points: Vector3[] = []

  let x = 2.25
  let z = 2.1
  let velX = 0
  let velZ = 0
  let meanX = 0
  let meanZ = 0
  let varX = 0
  let varZ = 0

  values.forEach((lossNorm, index) => {
    const { gx, gz } = surfaceGradient(x, z)
    const algorithm = optimizer.trim().toLowerCase()

    if (algorithm === 'sgd') {
      const step = 0.2
      const momentum = 0.78
      velX = momentum * velX - step * gx
      velZ = momentum * velZ - step * gz
    } else {
      const step = 0.13
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

    const liftedLoss = 0.28 + lossNorm * 1.18
    const y = Math.max(surfaceHeight(x, z) + 0.04, liftedLoss)
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

function surfaceHeight(x: number, z: number): number {
  const bowl = 0.09 * (x * x + z * z)
  const ridges = 0.15 * Math.sin(1.3 * x) * Math.cos(1.1 * z)
  return 0.22 + bowl + ridges
}

function surfaceGradient(x: number, z: number): { gx: number; gz: number } {
  const gx = 0.18 * x + 0.195 * Math.cos(1.3 * x) * Math.cos(1.1 * z)
  const gz = 0.18 * z - 0.165 * Math.sin(1.3 * x) * Math.sin(1.1 * z)
  return { gx, gz }
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
