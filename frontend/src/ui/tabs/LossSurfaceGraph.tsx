import { Line, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useMemo } from 'react'
import { Vector3 } from 'three'
import type { LossLandscapeData } from '../../store/trainingStore'

interface LossSurfaceGraphProps {
  landscape: LossLandscapeData | null
  currentEpoch: number
  totalEpochs: number
}

const FIXED_POLAR_ANGLE = 1.02

export function LossSurfaceGraph({
  landscape,
  currentEpoch,
  totalEpochs,
}: LossSurfaceGraphProps) {
  const scene = useMemo(() => {
    if (!landscape || !landscape.gridLoss || landscape.gridLoss.length === 0) {
      return null
    }
    return buildSceneData(landscape)
  }, [landscape])

  const safeTotalEpochs = Math.max(1, Math.floor(totalEpochs))
  const safeCurrentEpoch = clamp(Math.floor(currentEpoch), 0, safeTotalEpochs)

  return (
    <div className="loss-surface-shell">
      {scene ? (
        <Canvas
          className="loss-surface-canvas"
          camera={{ position: scene.cameraPosition, fov: 44 }}
        >
          <color attach="background" args={['#121212']} />
          <ambientLight intensity={0.72} />
          <directionalLight position={[4, 7, 3]} intensity={0.72} />
          <directionalLight position={[-3, 5, -4]} intensity={0.36} color="#ffd27a" />

          {scene.gridLines.map((points, index) => (
            <Line
              key={`surface-grid-${index}`}
              points={points}
              color="#9d7a3f"
              transparent
              opacity={0.35}
              lineWidth={1}
            />
          ))}

          {scene.pathPoints.length > 1 ? (
            <Line points={scene.pathPoints} color="#ffb429" lineWidth={2.2} />
          ) : null}

          <mesh position={scene.markerPoint.toArray()}>
            <sphereGeometry args={[0.09, 20, 20]} />
            <meshStandardMaterial color="#ffd27a" emissive="#8a5a12" emissiveIntensity={0.95} />
          </mesh>

          <OrbitControls
            enablePan={false}
            enableRotate
            enableZoom
            minPolarAngle={FIXED_POLAR_ANGLE}
            maxPolarAngle={FIXED_POLAR_ANGLE}
            minDistance={scene.minDistance}
            maxDistance={scene.maxDistance}
            rotateSpeed={0.75}
            zoomSpeed={0.85}
          />
        </Canvas>
      ) : (
        <div className="loss-surface-empty">
          Generating synthetic loss landscape preview...
          <br />
          Train to watch the marker path evolve epoch by epoch.
        </div>
      )}

      <div className="loss-surface-caption">
        {landscape ? formatLandscapeLabel(landscape) : 'Synthetic preview pending'}
      </div>
      <div className="loss-surface-epoch">
        Epoch {safeCurrentEpoch} of {safeTotalEpochs}
      </div>
    </div>
  )
}

function buildSceneData(landscape: LossLandscapeData): {
  gridLines: Vector3[][]
  pathPoints: Vector3[]
  markerPoint: Vector3
  cameraPosition: [number, number, number]
  minDistance: number
  maxDistance: number
} {
  const grid = landscape.gridLoss ?? []
  const rows = Math.min(grid.length, landscape.zAxis.length)
  const cols = rows > 0 ? Math.min(grid[0].length, landscape.xAxis.length) : 0
  if (rows === 0 || cols === 0) {
    return {
      gridLines: [],
      pathPoints: [],
      markerPoint: new Vector3(0, 0, 0),
      cameraPosition: [4.7, 3.5, 4.6],
      minDistance: 3.2,
      maxDistance: 10,
    }
  }

  const allLosses: number[] = []
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const loss = grid[row][col]
      if (Number.isFinite(loss)) {
        allLosses.push(loss)
      }
    }
  }
  for (const point of landscape.path) {
    if (Number.isFinite(point.loss)) {
      allLosses.push(point.loss)
    }
  }

  const minLoss = allLosses.length > 0 ? Math.min(...allLosses) : 0
  const maxLoss = allLosses.length > 0 ? Math.max(...allLosses) : 1
  const range = Math.max(maxLoss - minLoss, 1e-9)
  const lossToHeight = (loss: number) => 0.1 + ((loss - minLoss) / range) * 1.4

  const gridLines: Vector3[][] = []
  for (let row = 0; row < rows; row += 1) {
    const z = landscape.zAxis[row]
    const points: Vector3[] = []
    for (let col = 0; col < cols; col += 1) {
      const x = landscape.xAxis[col]
      const loss = grid[row][col]
      points.push(new Vector3(x, lossToHeight(loss), z))
    }
    gridLines.push(points)
  }
  for (let col = 0; col < cols; col += 1) {
    const x = landscape.xAxis[col]
    const points: Vector3[] = []
    for (let row = 0; row < rows; row += 1) {
      const z = landscape.zAxis[row]
      const loss = grid[row][col]
      points.push(new Vector3(x, lossToHeight(loss), z))
    }
    gridLines.push(points)
  }

  const pathPoints = landscape.path
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.z) && Number.isFinite(point.loss))
    .map((point) => new Vector3(point.x, lossToHeight(point.loss), point.z))

  const markerPoint =
    pathPoints[pathPoints.length - 1] ??
    new Vector3(
      landscape.xAxis[Math.floor(cols / 2)],
      lossToHeight(grid[Math.floor(rows / 2)][Math.floor(cols / 2)]),
      landscape.zAxis[Math.floor(rows / 2)]
    )

  const extent = Math.max(
    0.8,
    ...landscape.xAxis.map((value) => Math.abs(value)),
    ...landscape.zAxis.map((value) => Math.abs(value))
  )
  const cameraDistance = Math.max(3.8, extent * 3.2)

  return {
    gridLines,
    pathPoints,
    markerPoint,
    cameraPosition: [cameraDistance, Math.max(2.8, extent * 2.2), cameraDistance],
    minDistance: Math.max(2.4, extent * 1.4),
    maxDistance: Math.max(10, extent * 7.5),
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function formatLandscapeLabel(landscape: LossLandscapeData): string {
  const objective = formatObjectiveLabel(landscape.objective)
  const split = landscape.datasetSplit.toLowerCase()
  const sampleSuffix = landscape.sampleCount > 0 ? ` · n=${landscape.sampleCount}` : ''
  return `${objective} · ${split}${sampleSuffix}`
}

function formatObjectiveLabel(value: string): string {
  if (!value) return 'Loss'
  return value
    .split('_')
    .map((part) => {
      if (part.length === 0) return part
      if (part === 'mse') return 'MSE'
      if (part === 'ce') return 'CE'
      return part.charAt(0).toUpperCase() + part.slice(1)
    })
    .join(' ')
}
