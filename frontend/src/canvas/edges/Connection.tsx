import { useMemo } from 'react'
import * as THREE from 'three'
import type { LayerNode } from '../../store/graphStore'
import type { TrainingFlowPhase } from '../nodes/LayerNode'
import {
  LAYER_NODE_SPACING,
  LAYER_NODE_Z_OFFSET,
  buildLayerWorldNodePositions,
  getLayerGridSize,
} from '../../utils/layerLayout'
import { WeightVisual } from './WeightVisual'

const MAX_SEGMENTS_FULL_DETAIL = 6000
const LOW_DETAIL_ANCHORS: Array<[number, number]> = [
  [0.1, 0.1],
  [0.1, 0.9],
  [0.5, 0.5],
  [0.9, 0.1],
  [0.9, 0.9],
]

interface ConnectionProps {
  sourceNode: LayerNode
  targetNode: LayerNode
  lowDetailMode: boolean
  trainingFlow?: {
    active: boolean
    intensity: number
    value: number
    phase: TrainingFlowPhase | null
  }
}

const EDGE_VALUE_LOW_COLOR = new THREE.Color('#ff4747')
const EDGE_VALUE_MID_COLOR = new THREE.Color('#ffffff')
const EDGE_VALUE_HIGH_COLOR = new THREE.Color('#4eff8e')

export function Connection({
  sourceNode,
  targetNode,
  lowDetailMode,
  trainingFlow,
}: ConnectionProps) {
  const segmentPositions = useMemo(() => {
    if (lowDetailMode) {
      return buildLowDetailSegments(sourceNode, targetNode)
    }

    const sourcePoints = buildLayerWorldNodePositions(sourceNode)
    const targetPoints = buildLayerWorldNodePositions(targetNode)
    const totalSegments = sourcePoints.length * targetPoints.length
    const stride =
      totalSegments > MAX_SEGMENTS_FULL_DETAIL
        ? Math.ceil(Math.sqrt(totalSegments / MAX_SEGMENTS_FULL_DETAIL))
        : 1
    const values: number[] = []

    for (let sourceIndex = 0; sourceIndex < sourcePoints.length; sourceIndex += stride) {
      const source = sourcePoints[sourceIndex]
      for (let targetIndex = 0; targetIndex < targetPoints.length; targetIndex += stride) {
        const target = targetPoints[targetIndex]
        pushSegment(values, source, target)
      }
    }

    return new Float32Array(values)
  }, [sourceNode, targetNode, lowDetailMode])
  const lineColor = useMemo(
    () => getEdgeColor(trainingFlow?.value ?? 0),
    [trainingFlow?.value]
  )
  const flowIntensity = Math.min(1, Math.max(0, trainingFlow?.intensity ?? 0))
  const hasTrainingFlow = Boolean(trainingFlow)
  const lineOpacity = hasTrainingFlow
    ? 0.3 + flowIntensity * 0.56
    : lowDetailMode
      ? 0.55
      : 0.38
  const pulseCurve = useMemo(() => {
    const sourceCenter = new THREE.Vector3(
      sourceNode.position[0],
      sourceNode.position[1],
      sourceNode.position[2] + LAYER_NODE_Z_OFFSET
    )
    const targetCenter = new THREE.Vector3(
      targetNode.position[0],
      targetNode.position[1],
      targetNode.position[2] + LAYER_NODE_Z_OFFSET
    )
    if (trainingFlow?.phase === 'backward') {
      return new THREE.LineCurve3(targetCenter, sourceCenter)
    }
    return new THREE.LineCurve3(sourceCenter, targetCenter)
  }, [
    sourceNode.position,
    targetNode.position,
    trainingFlow?.phase,
  ])
  const pulseSpeed = trainingFlow?.active
    ? 0.65 + flowIntensity * 1.45
    : 0
  const pulseSize = trainingFlow?.active
    ? 0.035 + flowIntensity * 0.025
    : 0.035

  if (segmentPositions.length === 0) return null

  return (
    <group>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[segmentPositions, 3]} />
        </bufferGeometry>
        <lineBasicMaterial
          color={hasTrainingFlow ? lineColor : lowDetailMode ? '#5f6268' : '#525252'}
          transparent
          opacity={lineOpacity}
        />
      </lineSegments>

      {trainingFlow?.active ? (
        <WeightVisual curve={pulseCurve} speed={pulseSpeed} color={lineColor} size={pulseSize} />
      ) : null}
    </group>
  )
}

function buildLowDetailSegments(sourceNode: LayerNode, targetNode: LayerNode): Float32Array {
  const sourceGrid = getLayerGridSize(sourceNode.config, sourceNode.type)
  const targetGrid = getLayerGridSize(targetNode.config, targetNode.type)
  const values: number[] = []
  const seen = new Set<string>()

  LOW_DETAIL_ANCHORS.forEach(([rowRatio, colRatio]) => {
    const source = pointAtRatio(sourceNode, sourceGrid.rows, sourceGrid.cols, rowRatio, colRatio)
    const target = pointAtRatio(targetNode, targetGrid.rows, targetGrid.cols, rowRatio, colRatio)
    const key = `${source.join(',')}|${target.join(',')}`
    if (seen.has(key)) return
    seen.add(key)
    pushSegment(values, source, target)
  })

  return new Float32Array(values)
}

function pointAtRatio(
  node: LayerNode,
  rows: number,
  cols: number,
  rowRatio: number,
  colRatio: number
): [number, number, number] {
  const safeRows = Math.max(1, rows)
  const safeCols = Math.max(1, cols)
  const rowIndex = Math.round(clamp01(rowRatio) * (safeRows - 1))
  const colIndex = Math.round(clamp01(colRatio) * (safeCols - 1))
  return pointAtGridIndex(node, safeRows, safeCols, rowIndex, colIndex)
}

function pointAtGridIndex(
  node: LayerNode,
  rows: number,
  cols: number,
  rowIndex: number,
  colIndex: number
): [number, number, number] {
  const x = node.position[0] + (colIndex - (cols - 1) / 2) * LAYER_NODE_SPACING
  const y = node.position[1] + ((rows - 1) / 2 - rowIndex) * LAYER_NODE_SPACING
  const z = node.position[2] + LAYER_NODE_Z_OFFSET
  return [x, y, z]
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value))
}

function pushSegment(
  values: number[],
  source: [number, number, number],
  target: [number, number, number]
) {
  values.push(source[0], source[1], source[2], target[0], target[1], target[2])
}

function getEdgeColor(value: number): string {
  const clamped = Math.min(1, Math.max(-1, value))
  const color = EDGE_VALUE_MID_COLOR.clone()
  if (clamped >= 0) {
    color.lerp(EDGE_VALUE_HIGH_COLOR, clamped)
  } else {
    color.lerp(EDGE_VALUE_LOW_COLOR, Math.abs(clamped))
  }
  return `#${color.getHexString()}`
}
