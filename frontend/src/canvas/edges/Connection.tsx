import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { LayerNode } from '../../store/graphStore'
import type { TrainingFlowPhase } from '../nodes/LayerNode'
import {
  LAYER_NODE_SPACING,
  LAYER_NODE_Z_OFFSET,
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
const SEGMENT_SAMPLE_STD_DEV = 0.34
const SEGMENT_STEP_SCALE = 0.2

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

    return buildSampledSegments(sourceNode, targetNode, MAX_SEGMENTS_FULL_DETAIL)
  }, [sourceNode, targetNode, lowDetailMode])
  const segmentCount = Math.floor(segmentPositions.length / 6)
  const segmentStateRef = useRef<{
    count: number
    centers: Float32Array
    values: Float32Array
  } | null>(null)
  const [segmentColorVersion, setSegmentColorVersion] = useState(0)

  useEffect(() => {
    if (segmentCount <= 0) {
      segmentStateRef.current = null
      return
    }
    const currentState = segmentStateRef.current
    if (currentState && currentState.count === segmentCount) {
      return
    }

    const centers = new Float32Array(segmentCount)
    const values = new Float32Array(segmentCount)
    for (let index = 0; index < segmentCount; index += 1) {
      const center = randomSigned()
      centers[index] = center
      values[index] = center * 0.16
    }
    segmentStateRef.current = { count: segmentCount, centers, values }
    setSegmentColorVersion((value) => value + 1)
  }, [segmentCount])

  useEffect(() => {
    if (trainingFlow?.phase === null) return
    const state = segmentStateRef.current
    if (!state || state.count <= 0) return

    for (let index = 0; index < state.count; index += 1) {
      const sample = sampleNormal(state.centers[index], SEGMENT_SAMPLE_STD_DEV)
      state.values[index] = clampSigned(state.values[index] + sample * SEGMENT_STEP_SCALE)
    }
    setSegmentColorVersion((value) => value + 1)
  }, [trainingFlow?.phase])

  const segmentColorData = useMemo(() => {
    const state = segmentStateRef.current
    if (!state || state.count <= 0 || state.count !== segmentCount) {
      const fallback = clampSigned(trainingFlow?.value ?? 0)
      return {
        colors: undefined as Float32Array | undefined,
        meanValue: fallback,
        meanAbs: Math.abs(fallback),
      }
    }

    const colors = new Float32Array(state.count * 6)
    const color = new THREE.Color()
    let valueSum = 0
    let valueAbsSum = 0
    for (let index = 0; index < state.count; index += 1) {
      const signedValue = clampSigned(state.values[index])
      valueSum += signedValue
      valueAbsSum += Math.abs(signedValue)
      setColorFromSignedValue(color, signedValue)

      const base = index * 6
      colors[base] = color.r
      colors[base + 1] = color.g
      colors[base + 2] = color.b
      colors[base + 3] = color.r
      colors[base + 4] = color.g
      colors[base + 5] = color.b
    }

    return {
      colors,
      meanValue: valueSum / state.count,
      meanAbs: valueAbsSum / state.count,
    }
  }, [segmentCount, segmentColorVersion, trainingFlow?.value])

  const lineColor = useMemo(
    () => getEdgeColor(segmentColorData.meanValue),
    [segmentColorData.meanValue]
  )
  const flowIntensity = Math.min(1, Math.max(0, trainingFlow?.intensity ?? 0))
  const colorStrength = Math.min(1, segmentColorData.meanAbs)
  const hasTrainingFlow = Boolean(trainingFlow)
  const lineOpacity = hasTrainingFlow
    ? trainingFlow?.active
      ? 0.34 + flowIntensity * 0.5
      : 0.24 + colorStrength * 0.56
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
          {segmentColorData.colors ? (
            <bufferAttribute attach="attributes-color" args={[segmentColorData.colors, 3]} />
          ) : null}
        </bufferGeometry>
        <lineBasicMaterial
          color={segmentColorData.colors ? '#ffffff' : hasTrainingFlow ? lineColor : lowDetailMode ? '#5f6268' : '#525252'}
          vertexColors={Boolean(segmentColorData.colors)}
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

function buildSampledSegments(
  sourceNode: LayerNode,
  targetNode: LayerNode,
  maxSegments: number
): Float32Array {
  const sourceGrid = getLayerGridSize(sourceNode.config, sourceNode.type)
  const targetGrid = getLayerGridSize(targetNode.config, targetNode.type)
  const sourceCount = Math.max(1, sourceGrid.rows * sourceGrid.cols)
  const targetCount = Math.max(1, targetGrid.rows * targetGrid.cols)
  const totalSegments = sourceCount * targetCount

  let sourceSamples = sourceCount
  let targetSamples = targetCount
  if (totalSegments > maxSegments) {
    sourceSamples = Math.max(
      1,
      Math.floor(Math.sqrt((maxSegments * sourceCount) / Math.max(1, targetCount)))
    )
    targetSamples = Math.max(1, Math.floor(maxSegments / Math.max(1, sourceSamples)))
    sourceSamples = Math.min(sourceSamples, sourceCount)
    targetSamples = Math.min(targetSamples, targetCount)

    while (sourceSamples * targetSamples > maxSegments) {
      if (targetSamples > sourceSamples && targetSamples > 1) {
        targetSamples -= 1
        continue
      }
      if (sourceSamples > 1) {
        sourceSamples -= 1
        continue
      }
      if (targetSamples > 1) {
        targetSamples -= 1
        continue
      }
      break
    }
  }

  const sourceIndices = buildEvenFlatIndices(sourceCount, sourceSamples)
  const targetIndices = buildEvenFlatIndices(targetCount, targetSamples)
  const values: number[] = []

  sourceIndices.forEach((sourceIndex) => {
    const source = pointAtFlatIndex(
      sourceNode,
      sourceGrid.rows,
      sourceGrid.cols,
      sourceIndex
    )
    targetIndices.forEach((targetIndex) => {
      const target = pointAtFlatIndex(
        targetNode,
        targetGrid.rows,
        targetGrid.cols,
        targetIndex
      )
      pushSegment(values, source, target)
    })
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

function pointAtFlatIndex(
  node: LayerNode,
  rows: number,
  cols: number,
  flatIndex: number
): [number, number, number] {
  const safeRows = Math.max(1, rows)
  const safeCols = Math.max(1, cols)
  const clampedFlat = Math.min(Math.max(flatIndex, 0), safeRows * safeCols - 1)
  const rowIndex = Math.floor(clampedFlat / safeCols)
  const colIndex = clampedFlat % safeCols
  return pointAtGridIndex(node, safeRows, safeCols, rowIndex, colIndex)
}

function buildEvenFlatIndices(totalCount: number, sampleCount: number): number[] {
  const safeTotal = Math.max(1, totalCount)
  const safeSample = Math.max(1, Math.min(sampleCount, safeTotal))
  if (safeSample === safeTotal) {
    return Array.from({ length: safeTotal }, (_, index) => index)
  }
  if (safeSample === 1) {
    return [Math.floor((safeTotal - 1) / 2)]
  }

  const indices: number[] = []
  const seen = new Set<number>()
  for (let index = 0; index < safeSample; index += 1) {
    const ratio = index / (safeSample - 1)
    const flat = Math.round(ratio * (safeTotal - 1))
    if (seen.has(flat)) continue
    seen.add(flat)
    indices.push(flat)
  }

  if (indices.length === 0) return [0]
  return indices
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

function setColorFromSignedValue(target: THREE.Color, value: number) {
  const clamped = clampSigned(value)
  target.copy(EDGE_VALUE_MID_COLOR)
  if (clamped >= 0) {
    target.lerp(EDGE_VALUE_HIGH_COLOR, clamped)
  } else {
    target.lerp(EDGE_VALUE_LOW_COLOR, Math.abs(clamped))
  }
}

function clampSigned(value: number): number {
  return Math.min(1, Math.max(-1, value))
}

function randomSigned(): number {
  return Math.random() * 2 - 1
}

function sampleNormal(mean: number, stdDev: number): number {
  const safeStd = Math.max(stdDev, 0.000001)
  let u = 0
  let v = 0
  while (u === 0) {
    u = Math.random()
  }
  while (v === 0) {
    v = Math.random()
  }
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  return mean + z * safeStd
}
