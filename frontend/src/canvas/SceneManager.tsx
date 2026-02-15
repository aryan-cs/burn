import { useEffect, useMemo, useRef, useState } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import { useThree } from '@react-three/fiber'
import type { Intersection } from 'three'
import * as THREE from 'three'
import {
  EPOCH_WAVE_HOLD_MS,
  FADE_IN_MS,
  STAGGER_STEP_MS,
  computeStaggerProgress,
} from './animation/trainingPulse'
import { useGraphStore, type LayerNode } from '../store/graphStore'
import { useTrainingStore, type WeightSnapshot } from '../store/trainingStore'
import { getLayerRolesForColoring, getNeuralNetworkOrder } from '../utils/graphOrder'
import { Connection } from './edges/Connection'
import { ConnectionPreview } from './edges/ConnectionPreview'
import { LayerNode3D, type TrainingFlowPhase } from './nodes/LayerNode'

interface SceneManagerProps {
  lowDetailMode: boolean
}

export function SceneManager({ lowDetailMode }: SceneManagerProps) {
  const camera = useThree((s) => s.camera)
  const viewportSize = useThree((s) => s.size)
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const highlightSelectionActive = useGraphStore((s) => s.highlightSelectionActive)
  const highlightSelectionStart = useGraphStore((s) => s.highlightSelectionStart)
  const highlightSelectionEnd = useGraphStore((s) => s.highlightSelectionEnd)
  const setHighlightedNodes = useGraphStore((s) => s.setHighlightedNodes)
  const connectionSource = useGraphStore((s) => s.connectionSource)
  const connectionStart = useGraphStore((s) => s.connectionStart)
  const connectionCursor = useGraphStore((s) => s.connectionCursor)
  const updateConnectionCursor = useGraphStore((s) => s.updateConnectionCursor)
  const completeConnectionDrag = useGraphStore((s) => s.completeConnectionDrag)
  const trainingStatus = useTrainingStore((s) => s.status)
  const currentEpoch = useTrainingStore((s) => s.currentEpoch)
  const latestWeights = useTrainingStore((s) => s.latestWeights)
  const nodeRoles = getLayerRolesForColoring(nodes, edges)
  const orderedNodeIds = useMemo(() => getNeuralNetworkOrder(nodes, edges), [nodes, edges])
  const orderIndex = useMemo(() => {
    const value = new Map<string, number>()
    orderedNodeIds.forEach((nodeId, index) => {
      value.set(nodeId, index)
    })
    return value
  }, [orderedNodeIds])
  const currentNodeValueByNodeId = useMemo(
    () => buildNodeTrainingValues(nodes, orderedNodeIds, latestWeights),
    [latestWeights, nodes, orderedNodeIds]
  )
  const randomEdgeCenterByIdRef = useRef<Map<string, number>>(new Map())
  const randomEdgeValueByIdRef = useRef<Map<string, number>>(new Map())
  const previousWavePhaseRef = useRef<TrainingFlowPhase | null>(null)

  useEffect(() => {
    const activeEdgeIds = new Set(Object.keys(edges))
    const centers = randomEdgeCenterByIdRef.current
    const values = randomEdgeValueByIdRef.current

    Array.from(centers.keys()).forEach((edgeId) => {
      if (!activeEdgeIds.has(edgeId)) {
        centers.delete(edgeId)
      }
    })
    Array.from(values.keys()).forEach((edgeId) => {
      if (!activeEdgeIds.has(edgeId)) {
        values.delete(edgeId)
      }
    })

    Object.keys(edges).forEach((edgeId) => {
      if (!centers.has(edgeId)) {
        centers.set(edgeId, randomSigned())
      }
      if (!values.has(edgeId)) {
        values.set(edgeId, randomSigned() * 0.16)
      }
    })
  }, [edges])

  const epochWaveStartMsRef = useRef<number | null>(null)
  const prevEpochRef = useRef(0)
  const [waveNowMs, setWaveNowMs] = useState(() => performance.now())

  useEffect(() => {
    if (trainingStatus !== 'training') {
      prevEpochRef.current = currentEpoch
      epochWaveStartMsRef.current = null
      return
    }

    if (currentEpoch > prevEpochRef.current) {
      prevEpochRef.current = currentEpoch
      const now = performance.now()
      epochWaveStartMsRef.current = now
      setWaveNowMs(now)
    }
  }, [currentEpoch, trainingStatus])

  useEffect(() => {
    if (trainingStatus !== 'training') {
      return
    }
    if (epochWaveStartMsRef.current === null) {
      epochWaveStartMsRef.current = performance.now()
    }
    let rafId = 0
    const tick = () => {
      const now = performance.now()
      setWaveNowMs(now)
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(rafId)
    }
  }, [trainingStatus])

  const waveState = useMemo(() => {
    if (trainingStatus !== 'training') {
      return createIdleWaveState(orderedNodeIds.length)
    }
    const waveStart = epochWaveStartMsRef.current ?? waveNowMs
    const elapsedMs = Math.max(0, waveNowMs - waveStart)
    return computeTrainingWaveState(elapsedMs, orderedNodeIds.length)
  }, [orderedNodeIds.length, trainingStatus, waveNowMs])

  useEffect(() => {
    const previousPhase = previousWavePhaseRef.current
    previousWavePhaseRef.current = waveState.phase

    if (trainingStatus !== 'training') {
      return
    }
    if (waveState.phase === null || waveState.phase === previousPhase) return

    const centers = randomEdgeCenterByIdRef.current
    const values = randomEdgeValueByIdRef.current
    Object.keys(edges).forEach((edgeId) => {
      const center = centers.get(edgeId) ?? randomSigned()
      if (!centers.has(edgeId)) {
        centers.set(edgeId, center)
      }
      const current = values.get(edgeId) ?? 0
      const sample = sampleNormal(center, 0.34)
      const nextValue = clampSigned(current + sample * 0.2)
      values.set(edgeId, nextValue)
    })
  }, [edges, trainingStatus, waveState.phase])

  const trainingPulseByNodeId = useMemo(() => {
    const pulses = new Map<
      string,
      { active: boolean; intensity: number; value: number; phase: TrainingFlowPhase | null }
    >()
    if (trainingStatus !== 'training') return pulses

    orderedNodeIds.forEach((nodeId, index) => {
      const intensity = waveState.intensities[index] ?? 0
      pulses.set(nodeId, {
        active: intensity > 0,
        intensity,
        value: currentNodeValueByNodeId.get(nodeId) ?? 0,
        phase: waveState.phase,
      })
    })
    return pulses
  }, [currentNodeValueByNodeId, orderedNodeIds, trainingStatus, waveState])

  const trainingFlowByEdgeId = useMemo(() => {
    const flows = new Map<
      string,
      { active: boolean; intensity: number; value: number; phase: TrainingFlowPhase | null }
    >()
    const isTraining = trainingStatus === 'training'

    Object.values(edges).forEach((edge) => {
      const sourceIndex = orderIndex.get(edge.source)
      const targetIndex = orderIndex.get(edge.target)
      if (sourceIndex === undefined || targetIndex === undefined) return
      const isAdjacent = Math.abs(targetIndex - sourceIndex) === 1

      const sourceIntensity = waveState.intensities[sourceIndex] ?? 0
      const targetIntensity = waveState.intensities[targetIndex] ?? 0
      const intensity =
        !isTraining || waveState.phase === null || !isAdjacent
          ? 0
          : Math.min(sourceIntensity, targetIntensity)
      const edgeValue = randomEdgeValueByIdRef.current.get(edge.id) ?? 0
      const flowBlend = clamp01(intensity)
      const activeValue = clampSigned(
        edgeValue * (0.9 + flowBlend * 0.2)
      )

      flows.set(edge.id, {
        active: intensity > 0,
        intensity,
        value: isTraining ? activeValue : edgeValue,
        phase: isTraining ? waveState.phase : null,
      })
    })

    return flows
  }, [edges, orderIndex, trainingStatus, waveState])

  useEffect(() => {
    if (!highlightSelectionActive || !highlightSelectionStart || !highlightSelectionEnd) {
      return
    }

    const minX = Math.min(highlightSelectionStart[0], highlightSelectionEnd[0])
    const maxX = Math.max(highlightSelectionStart[0], highlightSelectionEnd[0])
    const minY = Math.min(highlightSelectionStart[1], highlightSelectionEnd[1])
    const maxY = Math.max(highlightSelectionStart[1], highlightSelectionEnd[1])

    const projected = new THREE.Vector3()
    const nextHighlightedNodeIds: string[] = []
    Object.values(nodes).forEach((node) => {
      projected.set(node.position[0], node.position[1], node.position[2]).project(camera)

      const screenX = (projected.x * 0.5 + 0.5) * viewportSize.width
      const screenY = (-projected.y * 0.5 + 0.5) * viewportSize.height
      if (screenX >= minX && screenX <= maxX && screenY >= minY && screenY <= maxY) {
        nextHighlightedNodeIds.push(node.id)
      }
    })

    setHighlightedNodes(nextHighlightedNodeIds)
  }, [
    highlightSelectionActive,
    highlightSelectionStart,
    highlightSelectionEnd,
    nodes,
    camera,
    viewportSize.width,
    viewportSize.height,
    setHighlightedNodes,
  ])

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (!connectionSource) return
    e.stopPropagation()
    updateConnectionCursor([e.point.x, e.point.y, e.point.z])
  }

  const handlePointerUp = (e: ThreeEvent<PointerEvent>) => {
    if (!connectionSource) return
    e.stopPropagation()
    const targetId = findTargetLayerId(e.intersections, connectionSource)
    completeConnectionDrag(targetId)
  }

  return (
    <group onPointerMove={handlePointerMove} onPointerUp={handlePointerUp}>
      {Object.values(nodes).map((node) => (
        <LayerNode3D
          key={node.id}
          node={node}
          role={nodeRoles.get(node.id) ?? 'hidden'}
          trainingPulse={trainingPulseByNodeId.get(node.id)}
          lowDetailMode={lowDetailMode}
        />
      ))}

      {Object.values(edges).map((edge) => {
        const sourceNode = nodes[edge.source]
        const targetNode = nodes[edge.target]
        if (!sourceNode || !targetNode) return null
        return (
          <Connection
            key={edge.id}
            sourceNode={sourceNode}
            targetNode={targetNode}
            lowDetailMode={lowDetailMode}
            trainingFlow={trainingFlowByEdgeId.get(edge.id)}
          />
        )
      })}

      {connectionSource && connectionStart ? (
        <mesh
          position={[0, 0, connectionStart[2]]}
          userData={{ interactionSurface: true }}
        >
          <planeGeometry args={[600, 600]} />
          <meshBasicMaterial
            transparent
            opacity={0}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      ) : null}

      {connectionSource && connectionStart && connectionCursor ? (
        <ConnectionPreview start={connectionStart} end={connectionCursor} />
      ) : null}
    </group>
  )
}

function findTargetLayerId(
  intersections: Intersection[],
  sourceLayerId: string
): string | null {
  for (const intersection of intersections) {
    const maybeLayerId = intersection.object.userData.layerId
    if (typeof maybeLayerId === 'string' && maybeLayerId !== sourceLayerId) {
      return maybeLayerId
    }
  }
  return null
}

function createIdleWaveState(nodeCount: number): {
  phase: TrainingFlowPhase | null
  intensities: number[]
} {
  return {
    phase: null,
    intensities: Array.from({ length: nodeCount }, () => 0),
  }
}

function computeTrainingWaveState(
  elapsedMs: number,
  nodeCount: number
): { phase: TrainingFlowPhase | null; intensities: number[] } {
  if (nodeCount <= 0) {
    return createIdleWaveState(0)
  }

  const forwardDurationMs = Math.max(
    FADE_IN_MS,
    (nodeCount - 1) * STAGGER_STEP_MS + FADE_IN_MS
  )
  const backwardDurationMs = forwardDurationMs
  const cycleDurationMs =
    forwardDurationMs +
    EPOCH_WAVE_HOLD_MS +
    backwardDurationMs +
    EPOCH_WAVE_HOLD_MS
  const elapsedCycleMs = elapsedMs % Math.max(1, cycleDurationMs)
  const intensities = Array.from({ length: nodeCount }, () => 0)

  if (elapsedCycleMs < forwardDurationMs) {
    for (let index = 0; index < nodeCount; index += 1) {
      intensities[index] = computeStaggerProgress(elapsedCycleMs, index)
    }
    return { phase: 'forward', intensities }
  }

  const backwardWindowStart = forwardDurationMs + EPOCH_WAVE_HOLD_MS
  const backwardWindowEnd = backwardWindowStart + backwardDurationMs
  if (elapsedCycleMs >= backwardWindowStart && elapsedCycleMs < backwardWindowEnd) {
    const backwardElapsedMs = elapsedCycleMs - backwardWindowStart
    for (let index = 0; index < nodeCount; index += 1) {
      const reverseIndex = nodeCount - 1 - index
      intensities[index] = computeStaggerProgress(backwardElapsedMs, reverseIndex)
    }
    return { phase: 'backward', intensities }
  }

  return { phase: null, intensities }
}

function buildNodeTrainingValues(
  nodes: Record<string, LayerNode>,
  orderedNodeIds: string[],
  weights: WeightSnapshot | null
): Map<string, number> {
  const directValues = new Map<string, number>()
  const trainableNodeIds = orderedNodeIds.filter((nodeId) =>
    isTrainableLayerType(nodes[nodeId]?.type)
  )
  const weightValues = extractNormalizedParameterValues(weights, 'weight')

  trainableNodeIds.forEach((nodeId, index) => {
    directValues.set(nodeId, weightValues[index] ?? 0)
  })

  const values = new Map<string, number>()
  orderedNodeIds.forEach((nodeId, index) => {
    const direct = directValues.get(nodeId)
    if (direct !== undefined) {
      values.set(nodeId, direct)
      return
    }

    let left: number | null = null
    for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
      const candidate = directValues.get(orderedNodeIds[cursor])
      if (candidate !== undefined) {
        left = candidate
        break
      }
    }

    let right: number | null = null
    for (let cursor = index + 1; cursor < orderedNodeIds.length; cursor += 1) {
      const candidate = directValues.get(orderedNodeIds[cursor])
      if (candidate !== undefined) {
        right = candidate
        break
      }
    }

    if (left !== null && right !== null) {
      values.set(nodeId, clampSigned((left + right) / 2))
    } else if (left !== null) {
      values.set(nodeId, left)
    } else if (right !== null) {
      values.set(nodeId, right)
    } else {
      values.set(nodeId, 0)
    }
  })

  return values
}

function extractNormalizedParameterValues(
  weights: WeightSnapshot | null,
  parameterKind: 'weight' | 'bias'
): number[] {
  if (!weights) return []

  return Object.entries(weights)
    .filter(([name]) => name.endsWith(`.${parameterKind}`))
    .sort(([nameA], [nameB]) => {
      const indexA = extractLayerParamIndex(nameA)
      const indexB = extractLayerParamIndex(nameB)
      if (indexA !== indexB) return indexA - indexB
      return nameA.localeCompare(nameB)
    })
    .map(([, stats]) =>
      normalizeWeightMean(stats.mean, stats.std, stats.min, stats.max)
    )
}

function extractLayerParamIndex(paramName: string): number {
  const match = paramName.match(/layers\.(\d+)\./)
  if (!match) return Number.MAX_SAFE_INTEGER
  return Number(match[1])
}

function normalizeWeightMean(mean: number, std: number, min: number, max: number): number {
  const safeSpread = Math.max(Math.abs(min), Math.abs(max), 1e-6)
  const spreadSignal = mean / safeSpread
  const safeStd = Math.max(Number.isFinite(std) ? std : 0, safeSpread * 0.04, 0.01)
  const zSignal = mean / safeStd
  const mixedSignal = spreadSignal * 0.3 + zSignal * 0.7
  const contrasted = Math.sign(mixedSignal) * Math.pow(Math.abs(mixedSignal), 0.78)
  return clampSigned(Math.tanh(contrasted * 4.2))
}

function clampSigned(value: number): number {
  return Math.min(1, Math.max(-1, value))
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value))
}

function randomSigned(): number {
  return Math.random() * 2 - 1
}

function sampleNormal(mean: number, stdDev: number): number {
  const safeStd = Math.max(0.000001, stdDev)
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

function isTrainableLayerType(type: LayerNode['type'] | undefined): boolean {
  return (
    type === 'Dense' ||
    type === 'Output' ||
    type === 'Conv2D' ||
    type === 'BatchNorm' ||
    type === 'LSTM' ||
    type === 'GRU'
  )
}
