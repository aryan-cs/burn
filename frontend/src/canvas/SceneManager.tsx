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
import { useGraphStore, type Edge, type LayerNode } from '../store/graphStore'
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
  const nodeValueByNodeId = useMemo(
    () => buildNodeTrainingValues(nodes, orderedNodeIds, latestWeights),
    [latestWeights, nodes, orderedNodeIds]
  )
  const edgeValueByEdgeId = useMemo(
    () => buildEdgeTrainingValues(edges, nodeValueByNodeId),
    [edges, nodeValueByNodeId]
  )
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
        value: nodeValueByNodeId.get(nodeId) ?? 0,
        phase: waveState.phase,
      })
    })
    return pulses
  }, [nodeValueByNodeId, orderedNodeIds, trainingStatus, waveState])

  const trainingFlowByEdgeId = useMemo(() => {
    const flows = new Map<
      string,
      { active: boolean; intensity: number; value: number; phase: TrainingFlowPhase | null }
    >()
    if (trainingStatus !== 'training') return flows

    Object.values(edges).forEach((edge) => {
      const sourceIndex = orderIndex.get(edge.source)
      const targetIndex = orderIndex.get(edge.target)
      if (sourceIndex === undefined || targetIndex === undefined) return
      const isAdjacent = Math.abs(targetIndex - sourceIndex) === 1

      const sourceIntensity = waveState.intensities[sourceIndex] ?? 0
      const targetIntensity = waveState.intensities[targetIndex] ?? 0
      const intensity =
        waveState.phase === null || !isAdjacent
          ? 0
          : Math.min(sourceIntensity, targetIntensity)

      flows.set(edge.id, {
        active: intensity > 0,
        intensity,
        value: edgeValueByEdgeId.get(edge.id) ?? 0,
        phase: waveState.phase,
      })
    })

    return flows
  }, [edgeValueByEdgeId, edges, orderIndex, trainingStatus, waveState])

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

function buildEdgeTrainingValues(
  edges: Record<string, Edge>,
  nodeValues: Map<string, number>
): Map<string, number> {
  const values = new Map<string, number>()
  Object.values(edges).forEach((edge) => {
    const sourceValue = nodeValues.get(edge.source) ?? 0
    const targetValue = nodeValues.get(edge.target) ?? 0
    values.set(edge.id, clampSigned((sourceValue + targetValue) / 2))
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
    .map(([, stats]) => normalizeWeightMean(stats.mean, stats.min, stats.max))
}

function extractLayerParamIndex(paramName: string): number {
  const match = paramName.match(/layers\.(\d+)\./)
  if (!match) return Number.MAX_SAFE_INTEGER
  return Number(match[1])
}

function normalizeWeightMean(mean: number, min: number, max: number): number {
  const range = max - min
  if (!Number.isFinite(range) || range <= 1e-8) {
    return clampSigned(mean)
  }
  const center = (max + min) / 2
  const normalized = (mean - center) / (range / 2)
  return clampSigned(Math.tanh(normalized * 1.7))
}

function clampSigned(value: number): number {
  return Math.min(1, Math.max(-1, value))
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
