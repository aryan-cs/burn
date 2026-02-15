/* eslint-disable react-refresh/only-export-components */
import { useEffect, useMemo, useRef, useState } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'
import { useGraphStore, type LayerNode } from '../../store/graphStore'
import {
  LAYER_NODE_RADIUS,
  LAYER_NODE_SPACING,
  LAYER_NODE_Z_OFFSET,
  getLayerGridSize,
} from '../../utils/layerLayout'

export type LayerRole = 'input' | 'hidden' | 'output'
export type TrainingFlowPhase = 'forward' | 'backward'
export interface LayerTrainingPulse {
  active: boolean
  intensity: number
  value: number
  phase: TrainingFlowPhase | null
}

export const DRAG_MOUSE_BUTTON = 0

export const NODE_SPHERE_WIDTH_SEGMENTS = 20
export const NODE_SPHERE_HEIGHT_SEGMENTS = 20

export const NODE_CORE_COLOR = '#6e6e6e'
export const NODE_CORE_EMISSIVE_COLOR = '#1a1a1a'
export const NODE_CORE_EMISSIVE_INTENSITY_ACTIVE = 0.5
export const NODE_CORE_EMISSIVE_INTENSITY_IDLE = 0.3
export const NODE_CORE_ROUGHNESS = 0.55
export const NODE_CORE_METALNESS = 0.1

export const NODE_OUTLINE_SCALE = 1.22
export const NODE_OUTLINE_OPACITY_SOURCE = 0.95
export const NODE_OUTLINE_OPACITY_SELECTED = 1
export const NODE_OUTLINE_OPACITY_HIGHLIGHTED = 0.9
export const NODE_OUTLINE_OPACITY_HOVERED = 0.7
export const NODE_OUTLINE_OPACITY_IDLE = 0.6
export const NODE_OUTLINE_SIDE = THREE.BackSide
export const NODE_OUTLINE_DEPTH_WRITE = false

export const LAYER_ROLE_COLORS: Record<LayerRole, string> = {
  input: '#ff8c2b',
  hidden: '#ffffff',
  output: '#4da3ff',
}

const NODE_VALUE_LOW_COLOR = new THREE.Color('#ff4747')
const NODE_VALUE_MID_COLOR = new THREE.Color('#ffffff')
const NODE_VALUE_HIGH_COLOR = new THREE.Color('#4eff8e')

export const IGNORE_RAYCAST: THREE.Mesh['raycast'] = () => undefined

const MAX_NEURON_INSTANCES_FULL_DETAIL = 3600
const MAX_NEURON_INSTANCES_LOW_DETAIL = 900

export function LayerNode3D({
  node,
  role,
  trainingPulse,
  lowDetailMode = false,
}: {
  node: LayerNode
  role: LayerRole
  trainingPulse?: LayerTrainingPulse
  lowDetailMode?: boolean
}) {
  const [hovered, setHovered] = useState(false)
  const isMovingRef = useRef(false)
  const dragDistanceRef = useRef(0)
  const dragOffsetRef = useRef(new THREE.Vector3())
  const draggedNodeIdsRef = useRef<string[]>([])
  const dragStartPositionsRef = useRef<Record<string, THREE.Vector3>>({})
  const primaryDragStartPositionRef = useRef(new THREE.Vector3())
  const hasRecordedMoveHistoryRef = useRef(false)

  const isSelected = useGraphStore((s) => s.selectedNodeId === node.id)
  const isDragging = useGraphStore((s) => s.draggingNodeId === node.id)
  const isHighlighted = useGraphStore((s) => s.highlightedNodeIds.includes(node.id))
  const isConnectionSource = useGraphStore((s) => s.connectionSource === node.id)
  const isConnectionTarget = useGraphStore(
    (s) => s.connectionSource !== null && s.connectionSource !== node.id
  )
  const selectNode = useGraphStore((s) => s.selectNode)
  const setDraggingNodeId = useGraphStore((s) => s.setDraggingNodeId)
  const setNodesPosition = useGraphStore((s) => s.setNodesPosition)
  const startConnectionDrag = useGraphStore((s) => s.startConnectionDrag)

  const { rows, cols } = getLayerGridSize(node.config, node.type)
  const nodePositions = useMemo(
    () =>
      buildSampledLayerLocalNodePositions(
        rows,
        cols,
        lowDetailMode ? MAX_NEURON_INSTANCES_LOW_DETAIL : MAX_NEURON_INSTANCES_FULL_DETAIL
      ),
    [rows, cols, lowDetailMode]
  )
  const coreInstancesRef = useRef<THREE.InstancedMesh>(null)
  const outlineInstancesRef = useRef<THREE.InstancedMesh>(null)
  const tempObject = useMemo(() => new THREE.Object3D(), [])
  const sphereSegments = getSphereSegments(nodePositions.length)
  const baseColor = getLayerColor(role)
  const outlineOpacity = isConnectionSource
    ? NODE_OUTLINE_OPACITY_SOURCE
    : isSelected || isDragging
      ? NODE_OUTLINE_OPACITY_SELECTED
      : isHighlighted
        ? NODE_OUTLINE_OPACITY_HIGHLIGHTED
      : hovered || isConnectionTarget
        ? NODE_OUTLINE_OPACITY_HOVERED
        : NODE_OUTLINE_OPACITY_IDLE
  const trainingPulseIntensity = trainingPulse?.active
    ? Math.min(1, Math.max(0, trainingPulse.intensity))
    : 0
  const trainingValue = clampSigned(trainingPulse?.value ?? 0)
  const trainingPhase = trainingPulse?.phase ?? null
  const valueTint = useMemo(() => getValueTintHex(trainingValue), [trainingValue])
  const backwardPhaseBoost = trainingPhase === 'backward' ? 0.14 : 0
  const coreEmissiveIntensity = Math.max(
    isSelected || isDragging || isHighlighted || hovered
      ? NODE_CORE_EMISSIVE_INTENSITY_ACTIVE
      : NODE_CORE_EMISSIVE_INTENSITY_IDLE,
    NODE_CORE_EMISSIVE_INTENSITY_IDLE +
      trainingPulseIntensity * 0.38 +
      Math.abs(trainingValue) * 0.28 +
      backwardPhaseBoost
  )
  const pulsedOutlineOpacity = Math.min(
    NODE_OUTLINE_OPACITY_SELECTED,
    Math.max(
      outlineOpacity,
      NODE_OUTLINE_OPACITY_IDLE +
        trainingPulseIntensity * 0.36 +
      backwardPhaseBoost * 0.5
    )
  )

  useEffect(() => {
    const coreMesh = coreInstancesRef.current
    const outlineMesh = outlineInstancesRef.current
    if (!coreMesh || !outlineMesh) return

    nodePositions.forEach((position, index) => {
      tempObject.position.set(position[0], position[1], position[2])
      tempObject.scale.set(1, 1, 1)
      tempObject.updateMatrix()
      coreMesh.setMatrixAt(index, tempObject.matrix)

      tempObject.scale.set(NODE_OUTLINE_SCALE, NODE_OUTLINE_SCALE, NODE_OUTLINE_SCALE)
      tempObject.updateMatrix()
      outlineMesh.setMatrixAt(index, tempObject.matrix)
    })

    coreMesh.instanceMatrix.needsUpdate = true
    outlineMesh.instanceMatrix.needsUpdate = true
    coreMesh.computeBoundingSphere()
    outlineMesh.computeBoundingSphere()
  }, [nodePositions, tempObject])

  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    if (e.button !== DRAG_MOUSE_BUTTON) return
    if (e.shiftKey) return
    e.stopPropagation()
    e.nativeEvent.preventDefault()
    e.nativeEvent.stopImmediatePropagation?.()
    selectNode(node.id)

    if (e.ctrlKey) {
      startConnectionDrag(node.id, [e.point.x, e.point.y, e.point.z])
      return
    }

    isMovingRef.current = true
    hasRecordedMoveHistoryRef.current = false
    setDraggingNodeId(node.id)
    ;(e.target as HTMLElement)?.setPointerCapture?.(e.pointerId)

    const highlightedNodeIds = useGraphStore.getState().highlightedNodeIds
    const dragNodeIds = highlightedNodeIds.includes(node.id)
      ? highlightedNodeIds
      : [node.id]
    const currentNodes = useGraphStore.getState().nodes
    const startPositions: Record<string, THREE.Vector3> = {}
    dragNodeIds.forEach((dragNodeId) => {
      const dragNode = currentNodes[dragNodeId]
      if (dragNode) {
        startPositions[dragNodeId] = new THREE.Vector3(...dragNode.position)
      }
    })
    if (!startPositions[node.id]) {
      startPositions[node.id] = new THREE.Vector3(...node.position)
    }
    draggedNodeIdsRef.current = Object.keys(startPositions)
    dragStartPositionsRef.current = startPositions
    primaryDragStartPositionRef.current.copy(startPositions[node.id])

    dragDistanceRef.current = primaryDragStartPositionRef.current.distanceTo(e.ray.origin)

    const intersection = new THREE.Vector3()
      .copy(e.ray.direction)
      .multiplyScalar(dragDistanceRef.current)
      .add(e.ray.origin)
    dragOffsetRef.current.copy(primaryDragStartPositionRef.current).sub(intersection)
  }

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (!isMovingRef.current) return
    e.stopPropagation()
    e.nativeEvent.preventDefault()

    const intersection = new THREE.Vector3()
      .copy(e.ray.direction)
      .multiplyScalar(dragDistanceRef.current)
      .add(e.ray.origin)
    const nextPrimaryPosition = intersection.add(dragOffsetRef.current)
    const delta = new THREE.Vector3().copy(nextPrimaryPosition).sub(primaryDragStartPositionRef.current)
    const nextPositions: Record<string, [number, number, number]> = {}

    draggedNodeIdsRef.current.forEach((dragNodeId) => {
      const startPosition = dragStartPositionsRef.current[dragNodeId]
      if (!startPosition) return
      nextPositions[dragNodeId] = [
        startPosition.x + delta.x,
        startPosition.y + delta.y,
        startPosition.z + delta.z,
      ]
    })

    const shouldRecordHistory = !hasRecordedMoveHistoryRef.current
    setNodesPosition(nextPositions, shouldRecordHistory)
    hasRecordedMoveHistoryRef.current = true
  }

  const handlePointerUp = (e: ThreeEvent<PointerEvent>) => {
    if (!isMovingRef.current) return
    e.stopPropagation()
    e.nativeEvent.preventDefault()
    isMovingRef.current = false
    setDraggingNodeId(null)
    draggedNodeIdsRef.current = []
    dragStartPositionsRef.current = {}
    hasRecordedMoveHistoryRef.current = false
    ;(e.target as HTMLElement)?.releasePointerCapture?.(e.pointerId)
  }

  return (
    <group position={node.position}>
      <instancedMesh
        ref={coreInstancesRef}
        userData={{ layerId: node.id }}
        args={[undefined, undefined, nodePositions.length]}
        onPointerDown={handlePointerDown}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      >
        <sphereGeometry args={[LAYER_NODE_RADIUS, sphereSegments, sphereSegments]} />
        <meshStandardMaterial
          color={NODE_CORE_COLOR}
          emissive={trainingPulse ? valueTint : NODE_CORE_EMISSIVE_COLOR}
          emissiveIntensity={coreEmissiveIntensity}
          roughness={NODE_CORE_ROUGHNESS}
          metalness={NODE_CORE_METALNESS}
        />
      </instancedMesh>

      <instancedMesh
        ref={outlineInstancesRef}
        raycast={IGNORE_RAYCAST}
        args={[undefined, undefined, nodePositions.length]}
      >
        <sphereGeometry args={[LAYER_NODE_RADIUS, sphereSegments, sphereSegments]} />
        <meshBasicMaterial
          color={baseColor}
          transparent
          opacity={pulsedOutlineOpacity}
          side={NODE_OUTLINE_SIDE}
          depthWrite={NODE_OUTLINE_DEPTH_WRITE}
        />
      </instancedMesh>
    </group>
  )
}

function getLayerColor(role: LayerRole): string {
  return LAYER_ROLE_COLORS[role]
}

function getSphereSegments(nodeCount: number): number {
  if (nodeCount >= 600) return 7
  if (nodeCount >= 250) return 10
  if (nodeCount >= 120) return 14
  return NODE_SPHERE_WIDTH_SEGMENTS
}

function clampSigned(value: number): number {
  return Math.min(1, Math.max(-1, value))
}

function getValueTintHex(value: number): string {
  const clamped = clampSigned(value)
  const color = NODE_VALUE_MID_COLOR.clone()
  if (clamped >= 0) {
    color.lerp(NODE_VALUE_HIGH_COLOR, clamped)
  } else {
    color.lerp(NODE_VALUE_LOW_COLOR, Math.abs(clamped))
  }
  return `#${color.getHexString()}`
}

function buildSampledLayerLocalNodePositions(
  rows: number,
  cols: number,
  maxInstances: number
): Array<[number, number, number]> {
  const safeRows = Math.max(1, rows)
  const safeCols = Math.max(1, cols)
  const total = safeRows * safeCols
  if (total <= maxInstances) {
    return buildLayerLocalNodePositionsFromIndices(
      buildEvenIndices(safeRows, safeRows),
      buildEvenIndices(safeCols, safeCols),
      safeRows,
      safeCols
    )
  }

  const rowToColRatio = safeRows / Math.max(1, safeCols)
  let sampledRows = Math.max(1, Math.floor(Math.sqrt(maxInstances * rowToColRatio)))
  let sampledCols = Math.max(1, Math.floor(maxInstances / sampledRows))
  sampledRows = Math.min(sampledRows, safeRows)
  sampledCols = Math.min(sampledCols, safeCols)

  while (sampledRows * sampledCols > maxInstances) {
    if (sampledRows >= sampledCols && sampledRows > 1) {
      sampledRows -= 1
      continue
    }
    if (sampledCols > 1) {
      sampledCols -= 1
      continue
    }
    break
  }

  const rowIndices = buildEvenIndices(safeRows, sampledRows)
  const colIndices = buildEvenIndices(safeCols, sampledCols)
  return buildLayerLocalNodePositionsFromIndices(rowIndices, colIndices, safeRows, safeCols)
}

function buildLayerLocalNodePositionsFromIndices(
  rowIndices: number[],
  colIndices: number[],
  sourceRows: number,
  sourceCols: number
): Array<[number, number, number]> {
  const positions: Array<[number, number, number]> = []
  rowIndices.forEach((rowIndex) => {
    colIndices.forEach((colIndex) => {
      const x = (colIndex - (sourceCols - 1) / 2) * LAYER_NODE_SPACING
      const y = ((sourceRows - 1) / 2 - rowIndex) * LAYER_NODE_SPACING
      positions.push([x, y, LAYER_NODE_Z_OFFSET])
    })
  })
  return positions
}

function buildEvenIndices(totalCount: number, sampleCount: number): number[] {
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
  for (let i = 0; i < safeSample; i += 1) {
    const ratio = i / (safeSample - 1)
    const index = Math.round(ratio * (safeTotal - 1))
    if (seen.has(index)) continue
    seen.add(index)
    indices.push(index)
  }

  if (indices.length === 0) return [0]
  return indices
}
