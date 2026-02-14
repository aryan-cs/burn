import { useEffect } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import { useThree } from '@react-three/fiber'
import type { Intersection } from 'three'
import * as THREE from 'three'
import { useGraphStore } from '../store/graphStore'
import { getLayerRolesForColoring } from '../utils/graphOrder'
import { Connection } from './edges/Connection'
import { ConnectionPreview } from './edges/ConnectionPreview'
import { LayerNode3D } from './nodes/LayerNode'

export function SceneManager() {
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
  const nodeRoles = getLayerRolesForColoring(nodes, edges)

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
