import { useRef, type ReactNode } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'
import { useGraphStore } from '../../store/graphStore'

interface DragControlsProps {
  nodeId: string
  position: [number, number, number]
  children: ReactNode
}

/**
 * Wraps a node and makes it draggable on the XZ plane (y stays constant).
 * Uses pointer events with a drag plane at the node's Y height.
 */
export function DragControls({ nodeId, position, children }: DragControlsProps) {
  const groupRef = useRef<THREE.Group>(null)
  const isDragging = useRef(false)
  const dragPlane = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0))
  const offset = useRef(new THREE.Vector3())
  const setNodePosition = useGraphStore((s) => s.setNodePosition)

  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    if (e.button !== 0) return
    e.stopPropagation()
    isDragging.current = true
    ;(e.target as HTMLElement)?.setPointerCapture?.(e.pointerId)

    // Set the drag plane at the node's current Y
    dragPlane.current.constant = -position[1]

    // Calculate offset between intersection and node position
    const intersection = new THREE.Vector3()
    e.ray.intersectPlane(dragPlane.current, intersection)
    offset.current.set(
      position[0] - intersection.x,
      0,
      position[2] - intersection.z
    )
  }

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (!isDragging.current) return
    e.stopPropagation()

    const intersection = new THREE.Vector3()
    e.ray.intersectPlane(dragPlane.current, intersection)

    const newPos: [number, number, number] = [
      intersection.x + offset.current.x,
      position[1],
      intersection.z + offset.current.z,
    ]
    setNodePosition(nodeId, newPos)
  }

  const handlePointerUp = (e: ThreeEvent<PointerEvent>) => {
    isDragging.current = false
    ;(e.target as HTMLElement)?.releasePointerCapture?.(e.pointerId)
  }

  return (
    <group
      ref={groupRef}
      position={position}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {children}
    </group>
  )
}
