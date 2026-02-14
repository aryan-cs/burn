import { useRef, type ReactNode } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'
import { useRFGraphStore } from '../../store/rfGraphStore'

interface RfDragControlsProps {
  nodeId: string
  position: [number, number, number]
  children: ReactNode
}

export function RfDragControls({ nodeId, position, children }: RfDragControlsProps) {
  const isDragging = useRef(false)
  const dragPlane = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0))
  const offset = useRef(new THREE.Vector3())
  const setNodePosition = useRFGraphStore((state) => state.setNodePosition)

  const handlePointerDown = (event: ThreeEvent<PointerEvent>) => {
    if (event.button !== 0) return
    event.stopPropagation()
    isDragging.current = true
    ;(event.target as HTMLElement)?.setPointerCapture?.(event.pointerId)

    dragPlane.current.constant = -position[1]
    const intersection = new THREE.Vector3()
    event.ray.intersectPlane(dragPlane.current, intersection)
    offset.current.set(position[0] - intersection.x, 0, position[2] - intersection.z)
  }

  const handlePointerMove = (event: ThreeEvent<PointerEvent>) => {
    if (!isDragging.current) return
    event.stopPropagation()

    const intersection = new THREE.Vector3()
    event.ray.intersectPlane(dragPlane.current, intersection)
    setNodePosition(nodeId, [
      intersection.x + offset.current.x,
      position[1],
      intersection.z + offset.current.z,
    ])
  }

  const handlePointerUp = (event: ThreeEvent<PointerEvent>) => {
    isDragging.current = false
    ;(event.target as HTMLElement)?.releasePointerCapture?.(event.pointerId)
  }

  return (
    <group
      position={position}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {children}
    </group>
  )
}
