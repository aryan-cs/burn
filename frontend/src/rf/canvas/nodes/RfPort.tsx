import { useState } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import { useRFGraphStore } from '../../store/rfGraphStore'

interface RfPortProps {
  nodeId: string
  type: 'input' | 'output'
  position: [number, number, number]
}

export function RfPort({ nodeId, type, position }: RfPortProps) {
  const [hovered, setHovered] = useState(false)
  const connectionSource = useRFGraphStore((state) => state.connectionSource)
  const setConnectionSource = useRFGraphStore((state) => state.setConnectionSource)
  const addEdge = useRFGraphStore((state) => state.addEdge)

  const isSource = connectionSource === nodeId && type === 'output'
  const canReceive = connectionSource !== null && connectionSource !== nodeId && type === 'input'

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    if (type === 'output') {
      setConnectionSource(nodeId)
      return
    }

    if (type === 'input' && connectionSource) {
      addEdge(connectionSource, nodeId)
      setConnectionSource(null)
    }
  }

  const color = isSource
    ? '#00f8aa'
    : canReceive && hovered
      ? '#68ffaf'
      : canReceive
        ? '#78dca8'
        : hovered
          ? '#f6fdff'
          : type === 'output'
            ? '#67b4ff'
            : '#ff8f6c'

  return (
    <mesh
      position={position}
      onClick={handleClick}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <sphereGeometry args={[0.11, 16, 16]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={hovered || isSource || canReceive ? 0.6 : 0.2} />
    </mesh>
  )
}
