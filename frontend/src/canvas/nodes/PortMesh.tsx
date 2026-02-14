import { useState } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import { useGraphStore } from '../../store/graphStore'

interface PortMeshProps {
  nodeId: string
  type: 'input' | 'output'
  position: [number, number, number]
  onConnect?: () => void
}

export function PortMesh({ nodeId, type, position, onConnect }: PortMeshProps) {
  const [hovered, setHovered] = useState(false)
  const connectionSource = useGraphStore((s) => s.connectionSource)
  const setConnectionSource = useGraphStore((s) => s.setConnectionSource)
  const addEdge = useGraphStore((s) => s.addEdge)

  const isSource = connectionSource === nodeId && type === 'output'
  const canReceive = connectionSource !== null && connectionSource !== nodeId && type === 'input'

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation()

    if (type === 'output') {
      // Start connection from this output port
      setConnectionSource(nodeId)
      onConnect?.()
    } else if (type === 'input' && connectionSource) {
      // Complete connection to this input port
      addEdge(connectionSource, nodeId)
      setConnectionSource(null)
    }
  }

  const color = isSource
    ? '#00ff88'
    : canReceive && hovered
      ? '#00ff88'
      : canReceive
        ? '#88ffbb'
        : hovered
          ? '#ffffff'
          : type === 'output'
            ? '#66aaff'
            : '#ff8866'

  return (
    <mesh
      position={position}
      onClick={handleClick}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <sphereGeometry args={[0.12, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={hovered || isSource || canReceive ? 0.6 : 0.2}
      />
    </mesh>
  )
}
