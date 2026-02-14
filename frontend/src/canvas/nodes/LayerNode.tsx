import { useRef, useState } from 'react'
import { ThreeEvent } from '@react-three/fiber'
import { Text } from '@react-three/drei'
import * as THREE from 'three'
import { useGraphStore, type LayerNode, type LayerType } from '../../store/graphStore'
import { PortMesh } from './PortMesh'
import { DragControls } from '../controls/DragControls'

// ── Visual mapping ─────────────────────────────────────

const LAYER_COLORS: Record<LayerType, string> = {
  Input: '#4A4A4A',
  Dense: '#4A90D9',
  Conv2D: '#7B61FF',
  MaxPool2D: '#50E3C2',
  LSTM: '#F5A623',
  GRU: '#F5A623',
  Dropout: '#9B9B9B',
  BatchNorm: '#9B9B9B',
  Flatten: '#50E3C2',
  Reshape: '#50E3C2',
  Output: '#D0021B',
}

const LAYER_SIZES: Record<LayerType, [number, number, number]> = {
  Input: [1.8, 0.2, 1.2],
  Dense: [1.6, 1.0, 1.0],
  Conv2D: [2.0, 0.6, 1.4],
  MaxPool2D: [1.4, 0.6, 1.0],
  LSTM: [1.4, 1.4, 1.4],
  GRU: [1.4, 1.4, 1.4],
  Dropout: [1.2, 0.8, 1.0],
  BatchNorm: [1.2, 0.8, 1.0],
  Flatten: [1.6, 0.3, 0.8],
  Reshape: [1.6, 0.3, 0.8],
  Output: [1.4, 1.2, 1.0],
}

export function LayerNode3D({ node }: { node: LayerNode }) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const selectNode = useGraphStore((s) => s.selectNode)
  const setConnectionSource = useGraphStore((s) => s.setConnectionSource)

  const isSelected = selectedNodeId === node.id
  const color = LAYER_COLORS[node.type]
  const size = LAYER_SIZES[node.type]

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation()
    selectNode(node.id)
  }

  const configSummary = getConfigSummary(node)

  return (
    <DragControls nodeId={node.id} position={node.position}>
      <group>
        {/* Main body */}
        <mesh
          ref={meshRef}
          onClick={handleClick}
          onPointerOver={() => setHovered(true)}
          onPointerOut={() => setHovered(false)}
        >
          <boxGeometry args={size} />
          <meshStandardMaterial
            color={color}
            transparent
            opacity={node.type === 'Dropout' ? 0.5 : 0.85}
            emissive={isSelected ? color : hovered ? color : '#000000'}
            emissiveIntensity={isSelected ? 0.4 : hovered ? 0.2 : 0}
          />
        </mesh>

        {/* Selection outline */}
        {isSelected && (
          <mesh>
            <boxGeometry args={[size[0] + 0.08, size[1] + 0.08, size[2] + 0.08]} />
            <meshBasicMaterial color="#ffffff" wireframe />
          </mesh>
        )}

        {/* Label */}
        <Text
          position={[0, size[1] / 2 + 0.3, 0]}
          fontSize={0.25}
          color="#ffffff"
          anchorX="center"
          anchorY="bottom"
          font={undefined}
        >
          {node.type}
        </Text>

        {/* Config summary */}
        {configSummary && (
          <Text
            position={[0, -size[1] / 2 - 0.2, 0]}
            fontSize={0.16}
            color="#aaaaaa"
            anchorX="center"
            anchorY="top"
            font={undefined}
          >
            {configSummary}
          </Text>
        )}

        {/* Ports */}
        <PortMesh
          nodeId={node.id}
          type="input"
          position={[-size[0] / 2 - 0.15, 0, 0]}
        />
        <PortMesh
          nodeId={node.id}
          type="output"
          position={[size[0] / 2 + 0.15, 0, 0]}
          onConnect={() => setConnectionSource(node.id)}
        />
      </group>
    </DragControls>
  )
}

function getConfigSummary(node: LayerNode): string {
  const c = node.config
  switch (node.type) {
    case 'Dense':
      return `${c.units} units, ${c.activation}`
    case 'Conv2D':
      return `${c.filters}f, ${c.kernel_size}x${c.kernel_size}`
    case 'Dropout':
      return `rate=${c.rate}`
    case 'Input':
      return c.shape ? `shape=[${c.shape.join(',')}]` : ''
    case 'Output':
      return `${c.num_classes} classes`
    case 'MaxPool2D':
      return `${c.kernel_size}x${c.kernel_size}`
    default:
      return ''
  }
}
