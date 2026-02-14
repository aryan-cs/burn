import { useState } from 'react'
import { Text } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import { RfDragControls } from '../controls/RfDragControls'
import { RfPort } from './RfPort'
import { useRFGraphStore } from '../../store/rfGraphStore'
import type { RFNode, RFNodeType } from '../../types'

const NODE_COLORS: Record<RFNodeType, string> = {
  RFInput: '#4f9ed9',
  RFFlatten: '#59b9a6',
  RandomForestClassifier: '#77d46f',
  RFOutput: '#d56c78',
}

const NODE_SIZES: Record<RFNodeType, [number, number, number]> = {
  RFInput: [1.8, 0.32, 1.1],
  RFFlatten: [1.35, 0.25, 0.9],
  RandomForestClassifier: [2.1, 1.2, 1.2],
  RFOutput: [1.5, 1.0, 1.0],
}

function getSummary(node: RFNode): string {
  if (node.type === 'RFInput') {
    const shape = Array.isArray(node.config.shape) ? node.config.shape : []
    return `shape=[${shape.join(',')}]`
  }
  if (node.type === 'RandomForestClassifier') {
    const trees = node.config.n_estimators
    const depth = node.config.max_depth
    return `trees=${String(trees)} depth=${depth === null ? 'none' : String(depth)}`
  }
  if (node.type === 'RFOutput') {
    return `classes=${String(node.config.num_classes)}`
  }
  return 'flatten'
}

export function RfNode3D({ node }: { node: RFNode }) {
  const [hovered, setHovered] = useState(false)
  const selectedNodeId = useRFGraphStore((state) => state.selectedNodeId)
  const selectNode = useRFGraphStore((state) => state.selectNode)
  const isSelected = selectedNodeId === node.id
  const size = NODE_SIZES[node.type]
  const color = NODE_COLORS[node.type]

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    selectNode(node.id)
  }

  return (
    <RfDragControls nodeId={node.id} position={node.position}>
      <group>
        <mesh onClick={handleClick} onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)}>
          <boxGeometry args={size} />
          <meshStandardMaterial
            color={color}
            transparent
            opacity={0.86}
            emissive={isSelected ? color : hovered ? color : '#000000'}
            emissiveIntensity={isSelected ? 0.36 : hovered ? 0.18 : 0}
          />
        </mesh>
        {isSelected && (
          <mesh>
            <boxGeometry args={[size[0] + 0.06, size[1] + 0.06, size[2] + 0.06]} />
            <meshBasicMaterial color="#ffffff" wireframe />
          </mesh>
        )}

        <Text position={[0, size[1] / 2 + 0.3, 0]} fontSize={0.22} color="#e7f7ff" anchorX="center" anchorY="bottom">
          {node.type}
        </Text>
        <Text position={[0, -size[1] / 2 - 0.22, 0]} fontSize={0.14} color="#b4d3ec" anchorX="center" anchorY="top">
          {getSummary(node)}
        </Text>

        <RfPort nodeId={node.id} type="input" position={[-size[0] / 2 - 0.14, 0, 0]} />
        <RfPort nodeId={node.id} type="output" position={[size[0] / 2 + 0.14, 0, 0]} />
      </group>
    </RfDragControls>
  )
}
