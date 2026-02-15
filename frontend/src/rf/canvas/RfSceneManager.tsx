import { useRFGraphStore } from '../store/rfGraphStore'
import { RfConnection } from './edges/RfConnection'
import { RfNode3D } from './nodes/RfNode3D'

export function RfSceneManager() {
  const nodes = useRFGraphStore((state) => state.nodes)
  const edges = useRFGraphStore((state) => state.edges)

  return (
    <group>
      {Object.values(nodes).map((node) => (
        <RfNode3D key={node.id} node={node} />
      ))}
      {Object.values(edges).map((edge) => {
        const sourceNode = nodes[edge.source]
        const targetNode = nodes[edge.target]
        if (!sourceNode || !targetNode) return null
        return (
          <RfConnection
            key={edge.id}
            edge={edge}
            sourcePos={sourceNode.position}
            targetPos={targetNode.position}
          />
        )
      })}
    </group>
  )
}
