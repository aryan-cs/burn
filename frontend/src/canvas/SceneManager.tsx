import { useGraphStore } from '../store/graphStore'
import { LayerNode3D } from './nodes/LayerNode'
import { Connection } from './edges/Connection'

export function SceneManager() {
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)

  return (
    <group>
      {Object.values(nodes).map((node) => (
        <LayerNode3D key={node.id} node={node} />
      ))}
      {Object.values(edges).map((edge) => {
        const sourceNode = nodes[edge.source]
        const targetNode = nodes[edge.target]
        if (!sourceNode || !targetNode) return null
        return (
          <Connection
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
