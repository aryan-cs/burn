import { useCallback } from 'react'
import { useGraphStore, type LayerType } from '../store/graphStore'

/**
 * Hook for handling palette-to-canvas drag & drop.
 * In the MVP, clicking the palette button directly adds a node.
 * A future version would use HTML5 drag events + raycasting.
 */
export function useDragToCanvas() {
  const addNode = useGraphStore((s) => s.addNode)
  const nodes = useGraphStore((s) => s.nodes)

  const addLayerAtDefaultPosition = useCallback(
    (type: LayerType) => {
      const count = Object.keys(nodes).length
      const x = count * 3 - 6
      addNode(type, [x, 0.5, 0])
    },
    [addNode, nodes]
  )

  return { addLayerAtDefaultPosition }
}
