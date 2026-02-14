import { useMemo } from 'react'
import type { LayerNode } from '../../store/graphStore'
import { buildLayerWorldNodePositions } from '../../utils/layerLayout'

interface ConnectionProps {
  sourceNode: LayerNode
  targetNode: LayerNode
}

export function Connection({ sourceNode, targetNode }: ConnectionProps) {
  const segmentPositions = useMemo(() => {
    const sourcePoints = buildLayerWorldNodePositions(sourceNode)
    const targetPoints = buildLayerWorldNodePositions(targetNode)
    const values: number[] = []

    for (const source of sourcePoints) {
      for (const target of targetPoints) {
        values.push(source[0], source[1], source[2], target[0], target[1], target[2])
      }
    }

    return new Float32Array(values)
  }, [sourceNode, targetNode])

  if (segmentPositions.length === 0) return null

  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[segmentPositions, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color="#525252" transparent opacity={0.38} />
    </lineSegments>
  )
}
