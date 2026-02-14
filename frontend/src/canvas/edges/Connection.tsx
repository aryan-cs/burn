import { useMemo } from 'react'
import * as THREE from 'three'
import type { Edge } from '../../store/graphStore'
import { weightToColor, weightToThickness } from '../../utils/colorScale'

interface ConnectionProps {
  edge: Edge
  sourcePos: [number, number, number]
  targetPos: [number, number, number]
}

export function Connection({ edge, sourcePos, targetPos }: ConnectionProps) {
  const { curve, color, thickness } = useMemo(() => {
    const src = new THREE.Vector3(...sourcePos)
    const tgt = new THREE.Vector3(...targetPos)
    const mid = new THREE.Vector3().lerpVectors(src, tgt, 0.5)
    // Slight upward arc
    mid.y += 0.5

    const c = new THREE.QuadraticBezierCurve3(src, mid, tgt)

    const col = edge.weightStats
      ? weightToColor(edge.weightStats.mean)
      : '#4A90D9'

    const thick = edge.weightStats
      ? weightToThickness(edge.weightStats.mean)
      : 0.04

    return { curve: c, color: col, thickness: thick }
  }, [sourcePos, targetPos, edge.weightStats])

  const tubeGeometry = useMemo(() => {
    return new THREE.TubeGeometry(curve, 32, thickness, 8, false)
  }, [curve, thickness])

  return (
    <mesh geometry={tubeGeometry}>
      <meshStandardMaterial
        color={color}
        transparent
        opacity={0.7}
        emissive={color}
        emissiveIntensity={0.3}
      />
    </mesh>
  )
}
