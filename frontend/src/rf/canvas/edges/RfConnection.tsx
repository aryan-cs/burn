import { useMemo } from 'react'
import * as THREE from 'three'
import type { RFEdge } from '../../types'

interface RfConnectionProps {
  edge: RFEdge
  sourcePos: [number, number, number]
  targetPos: [number, number, number]
}

export function RfConnection({ edge, sourcePos, targetPos }: RfConnectionProps) {
  const geometry = useMemo(() => {
    const source = new THREE.Vector3(...sourcePos)
    const target = new THREE.Vector3(...targetPos)
    const mid = new THREE.Vector3().lerpVectors(source, target, 0.5)
    mid.y += 0.55
    const curve = new THREE.QuadraticBezierCurve3(source, mid, target)
    return new THREE.TubeGeometry(curve, 32, 0.035, 8, false)
  }, [sourcePos, targetPos, edge.id])

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color="#d9d9d9" emissive="#8a8a8a" emissiveIntensity={0.35} transparent opacity={0.6} />
    </mesh>
  )
}
