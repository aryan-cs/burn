import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface WeightVisualProps {
  curve: THREE.Curve<THREE.Vector3>
  speed: number
  color: string
  size?: number
  opacity?: number
  offset?: number
}

/**
 * Animated particle that travels along an edge during training
 * to visualize gradient flow.
 */
export function WeightVisual({
  curve,
  speed,
  color,
  size = 0.045,
  opacity = 1,
  offset = 0,
}: WeightVisualProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const tRef = useRef(normalizeOffset(offset))

  useFrame((_, delta) => {
    if (!meshRef.current) return
    tRef.current += delta * speed
    while (tRef.current > 1) {
      tRef.current -= 1
    }
    while (tRef.current < 0) {
      tRef.current += 1
    }
    const point = curve.getPoint(tRef.current)
    meshRef.current.position.copy(point)
  })

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[size, 10, 10]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} />
    </mesh>
  )
}

function normalizeOffset(value: number): number {
  if (!Number.isFinite(value)) return 0
  let normalized = value
  while (normalized > 1) normalized -= 1
  while (normalized < 0) normalized += 1
  return normalized
}
