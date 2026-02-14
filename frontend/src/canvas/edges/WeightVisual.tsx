import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface WeightVisualProps {
  curve: THREE.Curve<THREE.Vector3>
  speed: number
  color: string
}

/**
 * Animated particle that travels along an edge during training
 * to visualize gradient flow.
 */
export function WeightVisual({ curve, speed, color }: WeightVisualProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const tRef = useRef(0)

  useFrame((_, delta) => {
    if (!meshRef.current) return
    tRef.current = (tRef.current + delta * speed) % 1
    const point = curve.getPoint(tRef.current)
    meshRef.current.position.copy(point)
  })

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.06, 8, 8]} />
      <meshBasicMaterial color={color} />
    </mesh>
  )
}
