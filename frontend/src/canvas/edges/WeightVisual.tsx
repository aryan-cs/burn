import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface WeightVisualProps {
  curve: THREE.Curve<THREE.Vector3>
  speed: number
  color: string
  size?: number
}

/**
 * Animated particle that travels along an edge during training
 * to visualize gradient flow.
 */
export function WeightVisual({ curve, speed, color, size = 0.045 }: WeightVisualProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const tRef = useRef(0)

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
      <meshBasicMaterial color={color} />
    </mesh>
  )
}
