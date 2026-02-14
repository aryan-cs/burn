import { OrbitControls } from '@react-three/drei'

export function CameraRig() {
  return (
    <OrbitControls
      makeDefault
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
      mouseButtons={{
        LEFT: undefined, // Left click reserved for selection/drag
        MIDDLE: 2,       // Middle click = pan
        RIGHT: 1,        // Right click = orbit
      }}
      minDistance={3}
      maxDistance={50}
    />
  )
}
