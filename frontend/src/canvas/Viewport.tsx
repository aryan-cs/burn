import { Canvas } from '@react-three/fiber'
import { SceneManager } from './SceneManager'
import { CameraRig } from './controls/CameraRig'

export function Viewport() {
  return (
    <Canvas
      camera={{ position: [0, 5, 12], fov: 50 }}
      style={{ width: '100%', height: '100%' }}
      onPointerMissed={() => {
        // Deselect when clicking empty space — handled by store
      }}
    >
      <color attach="background" args={['#0a0a0f']} />
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} />
      <pointLight position={[-10, -10, -5]} intensity={0.3} color="#4A90D9" />
      <gridHelper args={[40, 40, '#1a1a2e', '#1a1a2e']} />
      <SceneManager />
      <CameraRig />
    </Canvas>
  )
}
