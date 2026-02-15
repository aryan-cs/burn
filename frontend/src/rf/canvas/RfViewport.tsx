import { OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useRFGraphStore } from '../store/rfGraphStore'
import { RfSceneManager } from './RfSceneManager'

export function RfViewport() {
  const selectNode = useRFGraphStore((state) => state.selectNode)
  const setConnectionSource = useRFGraphStore((state) => state.setConnectionSource)

  return (
    <div className="rf-viewport-shell">
      <Canvas
        camera={{ position: [0, 5.5, 12], fov: 50 }}
        style={{ width: '100%', height: '100%' }}
        onPointerMissed={() => {
          selectNode(null)
          setConnectionSource(null)
        }}
      >
        <color attach="background" args={['#031122']} />
        <ambientLight intensity={0.45} />
        <directionalLight position={[10, 12, 8]} intensity={0.9} />
        <pointLight position={[-10, -8, -5]} intensity={0.4} color="#4ab1ff" />
        <gridHelper args={[36, 36, '#1d3450', '#16283f']} />
        <RfSceneManager />
        <OrbitControls
          makeDefault
          enablePan
          enableZoom
          enableRotate
          mouseButtons={{
            LEFT: undefined,
            MIDDLE: 2,
            RIGHT: 1,
          }}
          minDistance={3}
          maxDistance={45}
        />
      </Canvas>
    </div>
  )
}
