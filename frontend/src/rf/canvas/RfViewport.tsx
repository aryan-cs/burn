import { OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useRFGraphStore } from '../store/rfGraphStore'
import { RfSceneManager } from './RfSceneManager'

interface RfViewportProps {
  lowDetailMode: boolean
}

export function RfViewport({ lowDetailMode }: RfViewportProps) {
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
        <color attach="background" args={['#000000']} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 12, 8]} intensity={0.9} />
        <pointLight position={[-10, -8, -5]} intensity={0.35} color="#ffb429" />
        <gridHelper args={lowDetailMode ? [24, 24, '#3b2d14', '#1f1f1f'] : [36, 36, '#3b2d14', '#1f1f1f']} />
        <RfSceneManager lowDetailMode={lowDetailMode} />
        <OrbitControls
          makeDefault
          enablePan
          enableZoom
          enableRotate
          mouseButtons={{
            LEFT: 0,
            MIDDLE: 1,
            RIGHT: 2,
          }}
          minDistance={3}
          maxDistance={110}
        />
      </Canvas>
    </div>
  )
}
