import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { useGraphStore } from '../../store/graphStore'

export function CameraRig() {
  const isDraggingNode = useGraphStore((s) => s.draggingNodeId !== null)
  const isHighlightSelecting = useGraphStore((s) => s.highlightSelectionActive)
  const isConnecting = useGraphStore((s) => s.connectionSource !== null)

  return (
    <OrbitControls
      makeDefault
      enabled={!isConnecting && !isDraggingNode && !isHighlightSelecting}
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
      mouseButtons={{
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN,
      }}
      minDistance={3}
      maxDistance={50}
    />
  )
}
