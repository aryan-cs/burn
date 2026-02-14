import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react'
import { Canvas } from '@react-three/fiber'
import { useGraphStore } from '../store/graphStore'
import { SceneManager } from './SceneManager'
import { CameraRig } from './controls/CameraRig'

export function Viewport() {
  const draggingNodeId = useGraphStore((s) => s.draggingNodeId)
  const highlightSelectionActive = useGraphStore((s) => s.highlightSelectionActive)
  const highlightSelectionStart = useGraphStore((s) => s.highlightSelectionStart)
  const highlightSelectionEnd = useGraphStore((s) => s.highlightSelectionEnd)
  const connectionSource = useGraphStore((s) => s.connectionSource)
  const selectNode = useGraphStore((s) => s.selectNode)
  const selectEdge = useGraphStore((s) => s.selectEdge)
  const setDraggingNodeId = useGraphStore((s) => s.setDraggingNodeId)
  const startHighlightSelection = useGraphStore((s) => s.startHighlightSelection)
  const updateHighlightSelection = useGraphStore((s) => s.updateHighlightSelection)
  const endHighlightSelection = useGraphStore((s) => s.endHighlightSelection)
  const clearHighlightedNodes = useGraphStore((s) => s.clearHighlightedNodes)
  const completeConnectionDrag = useGraphStore((s) => s.completeConnectionDrag)

  const handlePointerMissed = () => {
    if (draggingNodeId) {
      setDraggingNodeId(null)
      return
    }
    if (connectionSource) {
      completeConnectionDrag(null)
      return
    }
    if (!highlightSelectionActive) {
      selectNode(null)
      selectEdge(null)
      clearHighlightedNodes()
    }
  }

  const handleWrapperPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || !event.shiftKey) return
    event.preventDefault()
    event.stopPropagation()

    setDraggingNodeId(null)
    completeConnectionDrag(null)
    clearHighlightedNodes()
    startHighlightSelection(getLocalPoint(event))
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const handleWrapperPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!highlightSelectionActive) return
    updateHighlightSelection(getLocalPoint(event))
  }

  const handleWrapperPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!highlightSelectionActive) return
    updateHighlightSelection(getLocalPoint(event))
    endHighlightSelection()
    event.currentTarget.releasePointerCapture(event.pointerId)
  }

  const selectionRectStyle = getSelectionRectStyle(
    highlightSelectionStart,
    highlightSelectionEnd
  )

  return (
    <div
      className="relative h-full w-full"
      onPointerDown={handleWrapperPointerDown}
      onPointerMove={handleWrapperPointerMove}
      onPointerUp={handleWrapperPointerUp}
      onPointerCancel={handleWrapperPointerUp}
    >
      <Canvas
        camera={{ position: [0, 5, 12], fov: 50 }}
        style={{ width: '100%', height: '100%' }}
        onPointerMissed={handlePointerMissed}
      >
        <color attach="background" args={['#181818']} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        <pointLight position={[-10, -10, -5]} intensity={0.25} color="#9a9a9a" />
        <gridHelper args={[40, 80, '#303030', '#242424']} />
        <SceneManager />
        <CameraRig />
      </Canvas>

      {highlightSelectionActive && selectionRectStyle ? (
        <div
          className="pointer-events-none absolute border border-[#9ebeff] bg-[#9ebeff]/15"
          style={selectionRectStyle}
        />
      ) : null}
    </div>
  )
}

function getLocalPoint(event: ReactPointerEvent<HTMLDivElement>): [number, number] {
  const rect = event.currentTarget.getBoundingClientRect()
  return [event.clientX - rect.left, event.clientY - rect.top]
}

function getSelectionRectStyle(
  start: [number, number] | null,
  end: [number, number] | null
): CSSProperties | null {
  if (!start || !end) return null
  const left = Math.min(start[0], end[0])
  const top = Math.min(start[1], end[1])
  const width = Math.abs(end[0] - start[0])
  const height = Math.abs(end[1] - start[1])

  return { left, top, width, height }
}
