import { useEffect } from 'react'
import { useGraphStore } from '../store/graphStore'

/**
 * Handles keyboard shortcuts for the graph (e.g. Escape to cancel connection,
 * Backspace/Delete to remove selected node/edge).
 */
export function useConnectionDraw() {
  const connectionSource = useGraphStore((s) => s.connectionSource)
  const setConnectionSource = useGraphStore((s) => s.setConnectionSource)
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const selectedEdgeId = useGraphStore((s) => s.selectedEdgeId)
  const removeNode = useGraphStore((s) => s.removeNode)
  const removeEdge = useGraphStore((s) => s.removeEdge)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setConnectionSource(null)
      }
      if (e.key === 'Backspace' || e.key === 'Delete') {
        // Don't delete if user is typing in an input
        if (
          e.target instanceof HTMLInputElement ||
          e.target instanceof HTMLSelectElement ||
          e.target instanceof HTMLTextAreaElement
        ) {
          return
        }
        if (selectedEdgeId) {
          removeEdge(selectedEdgeId)
        } else if (selectedNodeId) {
          removeNode(selectedNodeId)
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [
    connectionSource,
    setConnectionSource,
    selectedNodeId,
    selectedEdgeId,
    removeNode,
    removeEdge,
  ])
}
