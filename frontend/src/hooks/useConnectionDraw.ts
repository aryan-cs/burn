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
  const highlightedNodeIds = useGraphStore((s) => s.highlightedNodeIds)
  const removeNode = useGraphStore((s) => s.removeNode)
  const removeNodes = useGraphStore((s) => s.removeNodes)
  const removeEdge = useGraphStore((s) => s.removeEdge)
  const undo = useGraphStore((s) => s.undo)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isEditableTarget =
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLSelectElement ||
        e.target instanceof HTMLTextAreaElement ||
        (e.target instanceof HTMLElement && e.target.isContentEditable)

      const isUndoShortcut =
        (e.ctrlKey || e.metaKey) &&
        !e.shiftKey &&
        e.key.toLowerCase() === 'z'

      if (isUndoShortcut) {
        if (isEditableTarget) return
        e.preventDefault()
        undo()
        return
      }

      if (e.key === 'Escape') {
        setConnectionSource(null)
      }
      if (e.key === 'Backspace' || e.key === 'Delete') {
        // Don't delete if user is typing in an input
        if (isEditableTarget) {
          return
        }
        if (highlightedNodeIds.length > 0) {
          removeNodes(highlightedNodeIds)
        } else if (selectedEdgeId) {
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
    highlightedNodeIds,
    selectedNodeId,
    selectedEdgeId,
    removeNode,
    removeNodes,
    removeEdge,
    undo,
  ])
}
