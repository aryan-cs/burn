import { RfViewport } from '../canvas/RfViewport'
import { useRFGraphStore } from '../store/rfGraphStore'
import type { RFNodeType } from '../types'
import { RfNodeEditor } from './RfNodeEditor'

function nextNodePosition(index: number): [number, number, number] {
  const spacing = 2.8
  return [index * spacing - 4.2, 0, 0]
}

export function RfGraphView() {
  const nodesMap = useRFGraphStore((state) => state.nodes)
  const edgesMap = useRFGraphStore((state) => state.edges)
  const selectedNodeId = useRFGraphStore((state) => state.selectedNodeId)
  const connectionSource = useRFGraphStore((state) => state.connectionSource)
  const addNode = useRFGraphStore((state) => state.addNode)
  const removeNode = useRFGraphStore((state) => state.removeNode)
  const setNodeConfig = useRFGraphStore((state) => state.setNodeConfig)
  const autoConnectByX = useRFGraphStore((state) => state.autoConnectByX)
  const resetPreset = useRFGraphStore((state) => state.resetPreset)
  const clearGraph = useRFGraphStore((state) => state.clearGraph)
  const setConnectionSource = useRFGraphStore((state) => state.setConnectionSource)

  const nodes = Object.values(nodesMap)
  const edges = Object.values(edgesMap)
  const selectedNode = selectedNodeId ? nodesMap[selectedNodeId] : null

  const addNodeOfType = (type: RFNodeType) => {
    addNode(type, nextNodePosition(nodes.length))
  }

  return (
    <section className="rf-builder-surface">
      <RfViewport />

      <div className="rf-builder-top">
        <div className="rf-builder-title">3D RF Node Builder</div>
        <div className="rf-builder-subtitle">
          Drag, connect, and configure nodes directly in the scene.
        </div>
      </div>

      <div className="rf-builder-controls">
        <div className="rf-builder-toolbar">
          <button className="rf-btn" onClick={() => addNodeOfType('RFInput')}>
            + RFInput
          </button>
          <button className="rf-btn" onClick={() => addNodeOfType('RFFlatten')}>
            + RFFlatten
          </button>
          <button className="rf-btn" onClick={() => addNodeOfType('RandomForestClassifier')}>
            + RandomForestClassifier
          </button>
          <button className="rf-btn" onClick={() => addNodeOfType('RFOutput')}>
            + RFOutput
          </button>
          <button className="rf-btn" onClick={autoConnectByX}>
            Auto Connect by X
          </button>
          <button className="rf-btn" onClick={resetPreset}>
            Preset
          </button>
          <button className="rf-btn rf-btn-red" onClick={clearGraph}>
            Clear
          </button>
        </div>

        <div className="rf-builder-meta">
          <div>
            <div className="rf-meta-label">Nodes</div>
            <div className="rf-meta-value">{nodes.length}</div>
          </div>
          <div>
            <div className="rf-meta-label">Edges</div>
            <div className="rf-meta-value">{edges.length}</div>
          </div>
          <div>
            <div className="rf-meta-label">Connecting</div>
            <div className="rf-meta-value">{connectionSource ?? 'none'}</div>
          </div>
          <div>
            <div className="rf-meta-label">Selected</div>
            <div className="rf-meta-value">{selectedNode?.id ?? 'none'}</div>
          </div>
        </div>
      </div>

      {selectedNode ? (
        <article className="rf-node-card rf-selected-node-card">
          <div className="rf-node-header">
            <span className="rf-node-type">{selectedNode.type}</span>
            <span className="rf-node-id">{selectedNode.id}</span>
          </div>
          <RfNodeEditor node={selectedNode} onPatchConfig={setNodeConfig} />
          <div className="rf-button-row">
            <button className="rf-btn rf-btn-red" onClick={() => removeNode(selectedNode.id)}>
              Remove Node
            </button>
            <button className="rf-btn" onClick={() => setConnectionSource(null)}>
              Cancel Connection
            </button>
          </div>
        </article>
      ) : null}
    </section>
  )
}
