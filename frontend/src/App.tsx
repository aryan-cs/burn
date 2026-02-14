import { useEffect } from 'react'
import { Viewport } from './canvas/Viewport'
import { Palette } from './ui/Palette'
import { PropertiesPanel } from './ui/PropertiesPanel'
import { Toolbar } from './ui/Toolbar'
import { MetricsChart } from './ui/MetricsChart'
import { BackendWorkbench } from './ui/BackendWorkbench'
import { useConnectionDraw } from './hooks/useConnectionDraw'
import { useWebSocket } from './hooks/useWebSocket'
import { useGraphStore, type GraphJSON } from './store/graphStore'
import { useTrainingStore } from './store/trainingStore'

const MNIST_PRESET_GRAPH: GraphJSON = {
  nodes: [
    { id: 'node_1', type: 'Input', config: { shape: [1, 28, 28] } },
    { id: 'node_2', type: 'Flatten', config: {} },
    { id: 'node_3', type: 'Dense', config: { units: 128, activation: 'relu' } },
    { id: 'node_4', type: 'Output', config: { num_classes: 10, activation: 'softmax' } },
  ],
  edges: [
    { id: 'edge_1', source: 'node_1', target: 'node_2' },
    { id: 'edge_2', source: 'node_2', target: 'node_3' },
    { id: 'edge_3', source: 'node_3', target: 'node_4' },
  ],
}

function App() {
  // Keyboard shortcuts (Escape, Delete)
  useConnectionDraw()
  // WebSocket for training updates
  useWebSocket()

  useEffect(() => {
    const graphState = useGraphStore.getState()
    if (Object.keys(graphState.nodes).length === 0 && Object.keys(graphState.edges).length === 0) {
      graphState.fromJSON(MNIST_PRESET_GRAPH)
      const trainingState = useTrainingStore.getState()
      trainingState.setConfig({
        dataset: 'mnist',
        epochs: 20,
        batchSize: 64,
        optimizer: 'adam',
        learningRate: 0.001,
        loss: 'cross_entropy',
      })
    }
  }, [])

  return (
    <div className="relative w-full h-screen overflow-hidden bg-[#0a0a0f]">
      {/* 3D Viewport (full screen) */}
      <Viewport />

      {/* 2D Overlay panels */}
      <Palette />
      <BackendWorkbench />
      <PropertiesPanel />
      <Toolbar />
      <MetricsChart />
    </div>
  )
}

export default App
