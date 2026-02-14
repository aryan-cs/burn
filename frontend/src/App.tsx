import { Viewport } from './canvas/Viewport'
import { Palette } from './ui/Palette'
import { PropertiesPanel } from './ui/PropertiesPanel'
import { Toolbar } from './ui/Toolbar'
import { MetricsChart } from './ui/MetricsChart'
import { useConnectionDraw } from './hooks/useConnectionDraw'
import { useWebSocket } from './hooks/useWebSocket'

function App() {
  // Keyboard shortcuts (Escape, Delete)
  useConnectionDraw()
  // WebSocket for training updates
  useWebSocket()

  return (
    <div className="relative w-full h-screen overflow-hidden bg-[#0a0a0f]">
      {/* 3D Viewport (full screen) */}
      <Viewport />

      {/* 2D Overlay panels */}
      <Palette />
      <PropertiesPanel />
      <Toolbar />
      <MetricsChart />
    </div>
  )
}

export default App
