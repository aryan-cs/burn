import { useTrainingStore } from '../store/trainingStore'

/**
 * Minimal SVG-based metrics chart for loss and accuracy.
 * Renders inline without external chart libraries.
 */
export function MetricsChart() {
  const metrics = useTrainingStore((s) => s.metrics)
  const status = useTrainingStore((s) => s.status)

  if (metrics.length === 0 && status === 'idle') return null

  const width = 200
  const height = 50
  const padding = { top: 10, right: 10, bottom: 20, left: 35 }
  const plotW = width - padding.left - padding.right
  const plotH = height - padding.top - padding.bottom

  const maxLoss = Math.max(...metrics.map((m) => m.loss), 0.01)
  const maxEpoch = Math.max(metrics.length, 1)

  const lossPoints = metrics
    .map((m, i) => {
      const x = padding.left + (i / maxEpoch) * plotW
      const y = padding.top + (1 - m.loss / maxLoss) * plotH
      return `${x},${y}`
    })
    .join(' ')

  const accPoints = metrics
    .map((m, i) => {
      const x = padding.left + (i / maxEpoch) * plotW
      const y = padding.top + (1 - m.accuracy) * plotH
      return `${x},${y}`
    })
    .join(' ')

  return (
    <div className="absolute bottom-20 right-4 bg-[#121212]/90 backdrop-blur-md rounded-xl border border-white/10 p-3 z-10">
      <div className="flex items-center gap-4 mb-2">
        <span className="text-xs text-amber-300 flex items-center gap-1">
          <span className="w-3 h-0.5 bg-amber-300 inline-block" /> Loss
        </span>
        <span className="text-xs text-amber-500 flex items-center gap-1">
          <span className="w-3 h-0.5 bg-amber-500 inline-block" /> Accuracy
        </span>
      </div>
      <svg width={width} height={height}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <line
            key={t}
            x1={padding.left}
            y1={padding.top + t * plotH}
            x2={padding.left + plotW}
            y2={padding.top + t * plotH}
            stroke="rgba(255,255,255,0.05)"
          />
        ))}
        {/* Loss line */}
        {metrics.length > 1 && (
          <polyline
            points={lossPoints}
            fill="none"
            stroke="#ffb429"
            strokeWidth={2}
          />
        )}
        {/* Accuracy line */}
        {metrics.length > 1 && (
          <polyline
            points={accPoints}
            fill="none"
            stroke="#ffd27a"
            strokeWidth={2}
          />
        )}
        {/* X axis label */}
        <text
          x={padding.left + plotW / 2}
          y={height - 2}
          textAnchor="middle"
          fill="rgba(255,255,255,0.3)"
          fontSize={9}
        >
          Epoch
        </text>
      </svg>
    </div>
  )
}
