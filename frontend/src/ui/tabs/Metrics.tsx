interface MetricTileProps {
  label: string
  value: string
  compact?: boolean
}

export function MetricTile({ label, value, compact = false }: MetricTileProps) {
  return (
    <div className={`metric-tile ${compact ? 'metric-tile-compact' : ''}`}>
      <p className="metric-tile-label">{label}</p>
      <p className={`metric-tile-value ${compact ? 'metric-tile-value-compact' : ''}`}>
        {value}
      </p>
    </div>
  )
}

interface MetricLineChartProps {
  primaryLabel: string
  secondaryLabel: string
  primaryValues: number[]
  secondaryValues: number[]
  maxValue: number
}

export function MetricLineChart({
  primaryLabel,
  secondaryLabel,
  primaryValues,
  secondaryValues,
  maxValue,
}: MetricLineChartProps) {
  const valuesCount = Math.max(primaryValues.length, secondaryValues.length)
  if (valuesCount === 0) {
    return (
      <div className="metric-chart-empty">
        No metrics yet. Start training to populate this chart.
      </div>
    )
  }

  const width = 330
  const height = 140
  const padding = { top: 12, right: 10, bottom: 20, left: 26 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  const xDenominator = Math.max(valuesCount - 1, 1)
  const yMax = Math.max(maxValue, 0.0001)

  const primaryPoints = buildChartPoints(
    primaryValues,
    xDenominator,
    yMax,
    padding.left,
    padding.top,
    plotWidth,
    plotHeight
  )
  const secondaryPoints = buildChartPoints(
    secondaryValues,
    xDenominator,
    yMax,
    padding.left,
    padding.top,
    plotWidth,
    plotHeight
  )

  return (
    <div className="metric-chart">
      <div className="metric-chart-legend">
        <span className="metric-chart-legend-item">
          <span className="metric-chart-line metric-chart-line-primary" />
          {primaryLabel}
        </span>
        <span className="metric-chart-legend-item">
          <span className="metric-chart-line metric-chart-line-secondary" />
          {secondaryLabel}
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="metric-chart-svg">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const y = padding.top + tick * plotHeight
          return (
            <line
              key={tick}
              x1={padding.left}
              y1={y}
              x2={padding.left + plotWidth}
              y2={y}
              stroke="rgba(255,255,255,0.1)"
              strokeWidth={1}
            />
          )
        })}
        {secondaryValues.length > 1 ? (
          <polyline
            points={secondaryPoints}
            fill="none"
            stroke="#78f0b5"
            strokeWidth={2}
          />
        ) : null}
        {primaryValues.length > 1 ? (
          <polyline
            points={primaryPoints}
            fill="none"
            stroke="#8bd0ff"
            strokeWidth={2}
          />
        ) : null}
      </svg>
    </div>
  )
}

function buildChartPoints(
  values: number[],
  xDenominator: number,
  yMax: number,
  originX: number,
  originY: number,
  width: number,
  height: number
): string {
  return values
    .map((value, index) => {
      const x = originX + (index / xDenominator) * width
      const normalized = Math.max(0, Math.min(value / yMax, 1))
      const y = originY + (1 - normalized) * height
      return `${x},${y}`
    })
    .join(' ')
}
