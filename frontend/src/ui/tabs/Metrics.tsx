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
  xAxisLabel?: string
  yAxisLabel?: string
  xTickStep?: number
  yTickStep?: number
}

export function MetricLineChart({
  primaryLabel,
  secondaryLabel,
  primaryValues,
  secondaryValues,
  maxValue,
  xAxisLabel = 'Epoch',
  yAxisLabel = 'Value',
  xTickStep = 0.05,
  yTickStep = 0.05,
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
  const height = 160
  const padding = { top: 12, right: 10, bottom: 32, left: 38 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  const xDenominator = Math.max(valuesCount - 1, 1)
  const yMax = Math.max(maxValue, 0.0001)
  const xTicks = buildNormalizedTicks(xTickStep)
  const yTicks = buildNormalizedTicks(yTickStep)

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
        <line
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={padding.top + plotHeight}
          stroke="rgba(255,255,255,0.26)"
          strokeWidth={1}
        />
        <line
          x1={padding.left}
          y1={padding.top + plotHeight}
          x2={padding.left + plotWidth}
          y2={padding.top + plotHeight}
          stroke="rgba(255,255,255,0.26)"
          strokeWidth={1}
        />
        {xTicks.map((tick) => {
          const x = padding.left + tick * plotWidth
          return (
            <line
              key={`x-${tick}`}
              x1={x}
              y1={padding.top}
              x2={x}
              y2={padding.top + plotHeight}
              stroke="rgba(255,255,255,0.08)"
              strokeWidth={1}
            />
          )
        })}
        {yTicks.map((tick) => {
          const y = padding.top + (1 - tick) * plotHeight
          return (
            <line
              key={`y-${tick}`}
              x1={padding.left}
              y1={y}
              x2={padding.left + plotWidth}
              y2={y}
              stroke="rgba(255,255,255,0.08)"
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
        <text
          x={padding.left + plotWidth / 2}
          y={height - 8}
          textAnchor="middle"
          fill="rgba(255,255,255,0.7)"
          fontSize="10"
        >
          {xAxisLabel}
        </text>
        <text
          x={14}
          y={padding.top + plotHeight / 2}
          textAnchor="middle"
          fill="rgba(255,255,255,0.7)"
          fontSize="10"
          transform={`rotate(-90 14 ${padding.top + plotHeight / 2})`}
        >
          {yAxisLabel}
        </text>
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

function buildNormalizedTicks(step: number): number[] {
  const normalizedStep = Math.min(Math.max(step, 0.01), 1)
  const ticks: number[] = []

  for (let tick = 0; tick < 1; tick += normalizedStep) {
    ticks.push(Number(tick.toFixed(4)))
  }
  ticks.push(1)

  return ticks
}
