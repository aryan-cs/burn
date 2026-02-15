import type { ReactNode } from 'react'
import { InfoTooltip } from '../InfoTooltip'

interface MetricTileProps {
  label: string
  value: string
  compact?: boolean
  tooltip?: string
}

export function MetricTile({ label, value, compact = false, tooltip }: MetricTileProps) {
  return (
    <div className={`metric-tile ${compact ? 'metric-tile-compact' : ''}`}>
      <p className="metric-tile-label">
        {label}
        {tooltip && <InfoTooltip title={label} text={tooltip} position="top" />}
      </p>
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
  minValue?: number
  maxValue: number
  primaryColor?: string
  secondaryColor?: string
  xAxisLabel?: string
  yAxisLabel?: string
  xTickStep?: number
  yTickStep?: number
  bottomRightOverlay?: ReactNode
}

export function MetricLineChart({
  primaryLabel,
  secondaryLabel,
  primaryValues,
  secondaryValues,
  minValue = 0,
  maxValue,
  primaryColor = '#ffb429',
  secondaryColor = '#ffd89c',
  xAxisLabel = 'Epoch',
  yAxisLabel = 'Value',
  xTickStep = 0.05,
  yTickStep = 0.05,
  bottomRightOverlay,
}: MetricLineChartProps) {
  const valuesCount = Math.max(primaryValues.length, secondaryValues.length)
  const showSecondary = secondaryLabel.trim().length > 0 && secondaryValues.length > 0
  if (valuesCount === 0) {
    return (
      <div className="metric-chart-empty">
        No metrics yet. Start training to populate this chart.
      </div>
    )
  }

  const width = 140
  const height = 140
  const padding = { top: 4, right: 8, bottom: 24, left: 24 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  const xDenominator = Math.max(valuesCount - 1, 1)
  const yMin = Number.isFinite(minValue) ? minValue : 0
  const yMax = Math.max(maxValue, yMin + 0.0001)
  const yRange = Math.max(yMax - yMin, 0.0001)
  const xTicks = buildNormalizedTicks(xTickStep)
  const yTicks = buildNormalizedTicks(yTickStep)

  const primaryPoints = buildChartPoints(
    primaryValues,
    xDenominator,
    yMin,
    yRange,
    padding.left,
    padding.top,
    plotWidth,
    plotHeight
  )
  const secondaryPoints = buildChartPoints(
    secondaryValues,
    xDenominator,
    yMin,
    yRange,
    padding.left,
    padding.top,
    plotWidth,
    plotHeight
  )

  return (
    <div className="metric-chart">
      <div className="metric-chart-legend">
        <span className="metric-chart-legend-item">
          <span className="metric-chart-line" style={{ background: primaryColor }} />
          {primaryLabel}
        </span>
        {showSecondary ? (
          <span className="metric-chart-legend-item">
            <span className="metric-chart-line" style={{ background: secondaryColor }} />
            {secondaryLabel}
          </span>
        ) : null}
      </div>
      <div className="metric-chart-canvas">
        <svg viewBox={`0 0 ${width} ${height}`} className="metric-chart-svg">
          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={padding.top + plotHeight}
            stroke="var(--graph-grid-major)"
            strokeWidth={1}
          />
          <line
            x1={padding.left}
            y1={padding.top + plotHeight}
            x2={padding.left + plotWidth}
            y2={padding.top + plotHeight}
            stroke="var(--graph-grid-major)"
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
                stroke="var(--graph-grid-minor)"
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
                stroke="var(--graph-grid-minor)"
                strokeWidth={1}
              />
            )
          })}
          {showSecondary && secondaryValues.length > 1 ? (
          <polyline
            points={secondaryPoints}
            fill="none"
            stroke={secondaryColor}
            strokeWidth={2}
          />
        ) : null}
        {primaryValues.length > 1 ? (
          <polyline
            points={primaryPoints}
            fill="none"
            stroke={primaryColor}
            strokeWidth={2}
          />
        ) : null}
        <text
          x={padding.left + plotWidth / 2}
          y={height - 3}
          textAnchor="middle"
          fill="rgba(255,255,255,0.7)"
          fontSize="10"
        >
          {xAxisLabel}
        </text>
        <text
          x={13}
          y={padding.top + plotHeight / 2}
          textAnchor="middle"
          fill="rgba(255,255,255,0.7)"
          fontSize="9"
          transform={`rotate(-90 13 ${padding.top + plotHeight / 2})`}
        >
          {yAxisLabel}
        </text>
        </svg>
        {bottomRightOverlay ? (
          <div className="metric-chart-overlay">
            {bottomRightOverlay}
          </div>
        ) : null}
      </div>
    </div>
  )
}

function buildChartPoints(
  values: number[],
  xDenominator: number,
  yMin: number,
  yRange: number,
  originX: number,
  originY: number,
  width: number,
  height: number
): string {
  return values
    .map((value, index) => {
      const x = originX + (index / xDenominator) * width
      const normalized = Math.max(0, Math.min((value - yMin) / yRange, 1))
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
