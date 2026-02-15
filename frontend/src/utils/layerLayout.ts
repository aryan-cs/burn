import type { LayerConfig, LayerNode, LayerType } from '../store/graphStore'

export const LAYER_NODE_RADIUS = 0.15
export const LAYER_NODE_SPACING = 0.54
export const LAYER_PANEL_PADDING_X = 0.4
export const LAYER_PANEL_PADDING_Y = 0.4
export const LAYER_NODE_Z_OFFSET = 0.14

export function getLayerGridSize(config: LayerConfig, layerType?: LayerType) {
  const explicitRows = toPositiveInt(config.rows, 0)
  const explicitCols = toPositiveInt(config.cols, 0)
  if (explicitRows > 0 && explicitCols > 0) {
    return { rows: explicitRows, cols: explicitCols }
  }

  const inputGrid = inputShapeToGrid(config, layerType)
  if (inputGrid) {
    return inputGrid
  }

  const count = getPreferredNodeCount(config, layerType)
  if (count !== null) {
    return gridFromCount(count)
  }

  return { rows: 4, cols: 6 }
}

export function getLayerPanelSize(config: LayerConfig, layerType?: LayerType): [number, number] {
  const { rows, cols } = getLayerGridSize(config, layerType)
  const width = Math.max(
    1.4,
    (cols - 1) * LAYER_NODE_SPACING + LAYER_NODE_RADIUS * 2 + LAYER_PANEL_PADDING_X * 2
  )
  const height = Math.max(
    1.2,
    (rows - 1) * LAYER_NODE_SPACING + LAYER_NODE_RADIUS * 2 + LAYER_PANEL_PADDING_Y * 2
  )
  return [width, height]
}

export function buildLayerLocalNodePositions(
  rows: number,
  cols: number
): Array<[number, number, number]> {
  const positions: Array<[number, number, number]> = []
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const x = (col - (cols - 1) / 2) * LAYER_NODE_SPACING
      const y = ((rows - 1) / 2 - row) * LAYER_NODE_SPACING
      positions.push([x, y, LAYER_NODE_Z_OFFSET])
    }
  }
  return positions
}

export function buildLayerWorldNodePositions(node: LayerNode): Array<[number, number, number]> {
  const { rows, cols } = getLayerGridSize(node.config, node.type)
  const local = buildLayerLocalNodePositions(rows, cols)
  return local.map(([x, y, z]) => [
    node.position[0] + x,
    node.position[1] + y,
    node.position[2] + z,
  ])
}

function toPositiveInt(value: unknown, fallback: number): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback
  }
  return Math.floor(parsed)
}

function inputShapeToGrid(
  config: LayerConfig,
  layerType?: LayerType
): { rows: number; cols: number } | null {
  if (layerType !== 'Input') return null
  if (!Array.isArray(config.shape) || config.shape.length === 0) return null

  const dims = config.shape.map((entry) => toPositiveInt(entry, 0))
  if (dims.some((entry) => entry <= 0)) return null

  if (dims.length >= 3) {
    // Expected input shape for NN path: [channels, height, width]
    return { rows: dims[1], cols: dims[2] }
  }
  if (dims.length === 2) {
    return { rows: dims[0], cols: dims[1] }
  }
  return { rows: 1, cols: dims[0] }
}

function getPreferredNodeCount(config: LayerConfig, layerType?: LayerType): number | null {
  if (layerType === 'Output') {
    const classes = toPositiveInt(config.num_classes, 10)
    return classes > 0 ? classes : 10
  }

  const units = toPositiveInt(config.units, 0)
  if (units > 0) return units
  return null
}

function gridFromCount(count: number): { rows: number; cols: number } {
  if (count <= 1) {
    return { rows: 1, cols: 1 }
  }

  let bestRows = 1
  let bestCols = count
  let bestGap = Math.abs(bestCols - bestRows)

  for (let rows = 1; rows <= Math.floor(Math.sqrt(count)); rows += 1) {
    if (count % rows !== 0) continue
    const cols = count / rows
    const gap = Math.abs(cols - rows)
    if (gap < bestGap) {
      bestGap = gap
      bestRows = rows
      bestCols = cols
    }
  }

  return { rows: bestRows, cols: bestCols }
}
