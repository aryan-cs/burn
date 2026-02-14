import type { LayerConfig, LayerNode } from '../store/graphStore'

export const LAYER_NODE_RADIUS = 0.15
export const LAYER_NODE_SPACING = 0.54
export const LAYER_PANEL_PADDING_X = 0.4
export const LAYER_PANEL_PADDING_Y = 0.4
export const LAYER_NODE_Z_OFFSET = 0.14

export function getLayerGridSize(config: LayerConfig) {
  return {
    rows: toPositiveInt(config.rows, 4),
    cols: toPositiveInt(config.cols, 6),
  }
}

export function getLayerPanelSize(config: LayerConfig): [number, number] {
  const { rows, cols } = getLayerGridSize(config)
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
  const { rows, cols } = getLayerGridSize(node.config)
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
