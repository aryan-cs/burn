import {
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type PointerEvent,
  type SetStateAction,
} from 'react'

const GRID_SIZE = 28
const CANVAS_SIZE = 280
const CELL_SIZE = CANVAS_SIZE / GRID_SIZE
const ACTIVE_THRESHOLD = 0.05
const MNIST_TARGET_MAX_DIM = 20

type DrawMode = 'draw' | 'erase'

export function createEmptyInferenceGrid(): number[][] {
  return Array.from({ length: GRID_SIZE }, () => Array.from({ length: GRID_SIZE }, () => 0))
}

export function inferenceGridToPayload(grid: number[][]): number[][][] {
  const normalized = normalizeGridShape(grid)
  const preprocessed = preprocessForMnist(normalized)
  return [preprocessed.map((row) => row.map((value) => Number(value.toFixed(4))))]
}

export function countActivePixels(grid: number[][]): number {
  let total = 0
  for (const row of grid) {
    for (const value of row) {
      if (value > 0.01) total += 1
    }
  }
  return total
}

function normalizeGridShape(grid: number[][]): number[][] {
  return Array.from({ length: GRID_SIZE }, (_, row) =>
    Array.from({ length: GRID_SIZE }, (_, col) => {
      const value = Number(grid[row]?.[col] ?? 0)
      if (!Number.isFinite(value)) return 0
      if (value <= 0) return 0
      if (value >= 1) return 1
      return value
    })
  )
}

function preprocessForMnist(grid: number[][]): number[][] {
  const bbox = findActiveBoundingBox(grid)
  if (!bbox) {
    return grid
  }

  const cropHeight = bbox.maxRow - bbox.minRow + 1
  const cropWidth = bbox.maxCol - bbox.minCol + 1
  const scale = MNIST_TARGET_MAX_DIM / Math.max(cropHeight, cropWidth)
  const targetHeight = Math.max(1, Math.round(cropHeight * scale))
  const targetWidth = Math.max(1, Math.round(cropWidth * scale))

  const cropped = cropGrid(grid, bbox.minRow, bbox.maxRow, bbox.minCol, bbox.maxCol)
  const resized = resizeGridBilinear(cropped, targetHeight, targetWidth)
  const centered = placeAtCenter(resized)
  return centerByMass(centered)
}

function findActiveBoundingBox(
  grid: number[][]
): { minRow: number; maxRow: number; minCol: number; maxCol: number } | null {
  let minRow = GRID_SIZE
  let minCol = GRID_SIZE
  let maxRow = -1
  let maxCol = -1

  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      if ((grid[row]?.[col] ?? 0) <= ACTIVE_THRESHOLD) continue
      minRow = Math.min(minRow, row)
      minCol = Math.min(minCol, col)
      maxRow = Math.max(maxRow, row)
      maxCol = Math.max(maxCol, col)
    }
  }

  if (maxRow < 0 || maxCol < 0) return null
  return { minRow, maxRow, minCol, maxCol }
}

function cropGrid(
  grid: number[][],
  minRow: number,
  maxRow: number,
  minCol: number,
  maxCol: number
): number[][] {
  const out: number[][] = []
  for (let row = minRow; row <= maxRow; row += 1) {
    const nextRow: number[] = []
    for (let col = minCol; col <= maxCol; col += 1) {
      nextRow.push(grid[row]?.[col] ?? 0)
    }
    out.push(nextRow)
  }
  return out
}

function resizeGridBilinear(source: number[][], targetHeight: number, targetWidth: number): number[][] {
  const sourceHeight = source.length
  const sourceWidth = source[0]?.length ?? 0
  if (sourceHeight === 0 || sourceWidth === 0) {
    return Array.from({ length: targetHeight }, () =>
      Array.from({ length: targetWidth }, () => 0)
    )
  }

  const out: number[][] = Array.from({ length: targetHeight }, () =>
    Array.from({ length: targetWidth }, () => 0)
  )

  for (let row = 0; row < targetHeight; row += 1) {
    const srcY = ((row + 0.5) * sourceHeight) / targetHeight - 0.5
    const y0 = clampInt(Math.floor(srcY), 0, sourceHeight - 1)
    const y1 = clampInt(y0 + 1, 0, sourceHeight - 1)
    const wy = srcY - y0

    for (let col = 0; col < targetWidth; col += 1) {
      const srcX = ((col + 0.5) * sourceWidth) / targetWidth - 0.5
      const x0 = clampInt(Math.floor(srcX), 0, sourceWidth - 1)
      const x1 = clampInt(x0 + 1, 0, sourceWidth - 1)
      const wx = srcX - x0

      const v00 = source[y0]?.[x0] ?? 0
      const v01 = source[y0]?.[x1] ?? 0
      const v10 = source[y1]?.[x0] ?? 0
      const v11 = source[y1]?.[x1] ?? 0

      const top = v00 * (1 - wx) + v01 * wx
      const bottom = v10 * (1 - wx) + v11 * wx
      out[row][col] = top * (1 - wy) + bottom * wy
    }
  }

  return out
}

function placeAtCenter(source: number[][]): number[][] {
  const sourceHeight = source.length
  const sourceWidth = source[0]?.length ?? 0
  const out = createEmptyInferenceGrid()
  const top = Math.floor((GRID_SIZE - sourceHeight) / 2)
  const left = Math.floor((GRID_SIZE - sourceWidth) / 2)

  for (let row = 0; row < sourceHeight; row += 1) {
    for (let col = 0; col < sourceWidth; col += 1) {
      const rr = top + row
      const cc = left + col
      if (rr < 0 || rr >= GRID_SIZE || cc < 0 || cc >= GRID_SIZE) continue
      out[rr][cc] = source[row][col]
    }
  }
  return out
}

function centerByMass(grid: number[][]): number[][] {
  let mass = 0
  let rowMoment = 0
  let colMoment = 0

  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const value = grid[row]?.[col] ?? 0
      if (value <= 0) continue
      mass += value
      rowMoment += row * value
      colMoment += col * value
    }
  }

  if (mass <= 1e-6) return grid

  const centerRow = rowMoment / mass
  const centerCol = colMoment / mass
  const targetCenter = (GRID_SIZE - 1) / 2
  const shiftRow = Math.round(targetCenter - centerRow)
  const shiftCol = Math.round(targetCenter - centerCol)

  if (shiftRow === 0 && shiftCol === 0) return grid

  const out = createEmptyInferenceGrid()
  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const rr = row + shiftRow
      const cc = col + shiftCol
      if (rr < 0 || rr >= GRID_SIZE || cc < 0 || cc >= GRID_SIZE) continue
      out[rr][cc] = grid[row]?.[col] ?? 0
    }
  }
  return out
}

function clampInt(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

interface InferencePixelPadProps {
  grid: number[][]
  setGrid: Dispatch<SetStateAction<number[][]>>
  disabled?: boolean
}

export function InferencePixelPad({
  grid,
  setGrid,
  disabled = false,
}: InferencePixelPadProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const pointerDownRef = useRef(false)
  const [mode, setMode] = useState<DrawMode>('draw')
  const [brushSize, setBrushSize] = useState(1)

  const drawAt = (row: number, col: number): void => {
    const radius = Math.max(0, brushSize - 1)
    const nextValue = mode === 'draw' ? 1 : 0

    setGrid((prev) => {
      let changed = false
      const next = prev.map((prevRow) => prevRow.slice())

      for (let dr = -radius; dr <= radius; dr += 1) {
        for (let dc = -radius; dc <= radius; dc += 1) {
          const rr = row + dr
          const cc = col + dc
          if (rr < 0 || rr >= GRID_SIZE || cc < 0 || cc >= GRID_SIZE) continue

          if (radius > 0) {
            const distance = Math.sqrt(dr * dr + dc * dc)
            if (distance > radius + 0.1) continue
          }

          if (next[rr][cc] !== nextValue) {
            next[rr][cc] = nextValue
            changed = true
          }
        }
      }

      return changed ? next : prev
    })
  }

  const eventToCell = (event: PointerEvent<HTMLCanvasElement>): { row: number; col: number } => {
    const rect = event.currentTarget.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    const col = Math.min(GRID_SIZE - 1, Math.max(0, Math.floor((x / rect.width) * GRID_SIZE)))
    const row = Math.min(GRID_SIZE - 1, Math.max(0, Math.floor((y / rect.height) * GRID_SIZE)))

    return { row, col }
  }

  const paintFromEvent = (event: PointerEvent<HTMLCanvasElement>): void => {
    if (disabled) return
    const { row, col } = eventToCell(event)
    drawAt(row, col)
  }

  const handlePointerDown = (event: PointerEvent<HTMLCanvasElement>): void => {
    if (disabled || event.button !== 0) return
    pointerDownRef.current = true
    event.currentTarget.setPointerCapture(event.pointerId)
    paintFromEvent(event)
  }

  const handlePointerMove = (event: PointerEvent<HTMLCanvasElement>): void => {
    if (!pointerDownRef.current || disabled) return
    paintFromEvent(event)
  }

  const handlePointerUp = (event: PointerEvent<HTMLCanvasElement>): void => {
    pointerDownRef.current = false
    event.currentTarget.releasePointerCapture(event.pointerId)
  }

  const clearGrid = (): void => {
    if (disabled) return
    setGrid(createEmptyInferenceGrid())
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    canvas.width = CANVAS_SIZE
    canvas.height = CANVAS_SIZE

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.imageSmoothingEnabled = false
    ctx.fillStyle = '#03060b'
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)

    for (let row = 0; row < GRID_SIZE; row += 1) {
      for (let col = 0; col < GRID_SIZE; col += 1) {
        const value = grid[row]?.[col] ?? 0
        const shade = Math.round(value * 255)
        ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`
        ctx.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
      }
    }

    ctx.strokeStyle = 'rgba(125, 155, 190, 0.18)'
    ctx.lineWidth = 0.7
    for (let i = 0; i <= GRID_SIZE; i += 1) {
      const offset = i * CELL_SIZE
      ctx.beginPath()
      ctx.moveTo(offset + 0.5, 0)
      ctx.lineTo(offset + 0.5, CANVAS_SIZE)
      ctx.stroke()

      ctx.beginPath()
      ctx.moveTo(0, offset + 0.5)
      ctx.lineTo(CANVAS_SIZE, offset + 0.5)
      ctx.stroke()
    }
  }, [grid])

  return (
    <div className="inference-pad">
      <div className="inference-pad-header">
        <span>Inference Sketch (28x28)</span>
        <button
          type="button"
          onClick={clearGrid}
          disabled={disabled}
          className="inference-clear-button"
        >
          Clear
        </button>
      </div>

      <canvas
        ref={canvasRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={() => {
          pointerDownRef.current = false
        }}
        onContextMenu={(event) => event.preventDefault()}
        className="inference-canvas"
      />

      <div className="inference-controls">
        <div className="inference-mode-group">
          <button
            type="button"
            disabled={disabled}
            onClick={() => setMode('draw')}
            className={`inference-mode-button ${
              mode === 'draw' ? 'inference-mode-button-draw' : 'inference-mode-button-idle'
            }`}
          >
            Draw
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={() => setMode('erase')}
            className={`inference-mode-button ${
              mode === 'erase' ? 'inference-mode-button-erase' : 'inference-mode-button-idle'
            }`}
          >
            Erase
          </button>
        </div>

        <label className="inference-brush-control">
          <span className="inference-brush-label">Brush</span>
          <input
            type="range"
            min={1}
            max={4}
            step={1}
            disabled={disabled}
            value={brushSize}
            onChange={(event) => setBrushSize(Number(event.target.value))}
            className="inference-brush-slider"
          />
          <span>{brushSize}</span>
        </label>
      </div>
    </div>
  )
}
