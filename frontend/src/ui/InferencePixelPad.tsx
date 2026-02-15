import {
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type PointerEvent,
  type SetStateAction,
} from 'react'

const MNIST_GRID_SIZE = 28
const DEFAULT_GRID_ROWS = 28
const DEFAULT_GRID_COLS = 28
const MAX_CANVAS_SIZE = 280
const ACTIVE_THRESHOLD = 0.05
const MNIST_TARGET_MAX_DIM = 20

type DrawMode = 'draw' | 'erase'

export function createEmptyInferenceGrid(
  rows = DEFAULT_GRID_ROWS,
  cols = DEFAULT_GRID_COLS
): number[][] {
  const safeRows = Math.max(1, Math.floor(rows))
  const safeCols = Math.max(1, Math.floor(cols))
  return Array.from({ length: safeRows }, () => Array.from({ length: safeCols }, () => 0))
}

export function inferenceGridToPayload(grid: number[][]): number[][][] {
  const { rows, cols } = getGridDimensions(grid)
  const normalized = normalizeGridShape(grid, rows, cols)
  const preprocessed = maybePreprocessForMnist(normalized, rows, cols)
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

function normalizeGridShape(grid: number[][], rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, (_, row) =>
    Array.from({ length: cols }, (_, col) => {
      const value = Number(grid[row]?.[col] ?? 0)
      if (!Number.isFinite(value)) return 0
      if (value <= 0) return 0
      if (value >= 1) return 1
      return value
    })
  )
}

function maybePreprocessForMnist(
  grid: number[][],
  rows: number,
  cols: number
): number[][] {
  if (rows !== MNIST_GRID_SIZE || cols !== MNIST_GRID_SIZE) {
    return grid
  }
  return preprocessForMnist(grid)
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
  let minRow = MNIST_GRID_SIZE
  let minCol = MNIST_GRID_SIZE
  let maxRow = -1
  let maxCol = -1

  for (let row = 0; row < MNIST_GRID_SIZE; row += 1) {
    for (let col = 0; col < MNIST_GRID_SIZE; col += 1) {
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
  const out = createEmptyInferenceGrid(MNIST_GRID_SIZE, MNIST_GRID_SIZE)
  const top = Math.floor((MNIST_GRID_SIZE - sourceHeight) / 2)
  const left = Math.floor((MNIST_GRID_SIZE - sourceWidth) / 2)

  for (let row = 0; row < sourceHeight; row += 1) {
    for (let col = 0; col < sourceWidth; col += 1) {
      const rr = top + row
      const cc = left + col
      if (rr < 0 || rr >= MNIST_GRID_SIZE || cc < 0 || cc >= MNIST_GRID_SIZE) continue
      out[rr][cc] = source[row][col]
    }
  }
  return out
}

function centerByMass(grid: number[][]): number[][] {
  let mass = 0
  let rowMoment = 0
  let colMoment = 0

  for (let row = 0; row < MNIST_GRID_SIZE; row += 1) {
    for (let col = 0; col < MNIST_GRID_SIZE; col += 1) {
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
  const targetCenter = (MNIST_GRID_SIZE - 1) / 2
  const shiftRow = Math.round(targetCenter - centerRow)
  const shiftCol = Math.round(targetCenter - centerCol)

  if (shiftRow === 0 && shiftCol === 0) return grid

  const out = createEmptyInferenceGrid(MNIST_GRID_SIZE, MNIST_GRID_SIZE)
  for (let row = 0; row < MNIST_GRID_SIZE; row += 1) {
    for (let col = 0; col < MNIST_GRID_SIZE; col += 1) {
      const rr = row + shiftRow
      const cc = col + shiftCol
      if (rr < 0 || rr >= MNIST_GRID_SIZE || cc < 0 || cc >= MNIST_GRID_SIZE) continue
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
  const { rows: gridRows, cols: gridCols } = getGridDimensions(grid)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const pointerDownRef = useRef(false)
  const [mode, setMode] = useState<DrawMode>('draw')
  const [brushSize, setBrushSize] = useState(2)

  const drawAt = (row: number, col: number): void => {
    const radius = Math.max(0, brushSize - 1)
    const nextValue = mode === 'draw' ? 1 : 0

    setGrid((prev) => {
      const { rows: prevRows, cols: prevCols } = getGridDimensions(prev)
      let changed = false
      const next = prev.map((prevRow) => prevRow.slice())

      for (let dr = -radius; dr <= radius; dr += 1) {
        for (let dc = -radius; dc <= radius; dc += 1) {
          const rr = row + dr
          const cc = col + dc
          if (rr < 0 || rr >= prevRows || cc < 0 || cc >= prevCols) continue

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

    const col = Math.min(
      gridCols - 1,
      Math.max(0, Math.floor((x / rect.width) * gridCols))
    )
    const row = Math.min(
      gridRows - 1,
      Math.max(0, Math.floor((y / rect.height) * gridRows))
    )

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
    setGrid(createEmptyInferenceGrid(gridRows, gridCols))
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const normalizedExtent = Math.max(gridRows, gridCols)
    const cellSize = MAX_CANVAS_SIZE / normalizedExtent
    const canvasWidth = Math.max(1, Math.round(gridCols * cellSize))
    const canvasHeight = Math.max(1, Math.round(gridRows * cellSize))
    const cellWidth = canvasWidth / gridCols
    const cellHeight = canvasHeight / gridRows

    canvas.width = canvasWidth
    canvas.height = canvasHeight

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.imageSmoothingEnabled = false
    ctx.fillStyle = '#121212'
    ctx.fillRect(0, 0, canvasWidth, canvasHeight)

    for (let row = 0; row < gridRows; row += 1) {
      for (let col = 0; col < gridCols; col += 1) {
        const value = grid[row]?.[col] ?? 0
        const shade = Math.round(value * 255)
        ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`
        ctx.fillRect(col * cellWidth, row * cellHeight, cellWidth, cellHeight)
      }
    }

    ctx.strokeStyle = 'rgba(255, 200, 120, 0.18)'
    ctx.lineWidth = 0.7
    for (let i = 0; i <= gridCols; i += 1) {
      const offset = i * cellWidth
      ctx.beginPath()
      ctx.moveTo(offset + 0.5, 0)
      ctx.lineTo(offset + 0.5, canvasHeight)
      ctx.stroke()
    }

    for (let i = 0; i <= gridRows; i += 1) {
      const offset = i * cellHeight
      ctx.beginPath()
      ctx.moveTo(0, offset + 0.5)
      ctx.lineTo(canvasWidth, offset + 0.5)
      ctx.stroke()
    }
  }, [grid, gridRows, gridCols])

  return (
    <div className="inference-pad">
      <div className="inference-canvas-shell">
        <button
          type="button"
          onClick={clearGrid}
          disabled={disabled}
          className="inference-clear-button"
          aria-label="Clear sketch"
          title="Clear sketch"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            height="24px"
            viewBox="0 -960 960 960"
            width="24px"
            fill="#e3e3e3"
          >
            <path d="M267.33-120q-27.5 0-47.08-19.58-19.58-19.59-19.58-47.09V-740H160v-66.67h192V-840h256v33.33h192V-740h-40.67v553.33q0 27-19.83 46.84Q719.67-120 692.67-120H267.33Zm425.34-620H267.33v553.33h425.34V-740Zm-328 469.33h66.66v-386h-66.66v386Zm164 0h66.66v-386h-66.66v386ZM267.33-740v553.33V-740Z" />
          </svg>
        </button>
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
      </div>

      <div className="inference-controls">
        <div className="inference-mode-group">
          <button
            type="button"
            disabled={disabled}
            onClick={() => setMode('draw')}
            className={`inference-mode-button ${
              mode === 'draw' ? 'inference-mode-button-draw' : 'inference-mode-button-idle'
            }`}
            aria-label="Draw"
            title="Draw"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="24px"
              viewBox="0 -960 960 960"
              width="24px"
              fill="#e3e3e3"
            >
              <path d="M200-200h57l391-391-57-57-391 391v57Zm-80 80v-170l528-527q12-11 26.5-17t30.5-6q16 0 31 6t26 18l55 56q12 11 17.5 26t5.5 30q0 16-5.5 30.5T817-647L290-120H120Zm640-584-56-56 56 56Zm-141 85-28-29 57 57-29-28Z" />
            </svg>
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={() => setMode('erase')}
            className={`inference-mode-button ${
              mode === 'erase' ? 'inference-mode-button-erase' : 'inference-mode-button-idle'
            }`}
            aria-label="Erase"
            title="Erase"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="24px"
              viewBox="0 -960 960 960"
              width="24px"
              fill="#e3e3e3"
            >
              <path d="M690-240h190v80H610l80-80Zm-500 80-85-85q-23-23-23.5-57t22.5-58l440-456q23-24 56.5-24t56.5 23l199 199q23 23 23 57t-23 57L520-160H190Zm296-80 314-322-198-198-442 456 64 64h262Zm-6-240Z" />
            </svg>
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

function getGridDimensions(grid: number[][]): { rows: number; cols: number } {
  const rows = Math.max(1, grid.length)
  const cols = Math.max(1, grid.reduce((max, row) => Math.max(max, row.length), 0))
  return { rows, cols }
}
