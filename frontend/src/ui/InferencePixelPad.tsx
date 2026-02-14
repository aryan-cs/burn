import {
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type PointerEvent,
  type SetStateAction,
} from 'react'

const DEFAULT_GRID_ROWS = 28
const DEFAULT_GRID_COLS = 28
const MAX_CANVAS_SIZE = 280

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
  return [grid.map((row) => row.map((value) => Number(value.toFixed(4))))]
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
