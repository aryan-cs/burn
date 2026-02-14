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

type DrawMode = 'draw' | 'erase'

export function createEmptyInferenceGrid(): number[][] {
  return Array.from({ length: GRID_SIZE }, () => Array.from({ length: GRID_SIZE }, () => 0))
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
