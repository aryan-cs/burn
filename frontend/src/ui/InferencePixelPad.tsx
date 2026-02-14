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
    <div className="rounded-lg border border-cyan-300/15 bg-black/25 p-2">
      <div className="mb-2 flex items-center justify-between font-mono text-[11px] text-white/70">
        <span>Inference Sketch (28x28)</span>
        <button
          type="button"
          onClick={clearGrid}
          disabled={disabled}
          className="rounded border border-white/20 bg-white/5 px-2 py-0.5 text-[10px] text-white/80 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
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
        className="w-full rounded-md border border-white/15 bg-[#03060b] shadow-[inset_0_0_20px_rgba(0,0,0,0.35)] touch-none"
      />

      <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-[11px] text-white/75">
        <div className="flex items-center gap-1">
          <button
            type="button"
            disabled={disabled}
            onClick={() => setMode('draw')}
            className={`flex-1 rounded border px-2 py-1 transition ${
              mode === 'draw'
                ? 'border-emerald-300/40 bg-emerald-500/20 text-emerald-100'
                : 'border-white/15 bg-white/5 text-white/80 hover:bg-white/10'
            }`}
          >
            Draw
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={() => setMode('erase')}
            className={`flex-1 rounded border px-2 py-1 transition ${
              mode === 'erase'
                ? 'border-amber-300/40 bg-amber-500/20 text-amber-100'
                : 'border-white/15 bg-white/5 text-white/80 hover:bg-white/10'
            }`}
          >
            Erase
          </button>
        </div>

        <label className="flex items-center gap-2 rounded border border-white/15 bg-white/5 px-2 py-1">
          <span className="text-white/55">Brush</span>
          <input
            type="range"
            min={1}
            max={4}
            step={1}
            disabled={disabled}
            value={brushSize}
            onChange={(event) => setBrushSize(Number(event.target.value))}
            className="w-full"
          />
          <span>{brushSize}</span>
        </label>
      </div>
    </div>
  )
}
