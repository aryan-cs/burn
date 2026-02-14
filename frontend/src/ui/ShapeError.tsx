interface ShapeErrorProps {
  nodeId: string
  message: string
  expected?: string
  got?: string
}

export function ShapeError({ nodeId, message, expected, got }: ShapeErrorProps) {
  return (
    <div className="absolute bg-red-900/90 border border-red-500 rounded-lg px-3 py-2 text-xs text-red-200 max-w-60 z-20 pointer-events-none">
      <div className="font-semibold mb-1">Shape Error — {nodeId}</div>
      <div>{message}</div>
      {expected && got && (
        <div className="mt-1 text-red-300/70">
          Expected: {expected} | Got: {got}
        </div>
      )}
    </div>
  )
}
