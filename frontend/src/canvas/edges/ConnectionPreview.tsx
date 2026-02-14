import { useMemo } from 'react'

interface ConnectionPreviewProps {
  start: [number, number, number]
  end: [number, number, number]
}

export function ConnectionPreview({ start, end }: ConnectionPreviewProps) {
  const positions = useMemo(
    () =>
      new Float32Array([
        start[0],
        start[1],
        start[2],
        end[0],
        end[1],
        end[2],
      ]),
    [start, end]
  )

  return (
    <line>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color="#ffffff" transparent opacity={0.9} />
    </line>
  )
}
