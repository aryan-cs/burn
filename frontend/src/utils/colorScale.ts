/**
 * Maps a weight mean value to a color on a diverging blue ← 0 → red scale.
 */
export function weightToColor(mean: number): string {
  // Clamp to [-1, 1] for color mapping
  const clamped = Math.max(-1, Math.min(1, mean))

  if (clamped < 0) {
    // Negative: blue
    const t = -clamped
    const r = Math.round(74 * (1 - t))
    const g = Math.round(144 * (1 - t))
    const b = Math.round(217 + (255 - 217) * t)
    return `rgb(${r},${g},${b})`
  } else {
    // Positive: red
    const t = clamped
    const r = Math.round(217 + (255 - 217) * t)
    const g = Math.round(144 * (1 - t))
    const b = Math.round(74 * (1 - t))
    return `rgb(${r},${g},${b})`
  }
}

/**
 * Maps a weight magnitude to line thickness.
 */
export function weightToThickness(mean: number): number {
  const magnitude = Math.abs(mean)
  const minThick = 0.02
  const maxThick = 0.12
  return minThick + Math.min(magnitude, 1) * (maxThick - minThick)
}

/**
 * Returns a gradient speed for edge animation based on gradient magnitude.
 */
export function gradientToSpeed(gradMagnitude: number): number {
  return 0.2 + Math.min(gradMagnitude, 2) * 0.8
}
