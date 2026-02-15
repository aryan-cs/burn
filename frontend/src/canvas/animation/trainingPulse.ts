export const STAGGER_STEP_MS = 120
export const FADE_IN_MS = 220
export const PATH_PULSE_PERIOD_MS = 900
export const PATH_PULSE_MIN = 0.72
export const PATH_PULSE_MAX = 1
export const EPOCH_WAVE_HOLD_MS = 300

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

export function computeStaggerProgress(
  elapsedMs: number,
  index: number,
  stepMs: number = STAGGER_STEP_MS,
  fadeMs: number = FADE_IN_MS
): number {
  const startAt = index * stepMs
  const normalized = (elapsedMs - startAt) / Math.max(1, fadeMs)
  return clamp(normalized, 0, 1)
}

export function computePulse(
  elapsedMs: number,
  periodMs: number = PATH_PULSE_PERIOD_MS,
  min: number = PATH_PULSE_MIN,
  max: number = PATH_PULSE_MAX
): number {
  const safePeriod = Math.max(1, periodMs)
  const phase = ((elapsedMs % safePeriod) / safePeriod) * Math.PI * 2
  const wave01 = (Math.sin(phase) + 1) / 2
  return min + (max - min) * wave01
}
