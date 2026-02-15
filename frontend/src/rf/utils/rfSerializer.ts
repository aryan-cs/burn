import { useRFGraphStore } from '../store/rfGraphStore'
import type { RFGraphPayload } from '../types'

export function serializeRFGraph(): RFGraphPayload {
  return useRFGraphStore.getState().toPayload()
}
