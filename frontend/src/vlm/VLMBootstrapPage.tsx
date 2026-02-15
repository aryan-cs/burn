import { useMemo } from 'react'
import VLMPage, { type VLMTrainingConfig } from './VLMPage'

type VLMPreset = {
  training: Partial<VLMTrainingConfig>
}

const VLM_PRESETS: Record<string, VLMPreset> = {
  object_detection_demo: {
    training: {
      dataset: 'synthetic_boxes_tiny',
      modelId: 'hustvl/yolos-tiny',
      epochs: 1,
      batchSize: 1,
      stepsPerEpoch: 1,
      learningRate: 0.00001,
    },
  },
}

function resolveInitialConfig(search: string): Partial<VLMTrainingConfig> {
  const params = new URLSearchParams(search)
  const mode = (params.get('mode') ?? 'preset').toLowerCase()
  const template = (params.get('template') ?? 'object_detection_demo').toLowerCase()

  if (mode === 'scratch') {
    return {
      dataset: 'synthetic_boxes_tiny',
      modelId: 'hustvl/yolos-tiny',
      epochs: 1,
      batchSize: 1,
      stepsPerEpoch: 1,
      learningRate: 0.00001,
    }
  }

  const preset = VLM_PRESETS[template] ?? VLM_PRESETS.object_detection_demo
  return preset.training
}

export default function VLMBootstrapPage() {
  const initialConfig = useMemo(
    () => resolveInitialConfig(window.location.search),
    []
  )

  return (
    <VLMPage initialConfig={initialConfig} />
  )
}
