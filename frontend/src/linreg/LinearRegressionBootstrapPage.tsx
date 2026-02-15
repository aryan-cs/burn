import { useMemo } from 'react'
import LinearRegressionPage, { type LinearRegressionInitialConfig } from './LinearRegressionPage'

type LinearRegressionPreset = LinearRegressionInitialConfig

const LINEAR_REGRESSION_PRESETS: Record<string, LinearRegressionPreset> = {
  study_hours_baseline: {
    datasetId: 'study_hours',
    includeNormalization: true,
    fitIntercept: true,
    l2Penalty: 0,
    epochs: 420,
    learningRate: 0.03,
    testSplit: 0.2,
    randomSeed: 42,
  },
  home_value_baseline: {
    datasetId: 'home_value_tiny',
    includeNormalization: true,
    fitIntercept: true,
    l2Penalty: 0,
    epochs: 480,
    learningRate: 0.025,
    testSplit: 0.25,
    randomSeed: 21,
  },
}

const SCRATCH_DEFAULT: LinearRegressionInitialConfig = {
  datasetId: 'study_hours',
  includeNormalization: false,
  fitIntercept: true,
  l2Penalty: 0,
  epochs: 320,
  learningRate: 0.02,
  testSplit: 0.2,
  randomSeed: 42,
}

function resolveInitialConfig(search: string): LinearRegressionInitialConfig {
  const params = new URLSearchParams(search)
  const mode = (params.get('mode') ?? 'preset').toLowerCase()
  const template = (params.get('template') ?? 'study_hours_baseline').toLowerCase()

  if (mode === 'scratch') {
    return SCRATCH_DEFAULT
  }

  return LINEAR_REGRESSION_PRESETS[template] ?? LINEAR_REGRESSION_PRESETS.study_hours_baseline
}

export default function LinearRegressionBootstrapPage() {
  const initialConfig = useMemo(
    () => resolveInitialConfig(window.location.search),
    []
  )

  return (
    <LinearRegressionPage initialConfig={initialConfig} />
  )
}
