import { useMemo } from 'react'
import LinearRegressionPage, { type LinearRegressionInitialConfig } from './LinearRegressionPage'

type LinearRegressionPreset = LinearRegressionInitialConfig

const LINEAR_REGRESSION_PRESETS: Record<string, LinearRegressionPreset> = {
  diabetes_bmi_baseline: {
    datasetId: 'diabetes_bmi',
    includeNormalization: true,
    fitIntercept: true,
    l2Penalty: 0.0005,
    optimizer: 'bgd',
    epochs: 700,
    learningRate: 0.045,
    testSplit: 0.2,
    randomSeed: 42,
  },
  study_hours_baseline: {
    datasetId: 'study_hours',
    includeNormalization: true,
    fitIntercept: true,
    l2Penalty: 0,
    optimizer: 'bgd',
    epochs: 420,
    learningRate: 0.03,
    testSplit: 0.2,
    randomSeed: 42,
  },
  home_value_baseline: {
    datasetId: 'diabetes_bmi',
    includeNormalization: true,
    fitIntercept: true,
    l2Penalty: 0,
    optimizer: 'bgd',
    epochs: 700,
    learningRate: 0.045,
    testSplit: 0.2,
    randomSeed: 42,
  },
}

const SCRATCH_DEFAULT: LinearRegressionInitialConfig = {
  datasetId: 'diabetes_bmi',
  includeNormalization: true,
  fitIntercept: true,
  l2Penalty: 0,
  optimizer: 'bgd',
  epochs: 600,
  learningRate: 0.04,
  testSplit: 0.2,
  randomSeed: 42,
}

function resolveInitialConfig(search: string): LinearRegressionInitialConfig {
  const params = new URLSearchParams(search)
  const mode = (params.get('mode') ?? 'preset').toLowerCase()
  const template = (params.get('template') ?? 'diabetes_bmi_baseline').toLowerCase()

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
