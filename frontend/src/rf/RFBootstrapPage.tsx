import { useEffect, useRef } from 'react'
import RandomForestPage from './RandomForestPage'
import { useRFGraphStore } from './store/rfGraphStore'
import { useRFRunStore } from './store/rfRunStore'

type RFPreset = {
  dataset: string
  training: {
    testSize: number
    randomState: number
    stratify: boolean
    logEveryTrees: number
  }
  forest: {
    n_estimators: number
    max_depth: number | null
    criterion: string
    max_features: string | null
    min_samples_split: number
    min_samples_leaf: number
    bootstrap: boolean
    random_state: number
  }
}

const RF_PRESETS: Record<string, RFPreset> = {
  iris_basic: {
    dataset: 'iris',
    training: {
      testSize: 0.2,
      randomState: 42,
      stratify: true,
      logEveryTrees: 5,
    },
    forest: {
      n_estimators: 120,
      max_depth: 8,
      criterion: 'gini',
      max_features: 'sqrt',
      min_samples_split: 2,
      min_samples_leaf: 1,
      bootstrap: true,
      random_state: 42,
    },
  },
  wine_quality: {
    dataset: 'wine',
    training: {
      testSize: 0.25,
      randomState: 7,
      stratify: true,
      logEveryTrees: 10,
    },
    forest: {
      n_estimators: 220,
      max_depth: 14,
      criterion: 'entropy',
      max_features: 'sqrt',
      min_samples_split: 3,
      min_samples_leaf: 1,
      bootstrap: true,
      random_state: 7,
    },
  },
  breast_cancer_fast: {
    dataset: 'breast_cancer',
    training: {
      testSize: 0.2,
      randomState: 21,
      stratify: true,
      logEveryTrees: 8,
    },
    forest: {
      n_estimators: 180,
      max_depth: 12,
      criterion: 'gini',
      max_features: 'sqrt',
      min_samples_split: 2,
      min_samples_leaf: 1,
      bootstrap: true,
      random_state: 21,
    },
  },
}

const RF_DEFAULT_TRAINING = {
  dataset: 'iris',
  testSize: 0.2,
  randomState: 42,
  stratify: true,
  logEveryTrees: 5,
} as const

export default function RFBootstrapPage() {
  const initialized = useRef(false)

  useEffect(() => {
    if (initialized.current) return
    initialized.current = true

    const params = new URLSearchParams(window.location.search)
    const mode = (params.get('mode') ?? 'preset').toLowerCase()
    const template = (params.get('template') ?? 'iris_basic').toLowerCase()

    const graph = useRFGraphStore.getState()
    const run = useRFRunStore.getState()

    run.resetRun()
    run.clearLogs()
    run.setValidation(null)
    run.setCompileData(null)
    run.setStatusData(null)
    run.setInferenceData(null)
    run.setFinalResult(null)
    run.setError(null)
    graph.setTraining({ ...RF_DEFAULT_TRAINING })

    if (mode === 'scratch') {
      graph.clearGraph()
      return
    }

    const preset = RF_PRESETS[template] ?? RF_PRESETS.iris_basic
    graph.setDataset(preset.dataset)
    graph.setTraining({ ...preset.training, dataset: preset.dataset })
    graph.setNodeConfig('rf_node_3', preset.forest)
  }, [])

  return (
    <RandomForestPage />
  )
}
