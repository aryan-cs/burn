import { useEffect, useRef } from 'react'
import App from '../App'
import { useGraphStore, type GraphJSON } from '../store/graphStore'
import { useTrainingStore } from '../store/trainingStore'

type NNPreset = {
  graph: GraphJSON
  training: {
    dataset: string
    epochs: number
    batchSize: number
    optimizer: string
    learningRate: number
    loss: string
  }
}

const NN_PRESETS: Record<string, NNPreset> = {
  mnist_basic: {
    graph: {
      nodes: [
        { id: 'node_1', type: 'Input', config: { shape: [1, 28, 28], name: 'input layer' } },
        { id: 'node_2', type: 'Flatten', config: { name: 'flatten' } },
        { id: 'node_3', type: 'Dense', config: { rows: 4, cols: 8, units: 32, activation: 'relu', name: 'hidden 1' } },
        { id: 'node_4', type: 'Output', config: { num_classes: 10, activation: 'softmax', name: 'output layer' } },
      ],
      edges: [
        { id: 'edge_1', source: 'node_1', target: 'node_2' },
        { id: 'edge_2', source: 'node_2', target: 'node_3' },
        { id: 'edge_3', source: 'node_3', target: 'node_4' },
      ],
    },
    training: {
      dataset: 'mnist',
      epochs: 20,
      batchSize: 64,
      optimizer: 'adam',
      learningRate: 0.001,
      loss: 'cross_entropy',
    },
  },
  mnist_dropout: {
    graph: {
      nodes: [
        { id: 'node_1', type: 'Input', config: { shape: [1, 28, 28], name: 'input layer' } },
        { id: 'node_2', type: 'Flatten', config: { name: 'flatten' } },
        { id: 'node_3', type: 'Dense', config: { rows: 6, cols: 8, units: 48, activation: 'relu', name: 'hidden 1' } },
        { id: 'node_4', type: 'Dropout', config: { rate: 0.35, name: 'dropout' } },
        { id: 'node_5', type: 'Dense', config: { rows: 4, cols: 6, units: 24, activation: 'relu', name: 'hidden 2' } },
        { id: 'node_6', type: 'Output', config: { num_classes: 10, activation: 'softmax', name: 'output layer' } },
      ],
      edges: [
        { id: 'edge_1', source: 'node_1', target: 'node_2' },
        { id: 'edge_2', source: 'node_2', target: 'node_3' },
        { id: 'edge_3', source: 'node_3', target: 'node_4' },
        { id: 'edge_4', source: 'node_4', target: 'node_5' },
        { id: 'edge_5', source: 'node_5', target: 'node_6' },
      ],
    },
    training: {
      dataset: 'mnist',
      epochs: 30,
      batchSize: 64,
      optimizer: 'adam',
      learningRate: 0.0008,
      loss: 'cross_entropy',
    },
  },
}

const DEFAULT_TRAINING = {
  dataset: 'mnist',
  epochs: 20,
  batchSize: 64,
  optimizer: 'adam',
  learningRate: 0.001,
  loss: 'cross_entropy',
} as const

export default function NNBootstrapPage() {
  const initialized = useRef(false)

  useEffect(() => {
    if (initialized.current) return
    initialized.current = true

    const params = new URLSearchParams(window.location.search)
    const mode = (params.get('mode') ?? 'preset').toLowerCase()
    const template = (params.get('template') ?? 'mnist_basic').toLowerCase()

    const graph = useGraphStore.getState()
    const training = useTrainingStore.getState()

    training.reset()
    training.setConfig({ ...DEFAULT_TRAINING })

    if (mode === 'scratch') {
      graph.clear()
      return
    }

    const preset = NN_PRESETS[template] ?? NN_PRESETS.mnist_basic
    graph.fromJSON(preset.graph)
    training.setConfig(preset.training)
  }, [])

  return (
    <App />
  )
}
