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
  digits_basic: {
    graph: {
      nodes: [
        { id: 'node_1', type: 'Input', config: { shape: [1, 8, 8], name: 'input layer' } },
        { id: 'node_2', type: 'Flatten', config: { name: 'flatten' } },
        { id: 'node_3', type: 'Dense', config: { rows: 4, cols: 4, units: 16, activation: 'relu', name: 'hidden 1' } },
        { id: 'node_4', type: 'Output', config: { num_classes: 10, activation: 'softmax', name: 'output layer' } },
      ],
      edges: [
        { id: 'edge_1', source: 'node_1', target: 'node_2' },
        { id: 'edge_2', source: 'node_2', target: 'node_3' },
        { id: 'edge_3', source: 'node_3', target: 'node_4' },
      ],
    },
    training: {
      dataset: 'digits',
      epochs: 25,
      batchSize: 64,
      optimizer: 'adam',
      learningRate: 0.001,
      loss: 'cross_entropy',
    },
  },
  alexnet_cats_dogs: {
    graph: {
      nodes: [
        { id: 'node_1', type: 'Input', config: { shape: [3, 96, 96], name: 'input layer' } },
        { id: 'node_2', type: 'Conv2D', config: { filters: 96, kernel_size: 11, stride: 4, padding: 2, activation: 'relu', name: 'conv1' } },
        { id: 'node_3', type: 'MaxPool2D', config: { kernel_size: 3, stride: 2, padding: 0, name: 'pool1' } },
        { id: 'node_4', type: 'Conv2D', config: { filters: 256, kernel_size: 5, stride: 1, padding: 2, activation: 'relu', name: 'conv2' } },
        { id: 'node_5', type: 'MaxPool2D', config: { kernel_size: 3, stride: 2, padding: 0, name: 'pool2' } },
        { id: 'node_6', type: 'Conv2D', config: { filters: 384, kernel_size: 3, stride: 1, padding: 1, activation: 'relu', name: 'conv3' } },
        { id: 'node_7', type: 'Conv2D', config: { filters: 384, kernel_size: 3, stride: 1, padding: 1, activation: 'relu', name: 'conv4' } },
        { id: 'node_8', type: 'Conv2D', config: { filters: 256, kernel_size: 3, stride: 1, padding: 1, activation: 'relu', name: 'conv5' } },
        { id: 'node_9', type: 'MaxPool2D', config: { kernel_size: 3, stride: 2, padding: 0, name: 'pool5' } },
        { id: 'node_10', type: 'Flatten', config: { name: 'flatten' } },
        { id: 'node_11', type: 'Dense', config: { rows: 64, cols: 64, units: 4096, activation: 'relu', name: 'fc6' } },
        { id: 'node_12', type: 'Dropout', config: { rate: 0.5, name: 'dropout6' } },
        { id: 'node_13', type: 'Dense', config: { rows: 64, cols: 64, units: 4096, activation: 'relu', name: 'fc7' } },
        { id: 'node_14', type: 'Dropout', config: { rate: 0.5, name: 'dropout7' } },
        { id: 'node_15', type: 'Output', config: { num_classes: 2, activation: 'softmax', name: 'output layer' } },
      ],
      edges: [
        { id: 'edge_1', source: 'node_1', target: 'node_2' },
        { id: 'edge_2', source: 'node_2', target: 'node_3' },
        { id: 'edge_3', source: 'node_3', target: 'node_4' },
        { id: 'edge_4', source: 'node_4', target: 'node_5' },
        { id: 'edge_5', source: 'node_5', target: 'node_6' },
        { id: 'edge_6', source: 'node_6', target: 'node_7' },
        { id: 'edge_7', source: 'node_7', target: 'node_8' },
        { id: 'edge_8', source: 'node_8', target: 'node_9' },
        { id: 'edge_9', source: 'node_9', target: 'node_10' },
        { id: 'edge_10', source: 'node_10', target: 'node_11' },
        { id: 'edge_11', source: 'node_11', target: 'node_12' },
        { id: 'edge_12', source: 'node_12', target: 'node_13' },
        { id: 'edge_13', source: 'node_13', target: 'node_14' },
        { id: 'edge_14', source: 'node_14', target: 'node_15' },
      ],
    },
    training: {
      dataset: 'cats_vs_dogs',
      epochs: 20,
      batchSize: 32,
      optimizer: 'adam',
      learningRate: 0.0003,
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
    const resolvedTemplate =
      template === 'alexnet' || template === 'alexnet_inspired'
        ? 'alexnet_cats_dogs'
        : template

    const graph = useGraphStore.getState()
    const training = useTrainingStore.getState()

    training.reset()
    training.setConfig({ ...DEFAULT_TRAINING })

    if (mode === 'scratch') {
      graph.clear()
      return
    }

    const preset = NN_PRESETS[resolvedTemplate] ?? NN_PRESETS.mnist_basic
    graph.fromJSON(preset.graph)
    training.setConfig(preset.training)
  }, [])

  return (
    <App />
  )
}
