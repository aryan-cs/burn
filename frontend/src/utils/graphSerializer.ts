import { useGraphStore } from '../store/graphStore'
import { useTrainingStore } from '../store/trainingStore'

/**
 * Serializes the full graph + training config into the JSON format
 * expected by the backend API.
 */
export function serializeForBackend() {
  const graph = useGraphStore.getState().toJSON()
  const config = useTrainingStore.getState().config

  return {
    ...graph,
    training: {
      dataset: config.dataset,
      epochs: config.epochs,
      batch_size: config.batchSize,
      optimizer: config.optimizer,
      learning_rate: config.learningRate,
      loss: config.loss,
    },
  }
}

/**
 * Exports the graph as a downloadable JSON file.
 */
export function exportGraphJSON() {
  const data = serializeForBackend()
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'mlcanvas-model.json'
  a.click()
  URL.revokeObjectURL(url)
}
