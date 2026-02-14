import { useTrainingStore } from '../store/trainingStore'
import { useGraphStore } from '../store/graphStore'

export function Toolbar() {
  const status = useTrainingStore((s) => s.status)
  const currentEpoch = useTrainingStore((s) => s.currentEpoch)
  const totalEpochs = useTrainingStore((s) => s.totalEpochs)
  const metrics = useTrainingStore((s) => s.metrics)
  const config = useTrainingStore((s) => s.config)
  const setConfig = useTrainingStore((s) => s.setConfig)
  const graphToJSON = useGraphStore((s) => s.toJSON)
  const startTraining = useTrainingStore((s) => s.startTraining)
  const setStatus = useTrainingStore((s) => s.setStatus)
  const reset = useTrainingStore((s) => s.reset)

  const latestMetric = metrics[metrics.length - 1]

  const handleTrain = async () => {
    const graphJSON = graphToJSON()
    try {
      const res = await fetch('/api/model/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...graphJSON, training: config }),
      })
      const data = await res.json()
      startTraining(data.job_id, config.epochs)
    } catch {
      // Backend not connected — set a placeholder for development
      startTraining('dev-job', config.epochs)
    }
  }

  const handleStop = async () => {
    try {
      await fetch('/api/model/stop', { method: 'POST' })
    } catch {
      // noop if backend down
    }
    setStatus('idle')
  }

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-[#12121a]/90 backdrop-blur-md rounded-xl border border-white/10 px-5 py-3 flex items-center gap-4 z-10">
      {/* Dataset selector */}
      <select
        value={config.dataset}
        onChange={(e) => setConfig({ dataset: e.target.value })}
        disabled={status === 'training'}
        className="bg-white/5 border border-white/10 rounded-md px-2 py-1 text-xs text-white outline-none"
      >
        <option value="mnist">MNIST</option>
        <option value="cifar10">CIFAR-10</option>
        <option value="fashion_mnist">Fashion MNIST</option>
      </select>

      {/* Epochs */}
      <div className="flex items-center gap-1">
        <label className="text-xs text-white/50">Epochs:</label>
        <input
          type="number"
          value={config.epochs}
          onChange={(e) => setConfig({ epochs: Number(e.target.value) })}
          disabled={status === 'training'}
          className="w-12 bg-white/5 border border-white/10 rounded-md px-1 py-1 text-xs text-white outline-none text-center"
        />
      </div>

      {/* Learning rate */}
      <div className="flex items-center gap-1">
        <label className="text-xs text-white/50">LR:</label>
        <input
          type="number"
          value={config.learningRate}
          step={0.0001}
          onChange={(e) => setConfig({ learningRate: Number(e.target.value) })}
          disabled={status === 'training'}
          className="w-20 bg-white/5 border border-white/10 rounded-md px-1 py-1 text-xs text-white outline-none text-center"
        />
      </div>

      {/* Divider */}
      <div className="w-px h-6 bg-white/10" />

      {/* Train / Stop */}
      {status !== 'training' ? (
        <button
          onClick={handleTrain}
          className="bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium px-4 py-1.5 rounded-lg transition-colors"
        >
          Train
        </button>
      ) : (
        <button
          onClick={handleStop}
          className="bg-red-600 hover:bg-red-500 text-white text-sm font-medium px-4 py-1.5 rounded-lg transition-colors"
        >
          Stop
        </button>
      )}

      {/* Reset */}
      {status === 'complete' && (
        <button
          onClick={reset}
          className="text-xs text-white/50 hover:text-white transition-colors"
        >
          Reset
        </button>
      )}

      {/* Live metrics */}
      {status === 'training' && (
        <>
          <div className="w-px h-6 bg-white/10" />
          <div className="text-xs text-white/70">
            Epoch {currentEpoch}/{totalEpochs}
          </div>
          {latestMetric && (
            <>
              <div className="text-xs text-white/70">
                Loss: {latestMetric.loss.toFixed(4)}
              </div>
              <div className="text-xs text-white/70">
                Acc: {(latestMetric.accuracy * 100).toFixed(1)}%
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}
