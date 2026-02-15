import { useCallback, useEffect, useState } from 'react'
import { useMlStore } from '../../store/mlStore'
import { useMlWebSocket } from '../../hooks/useMlWebSocket'
import { MlBuildTab } from './MlBuildTab'
import { MlTrainTab } from './MlTrainTab'
import { MlTestTab } from './MlTestTab'
import { MlModelDiagram } from './MlModelDiagram'

type MlTab = 'build' | 'train' | 'test'

export function MlApp() {
  const store = useMlStore()
  const { sendStop } = useMlWebSocket()
  const [activeTab, setActiveTab] = useState<MlTab>('build')
  const [busy, setBusy] = useState(false)
  const [busyAction, setBusyAction] = useState<string | null>(null)

  // Fetch datasets on mount
  useEffect(() => {
    fetch('/api/ml/datasets')
      .then(async (r) => {
        if (!r.ok) return null
        return r.json() as Promise<{ datasets?: typeof store.datasets }>
      })
      .then((data) => {
        const incoming = data?.datasets
        store.setDatasets(Array.isArray(incoming) ? incoming : [])
      })
      .catch(() => {
        store.setDatasets([])
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Auto-switch to test tab on training complete
  useEffect(() => {
    if (store.status === 'complete') {
      const timer = setTimeout(() => setActiveTab('test'), 600)
      return () => clearTimeout(timer)
    }
  }, [store.status])

  // Dataset info for the selected dataset
  const dsInfo = store.datasets.find((d) => d.id === store.dataset) ?? null

  const handleTrain = useCallback(async () => {
    if (busy) return
    setBusy(true)
    setBusyAction('train')
    try {
      const body: Record<string, unknown> = {
        model_type: store.modelType,
        dataset: store.dataset,
        test_size: store.testSize,
        hyperparameters: store.hyperparams,
      }

      const res = await fetch('/api/ml/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `HTTP ${res.status}`)
      }

      const data = (await res.json()) as { job_id: string }
      store.startTraining(data.job_id)
      setActiveTab('train')
    } catch (e) {
      store.setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      setBusyAction(null)
    }
  }, [busy, store])

  const handleStop = useCallback(async () => {
    if (busy || !store.jobId) return
    setBusy(true)
    setBusyAction('stop')
    try {
      sendStop()
      await fetch('/api/ml/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: store.jobId }),
      })
    } catch {
      // Ignore stop errors
    } finally {
      setBusy(false)
      setBusyAction(null)
    }
  }, [busy, store.jobId, sendStop])

  const canOpenTrain = store.status !== 'idle' || store.jobId !== null
  const canOpenTest =
    (store.status === 'complete' || store.status === 'stopped') && store.jobId !== null

  const activeTabIndex = activeTab === 'build' ? 0 : activeTab === 'train' ? 1 : 2

  const isClassification = store.task
    ? store.task === 'classification'
    : store.modelType !== 'linear_regression'

  return (
    <div className="app-shell">
      <section className="app-sidebar">
        <div className="app-sidebar-inner">
          {/* Tab strip */}
          <div className="app-tab-strip">
            <div
              aria-hidden
              className="app-tab-indicator"
              style={{ transform: `translateX(${activeTabIndex * 100}%)` }}
            >
              <div className="app-tab-indicator-inner">
                <div className="app-tab-indicator-glow" />
                <div className="app-tab-indicator-line" />
              </div>
            </div>
            <button
              type="button"
              onClick={() => setActiveTab('build')}
              className={`app-tab-button ${activeTab === 'build' ? 'app-tab-button-active' : 'app-tab-button-inactive'}`}
            >
              Build
            </button>
            <button
              type="button"
              disabled={!canOpenTrain}
              onClick={() => canOpenTrain && setActiveTab('train')}
              className={`app-tab-button ${activeTab === 'train' ? 'app-tab-button-active' : 'app-tab-button-inactive'}`}
            >
              Train
            </button>
            <button
              type="button"
              disabled={!canOpenTest}
              onClick={() => canOpenTest && setActiveTab('test')}
              className={`app-tab-button ${activeTab === 'test' ? 'app-tab-button-active' : 'app-tab-button-inactive'}`}
            >
              Test
            </button>
          </div>

          {/* Tab content */}
          {activeTab === 'build' && (
            <MlBuildTab
              modelType={store.modelType}
              datasets={store.datasets}
              dataset={store.dataset}
              testSize={store.testSize}
              hyperparams={store.hyperparams}
              task={store.task}
              onModelTypeChange={(t) => store.setModelType(t)}
              onDatasetChange={(d) => store.setDataset(d)}
              onTestSizeChange={(s) => store.setTestSize(s)}
              onHyperparamChange={(p: Partial<typeof store.hyperparams>) => store.setHyperparams(p)}
              onTrain={handleTrain}
              trainDisabled={busy}
              trainLabel={busyAction === 'train' ? 'Starting...' : 'Train Model'}
            />
          )}

          {activeTab === 'train' && (
            <MlTrainTab
              modelType={store.modelType}
              status={store.status}
              progress={store.progress}
              progressHistory={store.progressHistory}
              trainMetrics={store.trainMetrics}
              testMetrics={store.testMetrics}
              featureImportances={store.featureImportances}
              trainingTime={store.trainingTime}
              isTraining={store.status === 'training' || store.status === 'loading_data'}
              onStop={handleStop}
              stopDisabled={busy || !store.jobId}
              stopLabel={busyAction === 'stop' ? 'Stopping...' : 'Stop Training'}
              onBackToBuild={() => setActiveTab('build')}
            />
          )}

          {activeTab === 'test' && (
            <MlTestTab
              featureNames={store.featureNames}
              targetNames={store.targetNames}
              isClassification={isClassification}
              jobId={store.jobId}
              trainingComplete={store.status === 'complete' || store.status === 'stopped'}
              onBackToBuild={() => setActiveTab('build')}
            />
          )}
        </div>
      </section>

      <section className="app-viewport-panel">
        <MlModelDiagram
          modelType={store.modelType}
          datasetName={dsInfo?.name ?? store.dataset}
          task={store.task}
          featureCount={dsInfo?.n_features ?? store.featureNames.length}
          status={store.status}
        />
      </section>
    </div>
  )
}
