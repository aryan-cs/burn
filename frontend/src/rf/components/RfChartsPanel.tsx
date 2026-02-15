import type { RFDoneMessage, RFProgressMessage } from '../types'

interface RfChartsPanelProps {
  progress: RFProgressMessage[]
  finalResult: RFDoneMessage | null
}

function accuracyWidth(value: number): string {
  const bounded = Math.max(0, Math.min(1, value))
  return `${Math.round(bounded * 100)}%`
}

export function RfChartsPanel({ progress, finalResult }: RfChartsPanelProps) {
  const latest = progress[progress.length - 1]
  const maxImportance = finalResult?.feature_importances.length
    ? Math.max(...finalResult.feature_importances, 1e-6)
    : 1

  return (
    <section className="rf-card">
      <div className="rf-card-title">Metrics & Charts</div>

      <div className="rf-card-subtitle">Progress (tree checkpoints)</div>
      <div className="rf-progress-list">
        {progress.length === 0 && <div className="rf-hint">No progress updates yet.</div>}
        {progress.slice(-8).map((entry) => (
          <div key={`${entry.trees_built}-${entry.elapsed_ms}`} className="rf-progress-item">
            <div className="rf-progress-head">
              trees {entry.trees_built}/{entry.total_trees}
            </div>
            <div className="rf-progress-bars">
              <div className="rf-progress-track">
                <div className="rf-progress-fill rf-progress-train" style={{ width: accuracyWidth(entry.train_accuracy) }} />
              </div>
              <div className="rf-progress-track">
                <div className="rf-progress-fill rf-progress-test" style={{ width: accuracyWidth(entry.test_accuracy) }} />
              </div>
            </div>
            <div className="rf-progress-caption">
              train {(entry.train_accuracy * 100).toFixed(2)}% | test {(entry.test_accuracy * 100).toFixed(2)}%
            </div>
          </div>
        ))}
      </div>

      <div className="rf-metric-grid">
        <div>
          <div className="rf-meta-label">Final Train Accuracy</div>
          <div className="rf-meta-value">
            {finalResult ? `${(finalResult.final_train_accuracy * 100).toFixed(2)}%` : '—'}
          </div>
        </div>
        <div>
          <div className="rf-meta-label">Final Test Accuracy</div>
          <div className="rf-meta-value">
            {finalResult ? `${(finalResult.final_test_accuracy * 100).toFixed(2)}%` : '—'}
          </div>
        </div>
        <div>
          <div className="rf-meta-label">Latest Trees</div>
          <div className="rf-meta-value">
            {latest ? `${latest.trees_built}/${latest.total_trees}` : '—'}
          </div>
        </div>
      </div>

      <div className="rf-card-subtitle">Confusion Matrix</div>
      <div className="rf-matrix">
        {finalResult?.confusion_matrix?.length ? (
          finalResult.confusion_matrix.map((row, rowIndex) => (
            <div key={rowIndex} className="rf-matrix-row">
              {row.map((value, colIndex) => (
                <div key={`${rowIndex}-${colIndex}`} className="rf-matrix-cell">
                  {value}
                </div>
              ))}
            </div>
          ))
        ) : (
          <div className="rf-hint">Confusion matrix will appear after training completes.</div>
        )}
      </div>

      <div className="rf-card-subtitle">Feature Importance</div>
      <div className="rf-importance-list">
        {finalResult?.feature_importances?.length ? (
          finalResult.feature_importances.map((value, index) => {
            const width = `${Math.round((value / maxImportance) * 100)}%`
            return (
              <div key={index} className="rf-importance-item">
                <div className="rf-importance-label">
                  {finalResult.feature_names[index] ?? `feature_${index}`}
                </div>
                <div className="rf-importance-track">
                  <div className="rf-importance-fill" style={{ width }} />
                </div>
                <div className="rf-importance-value">{value.toFixed(4)}</div>
              </div>
            )
          })
        ) : (
          <div className="rf-hint">Feature importance will appear after training completes.</div>
        )}
      </div>
    </section>
  )
}
