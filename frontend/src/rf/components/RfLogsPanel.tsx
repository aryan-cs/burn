import { levelClass, type RFLogEntry } from '../store/rfRunStore'

interface RfLogsPanelProps {
  logs: RFLogEntry[]
  onClear: () => void
}

export function RfLogsPanel({ logs, onClear }: RfLogsPanelProps) {
  return (
    <section className="rf-card">
      <div className="rf-card-title-row">
        <span className="rf-card-title">Operation Log</span>
        <button className="rf-btn rf-btn-sm" onClick={onClear}>
          Clear
        </button>
      </div>
      <div className="rf-log-list">
        {logs.length === 0 && <div className="rf-hint">No logs yet. Run a step to begin.</div>}
        {logs.map((entry) => (
          <div key={entry.id} className={`rf-log-line ${levelClass(entry.level)}`}>
            <span>[{entry.at}]</span> <span>{entry.message}</span>
          </div>
        ))}
      </div>
    </section>
  )
}
