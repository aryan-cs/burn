import { useGraphStore } from '../store/graphStore'

export function PropertiesPanel() {
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const nodes = useGraphStore((s) => s.nodes)
  const updateNodeConfig = useGraphStore((s) => s.updateNodeConfig)
  const removeNode = useGraphStore((s) => s.removeNode)

  if (!selectedNodeId) return null
  const node = nodes[selectedNodeId]
  if (!node) return null

  const config = node.config

  return (
    <div className="absolute right-4 top-4 w-64 bg-[#12121a]/90 backdrop-blur-md rounded-xl border border-white/10 p-4 z-10">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-white">{node.type} Layer</h2>
        <button
          onClick={() => removeNode(node.id)}
          className="text-xs text-red-400 hover:text-red-300 transition-colors"
        >
          Delete
        </button>
      </div>

      <div className="flex flex-col gap-3">
        {/* Units (Dense, LSTM, GRU) */}
        {config.units !== undefined && (
          <Field
            label="Units"
            type="number"
            value={config.units}
            onChange={(v) => updateNodeConfig(node.id, { units: Number(v) })}
          />
        )}

        {/* Activation */}
        {config.activation !== undefined && (
          <div>
            <label className="text-xs text-white/50 block mb-1">Activation</label>
            <select
              value={config.activation}
              onChange={(e) =>
                updateNodeConfig(node.id, { activation: e.target.value })
              }
              className="w-full bg-white/5 border border-white/10 rounded-md px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500"
            >
              <option value="relu">ReLU</option>
              <option value="sigmoid">Sigmoid</option>
              <option value="tanh">Tanh</option>
              <option value="softmax">Softmax</option>
              <option value="none">None</option>
            </select>
          </div>
        )}

        {/* Filters (Conv2D) */}
        {config.filters !== undefined && (
          <Field
            label="Filters"
            type="number"
            value={config.filters}
            onChange={(v) => updateNodeConfig(node.id, { filters: Number(v) })}
          />
        )}

        {/* Kernel Size */}
        {config.kernel_size !== undefined && (
          <Field
            label="Kernel Size"
            type="number"
            value={config.kernel_size}
            onChange={(v) =>
              updateNodeConfig(node.id, { kernel_size: Number(v) })
            }
          />
        )}

        {/* Dropout Rate */}
        {config.rate !== undefined && (
          <Field
            label="Dropout Rate"
            type="number"
            value={config.rate}
            step={0.1}
            min={0}
            max={1}
            onChange={(v) => updateNodeConfig(node.id, { rate: Number(v) })}
          />
        )}

        {/* Num Classes (Output) */}
        {config.num_classes !== undefined && (
          <Field
            label="Num Classes"
            type="number"
            value={config.num_classes}
            onChange={(v) =>
              updateNodeConfig(node.id, { num_classes: Number(v) })
            }
          />
        )}

        {/* Shape display */}
        {node.shape.input && (
          <div className="text-xs text-white/40">
            Input: [{node.shape.input.join(', ')}]
          </div>
        )}
        {node.shape.output && (
          <div className="text-xs text-white/40">
            Output: [{node.shape.output.join(', ')}]
          </div>
        )}
      </div>
    </div>
  )
}

function Field({
  label,
  type,
  value,
  onChange,
  step,
  min,
  max,
}: {
  label: string
  type: string
  value: string | number | undefined
  onChange: (value: string) => void
  step?: number
  min?: number
  max?: number
}) {
  return (
    <div>
      <label className="text-xs text-white/50 block mb-1">{label}</label>
      <input
        type={type}
        value={value ?? ''}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-white/5 border border-white/10 rounded-md px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500"
      />
    </div>
  )
}
