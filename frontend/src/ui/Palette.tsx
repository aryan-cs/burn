import { useGraphStore, type LayerType } from '../store/graphStore'

const LAYER_ITEMS: { type: LayerType; label: string; color: string }[] = [
  { type: 'Input', label: 'Input', color: '#4A4A4A' },
  { type: 'Dense', label: 'Dense', color: '#4A90D9' },
  { type: 'Conv2D', label: 'Conv2D', color: '#7B61FF' },
  { type: 'MaxPool2D', label: 'MaxPool', color: '#50E3C2' },
  { type: 'Flatten', label: 'Flatten', color: '#50E3C2' },
  { type: 'Dropout', label: 'Dropout', color: '#9B9B9B' },
  { type: 'BatchNorm', label: 'BatchNorm', color: '#9B9B9B' },
  { type: 'Output', label: 'Output', color: '#D0021B' },
]

export function Palette() {
  const addNode = useGraphStore((s) => s.addNode)
  const nodes = useGraphStore((s) => s.nodes)

  const handleAdd = (type: LayerType) => {
    // Place new nodes with some offset based on count
    const count = Object.keys(nodes).length
    const x = count * 3 - 6
    addNode(type, [x, 0.5, 0])
  }

  return (
    <div className="absolute left-4 top-4 bottom-4 w-44 bg-[#12121a]/90 backdrop-blur-md rounded-xl border border-white/10 p-3 flex flex-col gap-1 z-10 overflow-y-auto">
      <h2 className="text-xs font-semibold text-white/50 uppercase tracking-wider mb-2 px-1">
        Layers
      </h2>
      {LAYER_ITEMS.map((item) => (
        <button
          key={item.type}
          onClick={() => handleAdd(item.type)}
          className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-white/90 hover:bg-white/10 transition-colors cursor-grab active:cursor-grabbing text-left"
        >
          <span
            className="w-3 h-3 rounded-sm shrink-0"
            style={{ backgroundColor: item.color }}
          />
          {item.label}
        </button>
      ))}
    </div>
  )
}
