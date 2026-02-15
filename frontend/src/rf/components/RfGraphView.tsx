import { RfViewport } from '../canvas/RfViewport'

interface RfGraphViewProps {
  lowDetailMode: boolean
}

export function RfGraphView({ lowDetailMode }: RfGraphViewProps) {
  return (
    <section className="rf-builder-surface">
      <RfViewport lowDetailMode={lowDetailMode} />
    </section>
  )
}
