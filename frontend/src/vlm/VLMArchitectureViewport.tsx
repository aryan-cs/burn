import { Canvas } from '@react-three/fiber'
import { Line, OrbitControls } from '@react-three/drei'
import { useEffect, useMemo, useState } from 'react'
import * as THREE from 'three'
import { WeightVisual } from '../canvas/edges/WeightVisual'
import type {
  VLMArchitectureSpec,
  VLMArchitectureStage,
  VLMCnnStageBlueprint,
} from './architecture'

type Vec3 = [number, number, number]
type NodeKind = 'token' | 'stage'

interface VLMArchitectureViewportProps {
  architecture: VLMArchitectureSpec
  selectedStageId: string | null
  onSelectStage: (stageId: string) => void
  lowDetailMode: boolean
  activityPulse: number
}

interface VisualNode {
  id: string
  position: Vec3
  stageId: string
  radius: number
  kind: NodeKind
  segments?: number
}

interface VisualEdge {
  id: string
  from: Vec3
  to: Vec3
  weight: number
  sourceStageId: string
  targetStageId: string
  kind: 'attention' | 'token_chain' | 'stage_flow'
}

interface CnnMatrixPanel {
  id: string
  stageId: string
  label: string
  detail: string
  color: string
  trueRows: number
  trueCols: number
  rows: number
  cols: number
  channels: number
  sliceCount: number
  position: Vec3
  width: number
  height: number
  depth: number
}

interface CnnConvolutionOp {
  id: string
  stageId: string
  sourcePanelId: string
  targetPanelId: string
  kernelSize: number
  stride: number
  kernelPosition: Vec3
}

interface CnnPipeline {
  panels: CnnMatrixPanel[]
  ops: CnnConvolutionOp[]
}

interface PanelHighlight {
  row: number
  col: number
  rows: number
  cols: number
  color: string
}

interface StageRoles {
  input: string
  preprocess: string
  conv: string
  transformer: string
  head: string
}

interface AttentionState {
  edges: VisualEdge[]
  activeTokenIds: Set<string>
  activationById: Map<string, number>
}

interface StageFlowGraph {
  nodes: VisualNode[]
  edges: VisualEdge[]
}

interface TransformerRailGraph {
  leftTokens: VisualNode[]
  rightTokens: VisualNode[]
  railEdges: VisualEdge[]
  alignmentEdges: VisualEdge[]
}

const IGNORE_RAYCAST: THREE.Mesh['raycast'] = () => undefined

const TRANSFORMER_LEFT_X = 4.9
const TRANSFORMER_RIGHT_X = 9.8
const TRANSFORMER_TOP_Y = 2.55
const MAX_MATRIX_CELLS_HIGH = 144
const MAX_MATRIX_CELLS_LOW = 81

export function VLMArchitectureViewport({
  architecture,
  selectedStageId,
  onSelectStage,
  lowDetailMode,
  activityPulse,
}: VLMArchitectureViewportProps) {
  const roles = useMemo(() => resolveStageRoles(architecture.stages), [architecture.stages])
  const stageColorMap = useMemo(
    () => new Map(architecture.stages.map((stage) => [stage.id, stage.color])),
    [architecture.stages]
  )

  const cnnPipeline = useMemo(
    () => createCnnMatrixPipeline(architecture, roles, stageColorMap, lowDetailMode),
    [architecture, roles, stageColorMap, lowDetailMode]
  )

  const cnnPanelLookup = useMemo(
    () => new Map(cnnPipeline.panels.map((panel) => [panel.id, panel])),
    [cnnPipeline.panels]
  )

  const [convTick, setConvTick] = useState(0)
  useEffect(() => {
    setConvTick(0)
  }, [cnnPipeline.ops.length, architecture.id])

  useEffect(() => {
    if (cnnPipeline.ops.length === 0) return
    const timer = window.setInterval(() => {
      setConvTick((current) => current + 1)
    }, lowDetailMode ? 350 : 240)
    return () => window.clearInterval(timer)
  }, [cnnPipeline.ops.length, lowDetailMode])

  const animatedConvOpIndex =
    cnnPipeline.ops.length > 0 ? convTick % cnnPipeline.ops.length : -1
  const selectedConvOpIndex =
    selectedStageId === null
      ? -1
      : cnnPipeline.ops.findIndex((op) => op.stageId === selectedStageId)
  const focusedConvOpIndex = selectedConvOpIndex >= 0 ? selectedConvOpIndex : animatedConvOpIndex
  const focusedConvOp =
    focusedConvOpIndex >= 0 ? cnnPipeline.ops[focusedConvOpIndex] : null
  const convScanStep = Math.floor(convTick / Math.max(1, cnnPipeline.ops.length))

  const panelHighlights = useMemo(
    () => createPanelHighlights(focusedConvOp, cnnPanelLookup, convScanStep),
    [focusedConvOp, cnnPanelLookup, convScanStep]
  )

  const focusedSourcePanel = focusedConvOp
    ? (cnnPanelLookup.get(focusedConvOp.sourcePanelId) ?? null)
    : null
  const focusedTargetPanel = focusedConvOp
    ? (cnnPanelLookup.get(focusedConvOp.targetPanelId) ?? null)
    : null

  const showTransformer = false

  const totalTokens = Math.max(1, architecture.blueprint.transformer.token_count)
  const renderedTokenCount = showTransformer
    ? (lowDetailMode ? Math.min(22, totalTokens) : Math.min(36, totalTokens))
    : 0
  const transformerRailGraph = useMemo(
    () => createTransformerRailGraph(roles.transformer, renderedTokenCount, lowDetailMode),
    [roles.transformer, renderedTokenCount, lowDetailMode]
  )

  const [attentionStep, setAttentionStep] = useState(0)
  useEffect(() => {
    setAttentionStep(0)
  }, [transformerRailGraph.leftTokens.length, architecture.id])
  useEffect(() => {
    if (transformerRailGraph.leftTokens.length === 0) return
    const timer = window.setInterval(
      () => setAttentionStep((current) => (current + 1) % 100000),
      lowDetailMode ? 320 : 190
    )
    return () => window.clearInterval(timer)
  }, [transformerRailGraph.leftTokens.length, lowDetailMode])

  const attentionState = useMemo(
    () =>
      createAttentionState(
        transformerRailGraph.leftTokens,
        transformerRailGraph.rightTokens,
        architecture,
        attentionStep,
        lowDetailMode
      ),
    [transformerRailGraph.leftTokens, transformerRailGraph.rightTokens, architecture, attentionStep, lowDetailMode]
  )
  const attentionEdges = attentionState.edges
  const activeTokenIds = attentionState.activeTokenIds
  const tokenActivationById = attentionState.activationById

  const railEdgeSplit = useMemo(
    () => splitEdgesBySelection(transformerRailGraph.railEdges, selectedStageId),
    [transformerRailGraph.railEdges, selectedStageId]
  )
  const alignmentEdgeSplit = useMemo(
    () => splitEdgesBySelection(transformerRailGraph.alignmentEdges, selectedStageId),
    [transformerRailGraph.alignmentEdges, selectedStageId]
  )
  const attentionEdgeSplit = useMemo(
    () => splitEdgesBySelection(attentionEdges, selectedStageId),
    [attentionEdges, selectedStageId]
  )
  const stageFlowGraph = useMemo(
    () => createStageFlowGraph(architecture, lowDetailMode),
    [architecture, lowDetailMode]
  )
  const stageFlowEdgeSplit = useMemo(
    () => splitEdgesBySelection(stageFlowGraph.edges, selectedStageId),
    [stageFlowGraph.edges, selectedStageId]
  )
  const activeStageIds = useMemo(() => {
    const ids = new Set<string>()
    if (selectedStageId) {
      ids.add(selectedStageId)
    }
    if (focusedConvOp) {
      ids.add(focusedConvOp.stageId)
    }
    if (focusedSourcePanel) {
      ids.add(focusedSourcePanel.stageId)
    }
    if (focusedTargetPanel) {
      ids.add(focusedTargetPanel.stageId)
    }
    if (showTransformer) {
      ids.add(roles.transformer)
      ids.add(roles.head)
    }
    return ids
  }, [focusedConvOp, focusedSourcePanel, focusedTargetPanel, roles.head, roles.transformer, selectedStageId, showTransformer])

  const railDefaultOpacity = 0.2
  const railSelectedOpacity = clampOpacity(0.38 + activityPulse * 0.05)
  const alignmentDefaultOpacity = 0.18
  const alignmentSelectedOpacity = clampOpacity(0.34 + activityPulse * 0.06)
  const attentionDefaultOpacity = clampOpacity(0.44 + activityPulse * 0.08)
  const attentionSelectedOpacity = clampOpacity(0.62 + activityPulse * 0.14)

  const attentionPulseEdges = useMemo(() => {
    const ranked = [...attentionEdges].sort((a, b) => b.weight - a.weight)
    return ranked.slice(0, lowDetailMode ? 3 : 6)
  }, [attentionEdges, lowDetailMode])

  return (
    <div className="vlm-viewport-root">
      <Canvas camera={{ position: [0.2, 4.5, 15], fov: 48 }} style={{ width: '100%', height: '100%' }}>
        <color attach="background" args={['#000000']} />
        <ambientLight intensity={0.45} />
        <directionalLight position={[11, 10, 4]} intensity={0.9} />
        <pointLight position={[-10, -10, -5]} intensity={0.25} color="#9a9a9a" />
        <gridHelper args={[46, 92, '#2b2b2b', '#202020']} />

        <CnnPipelineChainScene
          panels={cnnPipeline.panels}
          ops={cnnPipeline.ops}
          panelHighlights={panelHighlights}
          selectedStageId={selectedStageId}
          activePairIndex={focusedConvOpIndex}
          activeStageIds={activeStageIds}
          lowDetailMode={lowDetailMode}
          activityPulse={activityPulse}
          onSelectStage={onSelectStage}
        />

        {false ? (
          <>
            <EdgeSegments
              edges={stageFlowEdgeSplit.rest}
              color="#3f454f"
              opacity={0.24}
            />
            <EdgeSegments
              edges={stageFlowEdgeSplit.selected}
              color="#ffb429"
              opacity={clampOpacity(0.52 + activityPulse * 0.1)}
            />
            {stageFlowGraph.nodes.map((node) => (
              <TokenNode
                key={node.id}
                node={node}
                selected={selectedStageId === node.stageId}
                active={activeStageIds.has(node.stageId)}
                activation={activeStageIds.has(node.stageId) ? 1 : 0}
                lowDetailMode={lowDetailMode}
                outlineColor={stageColorMap.get(node.stageId) ?? '#ffb429'}
                onSelectStage={onSelectStage}
              />
            ))}
          </>
        ) : null}

        {false && focusedConvOp && focusedSourcePanel && focusedTargetPanel ? (
          <CnnSymbolicScene
            op={focusedConvOp as CnnConvolutionOp}
            sourcePanel={focusedSourcePanel as CnnMatrixPanel}
            targetPanel={focusedTargetPanel as CnnMatrixPanel}
            sourceHighlight={panelHighlights.get((focusedSourcePanel as CnnMatrixPanel).id) ?? null}
            targetHighlight={panelHighlights.get((focusedTargetPanel as CnnMatrixPanel).id) ?? null}
            selectedStageId={selectedStageId}
            lowDetailMode={lowDetailMode}
            activityPulse={activityPulse}
            onSelectStage={onSelectStage}
          />
        ) : null}

        {showTransformer ? (
          <>
            <TransformerHeadStrip
              count={Math.max(2, lowDetailMode ? Math.min(6, architecture.blueprint.transformer.attention_heads) : Math.min(10, architecture.blueprint.transformer.attention_heads + 2))}
              lowDetailMode={lowDetailMode}
            />
            <EdgeSegments
              edges={railEdgeSplit.rest}
              color="#48525d"
              opacity={railDefaultOpacity}
            />
            <EdgeSegments
              edges={railEdgeSplit.selected}
              color="#8aa2bc"
              opacity={railSelectedOpacity}
            />
            <EdgeSegments
              edges={alignmentEdgeSplit.rest}
              color="#56616c"
              opacity={alignmentDefaultOpacity}
            />
            <EdgeSegments
              edges={alignmentEdgeSplit.selected}
              color="#b3c2d1"
              opacity={alignmentSelectedOpacity}
            />
            <WeightedEdgeSegments
              edges={attentionEdgeSplit.rest}
              lowColor="#2e343a"
              highColor="#ffb429"
              opacity={attentionDefaultOpacity}
            />
            <WeightedEdgeSegments
              edges={attentionEdgeSplit.selected}
              lowColor="#7e98b3"
              highColor="#ffd37a"
              opacity={attentionSelectedOpacity}
            />

            {[...transformerRailGraph.leftTokens, ...transformerRailGraph.rightTokens].map((node) => {
              const activation = tokenActivationById.get(node.id) ?? 0
              return (
                <TokenNode
                  key={node.id}
                  node={node}
                  selected={selectedStageId === node.stageId}
                  active={activeTokenIds.has(node.id)}
                  activation={activation}
                  lowDetailMode={lowDetailMode}
                  outlineColor={stageColorMap.get(node.stageId) ?? '#ffd37a'}
                  onSelectStage={onSelectStage}
                />
              )
            })}

            {attentionPulseEdges.map((edge, index) => {
              const curve = new THREE.LineCurve3(
                new THREE.Vector3(edge.from[0], edge.from[1], edge.from[2]),
                new THREE.Vector3(edge.to[0], edge.to[1], edge.to[2])
              )
              return (
                <WeightVisual
                  key={`att-pulse-${edge.id}`}
                  curve={curve}
                  speed={lowDetailMode ? 0.36 : 0.52}
                  color="#ffe29b"
                  size={0.045}
                  opacity={0.86}
                  offset={index * 0.12}
                />
              )
            })}
          </>
        ) : null}

        <OrbitControls
          makeDefault
          enablePan
          enableZoom
          enableRotate
          minDistance={5}
          maxDistance={72}
          target={[0.7, 0.1, 0]}
          mouseButtons={{
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN,
          }}
        />
      </Canvas>
    </div>
  )
}

interface CnnPipelineChainSceneProps {
  panels: CnnMatrixPanel[]
  ops: CnnConvolutionOp[]
  panelHighlights: Map<string, PanelHighlight>
  selectedStageId: string | null
  activePairIndex: number
  activeStageIds: Set<string>
  lowDetailMode: boolean
  activityPulse: number
  onSelectStage: (stageId: string) => void
}

function CnnPipelineChainScene({
  panels,
  ops,
  panelHighlights,
  selectedStageId,
  activePairIndex,
  activeStageIds,
  lowDetailMode,
  activityPulse,
  onSelectStage,
}: CnnPipelineChainSceneProps) {
  const chainEdges = useMemo(() => createCnnChainEdges(panels, lowDetailMode), [panels, lowDetailMode])
  const selectedSplit = useMemo(
    () => splitEdgesBySelection(chainEdges, selectedStageId),
    [chainEdges, selectedStageId]
  )

  const focusedPairIndex = useMemo(() => {
    if (panels.length < 2) return -1
    if (selectedStageId) {
      const selectedPair = panels.findIndex((panel, index) => {
        if (index >= panels.length - 1) return false
        const next = panels[index + 1]
        return panel.stageId === selectedStageId || next.stageId === selectedStageId
      })
      if (selectedPair >= 0) return selectedPair
    }
    if (activePairIndex >= 0 && activePairIndex < panels.length - 1) {
      return activePairIndex
    }
    if (ops.length > 0) return 0
    return -1
  }, [activePairIndex, ops.length, panels, selectedStageId])

  const activeEdges = useMemo(() => {
    if (focusedPairIndex < 0) return []
    return chainEdges.filter((edge) => edge.id.startsWith(`cnn-chain:${focusedPairIndex}:`))
  }, [chainEdges, focusedPairIndex])

  const pulseEdges = useMemo(
    () => activeEdges.slice(0, lowDetailMode ? 6 : 10),
    [activeEdges, lowDetailMode]
  )

  return (
    <group>
      <EdgeSegments
        edges={selectedSplit.rest}
        color="#343434"
        opacity={0.19}
      />
      <WeightedEdgeSegments
        edges={activeEdges}
        lowColor="#504a45"
        highColor="#ffb429"
        opacity={clampOpacity(0.5 + activityPulse * 0.2)}
      />
      <EdgeSegments
        edges={selectedSplit.selected}
        color="#ffd48a"
        opacity={clampOpacity(0.58 + activityPulse * 0.12)}
      />

      {pulseEdges.map((edge, index) => {
        const curve = new THREE.LineCurve3(
          new THREE.Vector3(edge.from[0], edge.from[1], edge.from[2]),
          new THREE.Vector3(edge.to[0], edge.to[1], edge.to[2])
        )
        return (
          <WeightVisual
            key={`vlm-chain-pulse-${edge.id}`}
            curve={curve}
            speed={lowDetailMode ? 0.38 : 0.5}
            color="#ffd88d"
            size={0.048}
            opacity={0.88}
            offset={index * 0.1}
          />
        )
      })}

      {panels.map((panel) => (
        <CnnChainPanel
          key={panel.id}
          panel={panel}
          selected={selectedStageId === panel.stageId}
          active={activeStageIds.has(panel.stageId)}
          highlight={panelHighlights.get(panel.id) ?? null}
          lowDetailMode={lowDetailMode}
          activityPulse={activityPulse}
          onSelectStage={onSelectStage}
        />
      ))}
    </group>
  )
}

interface CnnChainPanelProps {
  panel: CnnMatrixPanel
  selected: boolean
  active: boolean
  highlight: PanelHighlight | null
  lowDetailMode: boolean
  activityPulse: number
  onSelectStage: (stageId: string) => void
}

function CnnChainPanel({
  panel,
  selected,
  active,
  highlight,
  lowDetailMode,
  activityPulse,
  onSelectStage,
}: CnnChainPanelProps) {
  const [hovered, setHovered] = useState(false)
  const isInput = panel.id === 'cnn_input_matrix'
  const focused = selected || active || hovered
  const outlineColor = focused ? '#ffd999' : panel.color
  const slices = clampInt(panel.sliceCount + (isInput ? 1 : 0), 2, lowDetailMode ? 5 : 8)
  const nodePoints = useMemo(
    () => createPanelNodePoints(panel, lowDetailMode),
    [panel, lowDetailMode]
  )
  const highlightRect = useMemo(
    () => createPanelHighlightRect(panel, highlight),
    [panel, highlight]
  )

  return (
    <group position={panel.position} rotation={[0, isInput ? Math.PI * 0.5 : 0, 0]}>
      {Array.from({ length: slices }).map((_, index) => {
        const ratio = slices <= 1 ? 0 : index / (slices - 1)
        const xShift = -ratio * panel.depth * 0.38
        const zShift = -ratio * panel.depth * 0.18
        const opacity = ratio === 0 ? 0.85 : Math.max(0.18, 0.55 - ratio * 0.42)
        return (
          <mesh key={`${panel.id}-stack-${index}`} position={[xShift, 0, zShift]} raycast={index === 0 ? undefined : IGNORE_RAYCAST}>
            <boxGeometry args={[panel.width, panel.height, Math.max(0.18, panel.depth * 0.36)]} />
            <meshStandardMaterial
              color="#131313"
              emissive={panel.color}
              emissiveIntensity={ratio === 0 ? (focused ? 0.3 : 0.18) : 0.08}
              transparent
              opacity={opacity}
              roughness={0.45}
              metalness={0.08}
            />
          </mesh>
        )
      })}

      {nodePoints.map((point, index) => {
        const seeded = Math.abs(seededSigned(`chain-node:${panel.id}:${index}`))
        const pulse = active ? 0.24 + activityPulse * 0.16 : 0.04
        const emissiveIntensity = 0.16 + seeded * 0.38 + pulse + (selected ? 0.14 : 0)
        return (
          <mesh key={`${panel.id}-node-${index}`} position={point} raycast={IGNORE_RAYCAST}>
            <sphereGeometry args={[lowDetailMode ? 0.03 : 0.038, lowDetailMode ? 8 : 10, lowDetailMode ? 8 : 10]} />
            <meshStandardMaterial
              color="#111111"
              emissive={panel.color}
              emissiveIntensity={emissiveIntensity}
              roughness={0.32}
              metalness={0.1}
            />
          </mesh>
        )
      })}

      {highlightRect ? (
        <mesh position={highlightRect.position} raycast={IGNORE_RAYCAST}>
          <planeGeometry args={[highlightRect.width, highlightRect.height]} />
          <meshBasicMaterial
            color={highlightRect.color}
            transparent
            opacity={0.26}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      ) : null}

      {highlightRect ? (
        <mesh position={[highlightRect.position[0], highlightRect.position[1], highlightRect.position[2] + 0.003]} raycast={IGNORE_RAYCAST}>
          <planeGeometry args={[highlightRect.width, highlightRect.height]} />
          <meshBasicMaterial
            color={highlightRect.color}
            wireframe
            transparent
            opacity={0.9}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      ) : null}

      <mesh raycast={IGNORE_RAYCAST}>
        <boxGeometry args={[panel.width * 1.02, panel.height * 1.02, panel.depth * 1.02]} />
        <meshBasicMaterial
          color={outlineColor}
          wireframe
          transparent
          opacity={focused ? 0.72 : 0.36}
          depthWrite={false}
        />
      </mesh>

      <mesh
        onPointerOver={(event) => {
          event.stopPropagation()
          setHovered(true)
        }}
        onPointerOut={(event) => {
          event.stopPropagation()
          setHovered(false)
        }}
        onClick={(event) => {
          event.stopPropagation()
          onSelectStage(panel.stageId)
        }}
      >
        <boxGeometry args={[panel.width + 0.26, panel.height + 0.26, panel.depth + 0.26]} />
        <meshBasicMaterial transparent opacity={0} />
      </mesh>
    </group>
  )
}

function createPanelNodePoints(panel: CnnMatrixPanel, lowDetailMode: boolean): Vec3[] {
  const rows = clampInt(Math.round(Math.sqrt(panel.rows + 1)), 2, lowDetailMode ? 4 : 6)
  const cols = clampInt(Math.round(Math.sqrt(panel.cols + 1)), 2, lowDetailMode ? 4 : 6)
  const points: Vec3[] = []
  const x = panel.width * 0.51
  const ySpan = panel.height * 0.72
  const zSpan = panel.depth * 0.64

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const y = rows === 1 ? 0 : ySpan * (0.5 - row / (rows - 1))
      const z = cols === 1 ? 0 : zSpan * (col / (cols - 1) - 0.5)
      points.push([x, y, z])
    }
  }

  return points
}

function createPanelHighlightRect(
  panel: CnnMatrixPanel,
  highlight: PanelHighlight | null
): { position: Vec3; width: number; height: number; color: string } | null {
  if (!highlight) return null
  const safeRows = Math.max(1, panel.rows)
  const safeCols = Math.max(1, panel.cols)
  const cellHeight = panel.height / safeRows
  const cellWidth = panel.depth / safeCols
  const width = Math.max(0.04, highlight.cols * cellWidth)
  const height = Math.max(0.04, highlight.rows * cellHeight)
  const y = panel.height * 0.5 - (highlight.row + highlight.rows * 0.5) * cellHeight
  const z = -panel.depth * 0.5 + (highlight.col + highlight.cols * 0.5) * cellWidth
  return {
    position: [panel.width * 0.512, y, z],
    width,
    height,
    color: highlight.color,
  }
}

function createCnnChainEdges(panels: CnnMatrixPanel[], lowDetailMode: boolean): VisualEdge[] {
  if (panels.length < 2) return []
  const edges: VisualEdge[] = []
  const fanout = lowDetailMode ? 2 : 3

  for (let pairIndex = 0; pairIndex < panels.length - 1; pairIndex += 1) {
    const source = panels[pairIndex]
    const target = panels[pairIndex + 1]
    const sourcePoints = samplePanelConnectionPoints(source, 'right', lowDetailMode)
    const targetPoints = samplePanelConnectionPoints(target, 'left', lowDetailMode)
    if (sourcePoints.length === 0 || targetPoints.length === 0) continue

    sourcePoints.forEach((sourcePoint, sourceIndex) => {
      for (let branch = 0; branch < fanout; branch += 1) {
        const targetIndex = (sourceIndex * fanout + branch * 2) % targetPoints.length
        const targetPoint = targetPoints[targetIndex]
        const weight = 0.35 + Math.abs(seededSigned(`chain:${source.id}:${target.id}:${sourceIndex}:${branch}`)) * 0.65
        edges.push({
          id: `cnn-chain:${pairIndex}:${source.id}:${target.id}:${sourceIndex}:${branch}`,
          from: sourcePoint,
          to: targetPoint,
          weight,
          sourceStageId: source.stageId,
          targetStageId: target.stageId,
          kind: 'stage_flow',
        })
      }
    })
  }

  return edges
}

function samplePanelConnectionPoints(
  panel: CnnMatrixPanel,
  side: 'left' | 'right',
  lowDetailMode: boolean
): Vec3[] {
  const points: Vec3[] = []
  const isInput = panel.id === 'cnn_input_matrix'
  const rows = lowDetailMode ? 3 : 4
  const cols = lowDetailMode ? 3 : 4
  const xSpan = isInput ? panel.depth : panel.width
  const zSpan = isInput ? panel.width : panel.depth
  const x =
    panel.position[0] +
    (side === 'right' ? 1 : -1) * xSpan * 0.5

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const y = panel.position[1] + panel.height * 0.68 * (0.5 - row / (rows - 1))
      const z = panel.position[2] + zSpan * 0.64 * (col / (cols - 1) - 0.5)
      points.push([x, y, z])
    }
  }

  return points
}

interface CnnSymbolicSceneProps {
  op: CnnConvolutionOp
  sourcePanel: CnnMatrixPanel
  targetPanel: CnnMatrixPanel
  sourceHighlight: PanelHighlight | null
  targetHighlight: PanelHighlight | null
  selectedStageId: string | null
  lowDetailMode: boolean
  activityPulse: number
  onSelectStage: (stageId: string) => void
}

function CnnSymbolicScene({
  op,
  sourcePanel,
  targetPanel,
  sourceHighlight,
  targetHighlight,
  selectedStageId,
  lowDetailMode,
  activityPulse,
  onSelectStage,
}: CnnSymbolicSceneProps) {
  const sourcePosition: Vec3 = [-9.2, -0.2, 0]
  const kernelPosition: Vec3 = [-5.8, 1.1, 0.15]
  const outputPosition: Vec3 = [-2.1, -0.2, 0.2]

  const sourceRight: Vec3 = [sourcePosition[0] + 1.62, sourcePosition[1], sourcePosition[2] + 0.05]
  const kernelLeft: Vec3 = [kernelPosition[0] - 0.55, kernelPosition[1] - 0.08, kernelPosition[2]]
  const kernelRight: Vec3 = [kernelPosition[0] + 0.55, kernelPosition[1] - 0.08, kernelPosition[2]]
  const outputLeft: Vec3 = [outputPosition[0] - 1.62, outputPosition[1], outputPosition[2] + 0.05]

  const focused = selectedStageId === null || selectedStageId === op.stageId
  const connectorColor = focused ? '#ffd58a' : '#88a7c8'
  const connectorOpacity = clampOpacity((focused ? 0.84 : 0.52) + activityPulse * 0.08)

  const leftCurve = useMemo(
    () => new THREE.LineCurve3(
      new THREE.Vector3(sourceRight[0], sourceRight[1], sourceRight[2]),
      new THREE.Vector3(kernelLeft[0], kernelLeft[1], kernelLeft[2])
    ),
    [sourceRight, kernelLeft]
  )
  const rightCurve = useMemo(
    () => new THREE.LineCurve3(
      new THREE.Vector3(kernelRight[0], kernelRight[1], kernelRight[2]),
      new THREE.Vector3(outputLeft[0], outputLeft[1], outputLeft[2])
    ),
    [kernelRight, outputLeft]
  )

  return (
    <group>
      <CnnVolumePanel
        panel={sourcePanel}
        position={sourcePosition}
        highlight={sourceHighlight}
        selected={selectedStageId === sourcePanel.stageId}
        lowDetailMode={lowDetailMode}
        onClick={() => onSelectStage(sourcePanel.stageId)}
      />

      <CnnKernelCube
        op={op}
        position={kernelPosition}
        selected={selectedStageId === op.stageId}
        lowDetailMode={lowDetailMode}
        onClick={() => onSelectStage(op.stageId)}
      />

      <CnnVolumePanel
        panel={targetPanel}
        position={outputPosition}
        highlight={targetHighlight}
        selected={selectedStageId === targetPanel.stageId}
        lowDetailMode={lowDetailMode}
        onClick={() => onSelectStage(targetPanel.stageId)}
      />

      <Line
        points={[sourceRight, kernelLeft]}
        color={connectorColor}
        lineWidth={focused ? 2.2 : 1.35}
        transparent
        opacity={connectorOpacity}
        depthWrite={false}
        toneMapped={false}
      />
      <Line
        points={[kernelRight, outputLeft]}
        color={connectorColor}
        lineWidth={focused ? 2.2 : 1.35}
        transparent
        opacity={connectorOpacity}
        depthWrite={false}
        toneMapped={false}
      />

      <WeightVisual
        curve={leftCurve}
        speed={lowDetailMode ? 0.42 : 0.58}
        color="#ffe7bb"
        size={0.056}
        opacity={0.95}
        offset={0}
      />
      <WeightVisual
        curve={rightCurve}
        speed={lowDetailMode ? 0.45 : 0.62}
        color="#ffe7bb"
        size={0.056}
        opacity={0.95}
        offset={0.45}
      />
    </group>
  )
}

interface CnnVolumePanelProps {
  panel: CnnMatrixPanel
  position: Vec3
  highlight: PanelHighlight | null
  selected: boolean
  lowDetailMode: boolean
  onClick: () => void
}

function CnnVolumePanel({
  panel,
  position,
  highlight,
  selected,
  lowDetailMode,
  onClick,
}: CnnVolumePanelProps) {
  const [hovered, setHovered] = useState(false)
  const visualCellSize = lowDetailMode ? 0.115 : 0.13
  const gridWidth = panel.cols * visualCellSize
  const gridHeight = panel.rows * visualCellSize
  const panelWidth = Math.max(1.35, gridWidth + 0.38)
  const panelHeight = Math.max(1.15, gridHeight + 0.34)
  const isFocused = selected || hovered
  const depthSlices = clampInt(panel.sliceCount, 1, lowDetailMode ? 3 : 5)
  const sliceDepth = lowDetailMode ? 0.1 : 0.13

  const cells = useMemo(() => {
    const values: Array<{ position: Vec3; color: string }> = []
    for (let row = 0; row < panel.rows; row += 1) {
      for (let col = 0; col < panel.cols; col += 1) {
        const signal = seededSigned(`vol:${panel.id}:${row}:${col}`)
        const x = -gridWidth / 2 + visualCellSize / 2 + col * visualCellSize
        const y = gridHeight / 2 - visualCellSize / 2 - row * visualCellSize
        values.push({
          position: [x, y, 0.02],
          color: matrixCellColor(panel.color, signal, isFocused),
        })
      }
    }
    return values
  }, [panel.id, panel.color, panel.rows, panel.cols, gridWidth, gridHeight, visualCellSize, isFocused])

  const gridLines = useMemo(
    () => buildMatrixGridLinesWithCellSize(panel.rows, panel.cols, gridWidth, gridHeight, visualCellSize),
    [panel.rows, panel.cols, gridWidth, gridHeight, visualCellSize]
  )

  const highlightBox = useMemo(() => {
    if (!highlight) return null
    const width = highlight.cols * visualCellSize
    const height = highlight.rows * visualCellSize
    const x = -gridWidth / 2 + width / 2 + highlight.col * visualCellSize
    const y = gridHeight / 2 - height / 2 - highlight.row * visualCellSize
    return { x, y, width, height, color: highlight.color }
  }, [highlight, gridWidth, gridHeight, visualCellSize])

  return (
    <group position={position}>
      {Array.from({ length: depthSlices }).map((_, idx) => {
        const z = -(0.06 + idx * sliceDepth)
        const x = -idx * 0.05
        const opacity = idx === 0 ? 0.86 : Math.max(0.14, 0.42 - idx * 0.08)
        return (
          <mesh
            key={`${panel.id}-slice-${idx}`}
            position={[x, 0, z]}
            raycast={idx === 0 ? undefined : IGNORE_RAYCAST}
            renderOrder={1 + idx}
          >
            <planeGeometry args={[panelWidth, panelHeight]} />
            <meshStandardMaterial
              color="#071322"
              emissive={panel.color}
              emissiveIntensity={idx === 0 ? (isFocused ? 0.24 : 0.12) : 0.08}
              transparent
              opacity={opacity}
              roughness={0.52}
              metalness={0.08}
              side={THREE.DoubleSide}
              depthWrite={false}
            />
          </mesh>
        )
      })}

      <lineSegments raycast={IGNORE_RAYCAST}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[gridLines, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#9abbdc" transparent opacity={0.38} depthWrite={false} />
      </lineSegments>

      {cells.map((cell, index) => (
        <mesh
          key={`${panel.id}-value-${index}`}
          position={cell.position}
          raycast={IGNORE_RAYCAST}
          renderOrder={24}
        >
          <boxGeometry args={[visualCellSize * 0.78, visualCellSize * 0.78, lowDetailMode ? 0.028 : 0.036]} />
          <meshStandardMaterial
            color={cell.color}
            emissive={cell.color}
            emissiveIntensity={0.18}
            roughness={0.42}
            metalness={0.08}
          />
        </mesh>
      ))}

      {highlightBox ? (
        <mesh position={[highlightBox.x, highlightBox.y, 0.06]} raycast={IGNORE_RAYCAST} renderOrder={26}>
          <planeGeometry args={[highlightBox.width, highlightBox.height]} />
          <meshBasicMaterial color={highlightBox.color} transparent opacity={0.22} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      ) : null}

      {highlightBox ? (
        <mesh position={[highlightBox.x, highlightBox.y, 0.07]} raycast={IGNORE_RAYCAST} renderOrder={27}>
          <planeGeometry args={[highlightBox.width, highlightBox.height]} />
          <meshBasicMaterial color={highlightBox.color} wireframe transparent opacity={0.96} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      ) : null}

      <mesh
        onPointerOver={(event) => {
          event.stopPropagation()
          setHovered(true)
        }}
        onPointerOut={(event) => {
          event.stopPropagation()
          setHovered(false)
        }}
        onClick={(event) => {
          event.stopPropagation()
          onClick()
        }}
      >
        <planeGeometry args={[panelWidth + 0.2, panelHeight + 0.2]} />
        <meshBasicMaterial transparent opacity={0} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>

      <mesh raycast={IGNORE_RAYCAST} renderOrder={28}>
        <planeGeometry args={[panelWidth * 1.01, panelHeight * 1.01]} />
        <meshBasicMaterial
          color={isFocused ? '#f5fbff' : '#99b8d9'}
          wireframe
          transparent
          opacity={isFocused ? 0.52 : 0.25}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  )
}

interface CnnKernelCubeProps {
  op: CnnConvolutionOp
  position: Vec3
  selected: boolean
  lowDetailMode: boolean
  onClick: () => void
}

function CnnKernelCube({ op, position, selected, lowDetailMode, onClick }: CnnKernelCubeProps) {
  const [hovered, setHovered] = useState(false)
  const size = clampInt(op.kernelSize, 1, lowDetailMode ? 4 : 5)
  const depth = lowDetailMode ? 2 : 3
  const cell = lowDetailMode ? 0.11 : 0.12
  const isFocused = selected || hovered

  const cubes = useMemo(() => {
    const values: Array<{ position: Vec3; color: string }> = []
    for (let z = 0; z < depth; z += 1) {
      for (let row = 0; row < size; row += 1) {
        for (let col = 0; col < size; col += 1) {
          const weight = seededSigned(`kernel-cube:${op.id}:${z}:${row}:${col}`)
          const x = (col - (size - 1) / 2) * cell
          const y = ((size - 1) / 2 - row) * cell
          const zz = (z - (depth - 1) / 2) * cell
          values.push({
            position: [x, y, zz],
            color: weightToKernelColor(weight),
          })
        }
      }
    }
    return values
  }, [op.id, depth, size, cell])

  return (
    <group position={position}>
      <mesh
        onPointerOver={(event) => {
          event.stopPropagation()
          setHovered(true)
        }}
        onPointerOut={(event) => {
          event.stopPropagation()
          setHovered(false)
        }}
        onClick={(event) => {
          event.stopPropagation()
          onClick()
        }}
      >
        <boxGeometry args={[size * cell + 0.3, size * cell + 0.3, depth * cell + 0.3]} />
        <meshStandardMaterial
          color="#120d08"
          emissive="#ffcf8b"
          emissiveIntensity={isFocused ? 0.26 : 0.12}
          transparent
          opacity={0.36}
          roughness={0.36}
          metalness={0.08}
        />
      </mesh>

      {cubes.map((cube, index) => (
        <mesh key={`${op.id}-cube-${index}`} position={cube.position} raycast={IGNORE_RAYCAST}>
          <boxGeometry args={[cell * 0.7, cell * 0.7, cell * 0.7]} />
          <meshStandardMaterial color={cube.color} emissive={cube.color} emissiveIntensity={0.16} roughness={0.4} metalness={0.1} />
        </mesh>
      ))}
    </group>
  )
}

interface EdgeSegmentsProps {
  edges: VisualEdge[]
  color: string
  opacity: number
}

function EdgeSegments({ edges, color, opacity }: EdgeSegmentsProps) {
  const segmentPositions = useMemo(() => buildSegmentPositions(edges), [edges])
  if (segmentPositions.length === 0) return null

  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[segmentPositions, 3]} />
      </bufferGeometry>
      <lineBasicMaterial
        color={color}
        transparent
        opacity={clampOpacity(opacity)}
        depthWrite={false}
      />
    </lineSegments>
  )
}

interface WeightedEdgeSegmentsProps {
  edges: VisualEdge[]
  lowColor: string
  highColor: string
  opacity: number
}

function WeightedEdgeSegments({ edges, lowColor, highColor, opacity }: WeightedEdgeSegmentsProps) {
  const buffers = useMemo(
    () => buildWeightedSegmentBuffers(edges, lowColor, highColor),
    [edges, lowColor, highColor]
  )
  if (buffers.positions.length === 0) return null

  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[buffers.positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[buffers.colors, 3]} />
      </bufferGeometry>
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={clampOpacity(opacity)}
        depthWrite={false}
      />
    </lineSegments>
  )
}

interface TransformerHeadStripProps {
  count: number
  lowDetailMode: boolean
}

function TransformerHeadStrip({ count, lowDetailMode }: TransformerHeadStripProps) {
  const cubes = useMemo(() => {
    const safeCount = clampInt(count, 2, 14)
    const values: Array<{ position: Vec3; color: string }> = []
    const colors = ['#4ea1ff', '#ffb429', '#8ad96f', '#ff6f6f', '#b58cff', '#66d6ff', '#ffd37a', '#7da0ff']
    const width = safeCount * (lowDetailMode ? 0.28 : 0.32)
    for (let index = 0; index < safeCount; index += 1) {
      values.push({
        position: [7.35 - width / 2 + index * (lowDetailMode ? 0.28 : 0.32), 3.02, -0.08],
        color: colors[index % colors.length],
      })
    }
    return values
  }, [count, lowDetailMode])

  return (
    <group>
      {cubes.map((cube, index) => (
        <mesh key={`head-cube-${index}`} position={cube.position} raycast={IGNORE_RAYCAST}>
          <boxGeometry args={[lowDetailMode ? 0.22 : 0.25, 0.13, 0.12]} />
          <meshStandardMaterial
            color={cube.color}
            emissive={cube.color}
            emissiveIntensity={0.22}
            roughness={0.4}
            metalness={0.08}
          />
        </mesh>
      ))}
    </group>
  )
}

interface TokenNodeProps {
  node: VisualNode
  selected: boolean
  active: boolean
  activation: number
  lowDetailMode: boolean
  outlineColor: string
  onSelectStage: (stageId: string) => void
}

function TokenNode({
  node,
  selected,
  active,
  activation,
  lowDetailMode,
  outlineColor,
  onSelectStage,
}: TokenNodeProps) {
  const [hovered, setHovered] = useState(false)
  const segments = getSphereSegments(node, lowDetailMode)
  const outlineOpacity = active ? 0.95 : selected ? 0.86 : hovered ? 0.72 : 0.48
  const isStageNode = node.kind === 'stage'
  const baseColor = isStageNode ? '#5a5a5a' : '#6f6f6f'
  const baseEmissive = isStageNode ? '#1f1f1f' : '#191919'
  const activationClamped = THREE.MathUtils.clamp(activation, 0, 1)
  const coreIntensity = isStageNode
    ? (active ? 0.74 : selected || hovered ? 0.52 : 0.3)
    : (0.18 + activationClamped * 0.72 + (selected || hovered ? 0.1 : 0))

  return (
    <group position={node.position}>
      <mesh
        onPointerOver={(event) => {
          event.stopPropagation()
          setHovered(true)
        }}
        onPointerOut={(event) => {
          event.stopPropagation()
          setHovered(false)
        }}
        onClick={(event) => {
          event.stopPropagation()
          onSelectStage(node.stageId)
        }}
      >
        <sphereGeometry args={[node.radius, segments, segments]} />
        <meshStandardMaterial
          color={baseColor}
          emissive={baseEmissive}
          emissiveIntensity={coreIntensity}
          roughness={0.55}
          metalness={0.08}
        />
      </mesh>

      <mesh raycast={IGNORE_RAYCAST} scale={[1.2, 1.2, 1.2]}>
        <sphereGeometry args={[node.radius, segments, segments]} />
        <meshBasicMaterial
          color={active ? '#ffcb67' : outlineColor}
          transparent
          opacity={THREE.MathUtils.clamp(outlineOpacity + activationClamped * 0.18, 0.08, 0.98)}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </mesh>

      {active ? (
        <mesh raycast={IGNORE_RAYCAST}>
          <ringGeometry args={[node.radius * 1.3, node.radius * (isStageNode ? 1.92 : 1.66), 24]} />
          <meshBasicMaterial
            color={isStageNode ? outlineColor : '#ffe18a'}
            transparent
            opacity={0.86}
            side={THREE.DoubleSide}
          />
        </mesh>
      ) : null}
    </group>
  )
}

function createCnnMatrixPipeline(
  architecture: VLMArchitectureSpec,
  roles: StageRoles,
  colorMap: Map<string, string>,
  lowDetailMode: boolean
): CnnPipeline {
  const maxCells = lowDetailMode ? MAX_MATRIX_CELLS_LOW : MAX_MATRIX_CELLS_HIGH
  const input = architecture.blueprint.input

  const rawPanels: Array<Omit<CnnMatrixPanel, 'position' | 'width' | 'height' | 'depth'>> = []
  const rawOps: Array<Omit<CnnConvolutionOp, 'kernelPosition'>> = []

  const inputGrid = reduceMatrixGrid(input.height, input.width, maxCells)
  rawPanels.push({
    id: 'cnn_input_matrix',
    stageId: roles.input,
    label: 'Input Matrix',
    detail: `${input.height}x${input.width}`,
    color: colorMap.get(roles.input) ?? '#54baff',
    trueRows: input.height,
    trueCols: input.width,
    rows: inputGrid.rows,
    cols: inputGrid.cols,
    channels: Math.max(1, input.channels),
    sliceCount: 3,
  })

  let previousPanelId = rawPanels[0].id
  let currentHeight = Math.max(1, input.height)
  let currentWidth = Math.max(1, input.width)
  let currentChannels = Math.max(1, input.channels)

  architecture.blueprint.cnn.stages.forEach((stage, index) => {
    const stride = Math.max(1, Math.floor(stage.stride ?? 1))
    const kernelSize = clampInt(Math.floor(stage.kernel_size ?? 3), 1, 9)

    const outHeight = Math.max(1, Math.floor(currentHeight / stride))
    const outWidth = Math.max(1, Math.floor(currentWidth / stride))
    const outChannels = Math.max(1, stage.out_channels)

    const stageId = resolveCnnStageId(architecture.stages, stage, index, roles)
    const grid = reduceMatrixGrid(outHeight, outWidth, maxCells)

    const panelId = `cnn_${stage.id}_${index}`
    rawPanels.push({
      id: panelId,
      stageId,
      label: stage.label,
      detail: `${outHeight}x${outWidth} · ${outChannels}ch`,
      color:
        colorMap.get(stageId) ??
        (index === 0 ? colorMap.get(roles.preprocess) : colorMap.get(roles.conv)) ??
        '#7cf2c4',
      trueRows: outHeight,
      trueCols: outWidth,
      rows: grid.rows,
      cols: grid.cols,
      channels: outChannels,
      sliceCount: lowDetailMode
        ? clampInt(Math.round(Math.log2(outChannels + 1)), 1, 3)
        : clampInt(Math.round(Math.log2(outChannels + 1)), 1, 6),
    })

    rawOps.push({
      id: `conv_op_${stage.id}_${index}`,
      stageId,
      sourcePanelId: previousPanelId,
      targetPanelId: panelId,
      kernelSize,
      stride,
    })

    previousPanelId = panelId
    currentHeight = outHeight
    currentWidth = outWidth
    currentChannels = outChannels
  })

  const representedStages = new Set(rawPanels.map((panel) => panel.stageId))
  const tailStages = architecture.stages.filter((stage) => !representedStages.has(stage.id))
  tailStages.forEach((stage, index) => {
    const stageIdLower = stage.id.toLowerCase()
    const isHeadLike =
      stageIdLower.includes('head') ||
      stageIdLower.includes('output') ||
      stageIdLower.includes('prediction')
    const isTransformerLike =
      stageIdLower.includes('transformer') ||
      stageIdLower.includes('decoder') ||
      stageIdLower.includes('encoder') ||
      stageIdLower.includes('query')

    currentHeight = Math.max(1, Math.floor(currentHeight * (isHeadLike ? 0.5 : isTransformerLike ? 0.72 : 0.8)))
    currentWidth = Math.max(1, Math.floor(currentWidth * (isHeadLike ? 0.5 : isTransformerLike ? 0.72 : 0.8)))
    currentChannels = Math.max(1, Math.floor(currentChannels * (isHeadLike ? 0.45 : 0.8)))

    const grid = reduceMatrixGrid(currentHeight, currentWidth, maxCells)
    const panelId = `tail_${stage.id}_${index}`

    rawPanels.push({
      id: panelId,
      stageId: stage.id,
      label: stage.label,
      detail: stage.detail,
      color: colorMap.get(stage.id) ?? stage.color ?? '#ffd37a',
      trueRows: currentHeight,
      trueCols: currentWidth,
      rows: grid.rows,
      cols: grid.cols,
      channels: currentChannels,
      sliceCount: lowDetailMode
        ? clampInt(Math.round(Math.log2(currentChannels + 1)), 1, 3)
        : clampInt(Math.round(Math.log2(currentChannels + 1)), 1, 5),
    })

    rawOps.push({
      id: `tail_op_${stage.id}_${index}`,
      stageId: stage.id,
      sourcePanelId: previousPanelId,
      targetPanelId: panelId,
      kernelSize: 1,
      stride: 1,
    })
    previousPanelId = panelId
  })

  const maxSpatial = Math.max(
    1,
    ...rawPanels.map((panel) => Math.sqrt(panel.trueRows * panel.trueCols))
  )
  const panelShells = rawPanels.map((panel) => {
    const [width, height, depth] = matrixPanelSize(panel, maxSpatial, lowDetailMode)
    return { panel, width, height, depth }
  })
  const gap = lowDetailMode ? 1.02 : 1.18
  let cursorX = 0
  const placed = panelShells.map((entry) => {
    const centerX = cursorX + entry.width * 0.5
    cursorX += entry.width + gap
    return {
      ...entry,
      centerX,
    }
  })
  const firstCenter = placed[0]?.centerX ?? 0
  const lastCenter = placed[placed.length - 1]?.centerX ?? 0
  const shift = -1.3 - (firstCenter + lastCenter) * 0.5

  const panels: CnnMatrixPanel[] = placed.map((entry) => ({
    ...entry.panel,
    width: entry.width,
    height: entry.height,
    depth: entry.depth,
    position: [entry.centerX + shift, 0, 0],
  }))

  const panelLookup = new Map(panels.map((panel) => [panel.id, panel]))

  const ops: CnnConvolutionOp[] = rawOps.map((op) => {
    const source = panelLookup.get(op.sourcePanelId)
    const target = panelLookup.get(op.targetPanelId)
    const kernelX =
      source && target ? (source.position[0] + target.position[0]) / 2 : 0

    return {
      ...op,
      kernelPosition: [kernelX, 1.38, 0],
    }
  })

  return { panels, ops }
}

function resolveCnnStageId(
  stages: VLMArchitectureStage[],
  stage: VLMCnnStageBlueprint,
  index: number,
  roles: StageRoles
): string {
  const stageIdNorm = stage.id.toLowerCase()
  const labelTokens = tokenize(stage.label)

  const direct = stages.find((entry) => {
    const idNorm = entry.id.toLowerCase()
    if (idNorm.includes(stageIdNorm) || stageIdNorm.includes(idNorm)) {
      return true
    }
    return labelTokens.some((token) => idNorm.includes(token))
  })

  if (direct) return direct.id
  if (index === 0) return roles.preprocess
  return roles.conv
}

function reduceMatrixGrid(rows: number, cols: number, maxCells: number): { rows: number; cols: number } {
  const safeRows = Math.max(1, rows)
  const safeCols = Math.max(1, cols)
  const count = safeRows * safeCols

  if (count <= maxCells) {
    return {
      rows: clampInt(safeRows, 2, 16),
      cols: clampInt(safeCols, 2, 16),
    }
  }

  const scale = Math.sqrt(maxCells / count)
  let nextRows = clampInt(Math.floor(safeRows * scale), 2, 16)
  let nextCols = clampInt(Math.floor(safeCols * scale), 2, 16)

  while (nextRows * nextCols > maxCells) {
    if (nextRows >= nextCols && nextRows > 2) {
      nextRows -= 1
      continue
    }
    if (nextCols > 2) {
      nextCols -= 1
      continue
    }
    break
  }

  return { rows: nextRows, cols: nextCols }
}

function matrixPanelSize(
  panel: Omit<CnnMatrixPanel, 'position' | 'width' | 'height' | 'depth'>,
  maxSpatial: number,
  lowDetailMode: boolean
): [number, number, number] {
  const isInput = panel.id === 'cnn_input_matrix'
  const spatial = Math.sqrt(Math.max(1, panel.trueRows * panel.trueCols))
  const ratio = THREE.MathUtils.clamp(spatial / maxSpatial, 0.08, 1)
  const squareSize = (lowDetailMode ? 0.96 : 1.08) + ratio * (lowDetailMode ? 2.25 : 2.65)
  const depth = isInput
    ? (lowDetailMode ? 0.34 : 0.4)
    : 0.52 + Math.min(lowDetailMode ? 1.15 : 1.5, Math.log2(panel.channels + 1) * (lowDetailMode ? 0.17 : 0.2))
  const width = isInput ? (lowDetailMode ? 0.3 : 0.34) : squareSize
  const height = squareSize
  return [width, height, depth]
}

function createPanelHighlights(
  activeOp: CnnConvolutionOp | null,
  panelLookup: Map<string, CnnMatrixPanel>,
  scanStep: number
): Map<string, PanelHighlight> {
  const result = new Map<string, PanelHighlight>()
  if (!activeOp) return result

  const source = panelLookup.get(activeOp.sourcePanelId)
  const target = panelLookup.get(activeOp.targetPanelId)
  if (!source || !target) return result

  const windowRows = clampInt(activeOp.kernelSize, 1, source.rows)
  const windowCols = clampInt(activeOp.kernelSize, 1, source.cols)

  const availableRows = Math.max(1, source.rows - windowRows + 1)
  const availableCols = Math.max(1, source.cols - windowCols + 1)
  const totalWindows = availableRows * availableCols

  const index = scanStep % totalWindows
  const sourceRow = Math.floor(index / availableCols)
  const sourceCol = index % availableCols

  result.set(source.id, {
    row: sourceRow,
    col: sourceCol,
    rows: windowRows,
    cols: windowCols,
    color: '#ffd87b',
  })

  const targetRow =
    availableRows > 1
      ? Math.round((sourceRow / (availableRows - 1)) * Math.max(0, target.rows - 1))
      : Math.floor(target.rows / 2)
  const targetCol =
    availableCols > 1
      ? Math.round((sourceCol / (availableCols - 1)) * Math.max(0, target.cols - 1))
      : Math.floor(target.cols / 2)

  result.set(target.id, {
    row: clampInt(targetRow, 0, Math.max(0, target.rows - 1)),
    col: clampInt(targetCol, 0, Math.max(0, target.cols - 1)),
    rows: 1,
    cols: 1,
    color: '#78ffc1',
  })

  return result
}

function matrixCellColor(baseColor: string, value: number, active: boolean): string {
  const warm = new THREE.Color('#ffb56a')
  const cool = new THREE.Color('#62a0ff')
  const base = new THREE.Color(baseColor)
  const dark = new THREE.Color('#071320')

  const signal = value >= 0
    ? warm.clone().lerp(base, 0.35)
    : cool.clone().lerp(base, 0.35)

  const magnitude = Math.abs(value)
  const boost = active ? 0.1 : 0
  const color = dark.clone().lerp(signal, 0.24 + magnitude * 0.66 + boost)
  return color.getStyle()
}

function weightToKernelColor(value: number): string {
  const negative = new THREE.Color('#64a5ff')
  const positive = new THREE.Color('#ffb06b')
  const neutral = new THREE.Color('#111a28')

  const source = value >= 0 ? positive : negative
  const magnitude = Math.min(1, Math.abs(value))
  return neutral.clone().lerp(source, 0.2 + magnitude * 0.78).getStyle()
}

function buildMatrixGridLinesWithCellSize(
  rows: number,
  cols: number,
  gridWidth: number,
  gridHeight: number,
  cellSize: number
): Float32Array {
  const values: number[] = []
  const left = -gridWidth / 2
  const right = gridWidth / 2
  const top = gridHeight / 2
  const bottom = -gridHeight / 2

  for (let row = 0; row <= rows; row += 1) {
    const y = top - row * cellSize
    values.push(left, y, 0.032, right, y, 0.032)
  }

  for (let col = 0; col <= cols; col += 1) {
    const x = left + col * cellSize
    values.push(x, top, 0.032, x, bottom, 0.032)
  }

  return new Float32Array(values)
}

function createTransformerRailGraph(
  stageId: string,
  count: number,
  lowDetailMode: boolean
): TransformerRailGraph {
  const tokenCount = clampInt(count, 2, lowDetailMode ? 24 : 40)
  const spacing = lowDetailMode ? 0.24 : 0.27
  const radius = lowDetailMode ? 0.08 : 0.092
  const depthSpread = lowDetailMode ? 0.14 : 0.2
  const leftTokens: VisualNode[] = []
  const rightTokens: VisualNode[] = []
  const railEdges: VisualEdge[] = []
  const alignmentEdges: VisualEdge[] = []

  for (let index = 0; index < tokenCount; index += 1) {
    const y = TRANSFORMER_TOP_Y - index * spacing
    const jitter = seededSigned(`transformer-jitter:${index}`) * depthSpread
    const leftNode: VisualNode = {
      id: `tok_left_${index}`,
      stageId,
      kind: 'token',
      radius,
      segments: lowDetailMode ? 7 : 9,
      position: [TRANSFORMER_LEFT_X, y, -0.86 + jitter],
    }
    const rightNode: VisualNode = {
      id: `tok_right_${index}`,
      stageId,
      kind: 'token',
      radius,
      segments: lowDetailMode ? 7 : 9,
      position: [TRANSFORMER_RIGHT_X, y, 0.86 - jitter * 0.65],
    }
    leftTokens.push(leftNode)
    rightTokens.push(rightNode)
  }

  for (let index = 0; index < tokenCount; index += 1) {
    const left = leftTokens[index]
    const right = rightTokens[index]
    alignmentEdges.push({
      id: `align:${left.id}:${right.id}`,
      from: left.position,
      to: right.position,
      weight: 0.28 + Math.abs(seededSigned(`align:${index}`)) * 0.18,
      sourceStageId: stageId,
      targetStageId: stageId,
      kind: 'token_chain',
    })
  }

  for (let index = 0; index < tokenCount - 1; index += 1) {
    const leftA = leftTokens[index]
    const leftB = leftTokens[index + 1]
    const rightA = rightTokens[index]
    const rightB = rightTokens[index + 1]
    railEdges.push({
      id: `rail-left:${leftA.id}:${leftB.id}`,
      from: leftA.position,
      to: leftB.position,
      weight: 0.26,
      sourceStageId: stageId,
      targetStageId: stageId,
      kind: 'token_chain',
    })
    railEdges.push({
      id: `rail-right:${rightA.id}:${rightB.id}`,
      from: rightA.position,
      to: rightB.position,
      weight: 0.26,
      sourceStageId: stageId,
      targetStageId: stageId,
      kind: 'token_chain',
    })
  }

  return { leftTokens, rightTokens, railEdges, alignmentEdges }
}

function createAttentionState(
  sourceTokens: VisualNode[],
  targetTokens: VisualNode[],
  architecture: VLMArchitectureSpec,
  step: number,
  lowDetailMode: boolean
): AttentionState {
  const activeTokenIds = new Set<string>()
  const activationById = new Map<string, number>()
  if (sourceTokens.length <= 1 || targetTokens.length <= 1) {
    return { edges: [], activeTokenIds, activationById }
  }

  const configuredHeads = Math.max(1, architecture.blueprint.transformer.attention_heads)
  const headCount = lowDetailMode
    ? Math.max(1, Math.min(4, Math.ceil(configuredHeads / 2)))
    : configuredHeads
  const targetsPerHead = lowDetailMode ? 3 : 5
  const edges: VisualEdge[] = []

  for (let head = 0; head < headCount; head += 1) {
    const sourceIndex = (step * (head + 2) + head * 13) % sourceTokens.length
    const source = sourceTokens[sourceIndex]
    activeTokenIds.add(source.id)
    activationById.set(source.id, Math.max(activationById.get(source.id) ?? 0, 0.74))

    const candidates = targetTokens
      .map((target, targetIndex) => ({
        target,
        score: dynamicAttentionScore(source, target, head, step, targetIndex),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.min(targetsPerHead, targetTokens.length))

    candidates.forEach((entry, rank) => {
      activeTokenIds.add(entry.target.id)
      const targetActivation = entry.score * (1 - rank * 0.1)
      activationById.set(
        entry.target.id,
        Math.max(activationById.get(entry.target.id) ?? 0, targetActivation)
      )
      edges.push({
        id: `att:h${head}:${source.id}:${entry.target.id}`,
        from: source.position,
        to: entry.target.position,
        weight: targetActivation,
        sourceStageId: source.stageId,
        targetStageId: entry.target.stageId,
        kind: 'attention',
      })
    })
  }

  return { edges, activeTokenIds, activationById }
}

function dynamicAttentionScore(
  source: VisualNode,
  target: VisualNode,
  head: number,
  step: number,
  targetIndex: number
): number {
  const structural = attentionWeight(source.id, target.id)
  const phase = step * 0.22 + head * 1.31 + targetIndex * 0.77
  const oscillation = 0.5 + 0.5 * Math.sin(phase)
  const contrast = 0.5 + 0.5 * Math.cos(phase * 0.63 + structural * Math.PI)
  return THREE.MathUtils.clamp(
    structural * 0.35 + oscillation * 0.45 + contrast * 0.2,
    0.05,
    1
  )
}

function splitEdgesBySelection(
  edges: VisualEdge[],
  selectedStageId: string | null
): { selected: VisualEdge[]; rest: VisualEdge[] } {
  if (!selectedStageId) {
    return { selected: [], rest: edges }
  }

  const selected: VisualEdge[] = []
  const rest: VisualEdge[] = []
  edges.forEach((edge) => {
    if (edge.sourceStageId === selectedStageId || edge.targetStageId === selectedStageId) {
      selected.push(edge)
    } else {
      rest.push(edge)
    }
  })
  return { selected, rest }
}

function buildSegmentPositions(edges: VisualEdge[]): Float32Array {
  if (edges.length === 0) return new Float32Array()
  const values: number[] = []
  for (const edge of edges) {
    values.push(
      edge.from[0], edge.from[1], edge.from[2],
      edge.to[0], edge.to[1], edge.to[2]
    )
  }
  return new Float32Array(values)
}

function buildWeightedSegmentBuffers(
  edges: VisualEdge[],
  lowColor: string,
  highColor: string
): { positions: Float32Array; colors: Float32Array } {
  if (edges.length === 0) {
    return { positions: new Float32Array(), colors: new Float32Array() }
  }

  const positions: number[] = []
  const colors: number[] = []
  const low = new THREE.Color(lowColor)
  const high = new THREE.Color(highColor)

  for (const edge of edges) {
    const weight = THREE.MathUtils.clamp(edge.weight, 0, 1)
    const color = low.clone().lerp(high, weight)
    positions.push(
      edge.from[0], edge.from[1], edge.from[2],
      edge.to[0], edge.to[1], edge.to[2]
    )
    colors.push(
      color.r, color.g, color.b,
      color.r, color.g, color.b
    )
  }

  return {
    positions: new Float32Array(positions),
    colors: new Float32Array(colors),
  }
}

function createStageFlowGraph(architecture: VLMArchitectureSpec, lowDetailMode: boolean): StageFlowGraph {
  if (architecture.stages.length === 0) {
    return { nodes: [], edges: [] }
  }

  const nodes: VisualNode[] = []
  const edges: VisualEdge[] = []
  const stageNodeLookup = new Map<string, VisualNode[]>()
  const baseRadius = lowDetailMode ? 0.064 : 0.078
  const nodeGap = lowDetailMode ? 0.31 : 0.37

  architecture.stages.forEach((stage, stageIndex) => {
    const nodeCount = stageNodeCount(stage.id, stageIndex, lowDetailMode)
    const cols = clampInt(Math.ceil(Math.sqrt(nodeCount)), 2, 7)
    const rows = clampInt(Math.ceil(nodeCount / cols), 2, 6)
    const depthBand = stageDepthBand(stageIndex)
    const stageNodes: VisualNode[] = []

    for (let index = 0; index < nodeCount; index += 1) {
      const row = Math.floor(index / cols)
      const col = index % cols
      const x = stage.position[0] * 0.95
      const y = 2.8 - (rows - 1) * nodeGap * 0.5 + row * nodeGap
      const z = depthBand + (col - (cols - 1) / 2) * nodeGap * 0.68
      const node: VisualNode = {
        id: `stage:${stage.id}:${index}`,
        stageId: stage.id,
        kind: 'stage',
        radius: baseRadius,
        position: [x, y, z],
        segments: lowDetailMode ? 7 : 10,
      }
      nodes.push(node)
      stageNodes.push(node)
    }

    stageNodeLookup.set(stage.id, stageNodes)
  })

  for (let stageIndex = 0; stageIndex < architecture.stages.length - 1; stageIndex += 1) {
    const sourceStage = architecture.stages[stageIndex]
    const targetStage = architecture.stages[stageIndex + 1]
    const sourceNodes = stageNodeLookup.get(sourceStage.id) ?? []
    const targetNodes = stageNodeLookup.get(targetStage.id) ?? []
    const targetStep = lowDetailMode ? 2 : 3
    const sourceStride = lowDetailMode ? 2 : 1

    for (let sourceIndex = 0; sourceIndex < sourceNodes.length; sourceIndex += sourceStride) {
      const source = sourceNodes[sourceIndex]
      const base = Math.abs(Math.round(seededSigned(`${source.id}:${targetStage.id}`) * 10000))
      for (let offset = 0; offset < targetStep; offset += 1) {
        const target = targetNodes[(base + offset * 2) % Math.max(1, targetNodes.length)]
        if (!target) continue
        edges.push({
          id: `stageflow:${source.id}->${target.id}:${offset}`,
          from: source.position,
          to: target.position,
          weight: 0.58,
          sourceStageId: source.stageId,
          targetStageId: target.stageId,
          kind: 'stage_flow',
        })
      }
    }
  }

  return { nodes, edges }
}

function stageNodeCount(stageId: string, stageIndex: number, lowDetailMode: boolean): number {
  const id = stageId.toLowerCase()
  const base = lowDetailMode ? 10 : 16
  if (id.includes('input')) return lowDetailMode ? 12 : 20
  if (id.includes('preprocess') || id.includes('patch')) return lowDetailMode ? 10 : 14
  if (id.includes('backbone') || id.includes('conv') || id.includes('encoder')) {
    return lowDetailMode ? 14 : 24
  }
  if (id.includes('transformer') || id.includes('decoder') || id.includes('blocks') || id.includes('queries')) {
    return lowDetailMode ? 16 : 28
  }
  if (id.includes('head') || id.includes('output') || id.includes('prediction')) return lowDetailMode ? 10 : 16
  return base + (stageIndex % 3) * (lowDetailMode ? 2 : 3)
}

function stageDepthBand(stageIndex: number): number {
  const depthPattern = [-1.2, 0.85, -0.45, 1.15]
  return depthPattern[stageIndex % depthPattern.length]
}

function resolveStageRoles(stages: VLMArchitectureStage[]): StageRoles {
  const input = findStageId(stages, ['input'])
  const preprocess = findStageId(stages, ['preprocess', 'augment', 'patch', 'backbone'])
  const conv = findStageId(stages, ['backbone', 'patch', 'encoder'])
  const transformer = findStageId(stages, ['transformer', 'encoder', 'decoder', 'blocks', 'queries'])
  const head = findStageId(stages, ['head', 'output', 'prediction'])

  const fallback = stages[0]?.id ?? 'stage_default'
  return {
    input: input ?? fallback,
    preprocess: preprocess ?? stages[1]?.id ?? fallback,
    conv: conv ?? stages[Math.min(2, stages.length - 1)]?.id ?? fallback,
    transformer: transformer ?? stages[Math.max(0, stages.length - 2)]?.id ?? fallback,
    head: head ?? stages[stages.length - 1]?.id ?? fallback,
  }
}

function findStageId(stages: VLMArchitectureStage[], candidates: string[]): string | null {
  const lowered = candidates.map((value) => value.toLowerCase())
  const match = stages.find((stage) =>
    lowered.some((candidate) => stage.id.toLowerCase().includes(candidate))
  )
  return match?.id ?? null
}

function tokenize(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length >= 3)
}

function attentionWeight(sourceId: string, targetId: string): number {
  const raw = seededSigned(`att:${sourceId}->${targetId}`)
  return Math.abs(raw)
}

function seededSigned(seed: string): number {
  let hash = 2166136261
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  const normalized = (hash >>> 0) / 4294967295
  return normalized * 2 - 1
}

function getSphereSegments(node: VisualNode, lowDetailMode: boolean): number {
  if (typeof node.segments === 'number' && node.segments >= 3) {
    return node.segments
  }
  if (lowDetailMode) return 6
  return 8
}

function clampInt(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.floor(value)))
}

function clampOpacity(value: number): number {
  return THREE.MathUtils.clamp(value, 0.08, 0.96)
}
