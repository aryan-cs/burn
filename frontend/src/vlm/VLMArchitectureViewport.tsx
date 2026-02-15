import { Canvas } from '@react-three/fiber'
import { Line, OrbitControls, Text } from '@react-three/drei'
import { useEffect, useMemo, useState } from 'react'
import * as THREE from 'three'
import { WeightVisual } from '../canvas/edges/WeightVisual'
import type {
  VLMArchitectureSpec,
  VLMArchitectureStage,
  VLMCnnStageBlueprint,
} from './architecture'

type Vec3 = [number, number, number]
type NodeKind = 'token'

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
  kind: 'attention' | 'token_chain'
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
}

const IGNORE_RAYCAST: THREE.Mesh['raycast'] = () => undefined

const TOKEN_CENTER_X = 7.35
const TOKEN_CENTER_Z = 0
const CNN_X_START = -11.5
const CNN_X_END = 1.2

const MATRIX_CELL_SIZE = 0.145
const MATRIX_PADDING = 0.16
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

  const showTransformer = selectedStageId === null
    ? true
    : selectedStageId === roles.transformer || selectedStageId === roles.head

  const totalTokens = Math.max(1, architecture.blueprint.transformer.token_count)
  const renderedTokenCount = showTransformer
    ? (lowDetailMode ? Math.min(180, totalTokens) : Math.min(340, totalTokens))
    : 0
  const tokens = useMemo(
    () => createTokens(roles.transformer, renderedTokenCount),
    [roles.transformer, renderedTokenCount]
  )
  const tokenChainEdges = useMemo(
    () => createTokenChainEdges(tokens, lowDetailMode),
    [tokens, lowDetailMode]
  )

  const [attentionStep, setAttentionStep] = useState(0)
  useEffect(() => {
    setAttentionStep(0)
  }, [tokens.length, architecture.id])
  useEffect(() => {
    if (tokens.length === 0) return
    const timer = window.setInterval(
      () => setAttentionStep((current) => (current + 1) % 100000),
      lowDetailMode ? 320 : 190
    )
    return () => window.clearInterval(timer)
  }, [tokens.length, lowDetailMode])

  const attentionState = useMemo(
    () => createAttentionState(tokens, architecture, attentionStep, lowDetailMode),
    [tokens, architecture, attentionStep, lowDetailMode]
  )
  const attentionEdges = attentionState.edges
  const activeTokenIds = attentionState.activeTokenIds

  const chainEdgeSplit = useMemo(
    () => splitEdgesBySelection(tokenChainEdges, selectedStageId),
    [tokenChainEdges, selectedStageId]
  )
  const attentionEdgeSplit = useMemo(
    () => splitEdgesBySelection(attentionEdges, selectedStageId),
    [attentionEdges, selectedStageId]
  )

  const chainDefaultOpacity = 0.22
  const chainSelectedOpacity = clampOpacity(0.42 + activityPulse * 0.06)
  const attentionDefaultOpacity = clampOpacity(0.44 + activityPulse * 0.08)
  const attentionSelectedOpacity = clampOpacity(0.62 + activityPulse * 0.14)

  const tokenSuffix =
    !showTransformer
      ? ' (hidden in CNN focus mode)'
      : lowDetailMode && renderedTokenCount < totalTokens
        ? ` (showing ${renderedTokenCount.toLocaleString()})`
      : ''

  return (
    <div className="vlm-viewport-root">
      <Canvas camera={{ position: [0.2, 4.5, 15], fov: 48 }} style={{ width: '100%', height: '100%' }}>
        <color attach="background" args={['#000000']} />
        <ambientLight intensity={0.45} />
        <directionalLight position={[11, 10, 4]} intensity={0.9} />
        <pointLight position={[-10, -10, -5]} intensity={0.25} color="#9a9a9a" />
        <gridHelper args={[46, 92, '#2b2b2b', '#202020']} />

        <Text
          position={[-5.8, 3.34, 0]}
          fontSize={0.34}
          color="#e0f0ff"
          anchorX="center"
          anchorY="middle"
        >
          CNN Matrix Convolution
        </Text>
        <Text
          position={[-5.8, 2.88, 0]}
          fontSize={0.18}
          color="#98b5d4"
          anchorX="center"
          anchorY="middle"
        >
          input map × kernel matrix = output map (sliding window)
        </Text>
        <Text
          position={[-5.8, 2.54, 0]}
          fontSize={0.14}
          color="#7ea8cf"
          anchorX="center"
          anchorY="middle"
        >
          each square = one value, stacked sheets = channels, moving box = receptive field
        </Text>

        <Text
          position={[7.4, 3.34, 0]}
          fontSize={0.34}
          color="#e0f0ff"
          anchorX="center"
          anchorY="middle"
        >
          Transformer Attention
        </Text>
        <Text
          position={[7.4, 2.88, 0]}
          fontSize={0.18}
          color="#98b5d4"
          anchorX="center"
          anchorY="middle"
        >
          {`tokens: ${totalTokens.toLocaleString()}${tokenSuffix}`}
        </Text>

        {focusedConvOp && focusedSourcePanel && focusedTargetPanel ? (
          <CnnSymbolicScene
            op={focusedConvOp}
            sourcePanel={focusedSourcePanel}
            targetPanel={focusedTargetPanel}
            sourceHighlight={panelHighlights.get(focusedSourcePanel.id) ?? null}
            targetHighlight={panelHighlights.get(focusedTargetPanel.id) ?? null}
            selectedStageId={selectedStageId}
            stageIndex={focusedConvOpIndex + 1}
            totalStages={cnnPipeline.ops.length}
            lowDetailMode={lowDetailMode}
            activityPulse={activityPulse}
            onSelectStage={onSelectStage}
          />
        ) : (
          <Text
            position={[-5.8, 0, 0]}
            fontSize={0.22}
            color="#9db6d3"
            anchorX="center"
            anchorY="middle"
          >
            No CNN stages in architecture.
          </Text>
        )}

        {showTransformer ? (
          <>
            <EdgeSegments
              edges={chainEdgeSplit.rest}
              color="#4f5968"
              opacity={chainDefaultOpacity}
            />
            <EdgeSegments
              edges={chainEdgeSplit.selected}
              color="#80a4d8"
              opacity={chainSelectedOpacity}
            />
            <EdgeSegments
              edges={attentionEdgeSplit.rest}
              color="#ffb429"
              opacity={attentionDefaultOpacity}
            />
            <EdgeSegments
              edges={attentionEdgeSplit.selected}
              color="#ffd37a"
              opacity={attentionSelectedOpacity}
            />

            {tokens.map((node) => (
              <TokenNode
                key={node.id}
                node={node}
                selected={selectedStageId === node.stageId}
                active={activeTokenIds.has(node.id)}
                lowDetailMode={lowDetailMode}
                outlineColor={stageColorMap.get(node.stageId) ?? '#ffd37a'}
                onSelectStage={onSelectStage}
              />
            ))}
          </>
        ) : (
          <Text
            position={[7.4, -0.2, 0]}
            fontSize={0.16}
            color="#7f95ad"
            anchorX="center"
            anchorY="middle"
          >
            Transformer view hidden. Click a transformer stage to inspect attention.
          </Text>
        )}

        <OrbitControls
          makeDefault
          enablePan
          enableZoom
          enableRotate
          minDistance={5}
          maxDistance={32}
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

interface CnnSymbolicSceneProps {
  op: CnnConvolutionOp
  sourcePanel: CnnMatrixPanel
  targetPanel: CnnMatrixPanel
  sourceHighlight: PanelHighlight | null
  targetHighlight: PanelHighlight | null
  selectedStageId: string | null
  stageIndex: number
  totalStages: number
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
  stageIndex,
  totalStages,
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
      <Text
        position={[-5.8, 1.92, 0]}
        fontSize={0.19}
        color="#d7ebff"
        anchorX="center"
        anchorY="middle"
      >
        {`Stage ${stageIndex}/${Math.max(1, totalStages)} · ${sourcePanel.label} -> ${targetPanel.label}`}
      </Text>
      <Text
        position={[-5.8, 1.66, 0]}
        fontSize={0.12}
        color="#93b6d8"
        anchorX="center"
        anchorY="middle"
      >
        {'The filter scans the image, computes weighted sums, then writes one output value.'}
      </Text>

      <CnnVolumePanel
        panel={sourcePanel}
        position={sourcePosition}
        roleLabel="Input Feature Maps"
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
        roleLabel="Output Feature Maps"
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

      <Text
        position={[-6.9, 0.86, 0]}
        fontSize={0.25}
        color="#ffe0a5"
        anchorX="center"
        anchorY="middle"
      >
        ×
      </Text>
      <Text
        position={[-4.6, 0.86, 0]}
        fontSize={0.25}
        color="#ffe0a5"
        anchorX="center"
        anchorY="middle"
      >
        =
      </Text>
      <Text
        position={[-5.75, 0.64, 0]}
        fontSize={0.11}
        color="#b9d0e6"
        anchorX="center"
        anchorY="middle"
      >
        {'dot product + bias + activation'}
      </Text>

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
  roleLabel: string
  highlight: PanelHighlight | null
  selected: boolean
  lowDetailMode: boolean
  onClick: () => void
}

function CnnVolumePanel({
  panel,
  position,
  roleLabel,
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

      <Text
        position={[0, panelHeight / 2 + 0.25, 0]}
        fontSize={0.14}
        color="#dff1ff"
        anchorX="center"
        anchorY="middle"
      >
        {roleLabel}
      </Text>
      <Text
        position={[0, panelHeight / 2 + 0.05, 0]}
        fontSize={0.12}
        color="#b8d2ea"
        anchorX="center"
        anchorY="middle"
      >
        {panel.label}
      </Text>
      <Text
        position={[0, -panelHeight / 2 - 0.22, 0]}
        fontSize={0.1}
        color="#89aacb"
        anchorX="center"
        anchorY="middle"
      >
        {`${panel.trueRows}x${panel.trueCols} · ${panel.channels} channels`}
      </Text>
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

      <Text
        position={[0, size * cell / 2 + 0.33, 0]}
        fontSize={0.12}
        color="#ffe1b2"
        anchorX="center"
        anchorY="middle"
      >
        {'Filter Kernel'}
      </Text>
      <Text
        position={[0, -size * cell / 2 - 0.26, 0]}
        fontSize={0.1}
        color="#d8bb96"
        anchorX="center"
        anchorY="middle"
      >
        {`k${op.kernelSize} · stride ${op.stride}`}
      </Text>
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

interface TokenNodeProps {
  node: VisualNode
  selected: boolean
  active: boolean
  lowDetailMode: boolean
  outlineColor: string
  onSelectStage: (stageId: string) => void
}

function TokenNode({
  node,
  selected,
  active,
  lowDetailMode,
  outlineColor,
  onSelectStage,
}: TokenNodeProps) {
  const [hovered, setHovered] = useState(false)
  const segments = getSphereSegments(node, lowDetailMode)
  const outlineOpacity = active ? 0.95 : selected ? 0.86 : hovered ? 0.72 : 0.48

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
          color="#6f6f6f"
          emissive="#191919"
          emissiveIntensity={active ? 0.66 : selected || hovered ? 0.48 : 0.26}
          roughness={0.55}
          metalness={0.08}
        />
      </mesh>

      <mesh raycast={IGNORE_RAYCAST} scale={[1.2, 1.2, 1.2]}>
        <sphereGeometry args={[node.radius, segments, segments]} />
        <meshBasicMaterial
          color={active ? '#ffcb67' : outlineColor}
          transparent
          opacity={outlineOpacity}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </mesh>

      {active ? (
        <mesh raycast={IGNORE_RAYCAST}>
          <ringGeometry args={[node.radius * 1.3, node.radius * 1.66, 24]} />
          <meshBasicMaterial
            color="#ffe18a"
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

  const rawPanels: Array<Omit<CnnMatrixPanel, 'position' | 'width' | 'height'>> = []
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
  })

  const step =
    rawPanels.length > 1 ? (CNN_X_END - CNN_X_START) / (rawPanels.length - 1) : 0

  const panels: CnnMatrixPanel[] = rawPanels.map((panel, index) => {
    const [width, height] = matrixPanelSize(panel.rows, panel.cols)
    return {
      ...panel,
      width,
      height,
      position: [CNN_X_START + step * index, 0, 0],
    }
  })

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

function matrixPanelSize(rows: number, cols: number): [number, number] {
  const width = Math.max(0.95, cols * MATRIX_CELL_SIZE + MATRIX_PADDING * 2)
  const height = Math.max(0.95, rows * MATRIX_CELL_SIZE + MATRIX_PADDING * 2)
  return [width, height]
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

function createTokens(stageId: string, count: number): VisualNode[] {
  if (count <= 0) return []
  const safeCount = Math.max(1, count)
  const nodes: VisualNode[] = []
  const goldenAngle = Math.PI * (3 - Math.sqrt(5))
  const tokenSegments = safeCount > 1800 ? 5 : safeCount > 900 ? 6 : 8

  for (let index = 0; index < safeCount; index += 1) {
    const radius = 1.05 + 0.105 * Math.sqrt(index + 1)
    const angle = index * goldenAngle
    nodes.push({
      id: `tok_${index}`,
      stageId,
      kind: 'token',
      radius: 0.08,
      segments: tokenSegments,
      position: [
        TOKEN_CENTER_X + Math.cos(angle) * radius,
        Math.sin(angle * 0.55) * 0.22,
        TOKEN_CENTER_Z + Math.sin(angle) * radius,
      ],
    })
  }

  return nodes
}

function createTokenChainEdges(tokens: VisualNode[], lowDetailMode: boolean): VisualEdge[] {
  if (tokens.length <= 1) return []
  const stride = lowDetailMode ? Math.max(1, Math.floor(tokens.length / 120)) : 1
  const edges: VisualEdge[] = []

  for (let index = 0; index < tokens.length; index += stride) {
    const token = tokens[index]
    const next = tokens[(index + stride) % tokens.length]
    edges.push({
      id: `chain:${token.id}:${next.id}`,
      from: token.position,
      to: next.position,
      weight: 0.7,
      sourceStageId: token.stageId,
      targetStageId: next.stageId,
      kind: 'token_chain',
    })
  }

  return edges
}

function createAttentionState(
  tokens: VisualNode[],
  architecture: VLMArchitectureSpec,
  step: number,
  lowDetailMode: boolean
): AttentionState {
  const activeTokenIds = new Set<string>()
  if (tokens.length <= 1) return { edges: [], activeTokenIds }

  const configuredHeads = Math.max(1, architecture.blueprint.transformer.attention_heads)
  const headCount = lowDetailMode
    ? Math.max(1, Math.min(4, Math.ceil(configuredHeads / 2)))
    : configuredHeads
  const targetsPerHead = lowDetailMode ? 2 : 4
  const edges: VisualEdge[] = []

  for (let head = 0; head < headCount; head += 1) {
    const sourceIndex = (step * (head + 2) + head * 13) % tokens.length
    const source = tokens[sourceIndex]
    activeTokenIds.add(source.id)

    const candidates = tokens
      .filter((token) => token.id !== source.id)
      .map((target, targetIndex) => ({
        target,
        score: dynamicAttentionScore(source, target, head, step, targetIndex),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.min(targetsPerHead, tokens.length - 1))

    candidates.forEach((entry, rank) => {
      activeTokenIds.add(entry.target.id)
      edges.push({
        id: `att:h${head}:${source.id}:${entry.target.id}`,
        from: source.position,
        to: entry.target.position,
        weight: entry.score * (1 - rank * 0.12),
        sourceStageId: source.stageId,
        targetStageId: entry.target.stageId,
        kind: 'attention',
      })
    })
  }

  return { edges, activeTokenIds }
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
