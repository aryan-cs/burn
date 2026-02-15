import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import {
  FADE_IN_MS,
  PATH_PULSE_MAX,
  PATH_PULSE_MIN,
  PATH_PULSE_PERIOD_MS,
  STAGGER_STEP_MS,
  computePulse,
  computeStaggerProgress,
} from '../../canvas/animation/trainingPulse'
import { useRFGraphStore } from '../store/rfGraphStore'
import { useRFRunStore } from '../store/rfRunStore'

type EnsembleStrategy = 'bagging' | 'boosting' | 'stacking' | 'averaging'
type TreeNodeRole = 'root' | 'split' | 'leaf'

const NODE_MUTED_COLOR = '#505050'
const ROOT_CHECK_COLOR = '#ff8a65'
const FEATURE_SPLIT_COLOR = '#ffb429'
const LEAF_FALLBACK_COLOR = '#7cc6ff'
const ACTIVE_PATH_COLOR = '#f8f8f8'
const BROADCAST_LINE_COLOR = '#7a7a7a'

const LEAF_CLASS_COLORS = [
  '#4da3ff',
  '#34d399',
  '#a78bfa',
  '#f472b6',
  '#f97316',
  '#22d3ee',
  '#c4b5fd',
  '#fde047',
]

interface TreeNodePoint {
  position: THREE.Vector3
  role: TreeNodeRole
  highlighted: boolean
  classIndex?: number
}

interface TreeGeometryData {
  checkLines: number[]
  splitLines: number[]
  leafLines: number[]
  activeLines: number[]
  nodes: TreeNodePoint[]
}

function toPositiveInt(value: unknown, fallback: number): number {
  const asNumber = Number(value)
  if (!Number.isFinite(asNumber)) return fallback
  const rounded = Math.round(asNumber)
  return rounded > 0 ? rounded : fallback
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function parseStrategy(value: unknown): EnsembleStrategy {
  const raw = typeof value === 'string' ? value.toLowerCase() : 'bagging'
  if (raw === 'boosting' || raw === 'stacking' || raw === 'averaging') return raw
  return 'bagging'
}

function seededRandom(seed: number): () => number {
  let state = seed ^ 0x6d2b79f5
  return () => {
    state = Math.imul(state ^ (state >>> 15), state | 1)
    state ^= state + Math.imul(state ^ (state >>> 7), state | 61)
    return ((state ^ (state >>> 14)) >>> 0) / 4294967296
  }
}

function leafClassColor(classIndex: number | undefined): string {
  if (typeof classIndex !== 'number' || classIndex < 0) return LEAF_FALLBACK_COLOR
  return LEAF_CLASS_COLORS[classIndex % LEAF_CLASS_COLORS.length] ?? LEAF_FALLBACK_COLOR
}

function buildConceptTree(
  seed: number,
  depth: number,
  spread: number,
  classCount: number
): TreeGeometryData {
  const rand = seededRandom(seed)
  const normalizedDepth = clamp(Math.round(depth), 2, 6)
  const normalizedSpread = clamp(spread, 0.6, 2.2)
  const levelDrop = 0.56

  const rootNode: TreeNodePoint = {
    position: new THREE.Vector3(0, 0, 0),
    role: 'root',
    highlighted: true,
  }

  const checkLines: number[] = []
  const splitLines: number[] = []
  const leafLines: number[] = []
  const activeLines: number[] = []
  const nodes: TreeNodePoint[] = [rootNode]

  const pathChoices = Array.from({ length: normalizedDepth }, () => (rand() > 0.5 ? 1 : 0))
  let previousLevel: Array<{ position: THREE.Vector3; highlighted: boolean }> = [
    { position: rootNode.position, highlighted: true },
  ]

  for (let level = 1; level <= normalizedDepth; level += 1) {
    const isLeafLevel = level === normalizedDepth
    const span = ((normalizedDepth - level + 1) / normalizedDepth) * 1.3 * normalizedSpread
    const yOffset = -levelDrop + (rand() - 0.5) * 0.08
    const nextLevel: Array<{ position: THREE.Vector3; highlighted: boolean }> = []

    previousLevel.forEach((parent) => {
      ;[-1, 1].forEach((direction, sideIndex) => {
        const xOffset = direction * span * (0.72 + rand() * 0.3)
        const zOffset = direction * span * (0.42 + rand() * 0.35) + (rand() - 0.5) * 0.5

        const childPosition = new THREE.Vector3(
          parent.position.x + xOffset,
          parent.position.y + yOffset,
          parent.position.z + zOffset
        )
        const highlighted = parent.highlighted && pathChoices[level - 1] === sideIndex

        const role: TreeNodeRole = isLeafLevel ? 'leaf' : 'split'
        const classIndex = isLeafLevel ? Math.floor(rand() * Math.max(1, classCount)) : undefined

        const targetLines = highlighted
          ? activeLines
          : isLeafLevel
            ? leafLines
            : level === 1
              ? checkLines
              : splitLines

        targetLines.push(
          parent.position.x,
          parent.position.y,
          parent.position.z,
          childPosition.x,
          childPosition.y,
          childPosition.z
        )

        nodes.push({
          position: childPosition,
          role,
          highlighted,
          classIndex,
        })
        nextLevel.push({ position: childPosition, highlighted })
      })
    })

    previousLevel = nextLevel
  }

  return {
    checkLines,
    splitLines,
    leafLines,
    activeLines,
    nodes,
  }
}

function buildTreeOrigins(strategy: EnsembleStrategy, count: number, spread: number): THREE.Vector3[] {
  if (count <= 0) return []

  const normalizedSpread = clamp(spread, 0.6, 2.2)
  const rand = seededRandom(1200 + count * 17)
  const baseY = 2.2

  if (strategy === 'boosting') {
    return Array.from({ length: count }, (_, index) => {
      const t = count === 1 ? 0.5 : index / (count - 1)
      return new THREE.Vector3(
        -8 + t * 16,
        baseY + Math.sin(t * Math.PI * 3) * 0.35,
        -0.9 + (rand() - 0.5) * 1.8
      )
    })
  }

  if (strategy === 'stacking') {
    const firstRowCount = Math.max(2, Math.ceil(count * 0.6))
    const secondRowCount = Math.max(1, count - firstRowCount)
    const firstRow = Array.from({ length: firstRowCount }, (_, index) => {
      const t = firstRowCount === 1 ? 0.5 : index / (firstRowCount - 1)
      return new THREE.Vector3(
        -7 + t * 14,
        baseY + 0.25,
        1.4 + (rand() - 0.5) * 0.8
      )
    })
    const secondRow = Array.from({ length: secondRowCount }, (_, index) => {
      const t = secondRowCount === 1 ? 0.5 : index / (secondRowCount - 1)
      return new THREE.Vector3(
        -5 + t * 10,
        baseY - 0.95,
        -1.6 + (rand() - 0.5) * 1
      )
    })
    return [...firstRow, ...secondRow]
  }

  if (strategy === 'averaging') {
    const radius = 5.6 * normalizedSpread
    return Array.from({ length: count }, (_, index) => {
      const angle = (index / count) * Math.PI * 2
      return new THREE.Vector3(
        Math.cos(angle) * radius,
        baseY + Math.sin(angle * 2) * 0.45,
        Math.sin(angle) * radius * 0.68
      )
    })
  }

  const cols = Math.ceil(Math.sqrt(count))
  const rows = Math.ceil(count / cols)
  const spacingX = 2.7 * normalizedSpread
  const spacingZ = 2.45 * normalizedSpread
  return Array.from({ length: count }, (_, index) => {
    const row = Math.floor(index / cols)
    const col = index % cols
    const xCenter = ((cols - 1) * spacingX) / 2
    const zCenter = ((rows - 1) * spacingZ) / 2
    const jitterZ = (rand() - 0.5) * 0.8
    return new THREE.Vector3(
      col * spacingX - xCenter,
      baseY + (rand() - 0.5) * 0.18,
      row * spacingZ - zCenter + jitterZ
    )
  })
}

function LineSegments({
  positions,
  color,
  opacity,
}: {
  positions: number[]
  color: string
  opacity: number
}) {
  const geometry = useMemo(() => {
    const built = new THREE.BufferGeometry()
    built.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    return built
  }, [positions])

  useEffect(() => {
    return () => {
      geometry.dispose()
    }
  }, [geometry])

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={color} transparent opacity={opacity} />
    </lineSegments>
  )
}

function treeNodeColor(node: TreeNodePoint, activation: number): string {
  if (activation <= 0.01) return NODE_MUTED_COLOR
  if (node.highlighted) return ACTIVE_PATH_COLOR
  if (node.role === 'root') return ROOT_CHECK_COLOR
  if (node.role === 'split') return FEATURE_SPLIT_COLOR
  return leafClassColor(node.classIndex)
}

function computeAnimatedDisplayCount(
  elapsedMs: number,
  startDisplayCount: number,
  targetDisplayCount: number
): number {
  if (targetDisplayCount <= startDisplayCount) return targetDisplayCount

  const delta = targetDisplayCount - startDisplayCount
  const fullSteps = Math.floor(delta)
  const partialStep = delta - fullSteps

  let gained = 0
  for (let index = 0; index < fullSteps; index += 1) {
    gained += computeStaggerProgress(elapsedMs, index, STAGGER_STEP_MS, FADE_IN_MS)
  }
  if (partialStep > 0) {
    gained += partialStep * computeStaggerProgress(elapsedMs, fullSteps, STAGGER_STEP_MS, FADE_IN_MS)
  }

  return Math.min(targetDisplayCount, startDisplayCount + gained)
}

interface RfSceneManagerProps {
  lowDetailMode: boolean
}

export function RfSceneManager({ lowDetailMode }: RfSceneManagerProps) {
  const nodes = useRFGraphStore((state) => state.nodes)
  const training = useRFGraphStore((state) => state.training)
  const visualization = useRFGraphStore((state) => state.visualization)
  const status = useRFRunStore((state) => state.status)
  const progress = useRFRunStore((state) => state.progress)

  const strategy = parseStrategy(training.ensembleStrategy)
  const classifier = Object.values(nodes).find((node) => node.type === 'RandomForestClassifier') ?? null
  const outputNode = Object.values(nodes).find((node) => node.type === 'RFOutput') ?? null

  const totalTrees = toPositiveInt(classifier?.config.n_estimators, 100)
  const maxVisibleTrees = clamp(toPositiveInt(visualization.visibleTrees, 18), 3, 48)
  const requestedTreeCount = Math.min(maxVisibleTrees, Math.max(3, totalTrees))
  const displayTreeCount = lowDetailMode
    ? Math.min(requestedTreeCount, 10)
    : requestedTreeCount
  const requestedTreeDepth = clamp(toPositiveInt(visualization.treeDepth, 4), 2, 6)
  const treeDepth = lowDetailMode ? Math.min(requestedTreeDepth, 4) : requestedTreeDepth
  const treeSpread = clamp(Number(visualization.treeSpread) || 1, 0.6, 2.2)
  const nodeScale = clamp(Number(visualization.nodeScale) || 1, 0.6, 1.8)
  const classCount = Math.max(2, toPositiveInt(outputNode?.config.num_classes, 3))

  const lastCheckpointDisplayRef = useRef(0)
  const targetCheckpointDisplayRef = useRef(0)
  const checkpointStartMsRef = useRef(performance.now())
  const displayedCountRef = useRef(0)
  const [animationNowMs, setAnimationNowMs] = useState(() => performance.now())

  const latestProgress = progress[progress.length - 1]

  const nextBuiltTrees = useMemo(() => {
    if (status === 'idle' || status === 'error') return 0
    if (status === 'complete') return totalTrees
    return Math.max(
      0,
      Math.min(totalTrees, Number.isFinite(latestProgress?.trees_built) ? latestProgress.trees_built : 0)
    )
  }, [latestProgress?.trees_built, status, totalTrees])

  const targetDisplayCount = useMemo(
    () => (totalTrees > 0 ? (nextBuiltTrees / totalTrees) * displayTreeCount : 0),
    [displayTreeCount, nextBuiltTrees, totalTrees]
  )

  useEffect(() => {
    if (Math.abs(targetDisplayCount - targetCheckpointDisplayRef.current) < 0.0001) return

    lastCheckpointDisplayRef.current = displayedCountRef.current
    targetCheckpointDisplayRef.current = targetDisplayCount
    checkpointStartMsRef.current = performance.now()
    setAnimationNowMs(performance.now())
  }, [targetDisplayCount])

  useEffect(() => {
    let rafId = 0
    const tick = () => {
      const now = performance.now()
      setAnimationNowMs(now)

      const hasTransition =
        Math.abs(displayedCountRef.current - targetCheckpointDisplayRef.current) > 0.0005
      const shouldContinue = status === 'training' || hasTransition
      if (shouldContinue) {
        rafId = requestAnimationFrame(tick)
      }
    }

    const hasTransition =
      Math.abs(displayedCountRef.current - targetCheckpointDisplayRef.current) > 0.0005
    if (status === 'training' || hasTransition) {
      rafId = requestAnimationFrame(tick)
    }

    return () => {
      cancelAnimationFrame(rafId)
    }
  }, [latestProgress?.trees_built, status, targetDisplayCount])

  const displayedActiveTreeCount = useMemo(() => {
    const elapsedMs = animationNowMs - checkpointStartMsRef.current
    const animated = computeAnimatedDisplayCount(
      elapsedMs,
      lastCheckpointDisplayRef.current,
      targetCheckpointDisplayRef.current
    )
    displayedCountRef.current = animated
    return animated
  }, [animationNowMs])

  const hasActiveMotion =
    status === 'training' ||
    Math.abs(displayedActiveTreeCount - targetCheckpointDisplayRef.current) > 0.0005
  const pulseFactor = hasActiveMotion
    ? computePulse(animationNowMs, PATH_PULSE_PERIOD_MS, PATH_PULSE_MIN, PATH_PULSE_MAX)
    : 1

  const treeOrigins = useMemo(
    () => buildTreeOrigins(strategy, displayTreeCount, treeSpread),
    [displayTreeCount, strategy, treeSpread]
  )
  const treeShapes = useMemo(
    () =>
      Array.from({ length: displayTreeCount }, (_, index) =>
        buildConceptTree(900 + index * 31, treeDepth, treeSpread, classCount)
      ),
    [classCount, displayTreeCount, treeDepth, treeSpread]
  )

  const treeTransforms = useMemo(() => {
    const rand = seededRandom(7000 + displayTreeCount * 11)
    return Array.from({ length: displayTreeCount }, () => ({
      yaw: (rand() - 0.5) * 1.2,
      pitch: (rand() - 0.5) * 0.4,
      roll: (rand() - 0.5) * 0.18,
      scale: 0.92 + rand() * 0.2,
    }))
  }, [displayTreeCount])

  const sourcePosition = useMemo(() => new THREE.Vector3(0, 5.45, -0.8), [])
  const sourceLines = useMemo(() => {
    const lines: number[] = []
    treeOrigins.forEach((origin) => {
      lines.push(sourcePosition.x, sourcePosition.y, sourcePosition.z, origin.x, origin.y, origin.z)
    })
    return lines
  }, [sourcePosition, treeOrigins])

  const baseTreeScale = nodeScale * (1 - (treeDepth - 3) * 0.05)
  const sourceActivation = clamp(displayedActiveTreeCount / Math.max(1, displayTreeCount), 0, 1)

  return (
    <group>
      <LineSegments
        positions={sourceLines}
        color={BROADCAST_LINE_COLOR}
        opacity={0.12 + sourceActivation * 0.28}
      />

      <mesh position={[sourcePosition.x, sourcePosition.y, sourcePosition.z]}>
        <sphereGeometry args={[0.11 * nodeScale, lowDetailMode ? 10 : 16, lowDetailMode ? 10 : 16]} />
        <meshStandardMaterial
          color="#1f1f1f"
          emissive={ROOT_CHECK_COLOR}
          emissiveIntensity={0.2 + sourceActivation * 0.45}
          roughness={0.4}
          metalness={0.08}
        />
      </mesh>

      {treeOrigins.map((origin, treeIndex) => {
        const shape = treeShapes[treeIndex]
        const transform = treeTransforms[treeIndex]
        const treeActivation = clamp(displayedActiveTreeCount - treeIndex, 0, 1)

        const checkOpacity = 0.1 + treeActivation * 0.52
        const splitOpacity = 0.08 + treeActivation * 0.46
        const leafOpacity = 0.06 + treeActivation * 0.36
        const pathOpacity = (0.04 + treeActivation * 0.78) * pulseFactor

        return (
          <group
            key={`rf-tree-${treeIndex}`}
            position={[origin.x, origin.y, origin.z]}
            rotation={[transform.pitch, transform.yaw, transform.roll]}
            scale={[
              baseTreeScale * transform.scale,
              baseTreeScale * transform.scale,
              baseTreeScale * transform.scale,
            ]}
          >
            <LineSegments
              positions={shape.checkLines}
              color={treeActivation > 0.01 ? ROOT_CHECK_COLOR : NODE_MUTED_COLOR}
              opacity={checkOpacity}
            />
            <LineSegments
              positions={shape.splitLines}
              color={treeActivation > 0.01 ? FEATURE_SPLIT_COLOR : NODE_MUTED_COLOR}
              opacity={splitOpacity}
            />
            <LineSegments
              positions={shape.leafLines}
              color={treeActivation > 0.01 ? LEAF_FALLBACK_COLOR : NODE_MUTED_COLOR}
              opacity={leafOpacity}
            />
            <LineSegments
              positions={shape.activeLines}
              color={treeActivation > 0.01 ? ACTIVE_PATH_COLOR : NODE_MUTED_COLOR}
              opacity={pathOpacity}
            />

            {shape.nodes.map((node, nodeIndex) => {
              const radius =
                node.role === 'root'
                  ? 0.1 * nodeScale
                  : node.role === 'split'
                    ? 0.078 * nodeScale
                    : 0.064 * nodeScale

              const color = treeNodeColor(node, treeActivation)
              const nodeBaseIntensity = 0.18 + treeActivation * 0.5
              const pulsedIntensity = node.highlighted ? nodeBaseIntensity * pulseFactor : nodeBaseIntensity

              return (
                <mesh
                  key={`rf-tree-${treeIndex}-node-${nodeIndex}`}
                  position={[node.position.x, node.position.y, node.position.z]}
                >
                  <sphereGeometry args={[radius, lowDetailMode ? 8 : 12, lowDetailMode ? 8 : 12]} />
                  <meshStandardMaterial
                    color="#1f1f1f"
                    emissive={color}
                    emissiveIntensity={pulsedIntensity}
                    roughness={0.45}
                    metalness={0.08}
                  />
                </mesh>
              )
            })}
          </group>
        )
      })}

    </group>
  )
}
