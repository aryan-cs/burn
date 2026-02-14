import type { Edge, LayerNode } from '../store/graphStore'

export type LayerRole = 'input' | 'hidden' | 'output'

export function getNeuralNetworkOrder(
  nodes: Record<string, LayerNode>,
  edges: Record<string, Edge>
): string[] {
  const nodeIds = Object.keys(nodes)
  if (nodeIds.length <= 1) {
    return nodeIds
  }

  const directedEdges = Object.values(edges).filter(
    (edge) => Boolean(nodes[edge.source]) && Boolean(nodes[edge.target])
  )
  return orderNodesByDirectedEdges(nodeIds, directedEdges)
}

export function getLayerRolesForColoring(
  nodes: Record<string, LayerNode>,
  edges: Record<string, Edge>
): Map<string, LayerRole> {
  const roles = new Map<string, LayerRole>()
  const nodeIds = Object.keys(nodes)
  nodeIds.forEach((nodeId) => {
    roles.set(nodeId, 'hidden')
  })

  const typedInputIds = nodeIds.filter((nodeId) => nodes[nodeId]?.type === 'Input')
  const typedOutputIds = nodeIds.filter((nodeId) => nodes[nodeId]?.type === 'Output')
  if (typedInputIds.length === 1) {
    roles.set(typedInputIds[0], 'input')
  }
  if (typedOutputIds.length === 1 && typedOutputIds[0] !== typedInputIds[0]) {
    roles.set(typedOutputIds[0], 'output')
  }
  if (typedInputIds.length === 1 && typedOutputIds.length === 1) {
    return roles
  }

  const validEdges = Object.values(edges).filter(
    (edge) => Boolean(nodes[edge.source]) && Boolean(nodes[edge.target])
  )
  if (validEdges.length === 0) {
    return roles
  }

  const connectedNodeIds = Array.from(
    new Set(
      validEdges
        .flatMap((edge) => [edge.source, edge.target])
        .filter((nodeId) => Boolean(nodes[nodeId]))
    )
  )
  if (connectedNodeIds.length < 2) {
    return roles
  }

  const ordered = orderNodesByDirectedEdges(connectedNodeIds, validEdges)
  if (ordered.length < 2) {
    return roles
  }

  const inputNodeId = ordered[0]
  const outputNodeId = ordered[ordered.length - 1]
  roles.set(inputNodeId, 'input')
  if (outputNodeId !== inputNodeId) {
    roles.set(outputNodeId, 'output')
  }

  return roles
}

function orderNodesByDirectedEdges(nodeIds: string[], edges: Edge[]): string[] {
  const indegree = new Map<string, number>()
  const outgoing = new Map<string, string[]>()
  nodeIds.forEach((nodeId) => {
    indegree.set(nodeId, 0)
    outgoing.set(nodeId, [])
  })

  edges.forEach((edge) => {
    if (!indegree.has(edge.source) || !indegree.has(edge.target)) return
    if (edge.source === edge.target) return
    outgoing.get(edge.source)?.push(edge.target)
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1)
  })

  const queue = nodeIds
    .filter((nodeId) => (indegree.get(nodeId) ?? 0) === 0)
    .sort(compareNodeIds)
  const ordered: string[] = []

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const currentNodeId = queue[cursor]
    ordered.push(currentNodeId)

    const targets = outgoing.get(currentNodeId) ?? []
    targets.forEach((targetId) => {
      const nextInDegree = (indegree.get(targetId) ?? 0) - 1
      indegree.set(targetId, nextInDegree)
      if (nextInDegree === 0) {
        queue.push(targetId)
      }
    })
  }

  const seen = new Set(ordered)
  const remaining = nodeIds.filter((nodeId) => !seen.has(nodeId)).sort(compareNodeIds)
  return [...ordered, ...remaining]
}

function compareNodeIds(a: string, b: string): number {
  return getNodeNumericSuffix(a) - getNodeNumericSuffix(b)
}

function getNodeNumericSuffix(nodeId: string): number {
  const match = nodeId.match(/_(\d+)$/)
  if (!match) return Number.MAX_SAFE_INTEGER
  return Number(match[1])
}
