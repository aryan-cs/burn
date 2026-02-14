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

  const validEdges = Object.values(edges).filter(
    (edge) => Boolean(nodes[edge.source]) && Boolean(nodes[edge.target])
  )
  if (validEdges.length === 0) {
    return roles
  }

  const adjacency = new Map<string, string[]>()
  nodeIds.forEach((nodeId) => adjacency.set(nodeId, []))

  validEdges.forEach((edge) => {
    if (edge.source === edge.target) return
    adjacency.get(edge.source)?.push(edge.target)
    adjacency.get(edge.target)?.push(edge.source)
  })

  const visited = new Set<string>()
  nodeIds.forEach((startNodeId) => {
    if (visited.has(startNodeId)) return

    const queue = [startNodeId]
    const componentNodes: string[] = []
    visited.add(startNodeId)

    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const currentNodeId = queue[cursor]
      componentNodes.push(currentNodeId)

      const neighbors = adjacency.get(currentNodeId) ?? []
      neighbors.forEach((neighborId) => {
        if (visited.has(neighborId)) return
        visited.add(neighborId)
        queue.push(neighborId)
      })
    }

    if (componentNodes.length < 2) return

    const componentNodeSet = new Set(componentNodes)
    const componentEdges = validEdges.filter(
      (edge) =>
        componentNodeSet.has(edge.source) &&
        componentNodeSet.has(edge.target) &&
        edge.source !== edge.target
    )
    if (componentEdges.length === 0) return

    const ordered = orderNodesByDirectedEdges(componentNodes, componentEdges)
    if (ordered.length < 2) return

    roles.set(ordered[0], 'input')
    roles.set(ordered[ordered.length - 1], 'output')
  })

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
