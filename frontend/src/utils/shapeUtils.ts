import type { LayerNode } from '../store/graphStore'

/**
 * Quick client-side shape inference for display purposes.
 * The authoritative shape inference happens on the backend.
 */
export function inferOutputShape(
  node: LayerNode,
  inputShape: number[] | null
): number[] | null {
  if (!inputShape) return null
  const config = node.config

  switch (node.type) {
    case 'Input':
      return config.shape ?? null

    case 'Dense':
      return config.units ? [config.units] : null

    case 'Conv2D': {
      if (inputShape.length < 3 || !config.filters || !config.kernel_size)
        return null
      const [, h, w] = inputShape
      const pad = config.padding ?? 0
      const k = config.kernel_size
      const outH = h + 2 * pad - k + 1
      const outW = w + 2 * pad - k + 1
      return [config.filters, outH, outW]
    }

    case 'MaxPool2D': {
      if (inputShape.length < 3 || !config.kernel_size) return null
      const [ch, hh, ww] = inputShape
      const kk = config.kernel_size
      return [ch, Math.floor(hh / kk), Math.floor(ww / kk)]
    }

    case 'Flatten': {
      const total = inputShape.reduce((a, b) => a * b, 1)
      return [total]
    }

    case 'Dropout':
    case 'BatchNorm':
      return inputShape

    case 'Output':
      return config.num_classes ? [config.num_classes] : null

    default:
      return null
  }
}
