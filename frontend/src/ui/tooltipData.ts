import type { LayerType } from '../store/graphStore'

/**
 * Short descriptive tooltips for each layer type shown in the palette
 * and wherever layers are listed.
 */
export const LAYER_TOOLTIPS: Record<LayerType, string> = {
  Input:
    'The entry point of the network. Defines the shape of data fed into the model (e.g. image dimensions).',
  Dense:
    'A fully-connected layer where every neuron connects to every neuron in the previous layer. The core building block of MLPs.',
  Conv2D:
    'A 2D convolution layer that slides learned filters across the input to detect spatial features like edges and textures.',
  MaxPool2D:
    'Down-samples the input by taking the maximum value in each pooling window, reducing spatial dimensions while keeping dominant features.',
  Flatten:
    'Reshapes a multi-dimensional tensor into a 1D vector so it can be fed into Dense layers.',
  Dropout:
    'Randomly sets a fraction of inputs to zero during training to prevent overfitting and improve generalisation.',
  BatchNorm:
    'Normalises layer inputs to have zero mean and unit variance, stabilising and accelerating training.',
  Output:
    'The final layer of the network. Its size typically equals the number of classes for classification.',
  LSTM:
    'Long Short-Term Memory — a recurrent layer that can learn long-range dependencies in sequential data.',
  GRU:
    'Gated Recurrent Unit — a lighter recurrent layer similar to LSTM, with fewer parameters.',
  Reshape:
    'Changes the shape of the tensor without altering its data, useful for bridging between different layer types.',
}

/**
 * Tooltips for the editable fields shown in the Build tab's layer editor.
 */
export const FIELD_TOOLTIPS: Record<string, string> = {
  Size:
    'The grid size (rows × columns) determines the number of neurons in this layer. Total neurons = rows × cols.',
  Activation:
    'The activation function applied after the linear transformation. It introduces non-linearity so the network can learn complex patterns.',
  Layers:
    'Total number of layers in the current network architecture.',
  Neurons:
    'Total number of neurons (units) across all layers.',
  Weights:
    'Total number of trainable weight parameters connecting neurons between layers.',
  Biases:
    'Total number of bias parameters — one per neuron in each layer (except the input).',
  'Layer Type':
    'The predominant layer type used in this architecture (e.g. Dense, Conv2D).',
  'Shared Activation Function':
    'The activation function used across all hidden layers (excluding the output layer).',
}
