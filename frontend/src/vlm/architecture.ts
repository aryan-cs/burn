export interface VLMArchitectureStage {
  id: string
  label: string
  detail: string
  description: string
  color: string
  size: [number, number, number]
  position: [number, number, number]
}

export interface VLMInputBlueprint {
  channels: number
  height: number
  width: number
}

export interface VLMCnnStageBlueprint {
  id: string
  label: string
  blocks: number
  out_channels: number
  kernel_size?: number
  stride?: number
}

export interface VLMCnnBlueprint {
  stages: VLMCnnStageBlueprint[]
}

export interface VLMTransformerBlueprint {
  encoder_layers: number
  decoder_layers: number
  attention_heads: number
  hidden_size: number
  num_queries: number
  token_count: number
}

export interface VLMArchitectureBlueprint {
  input: VLMInputBlueprint
  cnn: VLMCnnBlueprint
  transformer: VLMTransformerBlueprint
}

export interface VLMArchitectureSpec {
  id: string
  name: string
  family: string
  source?: string
  warning?: string | null
  stages: VLMArchitectureStage[]
  blueprint: VLMArchitectureBlueprint
}

export function getVlmArchitecture(modelId: string): VLMArchitectureSpec {
  const normalized = modelId.toLowerCase()

  if (normalized.includes('detr')) {
    return {
      id: modelId,
      name: 'DETR Vision Pipeline',
      family: 'cnn-transformer',
      source: 'frontend_fallback',
      warning: null,
      stages: [
        stage('input', 'Image Input', '3x320x320', 'Raw RGB image enters the detector pipeline.', '#54baff', [1.0, 0.9, 1.0], [-8.0, 0.0, 0.0]),
        stage('preprocess', 'Preprocess', 'resize=320, normalize', 'Image processor normalization and tensor conversion.', '#65dbf6', [1.0, 0.75, 1.0], [-5.0, 0.0, 0.0]),
        stage('backbone', 'CNN Backbone', 'resnet50, blocks=[3,4,6,3]', 'Convolutional backbone feature extraction.', '#79f2c0', [2.1, 1.4, 1.4], [-1.7, 0.0, 0.0]),
        stage('encoder', 'Transformer Encoder', '6 layers, 8 heads', 'Global attention over backbone feature tokens.', '#ffd37a', [1.7, 1.05, 1.3], [1.6, 0.0, 0.0]),
        stage('decoder', 'Transformer Decoder', '6 layers, 8 heads', 'Object queries attend to encoder memory.', '#ffb85b', [1.6, 0.95, 1.2], [4.4, 0.0, 0.0]),
        stage('head', 'Detection Head', '100 queries -> boxes/classes', 'Final class logits and bounding box regressors.', '#ff8a6e', [1.25, 0.8, 1.05], [7.2, 0.0, 0.0]),
      ],
      blueprint: {
        input: { channels: 3, height: 320, width: 320 },
        cnn: {
          stages: [
            { id: 'stem', label: 'Stem Conv', blocks: 1, out_channels: 64, kernel_size: 7, stride: 2 },
            { id: 'res2', label: 'ResNet Stage 2', blocks: 3, out_channels: 256, kernel_size: 3, stride: 1 },
            { id: 'res3', label: 'ResNet Stage 3', blocks: 4, out_channels: 512, kernel_size: 3, stride: 2 },
            { id: 'res4', label: 'ResNet Stage 4', blocks: 6, out_channels: 1024, kernel_size: 3, stride: 2 },
            { id: 'res5', label: 'ResNet Stage 5', blocks: 3, out_channels: 2048, kernel_size: 3, stride: 2 },
          ],
        },
        transformer: {
          encoder_layers: 6,
          decoder_layers: 6,
          attention_heads: 8,
          hidden_size: 256,
          num_queries: 100,
          token_count: 200,
        },
      },
    }
  }

  if (normalized.includes('yolos')) {
    return {
      id: modelId,
      name: 'YOLOS Vision Pipeline',
      family: 'vit-detector',
      source: 'frontend_fallback',
      warning: null,
      stages: [
        stage('input', 'Image Input', '3x512x512', 'Raw RGB image enters the detector pipeline.', '#54baff', [1.0, 0.9, 1.0], [-7.5, 0.0, 0.0]),
        stage('preprocess', 'Preprocess', 'resize=512, normalize', 'Image processor normalization and tensor conversion.', '#65dbf6', [0.95, 0.72, 0.95], [-4.8, 0.0, 0.0]),
        stage('patch', 'Patch Embed Conv', 'kernel=16, stride=16, hidden=192', 'Conv projection converts image patches into embeddings.', '#79f2c0', [1.45, 1.0, 1.0], [-1.7, 0.0, 0.0]),
        stage('blocks', 'Transformer Encoder', '12 layers, 3 heads', 'Global self-attention over visual and detection tokens.', '#ffd37a', [2.4, 1.2, 1.4], [1.7, 0.0, 0.0]),
        stage('queries', 'Detection Tokens', '100 detection tokens', 'Learned detection tokens specialized for object localization.', '#ffb85b', [1.4, 0.92, 1.05], [4.9, 0.0, 0.0]),
        stage('head', 'Prediction Head', 'boxes + classes', 'Outputs class logits and bounding boxes.', '#ff8a6e', [1.2, 0.78, 0.98], [7.6, 0.0, 0.0]),
      ],
      blueprint: {
        input: { channels: 3, height: 512, width: 512 },
        cnn: {
          stages: [
            { id: 'patch_embed', label: 'Patch Embed Conv', blocks: 1, out_channels: 192, kernel_size: 16, stride: 16 },
          ],
        },
        transformer: {
          encoder_layers: 12,
          decoder_layers: 0,
          attention_heads: 3,
          hidden_size: 192,
          num_queries: 100,
          token_count: 1124,
        },
      },
    }
  }

  return {
    id: modelId,
    name: 'Detection Pipeline',
    family: 'generic',
    source: 'frontend_fallback',
    warning: null,
    stages: [
      stage('input', 'Image Input', '3x320x320', 'Input image tensor.', '#54baff', [1.0, 0.9, 1.0], [-5.8, 0.0, 0.0]),
      stage('encoder', 'Visual Encoder', '6 layers, hidden=256', 'Feature extraction and contextual encoding.', '#79f2c0', [1.8, 1.1, 1.2], [-1.9, 0.0, 0.0]),
      stage('aggregator', 'Attention Mixer', '8 heads', 'Attention-based feature mixing.', '#ffd37a', [1.7, 1.0, 1.1], [1.7, 0.0, 0.0]),
      stage('head', 'Detection Head', '100 queries', 'Predicts class logits and boxes.', '#ff8a6e', [1.2, 0.85, 1.0], [5.4, 0.0, 0.0]),
    ],
    blueprint: {
      input: { channels: 3, height: 320, width: 320 },
      cnn: {
        stages: [
          { id: 'encoder_conv', label: 'Feature Conv', blocks: 2, out_channels: 256, kernel_size: 3, stride: 2 },
        ],
      },
      transformer: {
        encoder_layers: 6,
        decoder_layers: 0,
        attention_heads: 8,
        hidden_size: 256,
        num_queries: 100,
        token_count: 500,
      },
    },
  }
}

function stage(
  id: string,
  label: string,
  detail: string,
  description: string,
  color: string,
  size: [number, number, number],
  position: [number, number, number]
): VLMArchitectureStage {
  return { id, label, detail, description, color, size, position }
}
