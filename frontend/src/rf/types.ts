export type RFNodeType = 'RFInput' | 'RFFlatten' | 'RandomForestClassifier' | 'RFOutput'

export interface RFNode {
  id: string
  type: RFNodeType
  position: [number, number, number]
  config: Record<string, unknown>
}

export interface RFEdge {
  id: string
  source: string
  target: string
}

export interface RFPayloadNode {
  id: string
  type: RFNodeType
  config: Record<string, unknown>
}

export interface RFTrainingConfig {
  dataset: string
  testSize: number
  randomState: number
  stratify: boolean
  logEveryTrees: number
  ensembleStrategy: 'bagging' | 'boosting' | 'stacking' | 'averaging'
}

export interface RFVisualizationConfig {
  visibleTrees: number
  treeDepth: number
  treeSpread: number
  nodeScale: number
}

export interface RFGraphPayload {
  nodes: RFPayloadNode[]
  edges: RFEdge[]
  training: {
    dataset: string
    test_size: number
    random_state: number
    stratify: boolean
    log_every_trees: number
    ensemble_strategy: 'bagging' | 'boosting' | 'stacking' | 'averaging'
  }
}

export interface RFValidationResponse {
  valid: boolean
  shapes: Record<string, { input: number[] | null; output: number[] | null }>
  errors: Array<{ message: string; node_id?: string }>
  execution_order: string[]
  warnings: string[]
}

export interface RFCompileResponse {
  valid: boolean
  summary: {
    model_family: string
    layers: Array<Record<string, unknown>>
    resolved_training: Record<string, unknown>
    resolved_hyperparameters: Record<string, unknown>
    dataset: string
    expected_feature_count: number
    num_classes: number
    input_shape: number[]
  }
  python_source: string
  warnings: string[]
}

export interface RFDatasetMeta {
  id: string
  name: string
  task: string
  source: string
  kaggle_dataset: string
  csv_filename: string
  target_column: string
  drop_columns: string[]
  delimiter: string
}

export interface RFTrainResponse {
  job_id: string
  status: string
}

export interface RFStatusResponse {
  job_id: string
  status: string
  terminal: boolean
  error: string | null
  final_metrics: Record<string, number> | null
  has_python_source: boolean
  has_artifact: boolean
}

export interface RFProgressMessage {
  type: 'rf_progress'
  stage: string
  trees_built: number
  total_trees: number
  train_accuracy: number
  test_accuracy: number
  oob_score: number | null
  elapsed_ms: number
}

export interface RFDoneMessage {
  type: 'rf_done'
  final_train_accuracy: number
  final_test_accuracy: number
  confusion_matrix: number[][]
  classes: string[]
  feature_importances: number[]
  feature_names: string[]
  model_path: string
}

export interface RFErrorMessage {
  type: 'rf_error'
  message: string
  details?: unknown
}

export interface RFInferResponse {
  job_id: string
  input_shape: number[]
  prediction_indices: number[]
  predictions: Array<string | number>
  classes: string[]
  probabilities?: number[][] | null
}
