export interface LinearRegressionDataset {
  id: string
  name: string
  description: string
  featureNames: string[]
  targetName: string
  samples: number[][]
  targets: number[]
}

function round(value: number, places = 4): number {
  const scale = 10 ** places
  return Math.round(value * scale) / scale
}

function generateStudyHoursDataset(): LinearRegressionDataset {
  const samples: number[][] = []
  const targets: number[] = []

  for (let index = 0; index < 42; index += 1) {
    const hours = round(0.5 + index * 0.28, 3)
    const sinusoid = Math.sin(hours * 1.25) * 4.2
    const examScore = round(27.5 + hours * 5.85 + sinusoid, 3)
    samples.push([hours])
    targets.push(examScore)
  }

  return {
    id: 'study_hours',
    name: 'Study Hours vs Exam Score',
    description:
      'Beginner 1-feature regression dataset. Predict exam score from hours studied.',
    featureNames: ['study_hours'],
    targetName: 'exam_score',
    samples,
    targets,
  }
}

function generateDiabetesBmiDataset(): LinearRegressionDataset {
  const bmiValues = [
    0.061696, -0.011595, -0.047163, 0.039062, -0.02884, -0.018062, -0.010517, -0.022373, 0.035829, 0.059541,
    0.044451, -0.050396, 0.022895, 0.014272, -0.010517, -0.035307, -0.041774, 0.016428, 0.025051, -0.06333,
    -0.00405, -0.037463, -0.018062, -0.046085, -0.00405, -0.030996, -0.057941, -0.022373, -0.039618, -0.006206,
    -0.025607, -0.064408, 0.009961, -0.064408, 0.028284, -0.032073, 0.045529, 0.004572, 0.110198, -0.021295,
    0.004572, 0.034751, -0.061174, 0.016428, -0.021295, 0.056307, 0.111276, 0.047685, 0.046607, 0.092953,
    0.070319, -0.043929, -0.010517, 0.043373, 0.03044, -0.066563, 0.059541, -0.046085, 0.009961, 0.027206,
    0.004572, -0.035307, 0.042296, -0.066563, -0.030996, 0.025051, 0.054152, -0.055785, 0.006728, 0.045529,
    0.014272, -0.070875, 0.035829, -0.041774, -0.025607, 0.032595, -0.012673, -0.030996, 0.039062, -0.068719,
    0.07463, -0.046085, -0.032073, 0.060618, 0.059541, -0.065486, -0.024529, -0.041774, 0.028284, -0.000817,
    0.03044, -0.020218, 0.026128, -0.024529, -0.005128, -0.020218, -0.016984, 0.018584, -0.030996, -0.008362,
    0.073552, 0.034751, -0.001895, 0.001339, -0.023451, -0.020218, 0.006728, 0.051996, 0.00565, 0.114509,
    0.03044, -0.006206, 0.085408, -0.007284, 0.021817, -0.002973, -0.020218, -0.040696, 0.045529, 0.04984,
    0.020739, -0.024529, 0.137143, 0.037984, -0.023451, -0.002973, 0.002417, -0.089197, -0.029918, 0.01535,
    0.069241, -0.046085, -0.030996, 0.022895, -0.033151, 0.123131, -0.050396, 0.058463, 0.006728, 0.071397,
    -0.036385, -0.039618, -0.034229, -0.033151, 0.055229, -0.023451, -0.015906, -0.07303,
  ]

  const progressionTargets = [
    151.0, 206.0, 138.0, 310.0, 179.0, 171.0, 97.0, 49.0, 184.0, 85.0,
    129.0, 87.0, 265.0, 90.0, 61.0, 53.0, 75.0, 225.0, 182.0, 37.0,
    61.0, 128.0, 150.0, 178.0, 202.0, 42.0, 252.0, 51.0, 65.0, 134.0,
    98.0, 96.0, 150.0, 83.0, 302.0, 53.0, 232.0, 59.0, 258.0, 281.0,
    200.0, 84.0, 99.0, 268.0, 107.0, 272.0, 336.0, 317.0, 174.0, 128.0,
    288.0, 71.0, 25.0, 195.0, 172.0, 59.0, 268.0, 74.0, 151.0, 225.0,
    107.0, 185.0, 137.0, 79.0, 91.0, 122.0, 142.0, 39.0, 277.0, 202.0,
    191.0, 49.0, 248.0, 185.0, 252.0, 208.0, 160.0, 154.0, 246.0, 72.0,
    275.0, 47.0, 78.0, 215.0, 91.0, 153.0, 89.0, 103.0, 145.0, 115.0,
    202.0, 241.0, 283.0, 200.0, 230.0, 233.0, 80.0, 248.0, 55.0, 31.0,
    275.0, 236.0, 44.0, 142.0, 144.0, 97.0, 109.0, 230.0, 249.0, 237.0,
    244.0, 164.0, 306.0, 95.0, 178.0, 139.0, 148.0, 71.0, 272.0, 221.0,
    281.0, 58.0, 233.0, 167.0, 71.0, 217.0, 245.0, 104.0, 69.0, 201.0,
    277.0, 69.0, 43.0, 232.0, 168.0, 281.0, 189.0, 136.0, 131.0, 55.0,
    146.0, 91.0, 120.0, 94.0, 173.0, 64.0, 104.0, 57.0,
  ]

  return {
    id: 'diabetes_bmi',
    name: 'Diabetes BMI vs Disease Progression',
    description:
      'Real subset from sklearn diabetes dataset. Predict one-year disease progression from BMI.',
    featureNames: ['bmi'],
    targetName: 'disease_progression',
    samples: bmiValues.map((value) => [value]),
    targets: progressionTargets,
  }
}

export const LINEAR_REGRESSION_DATASETS: LinearRegressionDataset[] = [
  generateDiabetesBmiDataset(),
  generateStudyHoursDataset(),
]

export function getLinearRegressionDataset(datasetId: string): LinearRegressionDataset {
  return (
    LINEAR_REGRESSION_DATASETS.find((dataset) => dataset.id === datasetId) ??
    LINEAR_REGRESSION_DATASETS[0]
  )
}
