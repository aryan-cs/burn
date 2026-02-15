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

function generateHomeValueDataset(): LinearRegressionDataset {
  const samples: number[][] = []
  const targets: number[] = []

  for (let index = 0; index < 38; index += 1) {
    const area = 620 + index * 52
    const curvature = Math.cos(index * 0.42) * 9500
    const price = round(95000 + area * 215 + curvature, 2)
    samples.push([round(area, 2)])
    targets.push(price)
  }

  return {
    id: 'home_value_tiny',
    name: 'Home Area vs Home Price',
    description:
      'Small synthetic housing-style dataset. Predict home price from area (sqft).',
    featureNames: ['area_sqft'],
    targetName: 'home_price_usd',
    samples,
    targets,
  }
}

export const LINEAR_REGRESSION_DATASETS: LinearRegressionDataset[] = [
  generateStudyHoursDataset(),
  generateHomeValueDataset(),
]

export function getLinearRegressionDataset(datasetId: string): LinearRegressionDataset {
  return (
    LINEAR_REGRESSION_DATASETS.find((dataset) => dataset.id === datasetId) ??
    LINEAR_REGRESSION_DATASETS[0]
  )
}
