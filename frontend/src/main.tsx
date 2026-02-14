import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import DeploymentsPage from './deployments/DeploymentsPage'
import ModelHubPage from './launch/ModelHubPage'
import NNBootstrapPage from './nn/NNBootstrapPage'
import RFBootstrapPage from './rf/RFBootstrapPage'

const path = window.location.pathname.toLowerCase()

function resolveTitle(currentPath: string): string {
  if (currentPath.startsWith('/deployments')) return 'Burn | Deployments'
  if (currentPath.startsWith('/rf')) return 'Burn | Random Forest Builder'
  if (currentPath.startsWith('/nn')) return 'Burn | Neural Network Builder'
  if (currentPath.startsWith('/vlm')) return 'Burn | VLM Builder'
  if (currentPath.startsWith('/svm')) return 'Burn | SVM Builder'
  if (currentPath.startsWith('/pca')) return 'Burn | PCA Builder'
  if (currentPath.startsWith('/linreg')) return 'Burn | Linear Regression Builder'
  if (currentPath.startsWith('/logreg')) return 'Burn | Logistic Regression Builder'
  return 'Burn | Home'
}

function resolveEntry() {
  if (path.startsWith('/deployments')) return <DeploymentsPage />
  if (path.startsWith('/rf')) return <RFBootstrapPage />
  if (path.startsWith('/nn')) return <NNBootstrapPage />
  return <ModelHubPage />
}

document.title = resolveTitle(path)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {resolveEntry()}
  </StrictMode>,
)
