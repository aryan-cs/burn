import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import DeploymentsPage from './deployments/DeploymentsPage'
import ModelHubPage from './launch/ModelHubPage'
import NNBootstrapPage from './nn/NNBootstrapPage'
import RFBootstrapPage from './rf/RFBootstrapPage'

const path = window.location.pathname.toLowerCase()

function resolveEntry() {
  if (path.startsWith('/deployments')) return <DeploymentsPage />
  if (path.startsWith('/rf')) return <RFBootstrapPage />
  if (path.startsWith('/nn')) return <NNBootstrapPage />
  return <ModelHubPage />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {resolveEntry()}
  </StrictMode>,
)
