import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import RandomForestPage from './rf/RandomForestPage'

const isRFRoute = window.location.pathname.startsWith('/rf')

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {isRFRoute ? <RandomForestPage /> : <App />}
  </StrictMode>,
)
