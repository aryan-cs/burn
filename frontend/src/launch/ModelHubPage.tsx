import { useState } from 'react'
import './modelHub.css'

const NN = 'Neural Network'
const RF = 'Random Forest'
const VLM = 'Vision-Language'
const LLM = 'LLM'
const SVM = 'SVM'
const PCA = 'PCA'
const LINREG = 'Linear Regression'
const LOGREG = 'Logistic Regression'

type AiProvider = 'openai' | 'gemini' | 'anthropic' | 'nvidia'

const PROVIDER_MODELS: Record<AiProvider, string[]> = {
  openai: ['gpt-4o-mini', 'gpt-4.1-mini', 'gpt-4o'],
  gemini: ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'],
  anthropic: ['claude-3-5-haiku-latest', 'claude-3-5-sonnet-latest'],
  nvidia: [
    'nvidia/nemotron-3-nano-30b-a3b',
    'nvidia/llama-3.3-nemotron-super-49b-v1.5',
    'nvidia/llama-3.1-nemotron-ultra-253b-v1',
    'nvidia/nemotron-nano-12b-v2-vl',
  ],
}

const AI_PROVIDER_KEY = 'burn.ai.provider'
const AI_MODEL_KEY = 'burn.ai.model'

function getStoredProvider(): AiProvider {
  const v = localStorage.getItem(AI_PROVIDER_KEY)
  if (v === 'openai' || v === 'gemini' || v === 'anthropic' || v === 'nvidia') return v
  return 'nvidia'
}

function getStoredModel(provider: AiProvider): string {
  const v = localStorage.getItem(AI_MODEL_KEY)
  if (v && PROVIDER_MODELS[provider].includes(v)) return v
  return PROVIDER_MODELS[provider][0]
}

type LaunchCard = {
  title: string
  subtitle: string
  description: string
  href: string
  badge: string
  tone: 'nn' | 'rf' | 'vlm' | 'llm' | 'svm' | 'pca' | 'linreg' | 'logreg'
}

const SCRATCH_CARDS: LaunchCard[] = [
  {
    title: 'Neural Network Builder',
    subtitle: 'Start From Scratch',
    description: 'Blank canvas with interactive 3D layer graph editing.',
    href: '/nn?mode=scratch',
    badge: NN,
    tone: 'nn',
  },
  {
    title: 'Random Forest Builder',
    subtitle: 'Start From Scratch',
    description: 'Blank workspace with 3D node pipeline and sklearn backend.',
    href: '/rf?mode=scratch',
    badge: RF,
    tone: 'rf',
  },
  {
    title: 'VLM Object Detection Lab',
    subtitle: 'Start From Scratch',
    description: 'Camera-based object detection workspace with Hugging Face model runtime.',
    href: '/vlm?mode=scratch',
    badge: VLM,
    tone: 'vlm',
  },
  {
    title: 'Support Vector Machine Builder',
    subtitle: 'Start From Scratch',
    description: 'Coming soon!',
    href: '#',
    badge: SVM,
    tone: 'svm',
  },
  {
    title: 'Principal Component Analysis Builder',
    subtitle: 'Start From Scratch',
    description: 'Coming soon!',
    href: '#',
    badge: PCA,
    tone: 'pca',
  },
  {
    title: 'Linear Regression Builder',
    subtitle: 'Start From Scratch',
    description: 'Build and train a real linear regression model with live fit visualization.',
    href: '/linreg?mode=scratch',
    badge: LINREG,
    tone: 'linreg',
  },
  {
    title: 'Logistic Regression Builder',
    subtitle: 'Start From Scratch',
    description: 'Coming soon!',
    href: '#',
    badge: LOGREG,
    tone: 'logreg',
  },
]

const PRESET_CARDS: LaunchCard[] = [
  {
    title: 'GPT-2 Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: LLM,
    tone: 'llm',
  },
  {
    title: 'BERT Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: LLM,
    tone: 'llm',
  },
  {
    title: 'NVIDIA Nemotron 3 Nano 30B Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: LLM,
    tone: 'llm',
  },
  {
    title: 'NVIDIA Nemotron Nano 9B v2 Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: LLM,
    tone: 'llm',
  },
  {
    title: 'Gemma 3 4B Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: LLM,
    tone: 'llm',
  },
  {
    title: 'GPT-OSS 120B Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: LLM,
    tone: 'llm',
  },
  {
    title: 'ResNet Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: LLM,
    tone: 'llm',
  },
  {
    title: 'MNIST Baseline MLP',
    subtitle: 'Preset Project',
    description: 'Input -> Flatten -> Dense -> Output. Good default for first compile/train.',
    href: '/nn?mode=preset&template=mnist_basic',
    badge: NN,
    tone: 'nn',
  },
  {
    title: 'AlexNet Cats vs Dogs',
    subtitle: 'Preset Project',
    description:
      'True AlexNet stack (conv/pool/fc) trained on Kaggle Cats vs Dogs at 96x96.',
    href: '/nn?mode=preset&template=alexnet_cats_dogs',
    badge: NN,
    tone: 'nn',
  },
  {
    title: 'MNIST Dropout Stack',
    subtitle: 'Preset Project',
    description: 'Adds dropout and deeper hidden path for experimentation.',
    href: '/nn?mode=preset&template=mnist_dropout',
    badge: NN,
    tone: 'nn',
  },
  {
    title: 'Digits Starter MLP',
    subtitle: 'Preset Project',
    description: '8x8 handwritten digits classifier. Great first non-MNIST NN example.',
    href: '/nn?mode=preset&template=digits_basic',
    badge: NN,
    tone: 'nn',
  },
  {
    title: 'AlexNet Cats vs Dogs',
    subtitle: 'Preset Project',
    description:
      'True AlexNet stack (conv/pool/fc) trained on Kaggle Cats vs Dogs at 96x96.',
    href: '/nn?mode=preset&template=alexnet_cats_dogs',
    badge: NN,
    tone: 'nn',
  },
  {
    title: 'Iris RF Classifier',
    subtitle: 'Preset Project',
    description: 'Classic low-dimensional classification with RandomForestClassifier.',
    href: '/rf?mode=preset&template=iris_basic',
    badge: RF,
    tone: 'rf',
  },
  {
    title: 'Wine Quality RF',
    subtitle: 'Preset Project',
    description: 'Multi-class RF setup tuned for wine quality tabular data.',
    href: '/rf?mode=preset&template=wine_quality',
    badge: RF,
    tone: 'rf',
  },
  {
    title: 'Breast Cancer RF',
    subtitle: 'Preset Project',
    description: 'Binary RF classifier preset for diagnostic feature vectors.',
    href: '/rf?mode=preset&template=breast_cancer_fast',
    badge: RF,
    tone: 'rf',
  },
  {
    title: 'Object Detection Camera Demo',
    subtitle: 'Preset Project',
    description: 'YOLOS-Tiny object detection starter with webcam testing flow.',
    href: '/vlm?mode=preset&template=object_detection_demo',
    badge: VLM,
    tone: 'vlm',
  },
  {
    title: 'Nemotron Nano 12B v2 [VL] Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: VLM,
    tone: 'vlm',
  },
  {
    title: 'Study Hours Regressor',
    subtitle: 'Preset Project',
    description:
      'Predict exam score from study hours with a beginner-friendly linear regression workflow.',
    href: '/linreg?mode=preset&template=study_hours_baseline',
    badge: LINREG,
    tone: 'linreg',
  },
  {
    title: 'Diabetes BMI Regressor',
    subtitle: 'Preset Project',
    description:
      'Real sklearn diabetes subset: predict one-year disease progression from BMI.',
    href: '/linreg?mode=preset&template=diabetes_bmi_baseline',
    badge: LINREG,
    tone: 'linreg',
  },
  {
    title: 'Ridge Regression Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: LINREG,
    tone: 'linreg',
  },
  {
    title: 'Lasso Regression Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: LINREG,
    tone: 'linreg',
  },
  {
    title: 'Elastic Net Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: LINREG,
    tone: 'linreg',
  },
  {
    title: 'Bayesian Linear Regression Visualizer',
    subtitle: 'Preset Project',
    description: 'Coming soon!',
    href: '#',
    badge: LINREG,
    tone: 'linreg',
  },
]

const LLM_PRESET_CARDS = PRESET_CARDS.filter((card) => card.tone === 'llm')
const NN_PRESET_CARDS = PRESET_CARDS.filter((card) => card.tone === 'nn')
const RF_PRESET_CARDS = PRESET_CARDS.filter((card) => card.tone === 'rf')
const VLM_PRESET_CARDS = PRESET_CARDS.filter((card) => card.tone === 'vlm')
const LINREG_PRESET_CARDS = PRESET_CARDS.filter((card) => card.tone === 'linreg')

function LaunchCardView({ card }: { card: LaunchCard }) {
  return (
    <a href={card.href} className={`hub-card hub-card-${card.tone} hub-card-link`}>
      <div className="hub-card-head">
        <div className="hub-card-subtitle">{card.subtitle}</div>
        <div className={`hub-badge hub-badge-${card.tone}`}>{card.badge}</div>
      </div>
      <h3 className="hub-card-title">{card.title}</h3>
      <p className="hub-card-description">{card.description}</p>
    </a>
  )
}

export default function ModelHubPage() {
  const [aiProvider, setAiProvider] = useState<AiProvider>(getStoredProvider)
  const [aiModel, setAiModel] = useState(() => getStoredModel(getStoredProvider()))
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [draftProvider, setDraftProvider] = useState<AiProvider>(aiProvider)
  const [draftModel, setDraftModel] = useState(aiModel)
  const [recOpen, setRecOpen] = useState(false)
  const [recPrompt, setRecPrompt] = useState('')
  const [recAnswer, setRecAnswer] = useState('')
  const [recLoading, setRecLoading] = useState(false)
  const [recSource, setRecSource] = useState('')
  const [recModel, setRecModel] = useState('')

  const openSettings = () => {
    setDraftProvider(aiProvider)
    setDraftModel(aiModel)
    setSettingsOpen(true)
  }

  const handleSettingsSave = () => {
    setAiProvider(draftProvider)
    setAiModel(draftModel)
    localStorage.setItem(AI_PROVIDER_KEY, draftProvider)
    localStorage.setItem(AI_MODEL_KEY, draftModel)
    setSettingsOpen(false)
  }

  const handleSettingsCancel = () => {
    setSettingsOpen(false)
  }

  const handleDraftProviderChange = (p: AiProvider) => {
    setDraftProvider(p)
    setDraftModel(PROVIDER_MODELS[p][0])
  }

  const handleRecommend = async () => {
    const text = recPrompt.trim()
    if (!text || recLoading) return
    setRecLoading(true)
    setRecAnswer('')
    setRecSource('')
    setRecModel('')
    try {
      const res = await fetch('/api/ai/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description: text,
          provider: aiProvider,
          model: aiModel,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = (await res.json()) as {
        recommendation?: string
        source?: string
        model?: string | null
      }
      setRecAnswer(data.recommendation || 'No recommendation received.')
      setRecSource(data.source || '')
      setRecModel((data.model ?? '').trim())
    } catch {
      setRecAnswer('Failed to get recommendation. Make sure the backend is running.')
      setRecSource('error')
      setRecModel('')
    } finally {
      setRecLoading(false)
    }
  }

  return (
    <div className="hub-root">
      <div className="hub-backdrop-grid" />
      <div className="hub-backdrop-glow" />
      <header className="hub-header">
        <div className="hub-header-top">
          <div>
            <div className="hub-kicker">Build, Tweak, Deploy.</div>
            <h1 className="hub-title">Welcome to Burn.</h1>
          </div>
        </div>
        <p className="hub-subtitle">
          Burn simplifies the process of building machine models from scratch, allowing anyone to
          harness the power of machine learning. Whether you're a student building their first
          classifier or a business designing an end-to-end demand forecasting engine, Burn is there
          to help.
        </p>
        {/* <p className="hub-subtitle">
          New: the Digits NN preset uses the built-in sklearn dataset and works without Kaggle setup.
        </p>
        <p className="hub-subtitle">
          VLM projects use Hugging Face object-detection models (default: YOLOS-Tiny) and include a
          camera testing interface.
        </p> */}
        <p className="hub-subtitle">
          Need to manage live model endpoints later? Open{' '}
          <a href="/deployments" className="hub-inline-link">
            Deployment Manager
          </a>
          .
        </p>
      </header>

      {/* Floating AI FAB (above settings) */}
      <button
        type="button"
        className="hub-ai-fab"
        onClick={() => setRecOpen(true)}
        title="AI Recommendation"
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 2l1.09 3.26L16 6l-2.91.74L12 10l-1.09-3.26L8 6l2.91-.74L12 2z" />
          <path d="M5 15l.54 1.63L7 17.5l-1.46.37L5 19.5l-.54-1.63L3 17.5l1.46-.37L5 15z" />
          <path d="M19 11l.54 1.63L21 13l-1.46.37L19 15l-.54-1.63L17 13l1.46-.37L19 11z" />
        </svg>
      </button>

      {/* Floating Settings FAB (bottom-right) */}
      <button type="button" className="hub-settings-fab" onClick={openSettings} title="AI Settings">
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
      </button>

      {/* AI Settings Modal */}
      {settingsOpen && (
        <div className="hub-recommend-overlay" onClick={handleSettingsCancel}>
          <div className="hub-recommend-modal" onClick={(e) => e.stopPropagation()}>
            <div className="hub-recommend-modal-header">
              <div className="hub-recommend-modal-title-row">
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="12" cy="12" r="3" />
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                </svg>
                <h2 className="hub-recommend-title">AI Model Settings</h2>
              </div>
              <button type="button" className="hub-recommend-close" onClick={handleSettingsCancel}>
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
            <p className="hub-recommend-subtitle">
              Choose the AI provider and model used for coaching and recommendations.
            </p>
            <div className="hub-settings-field">
              <label className="hub-settings-label">Provider</label>
              <select
                value={draftProvider}
                onChange={(e) => handleDraftProviderChange(e.target.value as AiProvider)}
                className="hub-settings-select"
              >
                <option value="openai">OpenAI</option>
                <option value="gemini">Gemini</option>
                <option value="anthropic">Anthropic</option>
                <option value="nvidia">NVIDIA</option>
              </select>
            </div>
            <div className="hub-settings-field">
              <label className="hub-settings-label">Model</label>
              <select
                value={draftModel}
                onChange={(e) => setDraftModel(e.target.value)}
                className="hub-settings-select"
              >
                {PROVIDER_MODELS[draftProvider].map((entry) => (
                  <option key={entry} value={entry}>
                    {entry}
                  </option>
                ))}
              </select>
            </div>
            <div className="hub-settings-actions">
              <button
                type="button"
                className="hub-settings-cancel-btn"
                onClick={handleSettingsCancel}
              >
                Cancel
              </button>
              <button type="button" className="hub-settings-save-btn" onClick={handleSettingsSave}>
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* AI Recommend Modal */}
      {recOpen && (
        <div className="hub-recommend-overlay" onClick={() => setRecOpen(false)}>
          <div className="hub-recommend-modal" onClick={(e) => e.stopPropagation()}>
            <div className="hub-recommend-modal-header">
              <div className="hub-recommend-modal-title-row">
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M12 2l1.09 3.26L16 6l-2.91.74L12 10l-1.09-3.26L8 6l2.91-.74L12 2z" />
                  <path d="M5 15l.54 1.63L7 17.5l-1.46.37L5 19.5l-.54-1.63L3 17.5l1.46-.37L5 15z" />
                  <path d="M19 11l.54 1.63L21 13l-1.46.37L19 15l-.54-1.63L17 13l1.46-.37L19 11z" />
                </svg>
                <h2 className="hub-recommend-title">AI Algorithm Advisor</h2>
              </div>
              <button type="button" className="hub-recommend-close" onClick={() => setRecOpen(false)}>
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
            <p className="hub-recommend-subtitle">
              Describe your problem, data, or goal and AI will recommend the best machine learning
              approach.
            </p>
            <div className="hub-recommend-input-row">
              <textarea
                className="hub-recommend-input"
                placeholder="e.g. I have 10,000 images of cats and dogs and want to classify them..."
                value={recPrompt}
                onChange={(e) => setRecPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    void handleRecommend()
                  }
                }}
                rows={3}
              />
            </div>
            <button
              type="button"
              className="hub-recommend-btn"
              disabled={recLoading || recPrompt.trim().length === 0}
              onClick={() => void handleRecommend()}
            >
              {recLoading ? 'Thinking…' : 'Get Recommendation'}
            </button>
            {recAnswer && (
              <div className="hub-recommend-answer">
                <div className="hub-recommend-answer-text">{recAnswer}</div>
                {recSource && recSource !== 'error' && recSource !== 'unavailable' && (
                  <div className="hub-recommend-answer-source">
                    Powered by {recSource.toUpperCase()}{recModel ? ` · ${recModel}` : ''}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      <main className="hub-main">
        <section className="hub-section">
          <h2 className="hub-section-title">Start From Scratch</h2>
          <p className="hub-section-subtitle">
            Open a clean workspace and build your pipeline layer-by-layer.
          </p>
          <div className="hub-grid hub-grid-scratch">
            {SCRATCH_CARDS.map((card) => (
              <LaunchCardView key={card.title} card={card} />
            ))}
          </div>
        </section>

        <section className="hub-section">
          <h2 className="hub-section-title">Preset Example Projects</h2>
          <p className="hub-section-subtitle">
            Ready-to-run projects grouped by model family so you can compare workflows quickly.
          </p>
          <div className="hub-family-grid">
            <article className="hub-family-panel hub-family-panel-llm">
              <header className="hub-family-head">
                <div className="hub-family-label">Large Language Models</div>
                <div className="hub-family-meta">Transformers · Text Generation</div>
              </header>
              <div className="hub-grid hub-grid-family">
                {LLM_PRESET_CARDS.map((card) => (
                  <LaunchCardView key={card.title} card={card} />
                ))}
              </div>
            </article>

            <article className="hub-family-panel hub-family-panel-nn">
              <header className="hub-family-head">
                <div className="hub-family-label">Neural Networks</div>
                <div className="hub-family-meta">PyTorch · 3D Layer Builder</div>
              </header>
              <div className="hub-grid hub-grid-family">
                {NN_PRESET_CARDS.map((card) => (
                  <LaunchCardView key={card.title} card={card} />
                ))}
              </div>
            </article>

            <article className="hub-family-panel hub-family-panel-rf">
              <header className="hub-family-head">
                <div className="hub-family-label">Random Forest</div>
                <div className="hub-family-meta">scikit-learn · 3D Node Pipeline</div>
              </header>
              <div className="hub-grid hub-grid-family">
                {RF_PRESET_CARDS.map((card) => (
                  <LaunchCardView key={card.title} card={card} />
                ))}
              </div>
            </article>

            <article className="hub-family-panel hub-family-panel-vlm">
              <header className="hub-family-head">
                <div className="hub-family-label">Vision-Language Models</div>
                <div className="hub-family-meta">Hugging Face · Camera Object Detection</div>
              </header>
              <div className="hub-grid hub-grid-family">
                {VLM_PRESET_CARDS.map((card) => (
                  <LaunchCardView key={card.title} card={card} />
                ))}
              </div>
            </article>

            <article className="hub-family-panel hub-family-panel-linreg">
              <header className="hub-family-head">
                <div className="hub-family-label">Linear Regression</div>
                <div className="hub-family-meta">Gradient Descent · Fit Visualization</div>
              </header>
              <div className="hub-grid hub-grid-family">
                {LINREG_PRESET_CARDS.map((card) => (
                  <LaunchCardView key={card.title} card={card} />
                ))}
              </div>
            </article>
          </div>
        </section>
      </main>
    </div>
  )
}
