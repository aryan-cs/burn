import './modelHub.css'

const NN = 'Neural Network'
const RF = 'Random Forest'
const VLM = 'Vision-Language'
const SVM = 'SVM'
const PCA = 'PCA'
const LINREG = 'Linear Regression'
const LOGREG = 'Logistic Regression'

type LaunchCard = {
  title: string
  subtitle: string
  description: string
  href: string
  badge: string
  tone: 'nn' | 'rf' | 'vlm' | 'svm' | 'pca' | 'linreg' | 'logreg'
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
    description: 'Coming soon!',
    href: '#',
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
    title: 'MNIST Baseline MLP',
    subtitle: 'Preset Project',
    description: 'Input -> Flatten -> Dense -> Output. Good default for first compile/train.',
    href: '/nn?mode=preset&template=mnist_basic',
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
]

const NN_PRESET_CARDS = PRESET_CARDS.filter((card) => card.tone === 'nn')
const RF_PRESET_CARDS = PRESET_CARDS.filter((card) => card.tone === 'rf')
const VLM_PRESET_CARDS = PRESET_CARDS.filter((card) => card.tone === 'vlm')

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
  return (
    <div className="hub-root">
      <div className="hub-backdrop-grid" />
      <div className="hub-backdrop-glow" />
      <header className="hub-header">
        <div className="hub-kicker">Build, Tweak, Deploy.</div>
        <h1 className="hub-title">Welcome to Burn.</h1>
        <p className="hub-subtitle">
          Burn simplifies the process of building machine models from scratch, allowing anyone to harness the power of machine learning.
          Whether you're a student building their first classifier or a business designing an end-to-end demand forecasting engine, Burn is there to help.
        </p>
        <p className="hub-subtitle">
          New: the Digits NN preset uses the built-in sklearn dataset and works without Kaggle setup.
        </p>
        <p className="hub-subtitle">
          VLM projects use Hugging Face object-detection models (default: YOLOS-Tiny) and include a camera testing interface.
        </p>
        <p className="hub-subtitle">
          Need to manage live model endpoints later? Open{' '}
          <a href="/deployments" className="hub-inline-link">
            Deployment Manager
          </a>
          .
        </p>
      </header>

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
          </div>
        </section>
      </main>
    </div>
  )
}
