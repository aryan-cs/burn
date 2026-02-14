import './modelHub.css'

type LaunchCard = {
  title: string
  subtitle: string
  description: string
  href: string
  badge: string
  tone: 'nn' | 'rf'
}

const SCRATCH_CARDS: LaunchCard[] = [
  {
    title: 'Neural Network Builder',
    subtitle: 'Start From Scratch',
    description: 'Blank NN canvas with interactive 3D layer graph editing.',
    href: '/nn?mode=scratch',
    badge: 'NN',
    tone: 'nn',
  },
  {
    title: 'Random Forest Builder',
    subtitle: 'Start From Scratch',
    description: 'Blank RF workspace with 3D node pipeline and sklearn backend.',
    href: '/rf?mode=scratch',
    badge: 'RF',
    tone: 'rf',
  },
]

const PRESET_CARDS: LaunchCard[] = [
  {
    title: 'MNIST Baseline MLP',
    subtitle: 'Preset Project',
    description: 'Input → Flatten → Dense → Output. Good default for first compile/train.',
    href: '/nn?mode=preset&template=mnist_basic',
    badge: 'NN',
    tone: 'nn',
  },
  {
    title: 'MNIST Dropout Stack',
    subtitle: 'Preset Project',
    description: 'Adds dropout and deeper hidden path for experimentation.',
    href: '/nn?mode=preset&template=mnist_dropout',
    badge: 'NN',
    tone: 'nn',
  },
  {
    title: 'Iris RF Classifier',
    subtitle: 'Preset Project',
    description: 'Classic low-dimensional classification with RandomForestClassifier.',
    href: '/rf?mode=preset&template=iris_basic',
    badge: 'RF',
    tone: 'rf',
  },
  {
    title: 'Wine Quality RF',
    subtitle: 'Preset Project',
    description: 'Multi-class RF setup tuned for wine quality tabular data.',
    href: '/rf?mode=preset&template=wine_quality',
    badge: 'RF',
    tone: 'rf',
  },
  {
    title: 'Breast Cancer RF',
    subtitle: 'Preset Project',
    description: 'Binary RF classifier preset for diagnostic feature vectors.',
    href: '/rf?mode=preset&template=breast_cancer_fast',
    badge: 'RF',
    tone: 'rf',
  },
]

function LaunchCardView({ card }: { card: LaunchCard }) {
  return (
    <article className={`hub-card hub-card-${card.tone}`}>
      <div className="hub-card-head">
        <div className="hub-card-subtitle">{card.subtitle}</div>
        <div className={`hub-badge hub-badge-${card.tone}`}>{card.badge}</div>
      </div>
      <h3 className="hub-card-title">{card.title}</h3>
      <p className="hub-card-description">{card.description}</p>
      <div className="hub-card-footer">
        <a href={card.href} className={`hub-launch-btn hub-launch-btn-${card.tone}`}>
          Launch Project
        </a>
      </div>
    </article>
  )
}

export default function ModelHubPage() {
  return (
    <div className="hub-root">
      <div className="hub-backdrop-grid" />
      <div className="hub-backdrop-glow" />
      <header className="hub-header">
        <div className="hub-kicker">Machine Learning Studio</div>
        <h1 className="hub-title">Choose Your Build Path</h1>
        <p className="hub-subtitle">
          Start from an empty builder or launch a preset project that is configured to run immediately.
          RF presets require Kaggle credentials in the backend environment.
        </p>
      </header>

      <main className="hub-main">
        <section className="hub-section">
          <h2 className="hub-section-title">Start From Scratch</h2>
          <div className="hub-grid hub-grid-scratch">
            {SCRATCH_CARDS.map((card) => (
              <LaunchCardView key={card.title} card={card} />
            ))}
          </div>
        </section>

        <section className="hub-section">
          <h2 className="hub-section-title">Preset Example Projects</h2>
          <div className="hub-grid hub-grid-preset">
            {PRESET_CARDS.map((card) => (
              <LaunchCardView key={card.title} card={card} />
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}
