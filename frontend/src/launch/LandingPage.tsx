import WaveBackground from './WaveBackground'
import './landing.css'

export default function LandingPage() {
  return (
    <div className='landing-root'>
      <main className='landing-shell'>
        <a href='/builders' className='landing-hero'>
          <WaveBackground className='landing-hero-wave' />
          <div className='landing-hero-tint' />
          <div className='landing-hero-content'>
            <p className='landing-kicker'>Build, Tweak, Deploy.</p>
            <h1 className='landing-title'>Welcome to Burn.</h1>

            {/* <p className='landing-subtitle'>
              Burn simplifies the process of building machine models from
              scratch, allowing anyone to harness the power of machine learning.
              Whether you are a student building your first classifier or a
              business designing an end-to-end forecasting engine, Burn gives
              you one visual, interactive workspace to move faster.
            </p> */}
            {/* <a  className='landing-primary-link'>
              Go To Builder
            </a> */}
          </div>
        </a>
      </main>
    </div>
  )
}
