import { useCallback, useEffect, useMemo, useState } from 'react'

type CoachTab = 'validate' | 'train' | 'infer'

interface NnAiCoachPanelProps {
  tab: CoachTab
  layerCount: number
  neuronCount: number
  weightCount: number
  activation: string
  currentEpoch: number
  totalEpochs: number
  trainLoss: number | null
  testLoss: number | null
  trainAccuracy: number | null
  testAccuracy: number | null
  trainingStatus: string
  inferenceTopPrediction: number | null
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

type AiProvider = 'openai' | 'gemini' | 'anthropic' | 'nvidia'

const PROVIDER_MODELS: Record<AiProvider, string[]> = {
  openai: ['gpt-4o-mini', 'gpt-4.1-mini', 'gpt-4o'],
  gemini: ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'],
  anthropic: ['claude-3-5-haiku-latest', 'claude-3-5-sonnet-latest'],
  nvidia: ['meta/llama-3.1-70b-instruct', 'meta/llama-3.1-8b-instruct'],
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

/* ── AI Coach Chat Panel (sparkle button) ── */

export function NnAiCoachPanel({
  tab,
  layerCount,
  neuronCount,
  weightCount,
  activation,
  currentEpoch,
  totalEpochs,
  trainLoss,
  testLoss,
  trainAccuracy,
  testAccuracy,
  trainingStatus,
  inferenceTopPrediction,
}: NnAiCoachPanelProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [initialAnswer, setInitialAnswer] = useState('')
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [prompt, setPrompt] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [source, setSource] = useState<'openai' | 'gemini' | 'anthropic' | 'nvidia' | 'unavailable' | 'error'>('unavailable')

  const provider = getStoredProvider()
  const model = getStoredModel(provider)

  const title = tab === 'validate'
    ? 'AI Build Coach'
    : tab === 'train'
      ? 'AI Train Coach'
      : 'AI Test Coach'

  const payload = useMemo(() => ({
    tab,
    layerCount,
    neuronCount,
    weightCount,
    activation,
    currentEpoch,
    totalEpochs,
    trainLoss,
    testLoss,
    trainAccuracy,
    testAccuracy,
    trainingStatus,
    inferenceTopPrediction,
    provider,
    model,
  }), [
    tab,
    layerCount,
    neuronCount,
    weightCount,
    activation,
    currentEpoch,
    totalEpochs,
    trainLoss,
    testLoss,
    trainAccuracy,
    testAccuracy,
    trainingStatus,
    inferenceTopPrediction,
    provider,
    model,
  ])

  const askCoach = useCallback(async (promptOverride?: string, history?: ChatMessage[]) => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/ai/nn-coach', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...payload,
          prompt: promptOverride ?? undefined,
          messages: history ?? [],
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const data = await response.json() as {
        tips?: string[]
        answer?: string
        source?: 'openai' | 'gemini' | 'anthropic' | 'nvidia' | 'unavailable' | 'error'
      }

      const nextSource =
        data.source === 'openai' ||
        data.source === 'gemini' ||
        data.source === 'anthropic' ||
        data.source === 'nvidia'
          ? data.source
          : data.source === 'error'
            ? 'error'
            : 'unavailable'
      setSource(nextSource)
      const incomingAnswer = (data.answer ?? '').trim()

      if (nextSource !== 'error' && nextSource !== 'unavailable' && incomingAnswer) {
        return incomingAnswer
      }

      return ''
    } catch {
      setSource('error')
      return ''
    } finally {
      setIsLoading(false)
    }
  }, [payload])

  useEffect(() => {
    setInitialAnswer('')
    setChatMessages([])
    setPrompt('')
    setSource('unavailable')
  }, [tab, provider, model])

  useEffect(() => {
    if (!isOpen) return

    let cancelled = false
    void (async () => {
      const first = await askCoach()
      if (!cancelled && first) {
        setInitialAnswer(first)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [askCoach, isOpen])

  const handleAsk = async () => {
    const userText = prompt.trim()
    if (!userText || isLoading) return

    const history: ChatMessage[] = [
      ...chatMessages,
      { role: 'user', content: userText },
    ]
    setChatMessages(history)
    setPrompt('')

    const reply = await askCoach(userText, history)
    if (reply) {
      setChatMessages((prev) => [...prev, { role: 'assistant', content: reply }])
    }
  }

  return (
    <div className="nn-ai-coach-dock" aria-label="AI model coach">
      <button
        type="button"
        className="nn-ai-coach-toggle"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
      >
        {isOpen ? '×' : (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2l1.09 3.26L16 6l-2.91.74L12 10l-1.09-3.26L8 6l2.91-.74L12 2z" />
            <path d="M5 15l.54 1.63L7 17.5l-1.46.37L5 19.5l-.54-1.63L3 17.5l1.46-.37L5 15z" />
            <path d="M19 11l.54 1.63L21 13l-1.46.37L19 15l-.54-1.63L17 13l1.46-.37L19 11z" />
          </svg>
        )}
      </button>

      {isOpen && (
        <aside className="nn-ai-coach">
          <div className="nn-ai-coach-header">
            <span className="nn-ai-coach-badge">AI</span>
            <h3 className="nn-ai-coach-title">{title}</h3>
            <button
              type="button"
              onClick={() => {
                setChatMessages([])
                setPrompt('')
                void (async () => {
                  const first = await askCoach()
                  setInitialAnswer(first)
                })()
              }}
              disabled={isLoading}
              className="nn-ai-coach-refresh"
            >
              {isLoading ? 'Thinking…' : 'Refresh'}
            </button>
          </div>

          <div className="nn-ai-coach-source">
            {formatSourceLabel(source)} &middot; {model}
          </div>

          {initialAnswer ? (
            <div className="nn-ai-coach-answer">{initialAnswer}</div>
          ) : isLoading && !initialAnswer ? (
            <div className="nn-ai-coach-answer nn-ai-coach-loading">Thinking…</div>
          ) : null}

          {chatMessages.length > 0 && (
            <div className="nn-ai-chat-list">
              {chatMessages.map((msg, index) => (
                <div key={`${msg.role}-${index}`} className={`nn-ai-chat-bubble nn-ai-chat-${msg.role}`}>
                  {msg.content}
                </div>
              ))}
            </div>
          )}

          <div className="nn-ai-coach-prompt-wrap">
            <textarea
              className="nn-ai-coach-prompt"
              placeholder="Ask AI coach…"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  void handleAsk()
                }
              }}
              rows={2}
            />
            <button
              type="button"
              className="nn-ai-coach-ask"
              disabled={isLoading || prompt.trim().length === 0}
              onClick={() => void handleAsk()}
            >
              {isLoading ? 'Asking…' : 'Ask AI'}
            </button>
          </div>
        </aside>
      )}
    </div>
  )
}

function formatSourceLabel(source: 'openai' | 'gemini' | 'anthropic' | 'nvidia' | 'unavailable' | 'error'): string {
  if (source === 'openai') return 'OpenAI'
  if (source === 'gemini') return 'Gemini'
  if (source === 'anthropic') return 'Anthropic'
  if (source === 'nvidia') return 'NVIDIA'
  if (source === 'error') return 'Error'
  return 'Unavailable'
}
