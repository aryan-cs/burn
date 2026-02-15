import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

type CoachTab = 'validate' | 'train' | 'infer'

type AiSource = 'openai' | 'gemini' | 'anthropic' | 'nvidia' | 'unavailable' | 'error' | 'idle'

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

interface RecordingSession {
  stream: MediaStream
  audioContext: AudioContext
  sourceNode: MediaStreamAudioSourceNode
  processorNode: ScriptProcessorNode
  gainNode: GainNode
  chunks: Float32Array[]
  sampleRate: number
}

interface PlaybackSession {
  audio: HTMLAudioElement
  url: string
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

/* AI Coach drawer */

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
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [prompt, setPrompt] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [source, setSource] = useState<AiSource>('idle')

  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isSynthesizing, setIsSynthesizing] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [voiceError, setVoiceError] = useState('')

  const recordingSessionRef = useRef<RecordingSession | null>(null)
  const playbackSessionRef = useRef<PlaybackSession | null>(null)

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

  const latestAssistantMessage = useMemo(
    () => [...chatMessages].reverse().find((msg) => msg.role === 'assistant')?.content ?? '',
    [chatMessages],
  )

  const stopPlayback = useCallback(() => {
    const session = playbackSessionRef.current
    if (!session) return

    session.audio.pause()
    session.audio.currentTime = 0
    session.audio.onended = null
    session.audio.onerror = null
    URL.revokeObjectURL(session.url)
    playbackSessionRef.current = null
    setIsSpeaking(false)
  }, [])

  const stopRecordingSession = useCallback((session?: RecordingSession | null) => {
    const current = session ?? recordingSessionRef.current
    if (!current) return

    current.processorNode.onaudioprocess = null
    current.processorNode.disconnect()
    current.sourceNode.disconnect()
    current.gainNode.disconnect()

    for (const track of current.stream.getTracks()) {
      track.stop()
    }

    void current.audioContext.close()

    if (recordingSessionRef.current === current) {
      recordingSessionRef.current = null
    }
  }, [])

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
    setChatMessages([])
    setPrompt('')
    setSource('idle')
    setVoiceError('')
    setIsRecording(false)
    stopRecordingSession()
    stopPlayback()
  }, [tab, provider, model, stopPlayback, stopRecordingSession])

  useEffect(() => {
    return () => {
      stopRecordingSession()
      stopPlayback()
    }
  }, [stopPlayback, stopRecordingSession])

  const handleAsk = async () => {
    const userText = prompt.trim()
    if (!userText || isLoading) return

    setVoiceError('')

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

  const handleRefresh = async () => {
    if (isLoading) return

    setVoiceError('')
    setChatMessages([])
    setPrompt('')

    const first = await askCoach()
    if (first) {
      setChatMessages([{ role: 'assistant', content: first }])
    }
  }

  const handleStartRecording = async () => {
    if (isRecording || isTranscribing) return

    setVoiceError('')

    if (!navigator.mediaDevices?.getUserMedia) {
      setVoiceError('Microphone capture is not supported in this browser.')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const AudioContextCtor = window.AudioContext ?? (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
      if (!AudioContextCtor) {
        stream.getTracks().forEach((track) => track.stop())
        throw new Error('Web Audio API is not available.')
      }

      const audioContext = new AudioContextCtor()
      const sourceNode = audioContext.createMediaStreamSource(stream)
      const processorNode = audioContext.createScriptProcessor(4096, 1, 1)
      const gainNode = audioContext.createGain()
      gainNode.gain.value = 0

      const chunks: Float32Array[] = []
      processorNode.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0)
        chunks.push(new Float32Array(input))
      }

      sourceNode.connect(processorNode)
      processorNode.connect(gainNode)
      gainNode.connect(audioContext.destination)

      recordingSessionRef.current = {
        stream,
        audioContext,
        sourceNode,
        processorNode,
        gainNode,
        chunks,
        sampleRate: audioContext.sampleRate,
      }

      setIsRecording(true)
    } catch {
      setVoiceError('Microphone access is blocked or unavailable.')
      setIsRecording(false)
      stopRecordingSession()
    }
  }

  const handleStopRecording = async () => {
    const session = recordingSessionRef.current
    if (!session || isTranscribing) return

    setIsRecording(false)
    stopRecordingSession(session)

    const merged = mergeAudioChunks(session.chunks)
    if (merged.length === 0) {
      setVoiceError('No audio was captured. Try recording again.')
      return
    }

    const wavBlob = encodeWavBlob(merged, session.sampleRate)
    const formData = new FormData()
    formData.append('audio', wavBlob, 'speech.wav')

    setIsTranscribing(true)
    try {
      const response = await fetch('/api/ai/stt/whisper', {
        method: 'POST',
        body: formData,
      })

      const body = await response.text()
      if (!response.ok) {
        throw new Error(parseApiError(body, response.status))
      }

      const parsed = body ? JSON.parse(body) as { text?: string } : {}
      const transcript = (parsed.text ?? '').trim()
      if (!transcript) {
        setVoiceError('Whisper did not detect speech from the recording.')
        return
      }

      setPrompt((prev) => {
        const existing = prev.trimEnd()
        return existing.length > 0 ? `${existing} ${transcript}` : transcript
      })
    } catch (error) {
      setVoiceError(error instanceof Error ? error.message : 'Speech-to-text failed.')
    } finally {
      setIsTranscribing(false)
    }
  }

  const handleToggleRecording = async () => {
    if (isRecording) {
      await handleStopRecording()
      return
    }
    await handleStartRecording()
  }

  const handleSpeak = async (text: string) => {
    const normalized = text.trim()
    if (!normalized || isSynthesizing) return

    setVoiceError('')
    stopPlayback()
    setIsSynthesizing(true)

    try {
      const response = await fetch('/api/ai/tts/chatterbox', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: normalized }),
      })

      if (!response.ok) {
        const body = await response.text()
        throw new Error(parseApiError(body, response.status))
      }

      const audioBlob = await response.blob()
      if (audioBlob.size === 0) {
        throw new Error('Text-to-speech returned empty audio.')
      }

      const url = URL.createObjectURL(audioBlob)
      const audio = new Audio(url)

      playbackSessionRef.current = { audio, url }
      audio.onended = () => {
        stopPlayback()
      }
      audio.onerror = () => {
        setVoiceError('Audio playback failed.')
        stopPlayback()
      }

      setIsSpeaking(true)
      await audio.play()
    } catch (error) {
      setVoiceError(error instanceof Error ? error.message : 'Text-to-speech failed.')
      stopPlayback()
    } finally {
      setIsSynthesizing(false)
    }
  }

  const handleSpeakLatest = async () => {
    if (!latestAssistantMessage) return

    if (isSpeaking) {
      stopPlayback()
      return
    }

    await handleSpeak(latestAssistantMessage)
  }

  return (
    <div
      className={`nn-ai-coach-dock ${isOpen ? 'nn-ai-coach-dock-open' : 'nn-ai-coach-dock-closed'}`}
      aria-label="AI model coach"
    >
      <button
        type="button"
        className="nn-ai-coach-side-toggle"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
        aria-label={isOpen ? 'Collapse AI coach panel' : 'Expand AI coach panel'}
        title={isOpen ? 'Collapse AI coach' : 'Expand AI coach'}
      >
        <span className="nn-ai-coach-side-icon">{isOpen ? '>' : '<'}</span>
        <span className="nn-ai-coach-side-text">AI</span>
      </button>

      <aside className="nn-ai-coach" aria-hidden={!isOpen}>
        <div className="nn-ai-coach-header">
          <span className="nn-ai-coach-badge">AI</span>
          <h3 className="nn-ai-coach-title">{title}</h3>
          <button
            type="button"
            onClick={() => void handleRefresh()}
            disabled={isLoading}
            className="nn-ai-coach-refresh"
          >
            {isLoading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>

        <div className="nn-ai-coach-source">
          {formatSourceLabel(source)} | {model}
        </div>

        {chatMessages.length > 0 ? (
          <div className="nn-ai-chat-list">
            {chatMessages.map((msg, index) => (
              <div
                key={`${msg.role}-${index}`}
                className={`nn-ai-chat-bubble nn-ai-chat-${msg.role}`}
              >
                <div className="nn-ai-chat-content">{msg.content}</div>
                {msg.role === 'assistant' ? (
                  <button
                    type="button"
                    className="nn-ai-chat-speak"
                    onClick={() => void handleSpeak(msg.content)}
                    disabled={isSynthesizing}
                  >
                    {isSynthesizing ? 'Synthesizing...' : 'Speak'}
                  </button>
                ) : null}
              </div>
            ))}
          </div>
        ) : (
          <div className="nn-ai-coach-empty">
            Ask a question, or use the mic to transcribe with Whisper.
          </div>
        )}

        {voiceError ? <div className="nn-ai-coach-error">{voiceError}</div> : null}

        <div className="nn-ai-coach-prompt-wrap">
          <textarea
            className="nn-ai-coach-prompt"
            placeholder="Ask AI coach..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void handleAsk()
              }
            }}
            rows={3}
          />

          <div className="nn-ai-coach-actions">
            <button
              type="button"
              className={`nn-ai-coach-voice ${isRecording ? 'nn-ai-coach-voice-recording' : ''}`}
              onClick={() => void handleToggleRecording()}
              disabled={isTranscribing || isLoading}
            >
              {isRecording ? 'Stop Mic' : isTranscribing ? 'Transcribing...' : 'Mic'}
            </button>

            <button
              type="button"
              className="nn-ai-coach-voice"
              onClick={() => void handleSpeakLatest()}
              disabled={!latestAssistantMessage || isSynthesizing}
            >
              {isSpeaking ? 'Stop Audio' : isSynthesizing ? 'Synthesizing...' : 'Speak Last'}
            </button>

            <button
              type="button"
              className="nn-ai-coach-ask"
              disabled={isLoading || prompt.trim().length === 0}
              onClick={() => void handleAsk()}
            >
              {isLoading ? 'Asking...' : 'Ask AI'}
            </button>
          </div>
        </div>
      </aside>
    </div>
  )
}

function parseApiError(body: string, status: number): string {
  if (!body) return `HTTP ${status}`

  try {
    const parsed = JSON.parse(body) as {
      detail?: { message?: string } | string
      message?: string
    }

    if (typeof parsed.detail === 'string') return parsed.detail
    if (parsed.detail?.message) return parsed.detail.message
    if (parsed.message) return parsed.message
  } catch {
    return body
  }

  return body
}

function mergeAudioChunks(chunks: Float32Array[]): Float32Array {
  if (chunks.length === 0) return new Float32Array(0)

  let totalLength = 0
  for (const chunk of chunks) {
    totalLength += chunk.length
  }

  const result = new Float32Array(totalLength)
  let offset = 0
  for (const chunk of chunks) {
    result.set(chunk, offset)
    offset += chunk.length
  }

  return result
}

function encodeWavBlob(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2
  const blockAlign = bytesPerSample
  const byteRate = sampleRate * blockAlign
  const dataSize = samples.length * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  writeAsciiString(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeAsciiString(view, 8, 'WAVE')
  writeAsciiString(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, 16, true)
  writeAsciiString(view, 36, 'data')
  view.setUint32(40, dataSize, true)

  let offset = 44
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
    view.setInt16(offset, int16, true)
    offset += bytesPerSample
  }

  return new Blob([buffer], { type: 'audio/wav' })
}

function writeAsciiString(view: DataView, offset: number, value: string): void {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i))
  }
}

function formatSourceLabel(source: AiSource): string {
  if (source === 'openai') return 'OpenAI'
  if (source === 'gemini') return 'Gemini'
  if (source === 'anthropic') return 'Anthropic'
  if (source === 'nvidia') return 'NVIDIA'
  if (source === 'error') return 'Error'
  if (source === 'unavailable') return 'Unavailable'
  return 'Ready'
}
