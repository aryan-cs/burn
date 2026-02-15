import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

type CoachTab = 'validate' | 'train' | 'infer'
type AiSource = 'openai' | 'gemini' | 'anthropic' | 'nvidia' | 'unavailable' | 'error' | 'idle'
type SttMode = 'probing' | 'whisper' | 'browser' | 'unavailable'

type ChatRole = 'user' | 'assistant'
type ChatMessageKind = 'normal' | 'status'

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
  role: ChatRole
  content: string
  source?: AiSource
  kind?: ChatMessageKind
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

interface CoachReply {
  answer: string
  source: AiSource
  reason?: string
}

interface AiCapabilitiesPayload {
  stt?: {
    whisper?: {
      available?: boolean
      reason?: string | null
      model?: string | null
    }
  }
}

interface BrowserSpeechRecognition extends EventTarget {
  lang: string
  continuous: boolean
  interimResults: boolean
  onresult: ((event: Event) => void) | null
  onerror: ((event: Event) => void) | null
  onend: (() => void) | null
  start(): void
  stop(): void
}

type BrowserSpeechRecognitionConstructor = new () => BrowserSpeechRecognition

type AiProvider = 'openai' | 'gemini' | 'anthropic' | 'nvidia'

const PROVIDER_MODELS: Record<AiProvider, string[]> = {
  openai: ['gpt-4o-mini', 'gpt-4.1-mini', 'gpt-4o'],
  gemini: ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'],
  anthropic: ['claude-3-5-haiku-latest', 'claude-3-5-sonnet-latest'],
  nvidia: ['meta/llama-3.1-70b-instruct', 'meta/llama-3.1-8b-instruct'],
}

const AI_PROVIDER_KEY = 'burn.ai.provider'
const AI_MODEL_KEY = 'burn.ai.model'
const COACH_TIMEOUT_MS = 20_000
const COACH_RETRIES = 1

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

function getSpeechRecognitionCtor(): BrowserSpeechRecognitionConstructor | null {
  if (typeof window === 'undefined') return null
  const withCtor = window as Window & {
    SpeechRecognition?: BrowserSpeechRecognitionConstructor
    webkitSpeechRecognition?: BrowserSpeechRecognitionConstructor
  }
  return withCtor.SpeechRecognition ?? withCtor.webkitSpeechRecognition ?? null
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
  const [isBrowserListening, setIsBrowserListening] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isSynthesizing, setIsSynthesizing] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [voiceError, setVoiceError] = useState('')

  const [sttMode, setSttMode] = useState<SttMode>('probing')
  const [sttReason, setSttReason] = useState('')

  const recordingSessionRef = useRef<RecordingSession | null>(null)
  const playbackSessionRef = useRef<PlaybackSession | null>(null)
  const browserRecognitionRef = useRef<BrowserSpeechRecognition | null>(null)

  const provider = getStoredProvider()
  const model = getStoredModel(provider)

  const title = tab === 'validate'
    ? 'AI Build Coach'
    : tab === 'train'
      ? 'AI Train Coach'
      : 'AI Test Coach'

  const normalizedLayerCount = toNonNegativeInt(layerCount)
  const normalizedNeuronCount = toNonNegativeInt(neuronCount)
  const normalizedWeightCount = toNonNegativeInt(weightCount)
  const normalizedTotalEpochs = Math.max(1, toNonNegativeInt(totalEpochs))
  const normalizedCurrentEpoch = Math.min(
    normalizedTotalEpochs,
    toNonNegativeInt(currentEpoch)
  )
  const normalizedInferenceTopPrediction = toOptionalInt(inferenceTopPrediction)

  const payload = useMemo(() => ({
    tab,
    layerCount: normalizedLayerCount,
    neuronCount: normalizedNeuronCount,
    weightCount: normalizedWeightCount,
    activation,
    currentEpoch: normalizedCurrentEpoch,
    totalEpochs: normalizedTotalEpochs,
    trainLoss,
    testLoss,
    trainAccuracy,
    testAccuracy,
    trainingStatus,
    inferenceTopPrediction: normalizedInferenceTopPrediction,
    provider,
    model,
  }), [
    tab,
    normalizedLayerCount,
    normalizedNeuronCount,
    normalizedWeightCount,
    activation,
    normalizedCurrentEpoch,
    normalizedTotalEpochs,
    trainLoss,
    testLoss,
    trainAccuracy,
    testAccuracy,
    trainingStatus,
    normalizedInferenceTopPrediction,
    provider,
    model,
  ])

  const latestAssistantMessage = useMemo(
    () => [...chatMessages].reverse().find((msg) => msg.role === 'assistant' && msg.kind !== 'status')?.content ?? '',
    [chatMessages],
  )

  const browserSpeechSupported = useMemo(() => getSpeechRecognitionCtor() !== null, [])

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

  const stopBrowserRecognition = useCallback(() => {
    const recognition = browserRecognitionRef.current
    if (!recognition) return

    recognition.onresult = null
    recognition.onerror = null
    recognition.onend = null
    recognition.stop()
    browserRecognitionRef.current = null
    setIsBrowserListening(false)
  }, [])

  const probeCapabilities = useCallback(async () => {
    setSttMode('probing')
    setSttReason('')

    const fallbackToBrowser = () => {
      if (browserSpeechSupported) {
        setSttMode('browser')
        setSttReason('Whisper unavailable. Using browser speech recognition fallback.')
      } else {
        setSttMode('unavailable')
        setSttReason('Speech-to-text is unavailable. Whisper is not reachable and browser STT is unsupported.')
      }
    }

    try {
      const controller = new AbortController()
      const timeoutId = window.setTimeout(() => controller.abort(), 7000)
      const response = await fetch('/api/ai/capabilities', {
        method: 'GET',
        headers: { Accept: 'application/json' },
        signal: controller.signal,
      })
      window.clearTimeout(timeoutId)

      if (!response.ok) {
        fallbackToBrowser()
        return
      }

      const data = await response.json() as AiCapabilitiesPayload
      const whisperAvailable = data.stt?.whisper?.available === true
      const whisperReason = (data.stt?.whisper?.reason ?? '').trim()

      if (whisperAvailable) {
        setSttMode('whisper')
        setSttReason('')
        return
      }

      if (browserSpeechSupported) {
        setSttMode('browser')
        setSttReason(whisperReason || 'Whisper unavailable. Using browser speech recognition fallback.')
        return
      }

      setSttMode('unavailable')
      setSttReason(whisperReason || 'Whisper unavailable and browser speech recognition is unsupported.')
    } catch {
      fallbackToBrowser()
    }
  }, [browserSpeechSupported])

  const askCoach = useCallback(async (promptOverride?: string, history?: ChatMessage[]) => {
    setIsLoading(true)

    const requestBody = JSON.stringify({
      ...payload,
      prompt: promptOverride ?? undefined,
      messages: (history ?? []).map((msg) => ({
        role: msg.role,
        content: msg.content,
      })),
    })

    for (let attempt = 0; attempt <= COACH_RETRIES; attempt += 1) {
      const controller = new AbortController()
      const timeoutId = window.setTimeout(() => controller.abort(), COACH_TIMEOUT_MS)

      try {
        const response = await fetch('/api/ai/nn-coach', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: requestBody,
          signal: controller.signal,
        })

        const body = await response.text()
        if (!response.ok) {
          if (attempt < COACH_RETRIES && response.status >= 500) {
            continue
          }

          const reason = parseApiError(body, response.status)
          setSource('error')
          return {
            answer: '',
            source: 'error',
            reason,
          } satisfies CoachReply
        }

        const data = body
          ? JSON.parse(body) as {
            answer?: string
            source?: AiSource
            reason?: string
          }
          : {}

        const nextSource = normalizeSource(data.source)
        setSource(nextSource)

        return {
          answer: compactText(data.answer ?? ''),
          source: nextSource,
          reason: compactText(data.reason ?? ''),
        } satisfies CoachReply
      } catch (error) {
        const transient = error instanceof DOMException && error.name === 'AbortError'
        if (attempt < COACH_RETRIES && transient) {
          continue
        }

        const reason = error instanceof Error ? error.message : 'Chat request failed.'
        setSource('error')
        return {
          answer: '',
          source: 'error',
          reason,
        } satisfies CoachReply
      } finally {
        window.clearTimeout(timeoutId)
      }
    }

    setSource('error')
    return {
      answer: '',
      source: 'error',
      reason: 'Chat request failed after retries.',
    } satisfies CoachReply
  }, [payload])

  useEffect(() => {
    setChatMessages([])
    setPrompt('')
    setSource('idle')
    setVoiceError('')
    setIsRecording(false)
    setIsBrowserListening(false)
    setIsTranscribing(false)
    stopRecordingSession()
    stopBrowserRecognition()
    stopPlayback()
  }, [tab, provider, model, stopPlayback, stopRecordingSession, stopBrowserRecognition])

  useEffect(() => {
    if (!isOpen) return
    void probeCapabilities()
  }, [isOpen, probeCapabilities])

  useEffect(() => {
    return () => {
      stopRecordingSession()
      stopBrowserRecognition()
      stopPlayback()
    }
  }, [stopPlayback, stopRecordingSession, stopBrowserRecognition])

  const appendAssistantStatus = useCallback((reply: CoachReply) => {
    setChatMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        content: buildAssistantStatusMessage(reply),
        source: reply.source,
        kind: 'status',
      },
    ])
  }, [])

  const handleAsk = async () => {
    const userText = compactText(prompt)
    if (!userText || isLoading) return

    setVoiceError('')

    const history: ChatMessage[] = [
      ...chatMessages,
      { role: 'user', content: userText, kind: 'normal' },
    ]

    setChatMessages(history)
    setPrompt('')

    const reply = await askCoach(userText, history)
    if (reply.answer) {
      setChatMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: compactText(reply.answer),
          source: reply.source,
          kind: 'normal',
        },
      ])
      return
    }

    appendAssistantStatus(reply)
  }

  const handleRefresh = async () => {
    if (isLoading) return

    setVoiceError('')
    setChatMessages([])
    setPrompt('')

    const reply = await askCoach()
    if (reply.answer) {
      setChatMessages([
        {
          role: 'assistant',
          content: compactText(reply.answer),
          source: reply.source,
          kind: 'normal',
        },
      ])
      return
    }

    appendAssistantStatus(reply)
  }

  const startBrowserStt = useCallback(() => {
    const SpeechCtor = getSpeechRecognitionCtor()
    if (!SpeechCtor) {
      setSttMode('unavailable')
      setVoiceError('Browser speech recognition is unsupported in this browser.')
      return
    }

    setVoiceError('')

    const recognition = new SpeechCtor()
    recognition.lang = 'en-US'
    recognition.continuous = true
    recognition.interimResults = true

    let transcript = ''

    recognition.onresult = (event: Event) => {
      const resultEvent = event as Event & {
        results?: ArrayLike<ArrayLike<{ transcript?: string; isFinal?: boolean }> & { isFinal?: boolean }>
      }
      const results = resultEvent.results
      if (!results) return

      for (let i = 0; i < results.length; i += 1) {
        const segment = results[i]
        if (!segment || !segment[0]) continue
        if ((segment as { isFinal?: boolean }).isFinal !== false) {
          transcript += ` ${segment[0].transcript ?? ''}`
        }
      }
    }

    recognition.onerror = () => {
      setVoiceError('Browser speech recognition failed. Try microphone permissions or retry.')
    }

    recognition.onend = () => {
      browserRecognitionRef.current = null
      setIsBrowserListening(false)

      const normalized = compactText(transcript)
      if (!normalized) return

      setPrompt((prev) => {
        const existing = compactText(prev)
        return existing.length > 0 ? `${existing} ${normalized}` : normalized
      })
    }

    browserRecognitionRef.current = recognition

    try {
      recognition.start()
      setIsBrowserListening(true)
    } catch {
      browserRecognitionRef.current = null
      setIsBrowserListening(false)
      setVoiceError('Unable to start browser speech recognition.')
    }
  }, [])

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

  const switchToBrowserFallback = useCallback((reason: string) => {
    if (!browserSpeechSupported) {
      setSttMode('unavailable')
      setSttReason(reason)
      setVoiceError(reason)
      return
    }

    setSttMode('browser')
    setSttReason(reason)
    setVoiceError(`${reason} Browser STT fallback enabled. Tap Mic again.`)
  }, [browserSpeechSupported])

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
        const reason = parseApiError(body, response.status)
        if (response.status === 404 || response.status === 503) {
          switchToBrowserFallback(`Whisper unavailable (${reason}).`)
          return
        }
        throw new Error(reason)
      }

      const parsed = body ? JSON.parse(body) as { text?: string } : {}
      const transcript = compactText(parsed.text ?? '')
      if (!transcript) {
        setVoiceError('Whisper did not detect speech from the recording.')
        return
      }

      setPrompt((prev) => {
        const existing = compactText(prev)
        return existing.length > 0 ? `${existing} ${transcript}` : transcript
      })
    } catch (error) {
      if (browserSpeechSupported) {
        switchToBrowserFallback('Whisper request failed.')
        return
      }
      setVoiceError(error instanceof Error ? error.message : 'Speech-to-text failed.')
    } finally {
      setIsTranscribing(false)
    }
  }

  const handleToggleRecording = async () => {
    if (sttMode === 'probing') {
      setVoiceError('Checking speech capabilities. Try again in a second.')
      return
    }

    if (sttMode === 'unavailable') {
      setVoiceError(sttReason || 'Speech-to-text is unavailable.')
      return
    }

    if (sttMode === 'browser') {
      if (isBrowserListening) {
        stopBrowserRecognition()
      } else {
        startBrowserStt()
      }
      return
    }

    if (isRecording) {
      await handleStopRecording()
      return
    }

    await handleStartRecording()
  }

  const handleSpeak = async (text: string) => {
    const normalized = compactText(text)
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

  const micBusy = isTranscribing || isLoading

  return (
    <div
      className={`nn-ai-coach-dock ${isOpen ? 'nn-ai-coach-dock-open' : 'nn-ai-coach-dock-closed'}`}
      aria-label="AI model coach"
    >
      <button
        type="button"
        className="sidebar-toggle-button nn-ai-coach-side-toggle"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
        aria-label={isOpen ? 'Collapse AI coach panel' : 'Expand AI coach panel'}
        title={isOpen ? 'Collapse AI coach' : 'Expand AI coach'}
      >
        <span className="nn-ai-coach-side-icon" aria-hidden="true">
          {isOpen ? (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="16px"
              viewBox="0 -960 960 960"
              width="16px"
              fill="currentColor"
            >
              <path d="m321-80-71-71 329-329-329-329 71-71 400 400L321-80Z" />
            </svg>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="16px"
              viewBox="0 -960 960 960"
              width="16px"
              fill="currentColor"
            >
              <path d="M400-80 0-480l400-400 71 71-329 329 329 329-71 71Z" />
            </svg>
          )}
        </span>
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
          <span>{formatSourceLabel(source)} | {model}</span>
          <span className={`nn-ai-stt-chip nn-ai-stt-${sttMode}`}>
            {formatSttModeLabel(sttMode, isBrowserListening)}
          </span>
        </div>

        {sttReason ? <div className="nn-ai-stt-note">{sttReason}</div> : null}

        {chatMessages.length > 0 ? (
          <div className="nn-ai-chat-list">
            {chatMessages.map((msg, index) => (
              <div
                key={`${msg.role}-${index}`}
                className={`nn-ai-chat-bubble nn-ai-chat-${msg.role} ${msg.kind === 'status' ? 'nn-ai-chat-status' : ''}`}
              >
                {msg.role === 'assistant' ? (
                  <div className="nn-ai-chat-meta">{formatSourceLabel(msg.source ?? source)}</div>
                ) : null}
                <div className="nn-ai-chat-content">{msg.content}</div>
                {msg.role === 'assistant' && msg.kind !== 'status' ? (
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
            Ask a question, or use the mic to transcribe with {sttMode === 'browser' ? 'browser STT' : 'Whisper'}.
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
              className={`nn-ai-coach-voice ${(isRecording || isBrowserListening) ? 'nn-ai-coach-voice-recording' : ''}`}
              onClick={() => void handleToggleRecording()}
              disabled={micBusy}
            >
              {isRecording || isBrowserListening ? 'Stop Mic' : isTranscribing ? 'Transcribing...' : 'Mic'}
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
              disabled={isLoading || compactText(prompt).length === 0}
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

function normalizeSource(source: string | undefined): AiSource {
  if (source === 'openai' || source === 'gemini' || source === 'anthropic' || source === 'nvidia') {
    return source
  }
  if (source === 'unavailable') return 'unavailable'
  if (source === 'error') return 'error'
  return 'idle'
}

function compactText(value: string): string {
  return value.replace(/\s+/g, ' ').trim()
}

function buildAssistantStatusMessage(reply: CoachReply): string {
  const detail = compactText(reply.reason ?? '')
  if (reply.source === 'unavailable') {
    return detail || 'AI provider unavailable. Configure at least one provider API key and retry.'
  }
  if (reply.source === 'error') {
    return detail || 'AI request failed. Retry in a few seconds.'
  }
  return detail || `No response returned from ${formatSourceLabel(reply.source)}. Retry to continue.`
}

function parseApiError(body: string, status: number): string {
  if (!body) return `HTTP ${status}`

  try {
    const parsed = JSON.parse(body) as {
      detail?: { message?: string } | string | Array<{ msg?: string; loc?: unknown[] }>
      message?: string
      reason?: string
    }

    if (typeof parsed.detail === 'string') return parsed.detail
    if (Array.isArray(parsed.detail) && parsed.detail.length > 0) {
      const first = parsed.detail[0] ?? {}
      const msg = typeof first.msg === 'string' ? first.msg : ''
      const location = Array.isArray(first.loc)
        ? first.loc
          .filter((entry): entry is string => typeof entry === 'string')
          .join('.')
        : ''
      if (location && msg) return `${location}: ${msg}`
      if (msg) return msg
    }
    if (
      parsed.detail &&
      typeof parsed.detail === 'object' &&
      !Array.isArray(parsed.detail) &&
      typeof parsed.detail.message === 'string'
    ) {
      return parsed.detail.message
    }
    if (parsed.reason) return parsed.reason
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

function formatSttModeLabel(mode: SttMode, isListening: boolean): string {
  if (mode === 'probing') return 'STT: Checking'
  if (mode === 'whisper') return 'STT: Whisper'
  if (mode === 'browser') return isListening ? 'STT: Browser (Listening)' : 'STT: Browser'
  return 'STT: Unavailable'
}

function toNonNegativeInt(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.floor(value))
}

function toOptionalInt(value: number | null): number | null {
  if (value === null) return null
  if (!Number.isFinite(value)) return null
  return Math.trunc(value)
}
