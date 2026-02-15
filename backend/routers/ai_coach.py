from __future__ import annotations

import asyncio
import io
import logging
import os
import wave
from threading import Lock
from typing import Any, Literal

import httpx
import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/ai", tags=["ai-coach"])
logger = logging.getLogger(__name__)

Provider = Literal["openai", "gemini", "anthropic", "nvidia"]
Source = Literal["openai", "gemini", "anthropic", "nvidia", "unavailable", "error"]
PROVIDER_FALLBACK_ORDER: tuple[Provider, ...] = ("nvidia", "gemini", "anthropic", "openai")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class NnCoachRequest(BaseModel):
    tab: Literal["validate", "train", "infer"]
    layerCount: int = Field(ge=0)
    neuronCount: int = Field(ge=0)
    weightCount: int = Field(ge=0)
    activation: str = ""
    currentEpoch: int = Field(ge=0)
    totalEpochs: int = Field(ge=1)
    trainLoss: float | None = None
    testLoss: float | None = None
    trainAccuracy: float | None = None
    testAccuracy: float | None = None
    trainingStatus: str
    inferenceTopPrediction: int | None = None
    prompt: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    provider: Provider = "openai"
    model: str | None = None


class NnCoachResponse(BaseModel):
    tips: list[str]
    answer: str
    source: Source
    model: str | None = None
    reason: str | None = None


class RecommendRequest(BaseModel):
    description: str = Field(min_length=1, max_length=2000)
    provider: Provider = "nvidia"
    model: str | None = None


class RecommendResponse(BaseModel):
    recommendation: str
    source: Source
    model: str | None = None
    reason: str | None = None


class ChatCapabilities(BaseModel):
    availableProviders: list[Provider]
    defaultProvider: Provider


class WhisperCapabilities(BaseModel):
    available: bool
    model: str | None = None
    reason: str | None = None


class SttCapabilities(BaseModel):
    whisper: WhisperCapabilities


class ChatterboxCapabilities(BaseModel):
    available: bool
    reason: str | None = None


class TtsCapabilities(BaseModel):
    chatterbox: ChatterboxCapabilities


class AiCapabilitiesResponse(BaseModel):
    chat: ChatCapabilities
    stt: SttCapabilities
    tts: TtsCapabilities


class WhisperSttResponse(BaseModel):
    text: str
    source: Literal["whisper"]
    model: str


class ChatterboxTtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1500)
    exaggeration: float = Field(default=0.45, ge=0.0, le=2.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=2.5)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)


class SpeechModelUnavailable(Exception):
    pass


class WhisperTranscriber:
    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return os.getenv("WHISPER_MODEL", "openai/whisper-small")

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                from transformers import pipeline
            except Exception as exc:
                raise SpeechModelUnavailable(
                    "Whisper dependencies are unavailable. Install torch + transformers."
                ) from exc

            device = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model_name,
                device=device,
            )

        return self._pipeline

    def transcribe_wav(self, audio_bytes: bytes, language: str | None = None) -> str:
        audio_array, sample_rate = _decode_wav_audio(audio_bytes)
        transcriber = self._ensure_pipeline()

        kwargs: dict[str, Any] = {}
        normalized_language = (language or "").strip()
        if normalized_language:
            kwargs["generate_kwargs"] = {"language": normalized_language}

        result = transcriber(
            {"array": audio_array, "sampling_rate": sample_rate},
            **kwargs,
        )
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()


class ChatterboxSynthesizer:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._sample_rate = 24_000
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return "chatterbox-tts"

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from chatterbox.tts import ChatterboxTTS
            except Exception as exc:
                raise SpeechModelUnavailable(
                    "Chatterbox is unavailable. Install the open-source package: chatterbox-tts."
                ) from exc

            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                model = ChatterboxTTS.from_pretrained(device=device)
            except Exception as exc:
                raise SpeechModelUnavailable(
                    "Failed to load Chatterbox model weights."
                ) from exc

            sample_rate = getattr(model, "sr", None) or getattr(model, "sample_rate", None)
            if isinstance(sample_rate, int) and sample_rate > 0:
                self._sample_rate = sample_rate

            self._model = model

        return self._model

    def synthesize(self, text: str, exaggeration: float, cfg_weight: float, temperature: float) -> tuple[bytes, int]:
        model = self._ensure_model()
        generated = model.generate(
            text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        waveform = _to_mono_waveform(generated)
        return _encode_wav_audio(waveform, self._sample_rate), self._sample_rate


whisper_transcriber = WhisperTranscriber()
chatterbox_synthesizer = ChatterboxSynthesizer()


def _provider_has_key(provider: Provider) -> bool:
    if provider == "openai":
        return len(_collect_openai_keys()) > 0
    if provider == "nvidia":
        return len(os.getenv("NVIDIA_API_KEY", "").strip()) > 0
    if provider == "gemini":
        return len(os.getenv("GEMINI_API_KEY", "").strip()) > 0
    return len(os.getenv("ANTHROPIC_API_KEY", "").strip() or os.getenv("ANTROPHIC_API_KEY", "").strip()) > 0


def get_available_chat_providers() -> list[Provider]:
    return [provider for provider in PROVIDER_FALLBACK_ORDER if _provider_has_key(provider)]


def warn_if_ai_provider_keys_missing() -> None:
    if get_available_chat_providers():
        return
    logger.warning(
        "No AI provider keys configured. Set one of NVIDIA_API_KEY, GEMINI_API_KEY, "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY (or PENAI_API_KEY alias)."
    )


def _build_provider_attempt_order(selected_provider: str) -> list[Provider]:
    normalized = selected_provider.strip().lower()
    attempts: list[Provider] = []
    if normalized in {"openai", "gemini", "anthropic", "nvidia"}:
        attempts.append(normalized)  # type: ignore[arg-type]
    for provider in PROVIDER_FALLBACK_ORDER:
        if provider not in attempts:
            attempts.append(provider)
    return attempts


async def _call_provider(
    client: httpx.AsyncClient,
    provider: Provider,
    history: list[dict[str, str]],
    model: str,
    system_prompt: str,
) -> str:
    if provider == "openai":
        return await _call_openai(client, history, model)
    if provider == "nvidia":
        return await _call_nvidia(client, history, model)
    if provider == "gemini":
        return await _call_gemini(client, history, model, system_prompt)
    return await _call_anthropic(client, history, model, system_prompt)


async def _run_provider_failover(
    *,
    selected_provider: str,
    selected_model: str | None,
    history: list[dict[str, str]],
    system_prompt: str,
) -> tuple[str, Source, str | None, str | None]:
    attempts = _build_provider_attempt_order(selected_provider)
    normalized_selected = selected_provider.strip().lower()
    selected_model_clean = (selected_model or "").strip() or None

    unavailable: list[str] = []
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=20.0) as client:
        for provider in attempts:
            model = selected_model_clean if provider == normalized_selected and selected_model_clean else _default_model(provider)
            try:
                answer = (await _call_provider(client, provider, history, model, system_prompt)).strip()
            except MissingProviderKey:
                unavailable.append(provider)
                continue
            except Exception as exc:
                errors.append(f"{provider}: {type(exc).__name__}")
                continue

            if answer:
                reason = None
                if provider != normalized_selected and normalized_selected in {"openai", "gemini", "anthropic", "nvidia"}:
                    reason = f"Primary provider '{normalized_selected}' was unavailable; used '{provider}' fallback."
                return answer, provider, reason, model  # type: ignore[return-value]

            errors.append(f"{provider}: empty response")

    if unavailable and not errors:
        return (
            "",
            "unavailable",
            f"No usable provider keys configured for attempted providers: {', '.join(unavailable)}.",
            None,
        )
    if unavailable and errors:
        return (
            "",
            "error",
            f"Unavailable providers: {', '.join(unavailable)}. Failed providers: {'; '.join(errors)}.",
            None,
        )
    if errors:
        return "", "error", f"All provider attempts failed: {'; '.join(errors)}.", None

    return "", "error", "No provider attempts were executed.", None


def _whisper_capability() -> tuple[bool, str | None]:
    try:
        import transformers  # noqa: F401
    except Exception:
        return False, "Whisper requires transformers in backend environment."
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "Whisper requires torch in backend environment."
    return True, None


def _chatterbox_capability() -> tuple[bool, str | None]:
    try:
        from chatterbox import tts as _tts  # noqa: F401
    except Exception:
        return False, "Chatterbox package is not installed in backend environment."
    return True, None


@router.get("/capabilities", response_model=AiCapabilitiesResponse)
async def ai_capabilities() -> AiCapabilitiesResponse:
    available_providers = get_available_chat_providers()
    configured_default = os.getenv("AI_PROVIDER", "nvidia").strip().lower()
    if configured_default in {"openai", "gemini", "anthropic", "nvidia"}:
        default_provider: Provider = configured_default  # type: ignore[assignment]
    else:
        default_provider = "nvidia"

    if default_provider not in available_providers and available_providers:
        default_provider = available_providers[0]

    whisper_available, whisper_reason = _whisper_capability()
    chatterbox_available, chatterbox_reason = _chatterbox_capability()

    return AiCapabilitiesResponse(
        chat=ChatCapabilities(
            availableProviders=available_providers,
            defaultProvider=default_provider,
        ),
        stt=SttCapabilities(
            whisper=WhisperCapabilities(
                available=whisper_available,
                model=whisper_transcriber.model_name if whisper_available else None,
                reason=whisper_reason,
            )
        ),
        tts=TtsCapabilities(
            chatterbox=ChatterboxCapabilities(
                available=chatterbox_available,
                reason=chatterbox_reason,
            )
        ),
    )


@router.post("/recommend", response_model=RecommendResponse)
async def recommend_algorithm(payload: RecommendRequest) -> RecommendResponse:
    provider = (payload.provider or "nvidia").strip().lower()
    model = (payload.model or _default_model(provider)).strip()

    if provider not in {"openai", "gemini", "anthropic", "nvidia"}:
        provider = "nvidia"

    system_prompt = (
        "You are Burn, an expert machine learning advisor. "
        "The user will describe their problem, data, or goal. "
        "Your job is to recommend the BEST machine learning algorithm for their use case from this list:\n"
        "1. Neural Network (deep learning, image classification, complex patterns)\n"
        "2. Random Forest (tabular data, classification, feature importance)\n"
        "3. Support Vector Machine / SVM (small-to-medium datasets, binary classification, high-dimensional)\n"
        "4. Linear Regression (continuous output, linear relationships)\n"
        "5. Logistic Regression (binary/multi-class classification, interpretability)\n"
        "6. PCA / Principal Component Analysis (dimensionality reduction, feature extraction)\n"
        "7. Vision-Language Model / VLM (image+text understanding)\n\n"
        "Structure your answer exactly like this:\n"
        "RECOMMENDED: <algorithm name>\n\n"
        "Then give a clear, concise explanation (4-8 lines) of WHY this algorithm fits their use case, "
        "what trade-offs to consider, and a brief mention of alternatives. "
        "Be practical and direct."
    )

    history: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload.description},
    ]

    answer, source, reason, used_model = await _run_provider_failover(
        selected_provider=provider,
        selected_model=model,
        history=history,
        system_prompt=system_prompt,
    )
    return RecommendResponse(recommendation=answer, source=source, model=used_model, reason=reason)


@router.post("/stt/whisper", response_model=WhisperSttResponse)
async def stt_whisper(
    audio: UploadFile = File(...),
    language: str | None = None,
) -> WhisperSttResponse:
    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Audio payload is empty.")

    try:
        transcript = await asyncio.to_thread(whisper_transcriber.transcribe_wav, payload, language)
    except SpeechModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Whisper transcription failed.") from exc

    return WhisperSttResponse(
        text=transcript,
        source="whisper",
        model=whisper_transcriber.model_name,
    )


@router.post("/tts/chatterbox")
async def tts_chatterbox(payload: ChatterboxTtsRequest) -> Response:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        audio_bytes, sample_rate = await asyncio.to_thread(
            chatterbox_synthesizer.synthesize,
            text,
            payload.exaggeration,
            payload.cfg_weight,
            payload.temperature,
        )
    except SpeechModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Chatterbox speech synthesis failed.") from exc

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "X-TTS-Source": "chatterbox",
            "X-TTS-Model": chatterbox_synthesizer.model_name,
            "X-Sample-Rate": str(sample_rate),
        },
    )


@router.post("/nn-coach", response_model=NnCoachResponse)
async def nn_coach(payload: NnCoachRequest) -> NnCoachResponse:
    provider = (payload.provider or os.getenv("AI_PROVIDER", "openai")).strip().lower()
    model = (payload.model or _default_model(provider)).strip()

    if provider not in {"openai", "gemini", "anthropic", "nvidia"}:
        provider = os.getenv("AI_PROVIDER", "nvidia").strip().lower()
        if provider not in {"openai", "gemini", "anthropic", "nvidia"}:
            provider = "nvidia"
        model = _default_model(provider)

    system_prompt = (
        "You are a practical neural-network training coach. "
        "Give concise, actionable guidance to improve architecture, training, and generalization. "
        "When the user asks a question, answer it directly first, then add short next steps."
    )

    context_block = (
        "Model context:\n"
        f"tab={payload.tab}\n"
        f"layerCount={payload.layerCount}, neuronCount={payload.neuronCount}, weightCount={payload.weightCount}\n"
        f"activation={payload.activation}\n"
        f"currentEpoch={payload.currentEpoch}, totalEpochs={payload.totalEpochs}, trainingStatus={payload.trainingStatus}\n"
        f"trainLoss={payload.trainLoss}, testLoss={payload.testLoss}\n"
        f"trainAccuracy={payload.trainAccuracy}, testAccuracy={payload.testAccuracy}\n"
        f"inferenceTopPrediction={payload.inferenceTopPrediction}"
    )

    user_prompt_text = (payload.prompt or "").strip()
    final_user_prompt = user_prompt_text or (
        "Give the first coaching response for this model state. "
        "Provide concise actionable guidance (4-8 lines)."
    )

    # Unified conversation history (system + context + chat)
    history: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Use this model context for every answer:\n{context_block}"},
    ]

    for msg in payload.messages[-12:]:
        content = msg.content.strip()
        if not content:
            continue
        history.append({"role": msg.role, "content": content})

    history.append({"role": "user", "content": final_user_prompt})

    answer, source, reason, used_model = await _run_provider_failover(
        selected_provider=provider,
        selected_model=model,
        history=history,
        system_prompt=system_prompt,
    )
    return NnCoachResponse(
        tips=_extract_tips(answer) if answer else [],
        answer=answer,
        source=source,
        model=used_model,
        reason=reason,
    )


def _decode_wav_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    if not audio_bytes:
        raise ValueError("Audio payload is empty.")

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)
    except wave.Error as exc:
        raise ValueError("Expected uncompressed WAV audio.") from exc

    if channels <= 0 or sample_rate <= 0:
        raise ValueError("Invalid WAV metadata.")
    if frame_count <= 0 or not raw:
        raise ValueError("WAV audio has no frames.")

    if sample_width == 1:
        waveform = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        waveform = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Unsupported WAV sample width. Use 8/16/32-bit PCM.")

    if waveform.size == 0:
        raise ValueError("No decoded audio samples found.")

    if channels > 1:
        usable_size = (waveform.size // channels) * channels
        waveform = waveform[:usable_size].reshape(-1, channels).mean(axis=1)

    return waveform.astype(np.float32, copy=False), sample_rate


def _to_mono_waveform(audio: Any) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        waveform = audio.detach().cpu().float().numpy()
    else:
        waveform = np.asarray(audio, dtype=np.float32)

    if waveform.ndim == 0:
        raise ValueError("Synthesized audio format is invalid.")

    if waveform.ndim == 1:
        mono = waveform
    elif waveform.ndim == 2:
        if waveform.shape[0] <= 4 and waveform.shape[1] > waveform.shape[0]:
            mono = waveform.mean(axis=0)
        elif waveform.shape[1] <= 4:
            mono = waveform.mean(axis=1)
        else:
            mono = waveform.reshape(-1)
    else:
        mono = waveform.reshape(-1)

    mono = np.nan_to_num(mono.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(mono, -1.0, 1.0)


def _encode_wav_audio(waveform: np.ndarray, sample_rate: int) -> bytes:
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    if waveform.size == 0:
        raise ValueError("No audio samples to encode.")

    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


class MissingProviderKey(Exception):
    pass


def _default_model(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    if provider == "nvidia":
        return os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
    return "gpt-4o-mini"


def _extract_tips(answer: str) -> list[str]:
    return [line.strip("- •\t ") for line in answer.splitlines() if line.strip()][:5]


def _collect_openai_keys() -> list[str]:
    keys: list[str] = []

    single_key = os.getenv("OPENAI_API_KEY", "").strip()
    if single_key:
        keys.append(single_key)

    legacy_key = os.getenv("PENAI_API_KEY", "").strip()
    if legacy_key:
        keys.append(legacy_key)

    multi_raw = os.getenv("OPENAI_API_KEYS", "")
    if multi_raw:
        normalized = multi_raw.replace("\n", ",").replace(" ", ",")
        keys.extend(part.strip() for part in normalized.split(",") if part.strip())

    for env_name, env_value in os.environ.items():
        if env_name.startswith("OPENAI_API_KEY_"):
            value = env_value.strip()
            if value:
                keys.append(value)

    seen: set[str] = set()
    deduped: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


async def _call_openai(client: httpx.AsyncClient, history: list[dict[str, str]], model: str) -> str:
    api_keys = _collect_openai_keys()
    if not api_keys:
        raise MissingProviderKey()

    for api_key in api_keys:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.3,
                "messages": history,
            },
        )

        if response.status_code >= 400:
            continue

        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if content:
            return content

    return ""


async def _call_nvidia(client: httpx.AsyncClient, history: list[dict[str, str]], model: str) -> str:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise MissingProviderKey()

    response = await client.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.3,
            "messages": history,
        },
    )

    if response.status_code >= 400:
        return ""

    data = response.json()
    return (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )


async def _call_gemini(
    client: httpx.AsyncClient,
    history: list[dict[str, str]],
    model: str,
    system_prompt: str,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise MissingProviderKey()

    contents = []
    for item in history:
        role = "model" if item["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": item["content"]}]})

    response = await client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.3},
        },
    )

    if response.status_code >= 400:
        return ""

    data = response.json()
    return (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )


async def _call_anthropic(
    client: httpx.AsyncClient,
    history: list[dict[str, str]],
    model: str,
    system_prompt: str,
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip() or os.getenv("ANTROPHIC_API_KEY", "").strip()
    if not api_key:
        raise MissingProviderKey()

    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item["role"] in {"user", "assistant"}
    ]

    response = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 700,
            "temperature": 0.3,
            "system": system_prompt,
            "messages": messages,
        },
    )

    if response.status_code >= 400:
        return ""

    data = response.json()
    content_items = data.get("content", [])
    text_parts = [item.get("text", "") for item in content_items if item.get("type") == "text"]
    return "\n".join(part for part in text_parts if part).strip()
