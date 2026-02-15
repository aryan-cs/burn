from __future__ import annotations

import io
import wave

from routers.ai_coach import SpeechModelUnavailable


def _build_test_wav(sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((b"\x00\x00") * sample_rate)
    return buffer.getvalue()


def test_whisper_stt_endpoint_returns_transcript(monkeypatch, client) -> None:
    def fake_transcribe(audio_bytes: bytes, language: str | None = None) -> str:
        assert audio_bytes.startswith(b"RIFF")
        assert language is None
        return "test transcript"

    monkeypatch.setattr("routers.ai_coach.whisper_transcriber.transcribe_wav", fake_transcribe)

    response = client.post(
        "/api/ai/stt/whisper",
        files={"audio": ("speech.wav", _build_test_wav(), "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "test transcript"
    assert payload["source"] == "whisper"


def test_whisper_stt_endpoint_unavailable(monkeypatch, client) -> None:
    def fake_transcribe(*_args, **_kwargs) -> str:
        raise SpeechModelUnavailable("missing whisper")

    monkeypatch.setattr("routers.ai_coach.whisper_transcriber.transcribe_wav", fake_transcribe)

    response = client.post(
        "/api/ai/stt/whisper",
        files={"audio": ("speech.wav", _build_test_wav(), "audio/wav")},
    )

    assert response.status_code == 503
    assert "missing whisper" in response.text


def test_chatterbox_tts_endpoint_returns_audio(monkeypatch, client) -> None:
    def fake_synthesize(text: str, exaggeration: float, cfg_weight: float, temperature: float) -> tuple[bytes, int]:
        assert text == "hello world"
        assert exaggeration > 0
        assert cfg_weight > 0
        assert temperature > 0
        return _build_test_wav(sample_rate=24000), 24000

    monkeypatch.setattr("routers.ai_coach.chatterbox_synthesizer.synthesize", fake_synthesize)

    response = client.post(
        "/api/ai/tts/chatterbox",
        json={"text": "hello world"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["x-tts-source"] == "chatterbox"
    assert response.content[:4] == b"RIFF"


def test_chatterbox_tts_endpoint_rejects_blank_text(client) -> None:
    response = client.post(
        "/api/ai/tts/chatterbox",
        json={"text": "   "},
    )

    assert response.status_code == 400
    assert "must not be empty" in response.text
