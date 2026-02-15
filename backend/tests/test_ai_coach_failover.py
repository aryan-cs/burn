from __future__ import annotations

from routers.ai_coach import _collect_openai_keys


def _nn_payload(provider: str, model: str) -> dict:
    return {
        "tab": "validate",
        "layerCount": 1,
        "neuronCount": 16,
        "weightCount": 32,
        "activation": "relu",
        "currentEpoch": 0,
        "totalEpochs": 1,
        "trainLoss": None,
        "testLoss": None,
        "trainAccuracy": None,
        "testAccuracy": None,
        "trainingStatus": "idle",
        "inferenceTopPrediction": None,
        "prompt": "hello",
        "messages": [],
        "provider": provider,
        "model": model,
    }


def test_nn_coach_failover_uses_secondary_provider_when_primary_unavailable(monkeypatch, client) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.delenv("PENAI_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")

    async def fake_nvidia(*_args, **_kwargs) -> str:
        return "fallback from nvidia"

    monkeypatch.setattr("routers.ai_coach._call_nvidia", fake_nvidia)

    response = client.post(
        "/api/ai/nn-coach",
        json=_nn_payload("openai", "gpt-4o-mini"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "nvidia"
    assert payload["answer"] == "fallback from nvidia"
    assert "fallback" in (payload.get("reason") or "")


def test_recommend_failover_uses_secondary_provider(monkeypatch, client) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.delenv("PENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")

    async def fake_nvidia(*_args, **_kwargs) -> str:
        return "RECOMMENDED: Random Forest"

    monkeypatch.setattr("routers.ai_coach._call_nvidia", fake_nvidia)

    response = client.post(
        "/api/ai/recommend",
        json={
            "description": "I have tabular data and want robust baseline performance.",
            "provider": "anthropic",
            "model": "claude-3-5-haiku-latest",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "nvidia"
    assert "Random Forest" in payload["recommendation"]


def test_openai_key_alias_penai_api_key_is_accepted(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.setenv("PENAI_API_KEY", "penai-alias-key")

    keys = _collect_openai_keys()
    assert "penai-alias-key" in keys


def test_capabilities_reports_chat_and_stt_status(monkeypatch, client) -> None:
    monkeypatch.setenv("AI_PROVIDER", "openai")

    monkeypatch.setattr("routers.ai_coach.get_available_chat_providers", lambda: ["nvidia", "gemini"])
    monkeypatch.setattr("routers.ai_coach._whisper_capability", lambda: (False, "whisper deps missing"))
    monkeypatch.setattr("routers.ai_coach._chatterbox_capability", lambda: (True, None))

    response = client.get("/api/ai/capabilities")
    assert response.status_code == 200

    payload = response.json()
    assert payload["chat"]["availableProviders"] == ["nvidia", "gemini"]
    assert payload["chat"]["defaultProvider"] == "nvidia"
    assert payload["stt"]["whisper"]["available"] is False
    assert "missing" in (payload["stt"]["whisper"].get("reason") or "")
    assert payload["tts"]["chatterbox"]["available"] is True
