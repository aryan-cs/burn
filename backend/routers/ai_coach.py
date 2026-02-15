from __future__ import annotations

import os
from typing import Literal

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/ai", tags=["ai-coach"])

Provider = Literal["openai", "gemini", "anthropic", "nvidia"]


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
    source: Literal["openai", "gemini", "anthropic", "nvidia", "unavailable", "error"]


@router.post("/nn-coach", response_model=NnCoachResponse)
async def nn_coach(payload: NnCoachRequest) -> NnCoachResponse:
    provider = (payload.provider or os.getenv("AI_PROVIDER", "openai")).strip().lower()
    model = (payload.model or _default_model(provider)).strip()

    if provider not in {"openai", "gemini", "anthropic", "nvidia"}:
        return NnCoachResponse(tips=[], answer="", source="error")

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

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            if provider == "openai":
                answer = await _call_openai(client, history, model)
            elif provider == "nvidia":
                answer = await _call_nvidia(client, history, model)
            elif provider == "gemini":
                answer = await _call_gemini(client, history, model, system_prompt)
            else:
                answer = await _call_anthropic(client, history, model, system_prompt)

        if not answer:
            return NnCoachResponse(tips=[], answer="", source="error")

        tips = _extract_tips(answer)
        return NnCoachResponse(tips=tips, answer=answer, source=provider)  # type: ignore[arg-type]
    except MissingProviderKey:
        return NnCoachResponse(tips=[], answer="", source="unavailable")
    except Exception:
        return NnCoachResponse(tips=[], answer="", source="error")


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
