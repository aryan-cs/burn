from __future__ import annotations

import os
from typing import Any

import httpx


class ComputeNodeError(RuntimeError):
    pass


class ComputeNodeClient:
    def __init__(self) -> None:
        self._base_url = os.getenv("VLM_COMPUTE_NODE_URL", "").strip().rstrip("/")
        timeout_raw = os.getenv("VLM_COMPUTE_NODE_TIMEOUT_SECONDS", "30").strip()
        try:
            self._timeout_seconds = max(1.0, float(timeout_raw))
        except ValueError:
            self._timeout_seconds = 30.0

    @property
    def enabled(self) -> bool:
        return self._base_url != ""

    @property
    def base_url(self) -> str:
        return self._base_url

    def websocket_url(self, path: str) -> str:
        if not self.enabled:
            raise ComputeNodeError("VLM_COMPUTE_NODE_URL is not configured")
        if path.startswith("/"):
            suffix = path
        else:
            suffix = f"/{path}"
        if self._base_url.startswith("https://"):
            return f"wss://{self._base_url[len('https://') :]}{suffix}"
        if self._base_url.startswith("http://"):
            return f"ws://{self._base_url[len('http://') :]}{suffix}"
        raise ComputeNodeError("VLM_COMPUTE_NODE_URL must start with http:// or https://")

    async def infer(
        self,
        *,
        image_base64: str,
        model_id: str,
        score_threshold: float,
        max_detections: int,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise ComputeNodeError("VLM_COMPUTE_NODE_URL is not configured")
        payload = {
            "image_base64": image_base64,
            "model_id": model_id,
            "score_threshold": score_threshold,
            "max_detections": max_detections,
        }
        url = f"{self._base_url}/api/v1/vlm/infer"
        timeout = httpx.Timeout(self._timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
            response.raise_for_status()
        except Exception as exc:
            raise ComputeNodeError(f"remote inference request failed: {exc}") from exc
        try:
            data = response.json()
        except Exception as exc:
            raise ComputeNodeError(f"remote inference response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ComputeNodeError("remote inference response must be a JSON object")
        return data


compute_node_client = ComputeNodeClient()
