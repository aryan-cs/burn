from __future__ import annotations

import os
from typing import Any

import httpx


class ComputeNodeError(RuntimeError):
    pass


class ComputeNodeClient:
    @property
    def enabled(self) -> bool:
        return self.base_url != ""

    @property
    def base_url(self) -> str:
        return os.getenv("VLM_COMPUTE_NODE_URL", "").strip().rstrip("/")

    @property
    def timeout_seconds(self) -> float:
        timeout_raw = os.getenv("VLM_COMPUTE_NODE_TIMEOUT_SECONDS", "30").strip()
        try:
            return max(1.0, float(timeout_raw))
        except ValueError:
            return 30.0

    def websocket_url(self, path: str) -> str:
        if not self.enabled:
            raise ComputeNodeError("VLM_COMPUTE_NODE_URL is not configured")
        if path.startswith("/"):
            suffix = path
        else:
            suffix = f"/{path}"
        base_url = self.base_url
        if base_url.startswith("https://"):
            return f"wss://{base_url[len('https://') :]}{suffix}"
        if base_url.startswith("http://"):
            return f"ws://{base_url[len('http://') :]}{suffix}"
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
        url = f"{self.base_url}/api/v1/vlm/infer"
        timeout = httpx.Timeout(self.timeout_seconds)
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
