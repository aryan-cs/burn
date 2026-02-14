"""
Remote Training Bridge
======================
Offloads training to a Jetson-hosted worker and relays progress events to the
local job registry so existing websocket consumers continue to work unchanged.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from urllib.parse import quote

import httpx
import torch

from core.job_registry import job_registry
from models.graph_schema import GraphSpec


TOKEN_HEADER = "X-Jetson-Token"


def _resolve_jetson_base_url() -> str:
    host = os.getenv("JETSON_HOST", "").strip()
    if not host:
        raise RuntimeError("JETSON_HOST is required for remote training")

    if host.startswith(("http://", "https://")):
        base = host
    else:
        port = os.getenv("JETSON_PORT", "8001").strip() or "8001"
        base = f"http://{host}:{port}"

    return base.rstrip("/")


def _jetson_headers() -> dict[str, str]:
    token = os.getenv("JETSON_PASS", "").strip()
    if not token:
        return {}
    return {TOKEN_HEADER: token}


async def run_remote_training_job(
    job_id: str,
    graph: GraphSpec,
    artifacts_dir: Path,
) -> None:
    """Submit graph to Jetson, relay events, and download final artifact."""
    entry = job_registry.get(job_id)
    if entry is None:
        return

    base = _resolve_jetson_base_url()
    headers = _jetson_headers()

    try:
        job_registry.set_status(job_id, "running")

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), headers=headers) as client:
            resp = await client.post(
                f"{base}/train",
                json=graph.model_dump(mode="json"),
            )

        if resp.status_code != 200:
            error_detail = resp.text
            await job_registry.publish(
                job_id,
                {"type": "error", "message": f"Jetson rejected training request: {error_detail}"},
            )
            await job_registry.mark_terminal(job_id, "failed", error=error_detail)
            return

        remote_data = resp.json()
        remote_job_id = str(remote_data["job_id"])
        remote_job_id_encoded = quote(remote_job_id, safe="")

        cursor = 0
        done = False
        poll_interval = 0.5
        stop_posted = False

        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0), headers=headers) as client:
            while not done:
                if entry.stop_event.is_set() and not stop_posted:
                    try:
                        await client.post(f"{base}/jobs/{remote_job_id_encoded}/stop")
                    finally:
                        stop_posted = True
                try:
                    resp = await client.get(
                        f"{base}/jobs/{remote_job_id_encoded}/events",
                        params={"after": cursor},
                    )
                    if resp.status_code != 200:
                        await asyncio.sleep(poll_interval)
                        poll_interval = min(5.0, poll_interval * 1.5)
                        continue

                    poll_interval = 0.5
                    payload = resp.json()
                    events: list[dict] = payload.get("events", [])
                    done = payload.get("done", False)

                    for event in events:
                        await job_registry.publish(job_id, event)
                        cursor += 1

                    if not done:
                        await asyncio.sleep(0.5)
                except (httpx.RequestError, httpx.TimeoutException):
                    await asyncio.sleep(poll_interval)
                    poll_interval = min(5.0, poll_interval * 1.5)
                    continue

        try:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifacts_dir / f"{job_id}.pt"

            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), headers=headers) as client:
                resp = await client.get(f"{base}/jobs/{remote_job_id_encoded}/artifact")

            if resp.status_code == 200:
                artifact_path.write_bytes(resp.content)
                if entry.model is not None:
                    state = torch.load(artifact_path, map_location="cpu")
                    entry.model.load_state_dict(state)
            else:
                artifact_path = None
        except (httpx.RequestError, httpx.TimeoutException):
            artifact_path = None

        final_status = "completed"
        final_metrics: dict[str, float] | None = None
        terminal_error: str | None = None

        if entry.stop_event.is_set():
            final_status = "stopped"

        for msg in reversed(entry.messages):
            if msg.get("type") == "training_done":
                final_metrics = {
                    k: v
                    for k, v in msg.items()
                    if k.startswith("final_") and isinstance(v, (int, float))
                }
                break
            if msg.get("type") == "error":
                final_status = "failed"
                terminal_error = str(msg.get("message", "unknown remote error"))
                break

        await job_registry.mark_terminal(
            job_id,
            final_status,
            final_metrics=final_metrics,
            error=terminal_error,
            artifact_path=artifact_path if artifact_path else None,
        )

    except Exception as exc:
        message = f"Remote training failed: {exc}"
        await job_registry.publish(job_id, {"type": "error", "message": message})
        await job_registry.mark_terminal(job_id, "failed", error=message)
