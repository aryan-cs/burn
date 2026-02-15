"""WebSocket endpoint for streaming classical ML training progress."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.ml_job_registry import ml_job_registry

router = APIRouter(tags=["ml-ws"])


@router.websocket("/ws/ml/training/{job_id}")
async def ml_training_ws(ws: WebSocket, job_id: str) -> None:
    entry = ml_job_registry.get(job_id)
    if entry is None:
        await ws.close(code=4004, reason="Job not found")
        return

    await ws.accept()
    cursor = 0

    # Background reader for client commands (e.g. "stop")
    async def _read_client() -> None:
        try:
            while True:
                data = await ws.receive_json()
                cmd = data.get("command")
                if cmd == "stop":
                    ml_job_registry.request_stop(job_id)
        except (WebSocketDisconnect, RuntimeError):
            pass

    reader_task = asyncio.create_task(_read_client())

    try:
        while True:
            messages, terminal = await ml_job_registry.wait_for_update(
                job_id, cursor, timeout=0.5,
            )
            for msg in messages:
                await ws.send_json(msg)
            cursor += len(messages)

            if terminal:
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        reader_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass
