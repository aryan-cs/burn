from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.vlm_registry import vlm_job_registry


router = APIRouter(tags=["vlm-websocket"])


@router.websocket("/ws/vlm/training/{job_id}")
async def vlm_training_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()

    entry = vlm_job_registry.get(job_id)
    if entry is None:
        await websocket.send_json({"type": "vlm_error", "message": f"Unknown VLM job_id: {job_id}"})
        await websocket.close(code=1008)
        return

    cursor = 0
    try:
        while True:
            messages, terminal = await vlm_job_registry.wait_for_update(job_id, cursor, timeout=0.25)
            for message in messages:
                await websocket.send_json(message)
                cursor += 1

            if terminal and cursor >= len(entry.messages):
                break

            try:
                incoming = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                payload = json.loads(incoming)
                if payload.get("command") == "stop":
                    vlm_job_registry.request_stop(job_id)
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                await websocket.send_json({"type": "vlm_error", "message": "Invalid websocket command payload"})
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        return
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
