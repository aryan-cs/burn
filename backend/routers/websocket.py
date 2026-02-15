from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.job_registry import job_registry


router = APIRouter(tags=["websocket"])


@router.websocket("/ws/training/{job_id}")
async def training_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()

    entry = job_registry.get(job_id)
    if entry is None:
        await websocket.send_json({"type": "error", "message": f"Unknown job_id: {job_id}"})
        await websocket.close(code=1008)
        return

    cursor = 0

    try:
        await websocket.send_json(
            {
                "type": "ws_connected",
                "job_id": job_id,
                "status": entry.status,
            }
        )
        while True:
            messages, terminal = await job_registry.wait_for_update(job_id, cursor, timeout=0.25)
            for message in messages:
                await websocket.send_json(message)
                cursor += 1

            if terminal and cursor >= len(entry.messages):
                break

    except WebSocketDisconnect:
        return
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            # Already closed by client.
            pass
