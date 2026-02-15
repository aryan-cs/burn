from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import websockets

from core.compute_node_client import ComputeNodeError, compute_node_client
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


@router.websocket("/ws/vlm/infer")
async def vlm_infer_proxy(websocket: WebSocket):
    await websocket.accept()
    if not compute_node_client.enabled:
        await websocket.send_json(
            {"type": "vlm_error", "message": "Remote compute node is not configured on backend"}
        )
        await websocket.close(code=1008)
        return

    try:
        remote_url = compute_node_client.websocket_url("/ws/v1/vlm/infer")
    except ComputeNodeError as exc:
        await websocket.send_json({"type": "vlm_error", "message": str(exc)})
        await websocket.close(code=1008)
        return

    try:
        async with websockets.connect(remote_url, ping_interval=20, ping_timeout=20) as remote_ws:
            client_to_remote = asyncio.create_task(_forward_client_to_remote(websocket, remote_ws))
            remote_to_client = asyncio.create_task(_forward_remote_to_client(remote_ws, websocket))
            done, pending = await asyncio.wait(
                {client_to_remote, remote_to_client},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                exception = task.exception()
                if exception is not None:
                    raise exception
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await websocket.send_json({"type": "vlm_error", "message": f"Infer proxy failed: {exc}"})
        except RuntimeError:
            pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


async def _forward_client_to_remote(
    websocket: WebSocket,
    remote_ws: websockets.ClientConnection,
) -> None:
    while True:
        payload = await websocket.receive_text()
        await remote_ws.send(payload)


async def _forward_remote_to_client(
    remote_ws: websockets.ClientConnection,
    websocket: WebSocket,
) -> None:
    async for message in remote_ws:
        if isinstance(message, str):
            await websocket.send_text(message)
        else:
            await websocket.send_bytes(message)
