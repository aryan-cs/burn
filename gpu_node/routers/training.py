from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse

from gpu_node.bootstrap import ensure_backend_path
from gpu_node.core.job_registry import remote_job_registry
from gpu_node.core.training_engine import run_training_job

ensure_backend_path()

from core.graph_compiler import GraphCompileError, compile_graph
from models.graph_schema import GraphSpec
from models.training_config import normalize_training_config


router = APIRouter(tags=["gpu-node-training"])
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
TOKEN_HEADER = "X-Jetson-Token"


def _auth_required(
    x_jetson_token: Annotated[str | None, Header(alias=TOKEN_HEADER)] = None,
) -> None:
    expected = os.getenv("JETSON_PASS", "").strip()
    if not expected:
        return
    if x_jetson_token != expected:
        raise HTTPException(status_code=401, detail={"message": "Unauthorized"})


@router.post("/train", dependencies=[Depends(_auth_required)])
async def start_training(graph: GraphSpec) -> dict[str, str]:
    training = normalize_training_config(graph.training)
    try:
        compiled = compile_graph(graph, training)
    except GraphCompileError as exc:
        raise HTTPException(status_code=400, detail={"errors": exc.errors}) from exc

    entry = remote_job_registry.create_job()
    task = asyncio.create_task(run_training_job(entry.job_id, compiled, training, ARTIFACTS_DIR))
    remote_job_registry.set_task(entry.job_id, task)
    return {"job_id": entry.job_id}


@router.get("/jobs/{job_id}/events", dependencies=[Depends(_auth_required)])
async def stream_events(job_id: str, after: Annotated[int, Query(ge=0)] = 0) -> dict:
    entry = remote_job_registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {job_id}"})
    events, done = await remote_job_registry.wait_for_update(job_id, after, timeout=10.0)
    return {"events": events, "done": done}


@router.post("/jobs/{job_id}/stop", dependencies=[Depends(_auth_required)])
async def stop_job(job_id: str) -> dict[str, str]:
    entry = remote_job_registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {job_id}"})
    remote_job_registry.request_stop(job_id)
    return {"job_id": job_id, "status": entry.status}


@router.get("/jobs/{job_id}/artifact", dependencies=[Depends(_auth_required)])
async def get_artifact(job_id: str):
    entry = remote_job_registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {job_id}"})
    artifact_path = entry.artifact_path
    if artifact_path is None or not artifact_path.exists():
        raise HTTPException(status_code=404, detail={"message": "No artifact available yet"})
    return FileResponse(path=artifact_path, filename=f"{job_id}.pt", media_type="application/octet-stream")
