from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from modal_worker import infer_deployment_payload


app = FastAPI(title="Burn Modal Sandbox Inference Server")
DEPLOYMENT_ID = os.getenv("BURN_MODAL_DEPLOYMENT_ID", "").strip()


class InferRequest(BaseModel):
    inputs: Any
    return_probabilities: bool = True
    deployment_id: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    if not DEPLOYMENT_ID:
        raise HTTPException(status_code=500, detail="Missing BURN_MODAL_DEPLOYMENT_ID")
    return {"status": "ok", "deployment_id": DEPLOYMENT_ID}


@app.post("/infer")
def infer(payload: InferRequest) -> dict[str, Any]:
    if not DEPLOYMENT_ID:
        raise HTTPException(status_code=500, detail="Missing BURN_MODAL_DEPLOYMENT_ID")
    if payload.deployment_id and payload.deployment_id != DEPLOYMENT_ID:
        raise HTTPException(status_code=400, detail="deployment_id does not match sandbox deployment")
    try:
        return infer_deployment_payload(
            {
                "deployment_id": DEPLOYMENT_ID,
                "inputs": payload.inputs,
                "return_probabilities": payload.return_probabilities,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sandbox inference failed: {exc}") from exc
