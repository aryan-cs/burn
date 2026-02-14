from __future__ import annotations

from fastapi import APIRouter

from datasets.registry import list_datasets


router = APIRouter(prefix="/api", tags=["datasets"])


@router.get("/datasets")
async def datasets_endpoint():
    return {"datasets": list_datasets()}
