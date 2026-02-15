from __future__ import annotations

from fastapi import APIRouter

from datasets.rf_registry import list_rf_datasets


router = APIRouter(prefix="/api/rf", tags=["rf-datasets"])


@router.get("/datasets")
async def rf_datasets_endpoint():
    return {"datasets": list_rf_datasets()}
