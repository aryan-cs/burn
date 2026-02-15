"""REST API router for classical ML models (Linear Reg, Logistic Reg, Random Forest)."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from core.ml_job_registry import ml_job_registry
from core.ml_training_engine import run_ml_training
from datasets.ml_loader import load_ml_dataset
from datasets.ml_registry import get_ml_dataset_info, list_ml_datasets
from models.ml_schema import (
    MLModelType,
    MLPredictRequest,
    MLPredictResponse,
    MLStatusResponse,
    MLTrainRequest,
    MLTrainResponse,
)

router = APIRouter(prefix="/api/ml", tags=["ml"])


# ── Datasets ──────────────────────────────────────────


@router.get("/datasets")
async def get_datasets(task: str | None = None) -> list[dict[str, Any]]:
    """List available datasets for classical ML, optionally filtered by task."""
    return list_ml_datasets(task)


@router.get("/datasets/{dataset_id}")
async def get_dataset_detail(dataset_id: str) -> dict[str, Any]:
    info = get_ml_dataset_info(dataset_id)
    if info is None:
        raise HTTPException(404, f"Dataset {dataset_id!r} not found")
    return info


@router.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, rows: int = 5) -> dict[str, Any]:
    """Return a small preview of the dataset (first N rows)."""
    try:
        data = load_ml_dataset(dataset_id, test_size=0.2)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    n = min(rows, len(data.X_train))
    return {
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "task": data.task,
        "samples": [
            {
                "features": data.X_train[i].tolist(),
                "target": data.y_train[i].item() if hasattr(data.y_train[i], "item") else float(data.y_train[i]),
            }
            for i in range(n)
        ],
    }


# ── Training ──────────────────────────────────────────


@router.post("/train", response_model=MLTrainResponse)
async def train_model(req: MLTrainRequest) -> MLTrainResponse:
    """Start training a classical ML model asynchronously."""
    # Validate dataset exists
    info = get_ml_dataset_info(req.dataset)
    if info is None:
        raise HTTPException(400, f"Unknown dataset: {req.dataset!r}")

    # Validate model/task compatibility
    task = info["task"]
    if req.model_type == MLModelType.linear_regression and task != "regression":
        raise HTTPException(
            400,
            f"Linear regression requires a regression dataset, but {req.dataset!r} is {task}",
        )
    if req.model_type == MLModelType.logistic_regression and task != "classification":
        raise HTTPException(
            400,
            f"Logistic regression requires a classification dataset, but {req.dataset!r} is {task}",
        )

    entry = ml_job_registry.create_job(
        model_type=req.model_type.value,
        dataset=req.dataset,
        hyperparameters=req.hyperparameters,
    )

    task_obj = asyncio.create_task(
        run_ml_training(
            registry=ml_job_registry,
            job_id=entry.job_id,
            model_type=req.model_type.value,
            dataset_id=req.dataset,
            hyperparams=req.hyperparameters,
            test_size=req.test_size,
        )
    )
    ml_job_registry.set_task(entry.job_id, task_obj)

    return MLTrainResponse(job_id=entry.job_id, status="queued")


@router.post("/stop")
async def stop_training(job_id: str | None = None) -> dict[str, Any]:
    jid = job_id or ml_job_registry.latest_job_id()
    if jid is None:
        raise HTTPException(404, "No active job")
    if not ml_job_registry.request_stop(jid):
        raise HTTPException(409, "Job already terminal or not found")
    return {"job_id": jid, "status": "stopping"}


@router.get("/status", response_model=MLStatusResponse)
async def get_status(job_id: str | None = None) -> MLStatusResponse:
    jid = job_id or ml_job_registry.latest_job_id()
    if jid is None:
        raise HTTPException(404, "No jobs found")
    entry = ml_job_registry.get(jid)
    if entry is None:
        raise HTTPException(404, f"Job {jid!r} not found")
    return MLStatusResponse(
        job_id=entry.job_id,
        status=entry.status,
        model_type=entry.model_type,
        dataset=entry.dataset,
        metrics=entry.final_metrics,
        error=entry.error,
        feature_names=entry.feature_names,
        target_names=entry.target_names,
    )


@router.get("/latest")
async def get_latest() -> dict[str, Any]:
    jid = ml_job_registry.latest_job_id()
    if jid is None:
        raise HTTPException(404, "No jobs found")
    entry = ml_job_registry.get(jid)
    if entry is None:
        raise HTTPException(404, "No jobs found")
    return {
        "job_id": entry.job_id,
        "status": entry.status,
        "model_type": entry.model_type,
        "dataset": entry.dataset,
        "metrics": entry.final_metrics,
    }


# ── Prediction ────────────────────────────────────────


@router.post("/predict", response_model=MLPredictResponse)
async def predict(req: MLPredictRequest) -> MLPredictResponse:
    entry = ml_job_registry.get(req.job_id)
    if entry is None:
        raise HTTPException(404, f"Job {req.job_id!r} not found")
    if entry.model is None:
        raise HTTPException(400, "Model has not been trained yet")

    try:
        X = np.array(req.features, dtype=np.float64)
        predictions = entry.model.predict(X).tolist()

        probabilities = None
        if hasattr(entry.model, "predict_proba"):
            probabilities = entry.model.predict_proba(X).tolist()

        importances = None
        if hasattr(entry.model, "feature_importances_") and entry.feature_names:
            importances = {
                name: float(imp)
                for name, imp in zip(entry.feature_names, entry.model.feature_importances_)
            }
        elif hasattr(entry.model, "coef_") and entry.feature_names:
            coef = np.abs(entry.model.coef_)
            if coef.ndim > 1:
                coef = coef.mean(axis=0)
            importances = {
                name: float(imp)
                for name, imp in zip(entry.feature_names, coef)
            }

        return MLPredictResponse(
            predictions=predictions,
            probabilities=probabilities,
            feature_importances=importances,
        )
    except Exception as exc:
        raise HTTPException(400, f"Prediction failed: {exc}") from exc
