"""Training engine for classical ML models (scikit-learn).

Runs training asynchronously, emitting progress messages that the
WebSocket can stream to the frontend.  Supports:
  - Linear Regression
  - Logistic Regression (with per-iteration progress)
  - Random Forest (with per-tree progress)
"""

from __future__ import annotations

import asyncio
import pickle
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from core.ml_job_registry import MLJobRegistry
from datasets.ml_loader import MLDataSplit, load_ml_dataset


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "ml"


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    average = "binary" if n_classes == 2 else "weighted"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _build_model(model_type: str, task: str, hyperparams: dict[str, Any]) -> Any:
    """Instantiate the scikit-learn estimator."""
    if model_type == "linear_regression":
        return LinearRegression(
            fit_intercept=hyperparams.get("fit_intercept", True),
        )
    if model_type == "logistic_regression":
        return LogisticRegression(
            C=hyperparams.get("C", 1.0),
            max_iter=hyperparams.get("max_iter", 200),
            penalty=hyperparams.get("penalty", "l2"),
            solver=hyperparams.get("solver", "lbfgs"),
            warm_start=True,  # enable incremental training for progress
        )
    if model_type == "random_forest":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            min_samples_split=hyperparams.get("min_samples_split", 2),
            min_samples_leaf=hyperparams.get("min_samples_leaf", 1),
            criterion=hyperparams.get("criterion", "gini" if task == "classification" else "squared_error"),
            max_features=hyperparams.get("max_features", "sqrt"),
            warm_start=True,  # enable incremental tree building
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model type: {model_type!r}")


def _get_feature_importances(model: Any, feature_names: list[str]) -> dict[str, float] | None:
    """Extract feature importances if the model supports them."""
    if hasattr(model, "feature_importances_"):
        return {
            name: float(imp)
            for name, imp in zip(feature_names, model.feature_importances_)
        }
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        if coef.ndim > 1:
            coef = coef.mean(axis=0)
        return {
            name: float(imp)
            for name, imp in zip(feature_names, coef)
        }
    return None


async def run_ml_training(
    registry: MLJobRegistry,
    job_id: str,
    model_type: str,
    dataset_id: str,
    hyperparams: dict[str, Any],
    test_size: float = 0.2,
) -> None:
    """Async wrapper that trains a classical ML model and publishes progress."""
    entry = registry.get(job_id)
    if entry is None:
        return

    try:
        entry.status = "loading_data"
        await registry.publish(job_id, {"type": "status", "status": "loading_data"})
        await asyncio.sleep(0)

        # Load data
        data: MLDataSplit = load_ml_dataset(dataset_id, test_size=test_size)
        entry.feature_names = data.feature_names
        entry.target_names = data.target_names
        entry.n_classes = data.n_classes

        await registry.publish(job_id, {
            "type": "data_loaded",
            "n_train": len(data.y_train),
            "n_test": len(data.y_test),
            "n_features": len(data.feature_names),
            "feature_names": data.feature_names,
            "target_names": data.target_names,
            "task": data.task,
        })

        # Check for stop
        if entry.stop_event.is_set():
            await registry.mark_terminal(job_id, "stopped")
            return

        # Build model
        entry.status = "training"
        await registry.publish(job_id, {"type": "status", "status": "training"})
        await asyncio.sleep(0)

        model = _build_model(model_type, data.task, hyperparams)
        t0 = time.perf_counter()

        # ── Train with progress ──
        if model_type == "logistic_regression":
            await _train_logistic_incremental(
                registry, job_id, entry, model, data, hyperparams,
            )
        elif model_type == "random_forest":
            await _train_rf_incremental(
                registry, job_id, entry, model, data, hyperparams,
            )
        else:
            # Linear regression — trains instantly
            model.fit(data.X_train, data.y_train)
            elapsed = time.perf_counter() - t0
            await registry.publish(job_id, {
                "type": "training_progress",
                "progress": 1.0,
                "step": 1,
                "total_steps": 1,
                "elapsed": elapsed,
            })

        if entry.stop_event.is_set():
            await registry.mark_terminal(job_id, "stopped")
            return

        # If model was set during incremental training, use that; otherwise set now
        if entry.model is None:
            entry.model = model

        # ── Evaluate ──
        train_pred = entry.model.predict(data.X_train)
        test_pred = entry.model.predict(data.X_test)

        if data.task == "classification":
            train_metrics = _classification_metrics(data.y_train, train_pred, data.n_classes)
            test_metrics = _classification_metrics(data.y_test, test_pred, data.n_classes)
        else:
            train_metrics = _regression_metrics(data.y_train, train_pred)
            test_metrics = _regression_metrics(data.y_test, test_pred)

        feature_importances = _get_feature_importances(entry.model, data.feature_names)

        elapsed = time.perf_counter() - t0
        final_metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "training_time": elapsed,
        }

        await registry.publish(job_id, {
            "type": "evaluation",
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importances": feature_importances,
            "elapsed": elapsed,
        })

        # ── Save artifact ──
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = ARTIFACTS_DIR / f"{job_id}.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(entry.model, f)

        await registry.publish(job_id, {"type": "training_done", "metrics": final_metrics})
        await registry.mark_terminal(
            job_id,
            "completed",
            final_metrics=final_metrics,
            artifact_path=artifact_path,
        )

    except Exception as exc:
        tb = traceback.format_exc()
        await registry.publish(job_id, {"type": "error", "error": str(exc), "traceback": tb})
        await registry.mark_terminal(job_id, "failed", error=str(exc))


async def _train_logistic_incremental(
    registry: MLJobRegistry,
    job_id: str,
    entry: Any,
    model: LogisticRegression,
    data: MLDataSplit,
    hyperparams: dict[str, Any],
) -> None:
    """Train logistic regression in chunks to emit progress updates."""
    total_iters = hyperparams.get("max_iter", 200)
    chunk_size = max(1, total_iters // 20)  # ~20 progress updates
    done_iters = 0

    while done_iters < total_iters:
        if entry.stop_event.is_set():
            break

        iters_this_round = min(chunk_size, total_iters - done_iters)
        model.max_iter = done_iters + iters_this_round
        model.fit(data.X_train, data.y_train)
        done_iters += iters_this_round

        # Intermediate accuracy
        train_pred = model.predict(data.X_train)
        train_acc = float(accuracy_score(data.y_train, train_pred))
        test_pred = model.predict(data.X_test)
        test_acc = float(accuracy_score(data.y_test, test_pred))

        progress = min(1.0, done_iters / total_iters)
        await registry.publish(job_id, {
            "type": "training_progress",
            "progress": progress,
            "step": done_iters,
            "total_steps": total_iters,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        })
        await asyncio.sleep(0)

        # Check convergence
        if hasattr(model, "n_iter_") and model.n_iter_[0] < model.max_iter:
            # Converged early
            await registry.publish(job_id, {
                "type": "training_progress",
                "progress": 1.0,
                "step": total_iters,
                "total_steps": total_iters,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "converged": True,
            })
            break

    entry.model = model


async def _train_rf_incremental(
    registry: MLJobRegistry,
    job_id: str,
    entry: Any,
    model: Any,
    data: MLDataSplit,
    hyperparams: dict[str, Any],
) -> None:
    """Train random forest incrementally, adding trees in batches."""
    total_trees = hyperparams.get("n_estimators", 100)
    batch_size = max(1, total_trees // 20)  # ~20 progress updates
    built_trees = 0

    while built_trees < total_trees:
        if entry.stop_event.is_set():
            break

        trees_this_round = min(batch_size, total_trees - built_trees)
        built_trees += trees_this_round
        model.n_estimators = built_trees
        model.fit(data.X_train, data.y_train)

        # Intermediate metrics
        train_pred = model.predict(data.X_train)
        test_pred = model.predict(data.X_test)

        if data.task == "classification":
            train_acc = float(accuracy_score(data.y_train, train_pred))
            test_acc = float(accuracy_score(data.y_test, test_pred))
            progress_metrics = {"train_accuracy": train_acc, "test_accuracy": test_acc}
        else:
            train_r2 = float(r2_score(data.y_train, train_pred))
            test_r2 = float(r2_score(data.y_test, test_pred))
            progress_metrics = {"train_r2": train_r2, "test_r2": test_r2}

        progress = min(1.0, built_trees / total_trees)
        await registry.publish(job_id, {
            "type": "training_progress",
            "progress": progress,
            "step": built_trees,
            "total_steps": total_trees,
            **progress_metrics,
        })
        await asyncio.sleep(0)

    entry.model = model
