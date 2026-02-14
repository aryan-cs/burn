from __future__ import annotations

import asyncio
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from core.rf_compiler import CompiledRFResult
from core.rf_job_registry import rf_job_registry
from datasets.rf_loader import load_rf_dataset, split_rf_dataset
from models.rf_training_config import RFTrainingConfig


def _progress_checkpoints(total_trees: int, log_every_trees: int) -> list[int]:
    step = max(1, min(total_trees, log_every_trees))
    checkpoints = list(range(step, total_trees + 1, step))
    if not checkpoints or checkpoints[-1] != total_trees:
        checkpoints.append(total_trees)
    return checkpoints


async def run_rf_training_job(
    job_id: str,
    compiled: CompiledRFResult,
    training: RFTrainingConfig,
    artifacts_dir: Path,
) -> None:
    entry = rf_job_registry.get(job_id)
    if entry is None:
        return

    try:
        rf_job_registry.set_status(job_id, "running")
        bundle = load_rf_dataset(training.dataset)
        train_x, test_x, train_y, test_y = split_rf_dataset(
            bundle,
            test_size=training.test_size,
            random_state=training.random_state,
            stratify=training.stratify,
        )

        if int(train_x.shape[1]) != compiled.expected_feature_count:
            message = (
                "Input feature mismatch between RF graph and dataset. "
                f"Graph expects {compiled.expected_feature_count}, dataset has {train_x.shape[1]}"
            )
            await rf_job_registry.publish(job_id, {"type": "rf_error", "message": message})
            await rf_job_registry.mark_terminal(job_id, "failed", error=message)
            return

        hyper = compiled.hyperparams
        classifier = RandomForestClassifier(
            n_estimators=1,
            criterion=hyper.criterion,
            max_depth=hyper.max_depth,
            min_samples_split=hyper.min_samples_split,
            min_samples_leaf=hyper.min_samples_leaf,
            max_features=hyper.max_features,
            bootstrap=hyper.bootstrap,
            random_state=hyper.random_state,
            warm_start=True,
            n_jobs=-1,
        )

        total_trees = hyper.n_estimators
        checkpoints = _progress_checkpoints(total_trees, training.log_every_trees)
        started_at = time.perf_counter()

        final_train_accuracy = 0.0
        final_test_accuracy = 0.0
        final_confusion: list[list[int]] = []
        fitted_once = False

        for tree_count in checkpoints:
            if entry.stop_event.is_set():
                break

            classifier.set_params(n_estimators=tree_count)
            classifier.fit(train_x, train_y)
            fitted_once = True

            train_pred = classifier.predict(train_x)
            test_pred = classifier.predict(test_x)

            final_train_accuracy = float(accuracy_score(train_y, train_pred))
            final_test_accuracy = float(accuracy_score(test_y, test_pred))
            final_confusion = confusion_matrix(test_y, test_pred).tolist()

            oob_score = getattr(classifier, "oob_score_", None)
            if oob_score is not None:
                try:
                    oob_score = float(oob_score)
                except Exception:
                    oob_score = None

            await rf_job_registry.publish(
                job_id,
                {
                    "type": "rf_progress",
                    "stage": "training",
                    "trees_built": tree_count,
                    "total_trees": total_trees,
                    "train_accuracy": final_train_accuracy,
                    "test_accuracy": final_test_accuracy,
                    "oob_score": oob_score,
                    "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                },
            )
            await asyncio.sleep(0)

        if not fitted_once:
            await rf_job_registry.mark_terminal(
                job_id,
                "stopped",
                final_metrics={"final_train_accuracy": 0.0, "final_test_accuracy": 0.0},
            )
            return

        artifact_dir = artifacts_dir / "rf"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{job_id}.pkl"

        artifact_payload = {
            "model": classifier,
            "dataset": bundle.dataset_id,
            "feature_names": bundle.feature_names,
            "class_names": bundle.class_names,
            "expected_feature_count": compiled.expected_feature_count,
            "hyperparameters": compiled.hyperparams.model_dump(),
        }
        joblib.dump(artifact_payload, artifact_path)
        rf_job_registry.set_model_data(
            job_id,
            model=classifier,
            feature_names=bundle.feature_names,
            class_names=bundle.class_names,
            expected_feature_count=compiled.expected_feature_count,
        )

        final_status = "stopped" if entry.stop_event.is_set() else "completed"
        await rf_job_registry.publish(
            job_id,
            {
                "type": "rf_done",
                "final_train_accuracy": final_train_accuracy,
                "final_test_accuracy": final_test_accuracy,
                "confusion_matrix": final_confusion,
                "classes": bundle.class_names,
                "feature_importances": [float(value) for value in classifier.feature_importances_.tolist()],
                "feature_names": bundle.feature_names,
                "model_path": str(artifact_path),
            },
        )
        await rf_job_registry.mark_terminal(
            job_id,
            final_status,
            final_metrics={
                "final_train_accuracy": final_train_accuracy,
                "final_test_accuracy": final_test_accuracy,
            },
            artifact_path=artifact_path,
        )
    except Exception as exc:  # pragma: no cover
        message = str(exc)
        await rf_job_registry.publish(job_id, {"type": "rf_error", "message": message})
        await rf_job_registry.mark_terminal(job_id, "failed", error=message)
