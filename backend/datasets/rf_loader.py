from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datasets.rf_registry import get_rf_dataset_meta


RF_KAGGLE_DATA_DIR = Path(__file__).resolve().parent / "data" / "rf_kaggle"


@dataclass
class RFDatasetBundle:
    dataset_id: str
    features: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    source_csv: Path


def _kaggle_cli_command() -> str:
    kaggle_exe = Path(sys.executable).resolve().parent / "kaggle"
    if kaggle_exe.exists():
        return str(kaggle_exe)

    which_kaggle = shutil.which("kaggle")
    if which_kaggle is None:
        raise RuntimeError("Kaggle CLI executable not found. Install `kaggle` in the backend environment.")
    return which_kaggle


def _resolve_csv_file(dataset_dir: Path, expected_csv: str) -> Path:
    explicit = dataset_dir / expected_csv
    if explicit.exists():
        return explicit

    csv_candidates = sorted(dataset_dir.rglob("*.csv"))
    if not csv_candidates:
        raise RuntimeError(f"No CSV files found in {dataset_dir}")
    return csv_candidates[0]


def _ensure_kaggle_dataset(dataset_id: str, meta: dict[str, Any]) -> Path:
    target_dir = RF_KAGGLE_DATA_DIR / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = str(meta["csv_filename"])
    existing_csv = _resolve_csv_file(target_dir, csv_filename) if any(target_dir.glob("**/*.csv")) else None
    if existing_csv is not None:
        return existing_csv

    cmd = [
        _kaggle_cli_command(),
        "datasets",
        "download",
        "-d",
        str(meta["kaggle_dataset"]),
        "-p",
        str(target_dir),
        "--unzip",
        "--force",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = (result.stderr or "").strip() or (result.stdout or "").strip() or "unknown Kaggle CLI error"
        raise RuntimeError(
            "Kaggle download failed. Ensure Kaggle credentials are configured "
            "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY). "
            f"CLI output: {details}"
        )

    return _resolve_csv_file(target_dir, csv_filename)


def load_rf_dataset(dataset_id: str) -> RFDatasetBundle:
    normalized = dataset_id.strip().lower()
    meta = get_rf_dataset_meta(normalized)
    if meta is None:
        raise RuntimeError(f"Unsupported RF dataset: {dataset_id}")

    csv_path = _ensure_kaggle_dataset(normalized, meta)
    delimiter = str(meta.get("delimiter", ","))
    frame = pd.read_csv(csv_path, sep=delimiter)

    target_column = str(meta["target_column"])
    if target_column not in frame.columns:
        raise RuntimeError(
            f"Target column '{target_column}' not found in {csv_path.name}. "
            f"Columns: {list(frame.columns)}"
        )

    drop_columns = [column for column in meta.get("drop_columns", []) if column in frame.columns]
    feature_frame = frame.drop(columns=[target_column, *drop_columns], errors="ignore")
    if feature_frame.shape[1] == 0:
        raise RuntimeError(f"No feature columns remain after dropping target columns for dataset '{normalized}'")

    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    if feature_frame.isna().any(axis=None):
        bad_columns = [column for column in feature_frame.columns if feature_frame[column].isna().any()]
        raise RuntimeError(
            f"Feature columns contain non-numeric values after coercion for dataset '{normalized}': {bad_columns}"
        )

    target_series = frame[target_column]
    if target_series.isna().any():
        raise RuntimeError(f"Target column '{target_column}' contains missing values for dataset '{normalized}'")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(target_series.to_numpy())
    class_names = [str(value) for value in encoder.classes_.tolist()]

    features = feature_frame.to_numpy(dtype=np.float32, copy=True)
    return RFDatasetBundle(
        dataset_id=normalized,
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        feature_names=[str(column) for column in feature_frame.columns.tolist()],
        class_names=class_names,
        source_csv=csv_path,
    )


def split_rf_dataset(
    bundle: RFDatasetBundle,
    *,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratify_labels = bundle.labels if stratify else None
    return train_test_split(
        bundle.features,
        bundle.labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
