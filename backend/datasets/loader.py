from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets.registry import get_dataset_meta


KAGGLE_DATASET_REF = "oddrationale/mnist-in-csv"
KAGGLE_DATA_DIR = Path(__file__).resolve().parent / "data" / "mnist_kaggle"
TRAIN_CSV = KAGGLE_DATA_DIR / "mnist_train.csv"
TEST_CSV = KAGGLE_DATA_DIR / "mnist_test.csv"


def _ensure_kaggle_mnist() -> tuple[Path, Path]:
    KAGGLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_CSV.exists() and TEST_CSV.exists():
        return TRAIN_CSV, TEST_CSV

    kaggle_exe = Path(sys.executable).resolve().parent / "kaggle"
    if kaggle_exe.exists():
        kaggle_cmd = str(kaggle_exe)
    else:
        which_kaggle = shutil.which("kaggle")
        if which_kaggle is None:
            raise RuntimeError(
                "Kaggle CLI executable not found. Install `kaggle` in the backend environment."
            )
        kaggle_cmd = which_kaggle

    cmd = [
        kaggle_cmd,
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET_REF,
        "-p",
        str(KAGGLE_DATA_DIR),
        "--unzip",
        "--force",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or "unknown Kaggle CLI error"
        raise RuntimeError(
            "Kaggle download failed. Ensure Kaggle credentials are configured "
            "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY). "
            f"CLI output: {details}"
        )

    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise RuntimeError(
            "Kaggle download completed but mnist_train.csv/mnist_test.csv were not found"
        )

    return TRAIN_CSV, TEST_CSV


def _load_mnist_tensor_dataset(csv_path: Path) -> torch.utils.data.TensorDataset:
    cache_path = csv_path.with_suffix(".tensor.pt")
    if cache_path.exists() and cache_path.stat().st_mtime >= csv_path.stat().st_mtime:
        cached = torch.load(cache_path, map_location="cpu")
        return torch.utils.data.TensorDataset(cached["features"], cached["labels"])

    matrix = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 785:
        raise RuntimeError(
            f"Unexpected MNIST CSV shape in {csv_path}: expected (*, 785), got {matrix.shape}"
        )

    labels = torch.from_numpy(matrix[:, 0].astype(np.int64, copy=False))
    features = torch.from_numpy(matrix[:, 1:] / 255.0).view(-1, 1, 28, 28)
    torch.save({"features": features, "labels": labels}, cache_path)
    return torch.utils.data.TensorDataset(features, labels)


def get_mnist_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_csv, test_csv = _ensure_kaggle_mnist()
    train_set = _load_mnist_tensor_dataset(train_csv)
    test_set = _load_mnist_tensor_dataset(test_csv)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def get_digits_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    digits = load_digits()
    features = digits.data.astype(np.float32, copy=False) / 16.0
    labels = digits.target.astype(np.int64, copy=False)

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_x_tensor = torch.from_numpy(train_x).view(-1, 1, 8, 8)
    test_x_tensor = torch.from_numpy(test_x).view(-1, 1, 8, 8)
    train_y_tensor = torch.from_numpy(train_y)
    test_y_tensor = torch.from_numpy(test_y)

    train_set = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
    test_set = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def get_dataset_dataloaders(dataset_id: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    meta = get_dataset_meta(dataset_id)
    if meta is None:
        raise RuntimeError(f"Unsupported dataset for v1: {dataset_id}")

    loader_id = str(meta.get("loader", "")).strip().lower()
    if loader_id == "kaggle_mnist_csv":
        return get_mnist_dataloaders(batch_size)
    if loader_id == "sklearn_digits":
        return get_digits_dataloaders(batch_size)

    raise RuntimeError(f"Unsupported dataset loader for '{dataset_id}': {loader_id}")
