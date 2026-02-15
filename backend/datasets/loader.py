from __future__ import annotations

import base64
import io
import os
import random
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets.registry import get_dataset_meta


MNIST_KAGGLE_REF = "oddrationale/mnist-in-csv"
MNIST_DATA_DIR = Path(__file__).resolve().parent / "data" / "mnist_kaggle"
MNIST_TRAIN_CSV = MNIST_DATA_DIR / "mnist_train.csv"
MNIST_TEST_CSV = MNIST_DATA_DIR / "mnist_test.csv"

CATS_DOGS_DATA_DIR = Path(__file__).resolve().parent / "data" / "cats_vs_dogs_kaggle"
CATS_DOGS_IMAGE_SIZE = int(os.getenv("CATS_DOGS_IMAGE_SIZE", "96"))
CATS_DOGS_TEST_SPLIT = float(os.getenv("CATS_DOGS_TEST_SPLIT", "0.2"))
CATS_DOGS_MAX_PER_CLASS = int(os.getenv("CATS_DOGS_MAX_PER_CLASS", "1200"))
CATS_DOGS_RANDOM_SEED = int(os.getenv("CATS_DOGS_RANDOM_SEED", "42"))
CATS_DOGS_NUM_WORKERS = int(os.getenv("CATS_DOGS_NUM_WORKERS", "0"))
CATS_DOGS_SAMPLE_LIMIT = int(os.getenv("CATS_DOGS_SAMPLE_LIMIT", "8"))
CATS_DOGS_REF_CANDIDATES = (
    os.getenv("CATS_DOGS_KAGGLE_REF", "").strip(),
    "karakaggle/kaggle-cat-vs-dog-dataset",
    "tongpython/cat-and-dog",
    "shaunthesheep/microsoft-catsvsdogs-dataset",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Some Kaggle image archives include files with truncated TIFF/EXIF payloads.
# We already skip unreadable samples in dataset access, so suppress this noisy warning.
warnings.filterwarnings(
    "ignore",
    message="Truncated File Read",
    category=UserWarning,
    module=r"PIL\.TiffImagePlugin",
)


class _ImagePathDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not self.samples:
            raise IndexError("Dataset is empty")

        index = int(index) % len(self.samples)
        path, label = self.samples[index]
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)
        except Exception:
            next_index = (index + 1) % len(self.samples)
            if next_index == index:
                raise
            return self.__getitem__(next_index)

        return image, int(label)


def _resolve_kaggle_cli() -> str:
    kaggle_exe = Path(sys.executable).resolve().parent / "kaggle"
    if kaggle_exe.exists():
        return str(kaggle_exe)

    which_kaggle = shutil.which("kaggle")
    if which_kaggle is None:
        raise RuntimeError(
            "Kaggle CLI executable not found. Install `kaggle` in the backend environment."
        )
    return which_kaggle


def _download_kaggle_dataset(dataset_ref: str, destination: Path) -> tuple[bool, str]:
    kaggle_cmd = _resolve_kaggle_cli()
    cmd = [
        kaggle_cmd,
        "datasets",
        "download",
        "-d",
        dataset_ref,
        "-p",
        str(destination),
        "--unzip",
        "--force",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return True, ""
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    details = stderr or stdout or "unknown Kaggle CLI error"
    return False, details


def _candidate_refs(values: Iterable[str]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        refs.append(normalized)
    return refs


def _ensure_kaggle_mnist() -> tuple[Path, Path]:
    MNIST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if MNIST_TRAIN_CSV.exists() and MNIST_TEST_CSV.exists():
        return MNIST_TRAIN_CSV, MNIST_TEST_CSV

    ok, details = _download_kaggle_dataset(MNIST_KAGGLE_REF, MNIST_DATA_DIR)
    if not ok:
        raise RuntimeError(
            "Kaggle download failed. Ensure Kaggle credentials are configured "
            "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY). "
            f"CLI output: {details}"
        )

    if not MNIST_TRAIN_CSV.exists() or not MNIST_TEST_CSV.exists():
        raise RuntimeError(
            "Kaggle download completed but mnist_train.csv/mnist_test.csv were not found"
        )

    return MNIST_TRAIN_CSV, MNIST_TEST_CSV


def _find_class_root(data_dir: Path) -> tuple[Path, str, str] | None:
    candidates: list[tuple[Path, str, str]] = []
    for directory in data_dir.rglob("*"):
        if not directory.is_dir():
            continue
        try:
            children = [entry.name for entry in directory.iterdir() if entry.is_dir()]
        except OSError:
            continue
        lower_children = {name.lower(): name for name in children}
        if "cats" in lower_children and "dogs" in lower_children:
            candidates.append(
                (directory, lower_children["cats"], lower_children["dogs"])
            )
        elif "cat" in lower_children and "dog" in lower_children:
            candidates.append(
                (directory, lower_children["cat"], lower_children["dog"])
            )

    if not candidates:
        return None

    # Prefer shallower roots that contain many files.
    candidates.sort(key=lambda item: (len(item[0].parts), str(item[0])))
    return candidates[0]


def _ensure_kaggle_cats_vs_dogs() -> tuple[Path, str, str, str]:
    CATS_DOGS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_root = _find_class_root(CATS_DOGS_DATA_DIR)
    if existing_root is not None:
        return (
            existing_root[0],
            existing_root[1],
            existing_root[2],
            "existing_local_data",
        )

    errors: list[str] = []
    for dataset_ref in _candidate_refs(CATS_DOGS_REF_CANDIDATES):
        ok, details = _download_kaggle_dataset(dataset_ref, CATS_DOGS_DATA_DIR)
        if not ok:
            errors.append(f"{dataset_ref}: {details}")
            continue
        found_root = _find_class_root(CATS_DOGS_DATA_DIR)
        if found_root is not None:
            return (
                found_root[0],
                found_root[1],
                found_root[2],
                dataset_ref,
            )
        errors.append(
            f"{dataset_ref}: downloaded but class folders were not found (expected cats/dogs or cat/dog)"
        )

    raise RuntimeError(
        "Kaggle download failed for Cats vs Dogs. Configure Kaggle credentials and set "
        "CATS_DOGS_KAGGLE_REF if needed. Attempted refs: "
        + "; ".join(errors)
    )


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


def _collect_class_images(root: Path, class_dir_name: str) -> list[Path]:
    class_dir = root / class_dir_name
    if not class_dir.exists() or not class_dir.is_dir():
        return []

    candidates = [
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    candidates.sort()
    return candidates


def _split_cats_dogs_samples(
    cat_paths: list[Path],
    dog_paths: list[Path],
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(CATS_DOGS_RANDOM_SEED)

    def split_class(paths: list[Path], label: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
        selected = paths[:]
        rng.shuffle(selected)
        if CATS_DOGS_MAX_PER_CLASS > 0:
            selected = selected[:CATS_DOGS_MAX_PER_CLASS]
        if len(selected) < 2:
            raise RuntimeError(
                "Cats vs Dogs dataset has too few usable images for one class; expected at least 2."
            )

        test_fraction = min(max(CATS_DOGS_TEST_SPLIT, 0.05), 0.45)
        test_count = max(1, int(round(len(selected) * test_fraction)))
        test_count = min(test_count, len(selected) - 1)
        split_idx = len(selected) - test_count
        train = [(path, label) for path in selected[:split_idx]]
        test = [(path, label) for path in selected[split_idx:]]
        return train, test

    train_cats, test_cats = split_class(cat_paths, 0)
    train_dogs, test_dogs = split_class(dog_paths, 1)
    train_samples = train_cats + train_dogs
    test_samples = test_cats + test_dogs
    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def _cats_dogs_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((CATS_DOGS_IMAGE_SIZE, CATS_DOGS_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _encode_preview_data_url(image: Image.Image) -> str:
    preview = image.resize((CATS_DOGS_IMAGE_SIZE, CATS_DOGS_IMAGE_SIZE))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def get_cats_vs_dogs_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    root_dir, cat_dir_name, dog_dir_name, source_ref = _ensure_kaggle_cats_vs_dogs()

    cat_paths = _collect_class_images(root_dir, cat_dir_name)
    dog_paths = _collect_class_images(root_dir, dog_dir_name)

    if not cat_paths or not dog_paths:
        raise RuntimeError(
            "Cats vs Dogs dataset is missing class image files. "
            f"Detected root={root_dir} source={source_ref}"
        )

    train_samples, test_samples = _split_cats_dogs_samples(cat_paths, dog_paths)
    train_transform = transforms.Compose(
        [
            transforms.Resize((CATS_DOGS_IMAGE_SIZE, CATS_DOGS_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    test_transform = _cats_dogs_eval_transform()

    train_set = _ImagePathDataset(train_samples, transform=train_transform)
    test_set = _ImagePathDataset(test_samples, transform=test_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CATS_DOGS_NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CATS_DOGS_NUM_WORKERS,
    )
    return train_loader, test_loader


def get_cats_vs_dogs_inference_samples(
    limit: int = CATS_DOGS_SAMPLE_LIMIT,
    split: str = "test",
) -> list[dict[str, object]]:
    normalized_split = split.strip().lower()
    if normalized_split not in {"train", "test"}:
        raise RuntimeError("split must be either 'train' or 'test'")

    safe_limit = max(1, min(int(limit), 32))
    root_dir, cat_dir_name, dog_dir_name, _ = _ensure_kaggle_cats_vs_dogs()
    cat_paths = _collect_class_images(root_dir, cat_dir_name)
    dog_paths = _collect_class_images(root_dir, dog_dir_name)
    if not cat_paths or not dog_paths:
        raise RuntimeError("Cats vs Dogs dataset is missing class image files.")

    train_samples, test_samples = _split_cats_dogs_samples(cat_paths, dog_paths)
    source_samples = test_samples if normalized_split == "test" else train_samples
    if not source_samples:
        raise RuntimeError(f"No samples available for split '{normalized_split}'")

    rng = random.Random(CATS_DOGS_RANDOM_SEED + (101 if normalized_split == "test" else 53))
    selected = source_samples[:]
    rng.shuffle(selected)
    selected = selected[:safe_limit]

    transform = _cats_dogs_eval_transform()
    label_names = {0: "cat", 1: "dog"}
    samples: list[dict[str, object]] = []
    for index, (path, label) in enumerate(selected):
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                tensor = transform(rgb)
                samples.append(
                    {
                        "id": f"{normalized_split}:{index}:{path.name}",
                        "filename": path.name,
                        "label": int(label),
                        "label_name": label_names.get(int(label), f"class_{label}"),
                        "image_data_url": _encode_preview_data_url(rgb),
                        "inputs": tensor.tolist(),
                    }
                )
        except Exception:
            continue

    if not samples:
        raise RuntimeError("Could not load any valid Cats vs Dogs samples for inference.")
    return samples


def get_dataset_dataloaders(dataset_id: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    meta = get_dataset_meta(dataset_id)
    if meta is None:
        raise RuntimeError(f"Unsupported dataset for v1: {dataset_id}")

    loader_id = str(meta.get("loader", "")).strip().lower()
    if loader_id == "kaggle_mnist_csv":
        return get_mnist_dataloaders(batch_size)
    if loader_id == "sklearn_digits":
        return get_digits_dataloaders(batch_size)
    if loader_id == "kaggle_cats_vs_dogs":
        return get_cats_vs_dogs_dataloaders(batch_size)

    raise RuntimeError(f"Unsupported dataset loader for '{dataset_id}': {loader_id}")
