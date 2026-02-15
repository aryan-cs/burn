from __future__ import annotations

import pytest
import torch

from datasets.loader import get_dataset_dataloaders


def test_digits_dataloaders_shape() -> None:
    train_loader, test_loader = get_dataset_dataloaders("digits", batch_size=32)
    train_x, train_y = next(iter(train_loader))
    test_x, test_y = next(iter(test_loader))

    assert list(train_x.shape[1:]) == [1, 8, 8]
    assert list(test_x.shape[1:]) == [1, 8, 8]
    assert train_y.dtype == torch.int64
    assert test_y.dtype == torch.int64


def test_unknown_dataset_loader_rejected() -> None:
    with pytest.raises(RuntimeError, match="Unsupported dataset for v1"):
        get_dataset_dataloaders("unknown_dataset", batch_size=16)
