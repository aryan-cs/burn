from __future__ import annotations

from models.training_config import normalize_training_config


def test_training_config_alias_normalization() -> None:
    cfg = normalize_training_config(
        {
            "dataset": "MNIST",
            "epochs": 5,
            "batchSize": 32,
            "optimizer": "Adam",
            "learningRate": 0.01,
            "loss": "Cross_Entropy",
        }
    )

    assert cfg.dataset == "mnist"
    assert cfg.epochs == 5
    assert cfg.batch_size == 32
    assert cfg.optimizer == "adam"
    assert cfg.learning_rate == 0.01
    assert cfg.loss == "cross_entropy"


def test_training_config_snake_case() -> None:
    cfg = normalize_training_config(
        {
            "dataset": "mnist",
            "epochs": 2,
            "batch_size": 16,
            "optimizer": "sgd",
            "learning_rate": 0.05,
            "loss": "cross_entropy",
        }
    )

    assert cfg.batch_size == 16
    assert cfg.learning_rate == 0.05
