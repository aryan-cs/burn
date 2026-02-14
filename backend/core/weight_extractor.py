from __future__ import annotations

import torch
import torch.nn as nn


def extract_weight_snapshot(model: nn.Module, bins: int = 20) -> dict[str, dict[str, float | list[float]]]:
    snapshot: dict[str, dict[str, float | list[float]]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        tensor = param.detach().float().cpu().view(-1)
        if tensor.numel() == 0:
            continue

        min_val = float(tensor.min().item())
        max_val = float(tensor.max().item())

        if min_val == max_val:
            histogram = [float(tensor.numel())] + [0.0] * (bins - 1)
        else:
            histogram = torch.histc(tensor, bins=bins, min=min_val, max=max_val).tolist()

        snapshot[name] = {
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item()),
            "min": min_val,
            "max": max_val,
            "histogram": [float(v) for v in histogram],
        }

    return snapshot
