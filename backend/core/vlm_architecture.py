from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

from transformers import AutoConfig


DEFAULT_IMAGE_SIZE = 320

RESNET_STAGE_DEPTHS: dict[int, list[int]] = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}

_ARCHITECTURE_CACHE: dict[str, dict[str, Any]] = {}


@dataclass
class VLMStageSpec:
    id: str
    label: str
    detail: str
    description: str
    color: str
    size: list[float]
    position: list[float]


def load_vlm_architecture_spec(model_id: str) -> dict[str, Any]:
    normalized = model_id.strip() or "hustvl/yolos-tiny"
    cached = _ARCHITECTURE_CACHE.get(normalized)
    if cached is not None:
        return copy.deepcopy(cached)

    try:
        config = AutoConfig.from_pretrained(normalized)
        spec = _build_from_config(normalized, config)
        _ARCHITECTURE_CACHE[normalized] = copy.deepcopy(spec)
        return spec
    except Exception as exc:
        fallback = _build_known_fallback(normalized)
        fallback["source"] = "fallback"
        fallback["warning"] = (
            "Could not load model config from Hugging Face; using built-in architecture fallback. "
            f"Details: {exc}"
        )
        _ARCHITECTURE_CACHE[normalized] = copy.deepcopy(fallback)
        return fallback


def _build_from_config(model_id: str, config: Any) -> dict[str, Any]:
    model_type = str(getattr(config, "model_type", "")).lower()
    normalized = model_id.lower()

    if "detr" in model_type or "detr" in normalized:
        return _build_detr_spec(model_id, config, source="huggingface_config")
    if "yolos" in model_type or "yolos" in normalized:
        return _build_yolos_spec(model_id, config, source="huggingface_config")

    return _build_generic_spec(model_id, config, source="huggingface_config")


def _build_detr_spec(model_id: str, config: Any, *, source: str) -> dict[str, Any]:
    image_size = _resolve_image_size(config, default=DEFAULT_IMAGE_SIZE)
    backbone_name = str(getattr(config, "backbone", "resnet50")).lower()
    depths = _resolve_resnet_depths(config, backbone_name)

    encoder_layers = _to_int(getattr(config, "encoder_layers", 6), 6)
    decoder_layers = _to_int(getattr(config, "decoder_layers", 6), 6)
    encoder_heads = _to_int(getattr(config, "encoder_attention_heads", 8), 8)
    decoder_heads = _to_int(getattr(config, "decoder_attention_heads", 8), 8)
    hidden_size = _to_int(getattr(config, "d_model", 256), 256)
    num_queries = _to_int(getattr(config, "num_queries", 100), 100)
    feature_tokens = max(1, (image_size // 32) * (image_size // 32))
    token_count = feature_tokens + num_queries

    stages = [
        _stage(
            "input",
            "Image Input",
            f"3x{image_size}x{image_size}",
            "Raw RGB image enters the detector pipeline.",
            "#54baff",
            [1.0, 0.9, 1.0],
            [-8.0, 0.0, 0.0],
        ),
        _stage(
            "preprocess",
            "Preprocess",
            f"resize={image_size}, normalize",
            "Image processor normalization and tensor conversion.",
            "#65dbf6",
            [1.0, 0.75, 1.0],
            [-5.0, 0.0, 0.0],
        ),
        _stage(
            "backbone",
            "CNN Backbone",
            f"{backbone_name}, blocks={depths}",
            "Convolutional backbone feature extraction.",
            "#79f2c0",
            [2.1, 1.4, 1.4],
            [-1.7, 0.0, 0.0],
        ),
        _stage(
            "encoder",
            "Transformer Encoder",
            f"{encoder_layers} layers, {encoder_heads} heads",
            "Global attention over backbone feature tokens.",
            "#ffd37a",
            [1.7, 1.05, 1.3],
            [1.6, 0.0, 0.0],
        ),
        _stage(
            "decoder",
            "Transformer Decoder",
            f"{decoder_layers} layers, {decoder_heads} heads",
            "Object queries attend to encoder memory.",
            "#ffb85b",
            [1.6, 0.95, 1.2],
            [4.4, 0.0, 0.0],
        ),
        _stage(
            "head",
            "Detection Head",
            f"{num_queries} queries -> boxes/classes",
            "Final class logits and bounding box regressors.",
            "#ff8a6e",
            [1.25, 0.8, 1.05],
            [7.2, 0.0, 0.0],
        ),
    ]

    return {
        "id": model_id,
        "name": "DETR Vision Pipeline",
        "family": "cnn-transformer",
        "source": source,
        "warning": None,
        "stages": [asdict(stage) for stage in stages],
        "blueprint": {
            "input": {
                "channels": 3,
                "height": image_size,
                "width": image_size,
            },
            "cnn": {
                "stages": [
                    {
                        "id": "stem",
                        "label": "Stem Conv",
                        "blocks": 1,
                        "out_channels": 64,
                        "kernel_size": 7,
                        "stride": 2,
                    },
                    {
                        "id": "res2",
                        "label": "ResNet Stage 2",
                        "blocks": depths[0],
                        "out_channels": 256,
                        "kernel_size": 3,
                        "stride": 1,
                    },
                    {
                        "id": "res3",
                        "label": "ResNet Stage 3",
                        "blocks": depths[1],
                        "out_channels": 512,
                        "kernel_size": 3,
                        "stride": 2,
                    },
                    {
                        "id": "res4",
                        "label": "ResNet Stage 4",
                        "blocks": depths[2],
                        "out_channels": 1024,
                        "kernel_size": 3,
                        "stride": 2,
                    },
                    {
                        "id": "res5",
                        "label": "ResNet Stage 5",
                        "blocks": depths[3],
                        "out_channels": 2048,
                        "kernel_size": 3,
                        "stride": 2,
                    },
                ]
            },
            "transformer": {
                "encoder_layers": encoder_layers,
                "decoder_layers": decoder_layers,
                "attention_heads": max(encoder_heads, decoder_heads),
                "hidden_size": hidden_size,
                "num_queries": num_queries,
                "token_count": token_count,
            },
        },
    }


def _build_yolos_spec(model_id: str, config: Any, *, source: str) -> dict[str, Any]:
    image_size = _resolve_image_size(config, default=512)
    patch_size = _to_int(getattr(config, "patch_size", 16), 16)
    hidden_size = _to_int(getattr(config, "hidden_size", 192), 192)
    encoder_layers = _to_int(getattr(config, "num_hidden_layers", 12), 12)
    attention_heads = _to_int(getattr(config, "num_attention_heads", 3), 3)
    num_queries = _to_int(
        getattr(config, "num_detection_tokens", getattr(config, "num_queries", 100)),
        100,
    )
    patch_tokens = max(1, (image_size // max(1, patch_size)) ** 2)
    token_count = patch_tokens + num_queries

    stages = [
        _stage(
            "input",
            "Image Input",
            f"3x{image_size}x{image_size}",
            "Raw RGB image enters the detector pipeline.",
            "#54baff",
            [1.0, 0.9, 1.0],
            [-7.5, 0.0, 0.0],
        ),
        _stage(
            "preprocess",
            "Preprocess",
            f"resize={image_size}, normalize",
            "Image processor normalization and tensor conversion.",
            "#65dbf6",
            [0.95, 0.72, 0.95],
            [-4.8, 0.0, 0.0],
        ),
        _stage(
            "patch",
            "Patch Embed Conv",
            f"kernel={patch_size}, stride={patch_size}, hidden={hidden_size}",
            "Conv projection converts image patches into embeddings.",
            "#79f2c0",
            [1.45, 1.0, 1.0],
            [-1.7, 0.0, 0.0],
        ),
        _stage(
            "blocks",
            "Transformer Encoder",
            f"{encoder_layers} layers, {attention_heads} heads",
            "Global self-attention over visual and detection tokens.",
            "#ffd37a",
            [2.4, 1.2, 1.4],
            [1.7, 0.0, 0.0],
        ),
        _stage(
            "queries",
            "Detection Tokens",
            f"{num_queries} detection tokens",
            "Learned detection tokens specialized for object localization.",
            "#ffb85b",
            [1.4, 0.92, 1.05],
            [4.9, 0.0, 0.0],
        ),
        _stage(
            "head",
            "Prediction Head",
            "boxes + classes",
            "Outputs class logits and bounding boxes.",
            "#ff8a6e",
            [1.2, 0.78, 0.98],
            [7.6, 0.0, 0.0],
        ),
    ]

    return {
        "id": model_id,
        "name": "YOLOS Vision Pipeline",
        "family": "vit-detector",
        "source": source,
        "warning": None,
        "stages": [asdict(stage) for stage in stages],
        "blueprint": {
            "input": {
                "channels": 3,
                "height": image_size,
                "width": image_size,
            },
            "cnn": {
                "stages": [
                    {
                        "id": "patch_embed",
                        "label": "Patch Embed Conv",
                        "blocks": 1,
                        "out_channels": hidden_size,
                        "kernel_size": patch_size,
                        "stride": patch_size,
                    }
                ]
            },
            "transformer": {
                "encoder_layers": encoder_layers,
                "decoder_layers": 0,
                "attention_heads": attention_heads,
                "hidden_size": hidden_size,
                "num_queries": num_queries,
                "token_count": token_count,
            },
        },
    }


def _build_generic_spec(model_id: str, config: Any, *, source: str) -> dict[str, Any]:
    image_size = _resolve_image_size(config, default=DEFAULT_IMAGE_SIZE)
    hidden_size = _to_int(getattr(config, "hidden_size", getattr(config, "d_model", 256)), 256)
    encoder_layers = _to_int(
        getattr(config, "num_hidden_layers", getattr(config, "encoder_layers", 6)),
        6,
    )
    attention_heads = _to_int(
        getattr(config, "num_attention_heads", getattr(config, "encoder_attention_heads", 8)),
        8,
    )
    num_queries = _to_int(getattr(config, "num_queries", 100), 100)
    token_count = max(1, (image_size // 16) ** 2) + num_queries

    stages = [
        _stage(
            "input",
            "Image Input",
            f"3x{image_size}x{image_size}",
            "Input image tensor.",
            "#54baff",
            [1.0, 0.9, 1.0],
            [-5.8, 0.0, 0.0],
        ),
        _stage(
            "encoder",
            "Visual Encoder",
            f"{encoder_layers} layers, hidden={hidden_size}",
            "Feature extraction and contextual encoding.",
            "#79f2c0",
            [1.8, 1.1, 1.2],
            [-1.9, 0.0, 0.0],
        ),
        _stage(
            "aggregator",
            "Attention Mixer",
            f"{attention_heads} heads",
            "Attention-based feature mixing.",
            "#ffd37a",
            [1.7, 1.0, 1.1],
            [1.7, 0.0, 0.0],
        ),
        _stage(
            "head",
            "Detection Head",
            f"{num_queries} queries",
            "Predicts class logits and boxes.",
            "#ff8a6e",
            [1.2, 0.85, 1.0],
            [5.4, 0.0, 0.0],
        ),
    ]

    return {
        "id": model_id,
        "name": "Detection Pipeline",
        "family": "generic",
        "source": source,
        "warning": None,
        "stages": [asdict(stage) for stage in stages],
        "blueprint": {
            "input": {
                "channels": 3,
                "height": image_size,
                "width": image_size,
            },
            "cnn": {
                "stages": [
                    {
                        "id": "encoder_conv",
                        "label": "Feature Conv",
                        "blocks": 2,
                        "out_channels": hidden_size,
                        "kernel_size": 3,
                        "stride": 2,
                    }
                ]
            },
            "transformer": {
                "encoder_layers": encoder_layers,
                "decoder_layers": 0,
                "attention_heads": attention_heads,
                "hidden_size": hidden_size,
                "num_queries": num_queries,
                "token_count": token_count,
            },
        },
    }


def _build_known_fallback(model_id: str) -> dict[str, Any]:
    normalized = model_id.lower()
    if "detr" in normalized:
        return _build_detr_spec(model_id, config=object(), source="fallback")
    if "yolos" in normalized:
        return _build_yolos_spec(model_id, config=object(), source="fallback")
    return _build_generic_spec(model_id, config=object(), source="fallback")


def _resolve_image_size(config: Any, *, default: int) -> int:
    raw = getattr(config, "image_size", default)
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    return max(1, _to_int(raw, default))


def _resolve_resnet_depths(config: Any, backbone_name: str) -> list[int]:
    # Some config variants expose backbone depths directly.
    backbone_config = getattr(config, "backbone_config", None)
    if backbone_config is not None:
        depths = getattr(backbone_config, "depths", None)
        if isinstance(depths, (list, tuple)) and len(depths) >= 4:
            return [max(1, _to_int(item, 1)) for item in list(depths)[:4]]

    depth_from_name = _extract_backbone_depth(backbone_name)
    if depth_from_name in RESNET_STAGE_DEPTHS:
        return RESNET_STAGE_DEPTHS[depth_from_name]
    return RESNET_STAGE_DEPTHS[50]


def _extract_backbone_depth(backbone_name: str) -> int:
    for depth in (18, 34, 50, 101, 152):
        if str(depth) in backbone_name:
            return depth
    return 50


def _to_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    if parsed <= 0:
        return default
    return parsed


def _stage(
    stage_id: str,
    label: str,
    detail: str,
    description: str,
    color: str,
    size: list[float],
    position: list[float],
) -> VLMStageSpec:
    return VLMStageSpec(
        id=stage_id,
        label=label,
        detail=detail,
        description=description,
        color=color,
        size=size,
        position=position,
    )
