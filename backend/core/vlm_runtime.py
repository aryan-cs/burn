from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


DEFAULT_VLM_MODEL_ID = "hustvl/yolos-tiny"


@dataclass
class RuntimeLoadResult:
    backend: str
    model_id: str
    pretrained: bool


class VLMRuntime:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self._backend: str | None = None
        self._model_id = DEFAULT_VLM_MODEL_ID
        self._hf_processor: Any = None
        self._base_model: torch.nn.Module | None = None
        self._label_map: dict[int, str] = {}
        self._artifact_cache: dict[str, torch.nn.Module] = {}
        self._load_warning: str | None = None

    @property
    def backend(self) -> str | None:
        return self._backend

    @property
    def load_warning(self) -> str | None:
        return self._load_warning

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def label_map(self) -> dict[int, str]:
        return self._label_map

    @property
    def hf_processor(self) -> Any:
        return self._hf_processor

    def ensure_model(self, model_id: str = DEFAULT_VLM_MODEL_ID) -> RuntimeLoadResult:
        if self._base_model is not None and model_id == self._model_id:
            return RuntimeLoadResult(
                backend=self._backend or "huggingface",
                model_id=self._model_id,
                pretrained=self._load_warning is None,
            )

        self._model_id = model_id
        self._artifact_cache.clear()
        self._load_warning = None

        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForObjectDetection.from_pretrained(model_id)
            model.to(self.device)
            model.eval()

            label_map = {
                int(label_id): str(label_name)
                for label_id, label_name in dict(model.config.id2label).items()
            }
            self._hf_processor = processor
            self._base_model = model
            self._label_map = label_map
            self._backend = "huggingface"
            return RuntimeLoadResult(backend="huggingface", model_id=model_id, pretrained=True)
        except Exception as exc:
            # Fallback keeps the endpoint functional if HF download/auth/cache fails.
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_320_fpn,
            )

            weights = None
            categories: list[str] = []
            fallback_warning = f"Hugging Face model load failed: {exc}"
            pretrained_weights_loaded = False

            try:
                weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
                categories = list(weights.meta.get("categories", []))
                pretrained_weights_loaded = True
            except Exception as fallback_exc:
                # Last-resort local model when pretrained torchvision weights are unavailable.
                model = fasterrcnn_mobilenet_v3_large_320_fpn(
                    weights=None,
                    weights_backbone=None,
                )
                categories = ["object"]
                fallback_warning = (
                    f"{fallback_warning}. Torchvision pretrained fallback also failed: {fallback_exc}. "
                    "Using randomly initialized detection model."
                )

            model.to(self.device)
            model.eval()
            self._base_model = model
            self._hf_processor = None
            self._label_map = {index: name for index, name in enumerate(categories)}
            self._backend = "torchvision_fallback"
            self._load_warning = fallback_warning
            return RuntimeLoadResult(
                backend="torchvision_fallback",
                model_id=model_id,
                pretrained=pretrained_weights_loaded,
            )

    def create_trainable_model(self) -> torch.nn.Module:
        self.ensure_model(self._model_id)
        assert self._base_model is not None
        model = copy.deepcopy(self._base_model)
        model.to(self.device)
        model.train()
        return model

    def save_state_dict(self, model: torch.nn.Module, artifact_path: Path) -> None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), artifact_path)

    def load_inference_model(self, artifact_path: Path | None) -> torch.nn.Module:
        self.ensure_model(self._model_id)
        assert self._base_model is not None

        if artifact_path is None:
            return self._base_model

        cache_key = str(artifact_path.resolve())
        cached = self._artifact_cache.get(cache_key)
        if cached is not None:
            return cached

        model = copy.deepcopy(self._base_model)
        state_dict = torch.load(artifact_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self._artifact_cache[cache_key] = model
        return model

    def detect(
        self,
        image: Image.Image,
        *,
        score_threshold: float,
        max_detections: int,
        artifact_path: Path | None,
    ) -> dict[str, Any]:
        model = self.load_inference_model(artifact_path)
        width, height = image.size

        if self._backend == "huggingface" and self._hf_processor is not None:
            inputs = self._hf_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            pixel_mask = inputs.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(self.device)

            with torch.no_grad():
                if pixel_mask is None:
                    outputs = model(pixel_values=pixel_values)
                else:
                    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            target_sizes = torch.tensor([[height, width]], device=self.device)
            results = self._hf_processor.post_process_object_detection(
                outputs,
                threshold=score_threshold,
                target_sizes=target_sizes,
            )[0]

            detections: list[dict[str, Any]] = []
            for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
                label_id = int(label.item())
                detections.append(
                    {
                        "label": self._label_map.get(label_id, f"class_{label_id}"),
                        "label_id": label_id,
                        "score": float(score.item()),
                        "box": [float(v) for v in box.tolist()],
                    }
                )
                if len(detections) >= max_detections:
                    break

            return {
                "backend": "huggingface",
                "model_id": self._model_id,
                "image_width": width,
                "image_height": height,
                "detections": detections,
                "warning": self._load_warning,
            }

        from torchvision.transforms.functional import to_tensor

        tensor = to_tensor(image).to(self.device)
        with torch.no_grad():
            output = model([tensor])[0]

        detections = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            score_value = float(score.item())
            if score_value < score_threshold:
                continue
            label_id = int(label.item())
            detections.append(
                {
                    "label": self._label_map.get(label_id, f"class_{label_id}"),
                    "label_id": label_id,
                    "score": score_value,
                    "box": [float(v) for v in box.tolist()],
                }
            )
            if len(detections) >= max_detections:
                break

        return {
            "backend": "torchvision_fallback",
            "model_id": self._model_id,
            "image_width": width,
            "image_height": height,
            "detections": detections,
            "warning": self._load_warning,
        }


vlm_runtime = VLMRuntime()
