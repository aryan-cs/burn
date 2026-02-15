from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


DEFAULT_VLM_MODEL_ID = "hustvl/yolos-tiny"


@dataclass
class RuntimeLoadResult:
    backend: str
    model_id: str
    device: str
    pretrained: bool


class VLMRuntime:
    def __init__(self) -> None:
        self.device = self._select_device()
        self._is_cuda = self.device.type == "cuda"
        self._use_autocast = self._is_cuda and os.getenv("COMPUTE_NODE_USE_AUTOCAST", "1").strip() != "0"
        self._use_fp16_model = self._is_cuda and os.getenv("COMPUTE_NODE_USE_FP16_MODEL", "0").strip() == "1"
        self._configure_acceleration_flags()
        self._backend: str | None = None
        self._model_id = DEFAULT_VLM_MODEL_ID
        self._hf_processor: Any = None
        self._base_model: torch.nn.Module | None = None
        self._label_map: dict[int, str] = {}
        self._load_warning: str | None = None

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def device_name(self) -> str:
        return str(self.device)

    @property
    def acceleration_profile(self) -> dict[str, Any]:
        return {
            "device": self.device_name,
            "autocast": self._use_autocast,
            "fp16_model": self._use_fp16_model,
        }

    def _configure_acceleration_flags(self) -> None:
        if not self._is_cuda:
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _autocast_context(self):
        if not self._use_autocast:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _maybe_cast_model_for_speed(self, model: torch.nn.Module) -> torch.nn.Module:
        if self._use_fp16_model:
            model = model.half()
        return model

    def _current_model_dtype(self) -> torch.dtype:
        assert self._base_model is not None
        try:
            first_param = next(self._base_model.parameters())
            return first_param.dtype
        except StopIteration:
            return torch.float32

    def ensure_model(self, model_id: str = DEFAULT_VLM_MODEL_ID) -> RuntimeLoadResult:
        if self._base_model is not None and model_id == self._model_id:
            return RuntimeLoadResult(
                backend=self._backend or "huggingface",
                model_id=self._model_id,
                device=self.device_name,
                pretrained=self._load_warning is None,
            )

        self._model_id = model_id
        self._load_warning = None
        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForObjectDetection.from_pretrained(model_id)
            model.to(self.device)
            model = self._maybe_cast_model_for_speed(model)
            model.eval()
            label_map = {
                int(label_id): str(label_name)
                for label_id, label_name in dict(model.config.id2label).items()
            }
            self._hf_processor = processor
            self._base_model = model
            self._label_map = label_map
            self._backend = "huggingface"
            return RuntimeLoadResult(
                backend="huggingface",
                model_id=model_id,
                device=self.device_name,
                pretrained=True,
            )
        except Exception as exc:
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_320_fpn,
            )

            weights = None
            categories: list[str] = []
            warning = f"Hugging Face model load failed: {exc}"
            pretrained_weights_loaded = False
            try:
                weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
                categories = list(weights.meta.get("categories", []))
                pretrained_weights_loaded = True
            except Exception as fallback_exc:
                model = fasterrcnn_mobilenet_v3_large_320_fpn(
                    weights=None,
                    weights_backbone=None,
                )
                categories = ["object"]
                warning = (
                    f"{warning}. Torchvision pretrained fallback also failed: {fallback_exc}. "
                    "Using randomly initialized detection model."
                )

            model.to(self.device)
            model = self._maybe_cast_model_for_speed(model)
            model.eval()
            self._base_model = model
            self._hf_processor = None
            self._label_map = {index: name for index, name in enumerate(categories)}
            self._backend = "torchvision_fallback"
            self._load_warning = warning
            return RuntimeLoadResult(
                backend="torchvision_fallback",
                model_id=model_id,
                device=self.device_name,
                pretrained=pretrained_weights_loaded,
            )

    def detect(
        self,
        image: Image.Image,
        *,
        score_threshold: float,
        max_detections: int,
        model_id: str = DEFAULT_VLM_MODEL_ID,
    ) -> dict[str, Any]:
        self.ensure_model(model_id)
        assert self._base_model is not None
        model_dtype = self._current_model_dtype()
        width, height = image.size

        if self._backend == "huggingface" and self._hf_processor is not None:
            inputs = self._hf_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(
                self.device,
                dtype=model_dtype if model_dtype == torch.float16 else None,
                non_blocking=True,
            )
            pixel_mask = inputs.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(self.device, non_blocking=True)
            with torch.inference_mode():
                with self._autocast_context():
                    if pixel_mask is None:
                        outputs = self._base_model(pixel_values=pixel_values)
                    else:
                        outputs = self._base_model(pixel_values=pixel_values, pixel_mask=pixel_mask)
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
                "runtime_backend": "huggingface",
                "runtime_model_id": self._model_id,
                "runtime_device": self.device_name,
                "image_width": width,
                "image_height": height,
                "detections": detections,
                "warning": self._load_warning,
            }

        from torchvision.transforms.functional import to_tensor

        tensor = to_tensor(image).to(
            self.device,
            dtype=model_dtype if model_dtype == torch.float16 else None,
            non_blocking=True,
        )
        with torch.inference_mode():
            with self._autocast_context():
                output = self._base_model([tensor])[0]
        detections: list[dict[str, Any]] = []
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
            "runtime_backend": "torchvision_fallback",
            "runtime_model_id": self._model_id,
            "runtime_device": self.device_name,
            "image_width": width,
            "image_height": height,
            "detections": detections,
            "warning": self._load_warning,
        }

    def warmup(self, model_id: str = DEFAULT_VLM_MODEL_ID) -> RuntimeLoadResult:
        load = self.ensure_model(model_id)
        image = Image.new("RGB", (320, 320), color=(0, 0, 0))
        self.detect(
            image,
            score_threshold=0.99,
            max_detections=1,
            model_id=model_id,
        )
        return load


vlm_runtime = VLMRuntime()
