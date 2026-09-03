from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    arch = str(arch).lower()
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = int(model.fc.in_features)
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported arch: {arch}")


class EvalImageClassifier:
    def __init__(
        self,
        model: nn.Module,
        classes: list[str],
        mean: list[float] | tuple[float, ...],
        std: list[float] | tuple[float, ...],
        image_size: int,
        device: str,
    ):
        self.model = model.to(device).eval()
        self.classes = list(classes)
        self.device = str(device)
        self.image_size = int(image_size)
        self.preprocess = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    @torch.no_grad()
    def predict_indices(self, batch_01: torch.Tensor) -> torch.Tensor:
        # Keep classifier inference in fp32 to avoid bf16/fp32 mismatch from outer autocast scopes.
        x = self.preprocess(batch_01.to(self.device, dtype=torch.float32))
        logits = self.model(x)
        return logits.argmax(dim=1)


def load_eval_image_classifier(ckpt_path: Path, device: str) -> EvalImageClassifier:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Invalid image classifier checkpoint format.")
    meta = payload.get("meta", {})
    classes = meta.get("classes", [])
    if not classes:
        raise ValueError("Checkpoint missing meta.classes")
    arch = str(meta.get("arch", "resnet50"))
    image_size = int(meta.get("image_size", 224))
    mean = meta.get("mean", [0.485, 0.456, 0.406])
    std = meta.get("std", [0.229, 0.224, 0.225])

    model = build_model(arch=arch, num_classes=len(classes), pretrained=False)
    state = payload.get("model_state_dict", None)
    if state is None:
        raise ValueError("Checkpoint missing model_state_dict")
    model.load_state_dict(state, strict=True)
    return EvalImageClassifier(model=model, classes=list(classes), mean=mean, std=std, image_size=image_size, device=device)


def write_report_json(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
