from __future__ import annotations

import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:
    from model import LatentAdaCUT


class _RobustStyleProbe(nn.Module):
    """Inference-only copy of src/utils/classify.py::RobustStyleProbe."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        def res_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)),
                nn.InstanceNorm2d(cout, affine=True),
                nn.Mish(inplace=True),
                nn.Dropout2d(0.1),
            )

        self.layer1 = res_block(in_channels, 32)
        self.layer2 = res_block(32, 64, stride=2)
        self.layer3 = res_block(64, 128)
        self.layer4 = res_block(128, 128, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.utils.spectral_norm(nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.gap(x).flatten(1)
        logits = self.classifier(feat)
        return feat, logits


def _tv_per_sample(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((x.shape[0],))
    tv_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    tv_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    return tv_x + tv_y


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})

        self.w_style_cls = float(loss_cfg.get("w_style_classifier", loss_cfg.get("w_style_cls", 1.0)))
        self.style_cls_on_xid_only = bool(loss_cfg.get("style_classifier_xid_only", True))
        self.style_cls_label_smoothing = float(loss_cfg.get("style_classifier_label_smoothing", 0.0))
        self.style_classifier_path = str(loss_cfg.get("style_classifier_path", "")).strip()
        if not self.style_classifier_path:
            self.style_classifier_path = str(Path(__file__).resolve().parent / "utils" / "robust_style_probe_final.pth")
        self._style_classifier: _RobustStyleProbe | None = None
        self._style_classifier_device: str | None = None
        self._style_classifier_missing = False

        self.w_identity = float(loss_cfg.get("w_identity", 10.0))
        self.w_delta_tv = float(loss_cfg.get("w_delta_tv", 0.0))
        self.w_delta_l2 = float(loss_cfg.get("w_delta_l2", 0.0))
        self.w_output_tv = float(loss_cfg.get("w_output_tv", 0.0))

        self.w_semigroup = float(loss_cfg.get("w_semigroup", 0.0))
        self.semigroup_every_n_steps = max(1, int(loss_cfg.get("semigroup_every_n_steps", 1)))
        self.semigroup_teacher_no_grad = bool(loss_cfg.get("semigroup_teacher_no_grad", True))
        self.semigroup_target_detach = bool(loss_cfg.get("semigroup_target_detach", True))
        self.semigroup_pool_size = max(0, int(loss_cfg.get("semigroup_pool_size", 8)))
        self.semigroup_num_steps = max(1, int(loss_cfg.get("semigroup_num_steps", 1)))
        self._compute_calls = 0

        self.train_num_steps_min = max(1, int(loss_cfg.get("train_num_steps_min", 1)))
        self.train_num_steps_max = max(1, int(loss_cfg.get("train_num_steps_max", self.train_num_steps_min)))
        self.train_step_size_min = float(loss_cfg.get("train_step_size_min", 1.0))
        self.train_step_size_max = float(loss_cfg.get("train_step_size_max", self.train_step_size_min))
        self.train_style_strength_min = float(loss_cfg.get("train_style_strength_min", 1.0))
        self.train_style_strength_max = float(loss_cfg.get("train_style_strength_max", self.train_style_strength_min))
        self.nsight_nvtx = bool(config.get("training", {}).get("nsight_nvtx", False))

    def _ensure_style_classifier(self, model: LatentAdaCUT, device: torch.device) -> None:
        if self.w_style_cls <= 0.0:
            return
        device_str = str(device)
        if self._style_classifier is not None and self._style_classifier_device == device_str:
            return

        ckpt_path = Path(self.style_classifier_path)
        if not ckpt_path.is_absolute():
            ckpt_path = (Path(__file__).resolve().parent / ckpt_path).resolve()
        if not ckpt_path.exists():
            self._style_classifier_missing = True
            raise FileNotFoundError(f"Style classifier checkpoint not found: {ckpt_path}")

        num_classes = int(getattr(model, "num_styles", 0))
        in_channels = int(getattr(model, "latent_channels", 4))
        clf = _RobustStyleProbe(in_channels=in_channels, num_classes=num_classes).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        clf.load_state_dict(state_dict, strict=True)
        for p in clf.parameters():
            p.requires_grad_(False)
        clf.eval()
        self._style_classifier = clf
        self._style_classifier_device = device_str

    @staticmethod
    def _sample_range(low: float, high: float) -> float:
        return float(random.uniform(low, high)) if high > low else float(low)

    @staticmethod
    def _sample_int_range(low: int, high: int) -> int:
        return int(random.randint(low, high)) if high > low else int(low)

    @staticmethod
    def _apply_model(model, x, style_id, step_size, style_strength, num_steps):
        steps = max(1, int(num_steps))
        if steps > 1:
            return model.integrate(x, style_id=style_id, num_steps=steps, step_size=step_size, style_strength=style_strength)
        return model(x, style_id=style_id, step_size=step_size, style_strength=style_strength)

    @contextmanager
    def _nvtx_range(self, name: str, enabled: bool):
        if not enabled:
            yield
            return
        try:
            torch.cuda.nvtx.range_push(name)
            yield
        finally:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        debug_timing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        nvtx_enabled = bool(self.nsight_nvtx and content.is_cuda)
        self._compute_calls += 1

        train_num_steps = self._sample_int_range(self.train_num_steps_min, self.train_num_steps_max)
        train_step_size = self._sample_range(self.train_step_size_min, self.train_step_size_max)
        train_style_strength = self._sample_range(self.train_style_strength_min, self.train_style_strength_max)

        if source_style_id is None:
            id_mask = torch.zeros_like(target_style_id, dtype=torch.bool)
        else:
            id_mask = source_style_id.long() == target_style_id.long()
        xid_mask = ~id_mask
        id_ratio = id_mask.float().mean()

        def _masked_mean(per_sample: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
            if per_sample.ndim != 1:
                per_sample = per_sample.reshape(per_sample.shape[0], -1).mean(dim=1)
            denom = mask_f.sum().clamp_min(1.0)
            return (per_sample * mask_f).sum() / denom

        with self._nvtx_range("loss/pred", nvtx_enabled):
            pred_student = self._apply_model(
                model,
                content,
                style_id=target_style_id,
                step_size=train_step_size,
                style_strength=train_style_strength,
                num_steps=train_num_steps,
            )

        # Always use raw latent in FP32 for loss computation.
        pred_f32 = pred_student.float()
        content_f32 = content.float()

        loss_style_cls = torch.tensor(0.0, device=content.device)
        style_cls_acc = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/style_classifier", nvtx_enabled):
            if self.w_style_cls > 0.0:
                self._ensure_style_classifier(model=model, device=content.device)
                assert self._style_classifier is not None
                if self.style_cls_on_xid_only and xid_mask.any():
                    valid_idx = torch.nonzero(xid_mask).squeeze(1)
                else:
                    valid_idx = torch.arange(pred_f32.shape[0], device=pred_f32.device, dtype=torch.long)
                if valid_idx.numel() > 0:
                    cls_in = pred_f32.index_select(0, valid_idx)
                    cls_target = target_style_id.index_select(0, valid_idx).long()
                    _, cls_logits = self._style_classifier(cls_in)
                    loss_style_cls = F.cross_entropy(
                        cls_logits,
                        cls_target,
                        label_smoothing=self.style_cls_label_smoothing,
                    )
                    pred_ids = cls_logits.argmax(dim=1)
                    style_cls_acc = (pred_ids == cls_target).float().mean()

        loss_identity = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/identity", nvtx_enabled):
            if self.w_identity > 0.0 and bool(id_mask.any().item()):
                id_per_sample = (pred_f32 - content_f32).abs().mean(dim=(1, 2, 3))
                loss_identity = _masked_mean(id_per_sample, id_mask.float())

        loss_delta_tv = torch.tensor(0.0, device=content.device)
        loss_delta_l2 = torch.tensor(0.0, device=content.device)
        loss_output_tv = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/reg", nvtx_enabled):
            delta = pred_f32 - content_f32
            if self.w_delta_tv > 0.0:
                loss_delta_tv = _tv_per_sample(delta).mean()
            if self.w_delta_l2 > 0.0:
                loss_delta_l2 = delta.pow(2).mean()
            if self.w_output_tv > 0.0:
                loss_output_tv = _tv_per_sample(pred_f32).mean()

        loss_semigroup = torch.tensor(0.0, device=content.device)
        semigroup_applied = False
        with self._nvtx_range("loss/semigroup", nvtx_enabled):
            if self.w_semigroup > 0.0 and train_style_strength > 0.1 and (self._compute_calls % self.semigroup_every_n_steps == 0):
                semigroup_applied = True
                split_ratio = 0.5
                str_a = train_style_strength * split_ratio
                str_b = train_style_strength - str_a
                sem_steps = self.semigroup_num_steps
                with torch.amp.autocast("cuda", enabled=content.is_cuda, dtype=torch.float16):
                    z_ab = pred_student.detach() if self.semigroup_target_detach else pred_student
                    if self.semigroup_teacher_no_grad:
                        with torch.no_grad():
                            z_a = self._apply_model(
                                model,
                                content,
                                style_id=target_style_id,
                                step_size=train_step_size,
                                style_strength=str_a,
                                num_steps=sem_steps,
                            )
                    else:
                        z_a = self._apply_model(
                            model,
                            content,
                            style_id=target_style_id,
                            step_size=train_step_size,
                            style_strength=str_a,
                            num_steps=sem_steps,
                        )
                    z_a_b = self._apply_model(
                        model,
                        z_a,
                        style_id=target_style_id,
                        step_size=train_step_size,
                        style_strength=str_b,
                        num_steps=sem_steps,
                    )
                    if self.semigroup_pool_size > 0:
                        z_ab_eval = F.adaptive_avg_pool2d(
                            z_ab.float(),
                            output_size=(self.semigroup_pool_size, self.semigroup_pool_size),
                        )
                        z_a_b_eval = F.adaptive_avg_pool2d(
                            z_a_b.float(),
                            output_size=(self.semigroup_pool_size, self.semigroup_pool_size),
                        )
                        semigroup_diff = (z_a_b_eval - z_ab_eval).abs()
                    else:
                        semigroup_diff = (z_a_b - z_ab).abs()
                loss_semigroup = semigroup_diff.float().mean()

        total = (
            self.w_style_cls * loss_style_cls
            + self.w_identity * loss_identity
            + self.w_delta_tv * loss_delta_tv
            + self.w_delta_l2 * loss_delta_l2
            + self.w_output_tv * loss_output_tv
            + self.w_semigroup * loss_semigroup
        )

        return {
            "loss": total,
            # Keep legacy keys for trainer logging compatibility.
            "moment": torch.tensor(0.0, device=content.device),
            "swd": loss_style_cls.detach(),
            "style_cls": loss_style_cls.detach(),
            "style_cls_acc": style_cls_acc.detach(),
            "identity": loss_identity.detach(),
            "identity_ratio": id_ratio.detach(),
            "delta_tv": loss_delta_tv.detach(),
            "delta_l2": loss_delta_l2.detach(),
            "output_tv": loss_output_tv.detach(),
            "semigroup": loss_semigroup.detach(),
            "semigroup_applied": torch.tensor(1.0 if semigroup_applied else 0.0, device=content.device),
            "train_num_steps": torch.tensor(float(train_num_steps), device=content.device),
            "train_step_size": torch.tensor(float(train_step_size), device=content.device),
            "train_style_strength": torch.tensor(float(train_style_strength), device=content.device),
        }
