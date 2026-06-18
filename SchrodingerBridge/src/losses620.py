from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from config_schema import ExperimentConfig


def _lowpass(x: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    return F.avg_pool2d(x.float(), kernel_size=k, stride=1, padding=k // 2).to(dtype=x.dtype)


def _sliced_wasserstein(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    dirs: torch.Tensor,
) -> torch.Tensor:
    bsz = a.shape[0]
    a_flat = a.float().reshape(bsz, -1)
    b_flat = b.float().reshape(bsz, -1)
    proj_a = torch.sort(a_flat @ dirs.t(), dim=0).values
    proj_b = torch.sort(b_flat @ dirs.t(), dim=0).values
    return (proj_a - proj_b).abs().mean()


class SpatialBridgeObjective620:
    """620 objective: vertical FM + single-step endpoint SWD/edge losses."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.bridge_cfg = config.bridge
        self.fm_weight = float(getattr(self.bridge_cfg, "w_flow", 1.0))
        self.single_step_swd_weight = float(getattr(self.bridge_cfg, "single_step_swd_weight", 8.0))
        self.single_step_edge_weight = float(getattr(self.bridge_cfg, "single_step_edge_weight", 0.1))
        self.lowpass_kernel = int(getattr(self.bridge_cfg, "training_target_projection_kernel", 5))
        self.num_projections = int(getattr(self.bridge_cfg, "semantic_swd_num_projections", 64))
        self.t_min = float(getattr(self.bridge_cfg, "t_min", 0.0))
        self.t_max = float(getattr(self.bridge_cfg, "t_max", 1.0))
        self.last_debug: dict[str, torch.Tensor] = {}
        self._projection_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

    def _projection_dirs(self, like: torch.Tensor) -> torch.Tensor:
        dim = int(like.reshape(like.shape[0], -1).shape[1])
        dtype = torch.float32
        key = (dim, str(like.device), dtype)
        dirs = self._projection_cache.get(key)
        if dirs is None or dirs.device != like.device:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(620 + dim + int(self.num_projections))
            dirs = torch.randn((max(1, self.num_projections), dim), generator=gen, device="cpu", dtype=dtype)
            dirs = F.normalize(dirs, p=2, dim=1, eps=1e-8).to(device=like.device)
            self._projection_cache[key] = dirs
        return dirs

    def _sample_t(self, content: torch.Tensor) -> torch.Tensor:
        lo = max(0.0, min(1.0, self.t_min))
        hi = max(lo + 1e-4, min(1.0, self.t_max))
        return torch.empty(content.shape[0], device=content.device, dtype=content.dtype).uniform_(lo, hi)

    def _vertical_state(self, content: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_low = _lowpass(content, self.lowpass_kernel)
        c_high = content - c_low
        t_high = target - _lowpass(target, self.lowpass_kernel)
        t4 = t.view(-1, 1, 1, 1).to(dtype=content.dtype)
        x_t = c_low + (1.0 - t4) * c_high + t4 * t_high
        target_velocity = t_high - c_high
        return x_t, target_velocity

    def compute(
        self,
        model,
        *,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        aux_target_style: torch.Tensor | None = None,
        aux_target_valid: torch.Tensor | None = None,
        conditioning: dict | None = None,
    ) -> Dict[str, torch.Tensor]:
        del source_style_id, aux_target_style, aux_target_valid
        conditioning = conditioning or {}
        style_patches = conditioning.get("target_style_dino_patches")
        style_cls = conditioning.get("target_style_dino_cls")
        if not torch.is_tensor(style_patches):
            style_patches = None
        if not torch.is_tensor(style_cls):
            style_cls = None
        t = self._sample_t(content)
        x_t, target_velocity = self._vertical_state(content, target_style, t)
        pred_velocity = model(
            x_t,
            t=t,
            style_id=target_style_id,
            style_dino_patches=style_patches,
            style_dino_cls=style_cls,
        )
        z_hat1 = x_t + (1.0 - t).view(-1, 1, 1, 1).to(dtype=x_t.dtype) * pred_velocity
        fm = F.mse_loss(pred_velocity.float(), target_velocity.float())
        swd_ss = _sliced_wasserstein(z_hat1, target_style, dirs=self._projection_dirs(z_hat1))
        edge_ss = F.l1_loss(
            (z_hat1 - _lowpass(z_hat1, self.lowpass_kernel)).float(),
            (target_style - _lowpass(target_style, self.lowpass_kernel)).float(),
        )
        loss = self.fm_weight * fm + self.single_step_swd_weight * swd_ss + self.single_step_edge_weight * edge_ss
        c_low = _lowpass(content, self.lowpass_kernel)
        low_leak = _lowpass(pred_velocity, self.lowpass_kernel).float().abs().mean()
        debug = getattr(model, "last_debug", {}) if hasattr(model, "last_debug") else {}
        zero = content.new_tensor(0.0)
        metrics = {
            "loss": loss,
            "flow": fm.detach(),
            "loss_fm": fm.detach(),
            "loss_swd_ss": swd_ss.detach(),
            "loss_edge_ss": edge_ss.detach(),
            "single_step_swd": (swd_ss * self.single_step_swd_weight).detach(),
            "single_step_edge": (edge_ss * self.single_step_edge_weight).detach(),
            "terminal_swd": zero,
            "ot_cost": zero,
            "ot_plan_entropy": zero,
            "ot_target_gini": zero,
            "t_mean": t.detach().float().mean(),
            "velocity_abs": pred_velocity.detach().float().abs().mean(),
            "target_velocity_abs": target_velocity.detach().float().abs().mean(),
            "endpoint_abs": z_hat1.detach().float().abs().mean(),
            "base_structural_drift": (_lowpass(z_hat1, self.lowpass_kernel) - c_low).detach().float().abs().mean(),
            "low_freq_leak": low_leak.detach(),
            "fiber_energy_ratio": ((target_velocity.float().square().mean()) / (target_style.float().square().mean().clamp_min(1e-8))).detach(),
            "bridge_sigma": content.new_tensor(float(getattr(model, "bridge_sigma", 0.0))),
            "style_dino_active": content.new_tensor(1.0 if style_patches is not None else 0.0),
            "style_gate_value": debug.get("style_gate_value", zero).detach() if torch.is_tensor(debug.get("style_gate_value", None)) else zero,
            "cross_attn_entropy": debug.get("cross_attn_entropy", zero).detach() if torch.is_tensor(debug.get("cross_attn_entropy", None)) else zero,
            "cross_attn_delta_abs": debug.get("cross_attn_delta_abs", zero).detach() if torch.is_tensor(debug.get("cross_attn_delta_abs", None)) else zero,
        }
        self.last_debug = {
            "x_t": x_t.detach(),
            "target_velocity": target_velocity.detach(),
            "pred_velocity": pred_velocity.detach(),
            "z_hat1": z_hat1.detach(),
        }
        return metrics

    def compute_debug(self, model, **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        metrics = self.compute(model, **kwargs)
        return {"metrics": metrics, "components": {}, "state": dict(self.last_debug)}
