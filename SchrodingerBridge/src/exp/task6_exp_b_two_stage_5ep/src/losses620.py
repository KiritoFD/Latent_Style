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
    noise_sigma: float = 0.0,
) -> torch.Tensor:
    bsz, c, h, w = a.shape
    # spatial feature format: (B, H*W, C)
    a_spatial = a.float().reshape(bsz, c, -1).transpose(1, 2)
    b_spatial = b.float().reshape(bsz, c, -1).transpose(1, 2)
    
    # Project: (B, H*W, C) @ (C, num_dirs) -> (B, H*W, num_dirs)
    proj_a = a_spatial @ dirs.t()
    proj_b = b_spatial @ dirs.t()
    
    if noise_sigma > 0.0:
        proj_a = proj_a + noise_sigma * torch.randn_like(proj_a)
        proj_b = proj_b + noise_sigma * torch.randn_like(proj_b)
        
    # Sort along the spatial dimension (dim=1)
    proj_a_sorted = torch.sort(proj_a, dim=1).values
    proj_b_sorted = torch.sort(proj_b, dim=1).values
    
    return (proj_a_sorted - proj_b_sorted).abs().mean()


class SpatialBridgeObjective620:
    """620 objective: vertical FM + single-step endpoint SWD/edge losses."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.bridge_cfg = config.bridge
        self.fm_weight = float(getattr(self.bridge_cfg, "w_flow", 1.0))
        self.single_step_swd_weight = float(getattr(self.bridge_cfg, "single_step_swd_weight", 8.0))
        self.single_step_edge_weight = float(getattr(self.bridge_cfg, "single_step_edge_weight", 0.1))
        self.endpoint_lowfreq_weight = float(getattr(self.bridge_cfg, "w_content_lowpass_anchor", 0.0))
        self.lowpass_kernel = int(getattr(self.bridge_cfg, "training_target_projection_kernel", 5))
        self.low_anchor = float(getattr(self.bridge_cfg, "training_target_projection_low_anchor", 1.0))
        self.training_target_projection_mode = str(
            getattr(self.bridge_cfg, "training_target_projection_mode", "source_low_target_high")
        ).strip().lower()
        if self.training_target_projection_mode not in {
            "legacy",
            "source_low_target_high",
            "wavelet_source_low_target_high",
            "pure_vertical_flow",
            "pure_vertical_flow_wavelet",
        }:
            self.training_target_projection_mode = "source_low_target_high"
        self.low_mode = str(getattr(self.bridge_cfg, "training_target_projection_low_mode", "all")).strip().lower()
        if self.low_mode not in {"all", "channel_mean", "target_linear"}:
            self.low_mode = "all"
        self.num_projections = int(getattr(self.bridge_cfg, "semantic_swd_num_projections", 64))
        self.t_min = float(getattr(self.bridge_cfg, "t_min", 0.0))
        self.t_max = float(getattr(self.bridge_cfg, "t_max", 1.0))
        self.t_sampling_power = max(1e-3, float(getattr(self.bridge_cfg, "t_sampling_power", 1.0)))
        self.source_endpoint_aux_weight = float(getattr(self.bridge_cfg, "source_endpoint_aux_weight", 0.0))
        self.endpoint_energy_band_weight = float(getattr(self.bridge_cfg, "endpoint_energy_band_weight", 0.0))
        self.swd_scale_mode = str(getattr(self.bridge_cfg, "swd_scale_mode", "global")).strip().lower()
        self.w_attn_entropy_reg = float(getattr(self.bridge_cfg, "w_attn_entropy_reg", 0.0))
        self.w_style_strength_reg = float(getattr(self.bridge_cfg, "w_style_strength_reg", 0.0))
        self.swd_noise_sigma = float(getattr(self.bridge_cfg, "swd_noise_sigma", 0.0))
        self.training_objective_mode = str(getattr(self.bridge_cfg, "training_objective_mode", "velocity")).strip().lower()
        if self.training_objective_mode not in {"velocity", "endpoint"}:
            self.training_objective_mode = "velocity"
        self.w_endpoint_content = float(getattr(self.bridge_cfg, "w_endpoint_content", 1.0))
        self.w_endpoint_style = float(getattr(self.bridge_cfg, "w_endpoint_style", 8.0))
        self.w_endpoint_velocity_reg = float(getattr(self.bridge_cfg, "w_endpoint_velocity_reg", 0.0))
        self.two_stage_enabled = bool(getattr(self.bridge_cfg, "two_stage_enabled", False))
        self.two_stage_s1_epochs = int(getattr(self.bridge_cfg, "two_stage_s1_epochs", 2))
        self.two_stage_s1_w_endpoint_content = float(getattr(self.bridge_cfg, "two_stage_s1_w_endpoint_content", 0.3))
        self.two_stage_s1_w_endpoint_style = float(getattr(self.bridge_cfg, "two_stage_s1_w_endpoint_style", 16.0))
        self.two_stage_s1_w_style_strength_reg = float(getattr(self.bridge_cfg, "two_stage_s1_w_style_strength_reg", 0.5))
        self.two_stage_s2_w_endpoint_content = float(getattr(self.bridge_cfg, "two_stage_s2_w_endpoint_content", 1.0))
        self.two_stage_s2_w_endpoint_style = float(getattr(self.bridge_cfg, "two_stage_s2_w_endpoint_style", 8.0))
        self.two_stage_s2_w_style_strength_reg = float(getattr(self.bridge_cfg, "two_stage_s2_w_style_strength_reg", 0.5))
        self._base_w_endpoint_content = self.w_endpoint_content
        self._base_w_endpoint_style = self.w_endpoint_style
        self._base_w_style_strength_reg = self.w_style_strength_reg
        self.last_debug: dict[str, torch.Tensor] = {}
        self._projection_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

    def update_weights_for_epoch(self, epoch: int) -> dict[str, float]:
        if not self.two_stage_enabled:
            return {
                "w_endpoint_content": self.w_endpoint_content,
                "w_endpoint_style": self.w_endpoint_style,
                "w_style_strength_reg": self.w_style_strength_reg,
                "stage": 0,
            }
        epoch_idx = epoch - 1
        if epoch_idx < self.two_stage_s1_epochs:
            self.w_endpoint_content = self.two_stage_s1_w_endpoint_content
            self.w_endpoint_style = self.two_stage_s1_w_endpoint_style
            self.w_style_strength_reg = self.two_stage_s1_w_style_strength_reg
            stage = 1
        else:
            self.w_endpoint_content = self.two_stage_s2_w_endpoint_content
            self.w_endpoint_style = self.two_stage_s2_w_endpoint_style
            self.w_style_strength_reg = self.two_stage_s2_w_style_strength_reg
            stage = 2
        return {
            "w_endpoint_content": self.w_endpoint_content,
            "w_endpoint_style": self.w_endpoint_style,
            "w_style_strength_reg": self.w_style_strength_reg,
            "stage": stage,
        }

    def _projection_dirs(self, like: torch.Tensor) -> torch.Tensor:
        dim = like.shape[1] if like.dim() == 4 else like.shape[-1]
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
        u = torch.empty(content.shape[0], device=content.device, dtype=content.dtype).uniform_(0.0, 1.0)
        u = u.pow(self.t_sampling_power)
        return lo + (hi - lo) * u

    def _wavelet_lowpass(self, x: torch.Tensor) -> torch.Tensor:
        down = F.avg_pool2d(x.float(), kernel_size=2, stride=2, ceil_mode=False)
        return F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False).to(dtype=x.dtype)

    def _split_base_fiber(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training_target_projection_mode in {"wavelet_source_low_target_high", "pure_vertical_flow_wavelet"}:
            low = self._wavelet_lowpass(x)
        else:
            low = _lowpass(x, self.lowpass_kernel)
        high = x - low
        return low, high

    def _project_training_target(
        self,
        content: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = content.new_tensor(0.0)
        metrics = {
            "training_target_projection_active": content.new_tensor(0.0),
            "training_target_projection_mode_source_low_target_high": content.new_tensor(0.0),
            "training_target_projection_mode_wavelet_source_low_target_high": content.new_tensor(0.0),
            "training_target_projection_mode_pure_vertical_flow": content.new_tensor(0.0),
            "training_target_projection_mode_pure_vertical_flow_wavelet": content.new_tensor(0.0),
            "training_target_projection_low_anchor": content.new_tensor(max(0.0, min(1.0, self.low_anchor))),
            "training_target_projection_low_drift": zero,
            "training_target_projection_target_delta": zero,
            "training_target_projection_high_energy_ratio": zero,
        }
        if self.training_target_projection_mode == "legacy":
            return target, metrics

        c_low, c_high = self._split_base_fiber(content)
        t_low, t_high = self._split_base_fiber(target)
        if self.low_mode == "channel_mean":
            anchor_low = c_low.mean(dim=1, keepdim=True).expand_as(c_low)
        else:
            anchor_low = c_low
        low_anchor = max(0.0, min(1.0, self.low_anchor))

        if self.training_target_projection_mode in {"pure_vertical_flow", "pure_vertical_flow_wavelet"}:
            projected = anchor_low + t_high
            key = "training_target_projection_mode_pure_vertical_flow_wavelet"
            if self.training_target_projection_mode == "pure_vertical_flow":
                key = "training_target_projection_mode_pure_vertical_flow"
            metrics[key] = content.new_tensor(1.0)
        else:
            projected_low = t_low.lerp(anchor_low, low_anchor)
            projected = projected_low + t_high
            key = "training_target_projection_mode_wavelet_source_low_target_high"
            if self.training_target_projection_mode == "source_low_target_high":
                key = "training_target_projection_mode_source_low_target_high"
            metrics[key] = content.new_tensor(1.0)

        proj_low, proj_high = self._split_base_fiber(projected)
        metrics.update(
            {
                "training_target_projection_active": content.new_tensor(1.0),
                "training_target_projection_low_drift": (proj_low - c_low).detach().float().abs().mean(),
                "training_target_projection_target_delta": (projected - target).detach().float().abs().mean(),
                "training_target_projection_high_energy_ratio": (
                    proj_high.detach().float().std(dim=(1, 2, 3), unbiased=False).mean()
                    / c_high.detach().float().std(dim=(1, 2, 3), unbiased=False).mean().clamp_min(1e-8)
                ),
            }
        )
        return projected.to(dtype=target.dtype), metrics

    def _vertical_state(self, content: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected_target, _ = self._project_training_target(content, target)
        c_low, c_high = self._split_base_fiber(content)
        p_low, p_high = self._split_base_fiber(projected_target)
        low_anchor = max(0.0, min(1.0, self.low_anchor))
        t4 = t.view(-1, 1, 1, 1).to(dtype=content.dtype)
        if self.low_mode == "target_linear":
            x_low = (1.0 - t4) * c_low + t4 * p_low
            target_low_velocity = p_low - c_low
        else:
            x_low = low_anchor * c_low + (1.0 - low_anchor) * p_low
            target_low_velocity = torch.zeros_like(c_low)
            if self.low_mode == "channel_mean":
                c_mean = c_low.mean(dim=(2, 3), keepdim=True)
                p_mean = p_low.mean(dim=(2, 3), keepdim=True)
                x_low = x_low + t4 * (p_mean - c_mean)
                target_low_velocity = p_mean - c_mean
        x_t = x_low + (1.0 - t4) * c_high + t4 * p_high
        target_velocity = (p_high - c_high) + target_low_velocity
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
        content_patches = conditioning.get("content_dino_patches")
        style_text_tokens = conditioning.get("target_style_text_tokens")
        if not torch.is_tensor(style_patches):
            style_patches = None
        if not torch.is_tensor(style_cls):
            style_cls = None
        if not torch.is_tensor(content_patches):
            content_patches = None
        if not torch.is_tensor(style_text_tokens):
            style_text_tokens = None

        projected_target, projection_metrics = self._project_training_target(content, target_style)
        t = self._sample_t(content)
        x_t, target_velocity = self._vertical_state(content, target_style, t)
        pred_velocity = model(
            x_t,
            t=t,
            style_id=target_style_id,
            style_dino_patches=style_patches,
            style_dino_cls=style_cls,
            content_dino_patches=content_patches,
            style_latent=target_style,
            style_text_tokens=style_text_tokens,
        )
        z_hat1 = x_t + (1.0 - t).view(-1, 1, 1, 1).to(dtype=x_t.dtype) * pred_velocity
        fm = F.mse_loss(pred_velocity.float(), target_velocity.float())

        source_endpoint_aux = content.new_tensor(0.0)
        if self.source_endpoint_aux_weight > 0.0:
            source_endpoint = model.predict_endpoint(
                content,
                t=torch.zeros((content.shape[0],), device=content.device, dtype=content.dtype),
                style_id=target_style_id,
                style_dino_patches=style_patches,
                style_dino_cls=style_cls,
                style_text_tokens=style_text_tokens,
            )
            source_endpoint_aux = (
                F.l1_loss(_lowpass(source_endpoint, self.lowpass_kernel).float(), _lowpass(projected_target, self.lowpass_kernel).float())
                + _sliced_wasserstein(source_endpoint, projected_target, dirs=self._projection_dirs(source_endpoint), noise_sigma=self.swd_noise_sigma)
                + F.l1_loss(
                    (source_endpoint - _lowpass(source_endpoint, self.lowpass_kernel)).float(),
                    (projected_target - _lowpass(projected_target, self.lowpass_kernel)).float(),
                )
            ) / 3.0

        endpoint_energy_band = content.new_tensor(0.0)
        if self.endpoint_energy_band_weight > 0.0:
            z_abs = z_hat1.float().abs().mean(dim=(1, 2, 3))
            src_abs = content.float().abs().mean(dim=(1, 2, 3))
            tgt_abs = target_style.float().abs().mean(dim=(1, 2, 3))
            lower = torch.minimum(src_abs, tgt_abs)
            upper = torch.maximum(src_abs, tgt_abs)
            endpoint_energy_band = (
                F.relu(z_abs - upper).mean()
                + F.relu(lower - z_abs).mean()
            )

        delta_target = projected_target - content
        delta_pred = z_hat1 - content
        alpha_num = (delta_pred.float() * delta_target.float()).sum(dim=[1, 2, 3])
        alpha_den = (delta_target.float() * delta_target.float()).sum(dim=[1, 2, 3]).clamp_min(1e-6)
        style_strength_alpha = (alpha_num / alpha_den).mean()
        style_strength_loss = -self.w_style_strength_reg * style_strength_alpha

        # SWD scale mode handling
        if self.swd_scale_mode == "2-scale":
            swd_64 = _sliced_wasserstein(z_hat1, projected_target, dirs=self._projection_dirs(z_hat1), noise_sigma=self.swd_noise_sigma)
            z_hat1_32 = F.avg_pool2d(z_hat1, kernel_size=2, stride=2)
            target_style_32 = F.avg_pool2d(projected_target, kernel_size=2, stride=2)
            swd_32 = _sliced_wasserstein(z_hat1_32, target_style_32, dirs=self._projection_dirs(z_hat1_32), noise_sigma=self.swd_noise_sigma)
            swd_ss = 0.5 * swd_64 + 0.5 * swd_32
        elif self.swd_scale_mode == "3-scale":
            swd_64 = _sliced_wasserstein(z_hat1, projected_target, dirs=self._projection_dirs(z_hat1), noise_sigma=self.swd_noise_sigma)
            z_hat1_32 = F.avg_pool2d(z_hat1, kernel_size=2, stride=2)
            target_style_32 = F.avg_pool2d(projected_target, kernel_size=2, stride=2)
            swd_32 = _sliced_wasserstein(z_hat1_32, target_style_32, dirs=self._projection_dirs(z_hat1_32), noise_sigma=self.swd_noise_sigma)
            z_hat1_16 = F.avg_pool2d(z_hat1, kernel_size=4, stride=4)
            target_style_16 = F.avg_pool2d(projected_target, kernel_size=4, stride=4)
            swd_16 = _sliced_wasserstein(z_hat1_16, target_style_16, dirs=self._projection_dirs(z_hat1_16), noise_sigma=self.swd_noise_sigma)
            swd_ss = 0.4 * swd_64 + 0.4 * swd_32 + 0.2 * swd_16
        elif self.swd_scale_mode == "attention-weighted" and getattr(model, "last_pixel_entropy", None) is not None:
            weight = model.last_pixel_entropy.to(device=z_hat1.device, dtype=z_hat1.dtype)
            weight = weight / weight.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            swd_ss = _sliced_wasserstein(z_hat1 * weight, projected_target * weight, dirs=self._projection_dirs(z_hat1), noise_sigma=self.swd_noise_sigma)
        else:
            swd_ss = _sliced_wasserstein(z_hat1, projected_target, dirs=self._projection_dirs(z_hat1), noise_sigma=self.swd_noise_sigma)

        edge_ss = F.l1_loss(
            (z_hat1 - _lowpass(z_hat1, self.lowpass_kernel)).float(),
            (projected_target - _lowpass(projected_target, self.lowpass_kernel)).float(),
        )
        endpoint_lowfreq = F.l1_loss(_lowpass(z_hat1, self.lowpass_kernel).float(), _lowpass(projected_target, self.lowpass_kernel).float())

        loss_endpoint_content = content.new_tensor(0.0)
        loss_endpoint_style = content.new_tensor(0.0)
        loss_endpoint_vel_reg = content.new_tensor(0.0)

        if self.training_objective_mode == "endpoint":
            z1_low = _lowpass(z_hat1, self.lowpass_kernel)
            y_low = _lowpass(projected_target, self.lowpass_kernel)
            loss_endpoint_content = F.mse_loss(z1_low.float(), y_low.float())
            loss_endpoint_style = swd_ss
            loss_endpoint_vel_reg = fm
            loss = (
                self.w_endpoint_content * loss_endpoint_content
                + self.w_endpoint_style * loss_endpoint_style
                + self.w_endpoint_velocity_reg * loss_endpoint_vel_reg
                + self.source_endpoint_aux_weight * source_endpoint_aux
                + self.endpoint_energy_band_weight * endpoint_energy_band
                + style_strength_loss
            )
        else:
            loss = (
                self.fm_weight * fm
                + self.single_step_swd_weight * swd_ss
                + self.single_step_edge_weight * edge_ss
                + self.endpoint_lowfreq_weight * endpoint_lowfreq
                + self.source_endpoint_aux_weight * source_endpoint_aux
                + self.endpoint_energy_band_weight * endpoint_energy_band
                + style_strength_loss
            )

        entropy_loss = content.new_tensor(0.0)
        if self.w_attn_entropy_reg > 0.0 and getattr(model, "last_cross_attn_entropy", None) is not None:
            entropy_loss = self.w_attn_entropy_reg * model.last_cross_attn_entropy
            loss = loss + entropy_loss
        c_low = _lowpass(content, self.lowpass_kernel)
        t_low = _lowpass(projected_target, self.lowpass_kernel)
        z_low = _lowpass(z_hat1, self.lowpass_kernel)
        z_high = z_hat1 - z_low
        target_high = projected_target - t_low
        low_to_source = (z_low - c_low).detach().float().abs().mean()
        low_to_target = (z_low - t_low).detach().float().abs().mean()
        high_to_target = (z_high - target_high).detach().float().abs().mean()
        low_target_ratio = low_to_target / low_to_source.clamp_min(1e-8)
        low_leak = _lowpass(pred_velocity, self.lowpass_kernel).float().abs().mean()
        debug = getattr(model, "last_debug", {}) if hasattr(model, "last_debug") else {}
        zero = content.new_tensor(0.0)
        metrics = {
            "loss": loss,
            "flow": fm.detach(),
            "loss_fm": fm.detach(),
            "loss_swd_ss": swd_ss.detach(),
            "loss_edge_ss": edge_ss.detach(),
            "loss_endpoint_lowfreq": endpoint_lowfreq.detach(),
            "loss_endpoint_content": loss_endpoint_content.detach(),
            "loss_endpoint_style": loss_endpoint_style.detach(),
            "loss_endpoint_vel_reg": loss_endpoint_vel_reg.detach(),
            "training_objective_mode": content.new_tensor(1.0 if self.training_objective_mode == "endpoint" else 0.0),
            "loss_source_endpoint_aux": source_endpoint_aux.detach(),
            "loss_endpoint_energy_band": endpoint_energy_band.detach(),
            "loss_style_strength_reg": style_strength_loss.detach(),
            "style_strength_alpha": style_strength_alpha.detach(),
            "loss_attn_entropy": entropy_loss.detach(),
            "single_step_swd": (swd_ss * self.single_step_swd_weight).detach(),
            "single_step_edge": (edge_ss * self.single_step_edge_weight).detach(),
            "endpoint_lowfreq": (endpoint_lowfreq * self.endpoint_lowfreq_weight).detach(),
            "source_endpoint_aux": (source_endpoint_aux * self.source_endpoint_aux_weight).detach(),
            "endpoint_energy_band": (endpoint_energy_band * self.endpoint_energy_band_weight).detach(),
            "terminal_swd": zero,
            "ot_cost": zero,
            "ot_plan_entropy": zero,
            "ot_target_gini": zero,
            "t_mean": t.detach().float().mean(),
            "velocity_abs": pred_velocity.detach().float().abs().mean(),
            "target_velocity_abs": target_velocity.detach().float().abs().mean(),
            "endpoint_abs": z_hat1.detach().float().abs().mean(),
            "base_structural_drift": low_to_source.detach(),
            "endpoint_low_to_source": low_to_source.detach(),
            "endpoint_low_to_target": low_to_target.detach(),
            "endpoint_high_to_target": high_to_target.detach(),
            "endpoint_low_target_ratio": low_target_ratio.detach(),
            "low_freq_leak": low_leak.detach(),
            "fiber_energy_ratio": ((target_velocity.float().square().mean()) / (target_style.float().square().mean().clamp_min(1e-8))).detach(),
            "target_base_shift": (t_low - c_low).detach().float().abs().mean(),
            "training_target_projection_low_mode_target_linear": content.new_tensor(1.0 if self.low_mode == "target_linear" else 0.0),
            "training_target_projection_low_mode_channel_mean": content.new_tensor(1.0 if self.low_mode == "channel_mean" else 0.0),
            "training_target_projection_low_mode_all": content.new_tensor(1.0 if self.low_mode == "all" else 0.0),
            "bridge_sigma": content.new_tensor(float(getattr(model, "bridge_sigma", 0.0))),
            "swd_noise_sigma": content.new_tensor(self.swd_noise_sigma),
            "style_dino_active": content.new_tensor(1.0 if style_patches is not None else 0.0),
            "style_gate_value": debug.get("style_gate_value", zero).detach() if torch.is_tensor(debug.get("style_gate_value", None)) else zero,
            "cross_attn_entropy": debug.get("cross_attn_entropy", zero).detach() if torch.is_tensor(debug.get("cross_attn_entropy", None)) else zero,
            "cross_attn_delta_abs": debug.get("cross_attn_delta_abs", zero).detach() if torch.is_tensor(debug.get("cross_attn_delta_abs", None)) else zero,
            "endpoint_head_mode_lowhigh": debug.get("endpoint_head_mode_lowhigh", zero).detach() if torch.is_tensor(debug.get("endpoint_head_mode_lowhigh", None)) else zero,
            "endpoint_pred_abs_debug": debug.get("endpoint_pred_abs", zero).detach() if torch.is_tensor(debug.get("endpoint_pred_abs", None)) else zero,
            "endpoint_low_abs_debug": debug.get("endpoint_low_abs", zero).detach() if torch.is_tensor(debug.get("endpoint_low_abs", None)) else zero,
            "endpoint_high_abs_debug": debug.get("endpoint_high_abs", zero).detach() if torch.is_tensor(debug.get("endpoint_high_abs", None)) else zero,
            "endpoint_style_low_abs_debug": debug.get("endpoint_style_low_abs", zero).detach() if torch.is_tensor(debug.get("endpoint_style_low_abs", None)) else zero,
            "endpoint_style_high_abs_debug": debug.get("endpoint_style_high_abs", zero).detach() if torch.is_tensor(debug.get("endpoint_style_high_abs", None)) else zero,
        }
        self.last_debug = {
            "x_t": x_t.detach(),
            "target_velocity": target_velocity.detach(),
            "pred_velocity": pred_velocity.detach(),
            "z_hat1": z_hat1.detach(),
            "projected_target": projected_target.detach(),
        }
        metrics.update(projection_metrics)
        return metrics

    def compute_debug(self, model, **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        metrics = self.compute(model, **kwargs)
        return {"metrics": metrics, "components": {}, "state": dict(self.last_debug)}
