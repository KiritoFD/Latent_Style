from __future__ import annotations

import random
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
        self.training = True
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
            "tri_band_wavelet",
        }:
            self.training_target_projection_mode = "source_low_target_high"
        self.low_mode = str(getattr(self.bridge_cfg, "training_target_projection_low_mode", "all")).strip().lower()
        if self.low_mode not in {"all", "channel_mean", "target_linear"}:
            self.low_mode = "all"
        self.num_projections = int(getattr(self.bridge_cfg, "semantic_swd_num_projections", 64))
        self.t_min = float(getattr(self.bridge_cfg, "t_min", 0.0))
        self.t_max = float(getattr(self.bridge_cfg, "t_max", 1.0))
        self.t_sampling_power = max(1e-3, float(getattr(self.bridge_cfg, "t_sampling_power", 1.0)))
        self.t_sampling_beta_a = max(0.0, float(getattr(self.bridge_cfg, "t_sampling_beta_a", 0.0)))
        self.t_sampling_beta_b = max(0.0, float(getattr(self.bridge_cfg, "t_sampling_beta_b", 0.0)))
        # === FC-SB Phase 4 A3: Logit-Normal 时间采样 ===
        self.t_sampling_mode = str(getattr(self.bridge_cfg, "t_sampling_mode", "uniform_power")).strip().lower()
        self.t_sampling_logit_mean = float(getattr(self.bridge_cfg, "t_sampling_logit_mean", 0.0))  # 正值偏向 t→1
        self.t_sampling_logit_std = max(1e-3, float(getattr(self.bridge_cfg, "t_sampling_logit_std", 1.0)))  # 越小越集中
        self.source_endpoint_aux_weight = float(getattr(self.bridge_cfg, "source_endpoint_aux_weight", 0.0))
        self.endpoint_energy_band_weight = float(getattr(self.bridge_cfg, "endpoint_energy_band_weight", 0.0))
        self.swd_scale_mode = str(getattr(self.bridge_cfg, "swd_scale_mode", "global")).strip().lower()
        self.w_attn_entropy_reg = float(getattr(self.bridge_cfg, "w_attn_entropy_reg", 0.0))
        self.w_style_strength_reg = float(getattr(self.bridge_cfg, "w_style_strength_reg", 0.0))
        self.swd_noise_sigma = float(getattr(self.bridge_cfg, "swd_noise_sigma", 0.0))
        self.bridge_sigma = float(getattr(self.bridge_cfg, "bridge_sigma", 0.0))
        self._base_bridge_sigma = self.bridge_sigma
        self.bridge_sigma_schedule = str(getattr(self.bridge_cfg, "bridge_sigma_schedule", "constant")).strip().lower()
        if self.bridge_sigma_schedule not in {"constant", "curriculum", "linear_ramp", "brownian_bridge"}:
            self.bridge_sigma_schedule = "constant"
        self.training_sde_noise_mode = str(getattr(self.bridge_cfg, "training_sde_noise_mode", "subtractive")).strip().lower()
        if self.training_sde_noise_mode not in {"subtractive", "additive"}:
            self.training_sde_noise_mode = "subtractive"
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
        self.w_contrast_preserve = float(getattr(self.bridge_cfg, "w_contrast_preserve", 0.0))
        self.contrast_preserve_threshold = max(0.01, min(1.0, float(getattr(self.bridge_cfg, "contrast_preserve_threshold", 0.8))))
        self.w_channel_variance = float(getattr(self.bridge_cfg, "w_channel_variance", 0.0))
        self.w_hf_energy = float(getattr(self.bridge_cfg, "w_hf_energy", 0.0))
        self.hf_energy_threshold = max(0.01, min(1.0, float(getattr(self.bridge_cfg, "hf_energy_threshold", 0.5))))
        self.w_velocity_magnitude = float(getattr(self.bridge_cfg, "w_velocity_magnitude", 0.0))
        self.w_pixel_color_match = float(getattr(self.bridge_cfg, "w_pixel_color_match", 0.0))
        self.w_hsv_saturation = float(getattr(self.bridge_cfg, "w_hsv_saturation", 0.0))
        self.hsv_sat_threshold = max(0.01, min(1.0, float(getattr(self.bridge_cfg, "hsv_sat_threshold", 0.8))))
        self.w_flow_scale = max(0.01, min(2.0, float(getattr(self.bridge_cfg, "w_flow_scale", 1.0))))
        self.w_directional_cosine = max(0.0, float(getattr(self.bridge_cfg, "w_directional_cosine", 0.0)))
        self.w_freq_split_cosine = max(0.0, float(getattr(self.bridge_cfg, "w_freq_split_cosine", 0.0)))
        self.w_style_contrastive = float(getattr(self.bridge_cfg, "w_style_contrastive", 0.0))
        self.contrastive_margin = max(0.01, min(1.0, float(getattr(self.bridge_cfg, "contrastive_margin", 0.1))))
        self.contrastive_temperature = max(1e-4, float(getattr(self.bridge_cfg, "contrastive_temperature", 0.1)))
        # === FC-SB Phase 3 W: 风格排斥 Loss ===
        self.w_fiber_repulsion = float(getattr(self.bridge_cfg, "w_fiber_repulsion", 0.0))
        self.fiber_repulsion_margin = max(0.01, float(getattr(self.bridge_cfg, "fiber_repulsion_margin", 0.5)))
        self.w_anti_input_style = float(getattr(self.bridge_cfg, "w_anti_input_style", 0.0))
        self.anti_input_margin = max(0.01, float(getattr(self.bridge_cfg, "anti_input_margin", 0.3)))
        self.w_style_disc = float(getattr(self.bridge_cfg, "w_style_disc", 0.0))
        self.style_disc_dim = max(8, int(getattr(self.bridge_cfg, "style_disc_dim", 128)))
        # === FC-SB Phase 4 A4: Output Variance Matching (W 方向重生) ===
        # 理论: W2 hinge loss 失效（step=1 归零），白化根因是输出 fiber 方差被洗掉.
        # 改为约束输出 fiber 的 per-channel 标准差对齐 target style fiber 的标准差.
        self.w_output_variance = float(getattr(self.bridge_cfg, "w_output_variance", 0.0))
        self.output_variance_band: str = str(getattr(self.bridge_cfg, "output_variance_band", "hh")).strip().lower()
        # "hh" = 仅匹配 HH 频带方差; "mid" = 仅匹配 Mid; "all" = 匹配全 fiber
        # === FC-SB Phase 4 B4: Fiber-MoE Load Balancing ===
        self.fiber_moe_load_balance_weight = float(getattr(self.bridge_cfg, "fiber_moe_load_balance_weight", 0.0))
        self.bridge_path_mode = str(getattr(self.bridge_cfg, "bridge_path_mode", "linear")).strip().lower()
        if self.bridge_path_mode not in {"linear", "spherical_vp"}:
            self.bridge_path_mode = "linear"
        # CFG dropout: replace style tokens with null tokens during training
        self.cfg_dropout_prob = max(0.0, min(1.0, float(getattr(self.bridge_cfg, "cfg_dropout_prob", 0.0))))
        self.cfg_null_token_init_std = max(1e-6, float(getattr(self.bridge_cfg, "cfg_null_token_init_std", 0.02)))
        self._null_style_tokens: torch.Tensor | None = None
        # === D3: per-subband FM loss（频域解耦）===
        # 默认 spectral_w_ll=0 → 不启用，走原 FM loss（向后兼容）
        # 用户显式设 spectral_w_ll>0 → 启用 per-subband 加权 FM loss
        self.spectral_w_ll = float(getattr(self.bridge_cfg, "spectral_w_ll", 0.0))
        self.spectral_w_lh = float(getattr(self.bridge_cfg, "spectral_w_lh", 1.0))
        self.spectral_w_hl = float(getattr(self.bridge_cfg, "spectral_w_hl", 1.0))
        self.spectral_w_hh = float(getattr(self.bridge_cfg, "spectral_w_hh", 2.0))
        self.spectral_fm_enabled = self.spectral_w_ll > 0.0

    def update_weights_for_epoch(self, epoch: int, num_epochs: int = 3) -> dict[str, float]:
        # Sigma schedule (curriculum / linear ramp)
        if self.bridge_sigma_schedule == "curriculum":
            if epoch <= max(1, num_epochs // 3):
                self.bridge_sigma = self._base_bridge_sigma * 0.25
            elif epoch <= max(1, 2 * num_epochs // 3):
                self.bridge_sigma = self._base_bridge_sigma * 0.6
            else:
                self.bridge_sigma = self._base_bridge_sigma * 1.0
        elif self.bridge_sigma_schedule == "linear_ramp":
            t = max(0.0, min(1.0, (epoch - 1) / max(1, num_epochs - 1)))
            self.bridge_sigma = self._base_bridge_sigma * (0.2 + 0.8 * t)
        elif self.bridge_sigma_schedule == "brownian_bridge":
            # Brownian Bridge: sigma peaks at middle epoch, vanishes at start/end
            # Matches the theoretical σ²·t·(1-t) bridge variance
            t = max(0.0, min(1.0, (epoch - 1) / max(1, num_epochs - 1)))
            self.bridge_sigma = self._base_bridge_sigma * (4.0 * t * (1.0 - t))
        else:
            self.bridge_sigma = self._base_bridge_sigma

        if not self.two_stage_enabled:
            return {
                "w_endpoint_content": self.w_endpoint_content,
                "w_endpoint_style": self.w_endpoint_style,
                "w_style_strength_reg": self.w_style_strength_reg,
                "stage": 0,
                "bridge_sigma": self.bridge_sigma,
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
            "bridge_sigma": self.bridge_sigma,
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

        if self.t_sampling_mode == "logit_normal":
            # === FC-SB Phase 4 A3: Logit-Normal 时间采样 ===
            # 理论: 模型 99% 精力浪费在 t∈[0,0.3] 无意义区间.
            # Logit-Normal: u = sigmoid(N(μ, σ²)), 集中在 μ 附近, σ 控制集中度.
            # μ>0 偏向后段（笔触生成关键期 t∈[0.6,0.95]）, μ<0 偏向前段.
            u_normal = torch.randn(content.shape[0], device=content.device, dtype=content.dtype)
            u_normal = u_normal * self.t_sampling_logit_std + self.t_sampling_logit_mean
            u = torch.sigmoid(u_normal).clamp(1e-6, 1.0 - 1e-6)
        elif self.t_sampling_beta_a > 0 and self.t_sampling_beta_b > 0:
            # Beta distribution sampling for late-stage focus
            a = torch.tensor(self.t_sampling_beta_a, device=content.device)
            b = torch.tensor(self.t_sampling_beta_b, device=content.device)
            dist = torch.distributions.Beta(a, b)
            u = dist.sample([content.shape[0]]).to(dtype=content.dtype)
            # Clamp to valid range
            u = u.clamp(1e-6, 1.0 - 1e-6)
        else:
            # Original uniform power sampling (backward compatible)
            u = torch.empty(content.shape[0], device=content.device, dtype=content.dtype).uniform_(0.0, 1.0)
            u = u.pow(self.t_sampling_power)

        return lo + (hi - lo) * u

    def _wavelet_lowpass(self, x: torch.Tensor) -> torch.Tensor:
        down = F.avg_pool2d(x.float(), kernel_size=2, stride=2, ceil_mode=False)
        return F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False).to(dtype=x.dtype)

    def _split_base_fiber(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training_target_projection_mode in {"wavelet_source_low_target_high", "pure_vertical_flow_wavelet"}:
            low = self._wavelet_lowpass(x)
        elif self.training_target_projection_mode == "tri_band_wavelet":
            # Tri-band: low = broad structure (large kernel), high = everything else
            low = _lowpass(x, int(getattr(self.bridge_cfg, "tri_band_low_kernel", 11)))
        else:
            low = _lowpass(x, self.lowpass_kernel)
        high = x - low
        return low, high

    def _split_tri_band(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Three-band decomposition: LL (structure), Mid (edges), HH (texture).
        LL = large-kernel lowpass (color/illumination)
        Mid = mid-kernel lowpass - LL (edges/contours — content structure)
        HH = x - mid-kernel lowpass (fine texture/strokes — style)
        """
        low_kernel = int(getattr(self.bridge_cfg, "tri_band_low_kernel", 11))
        mid_kernel = int(getattr(self.bridge_cfg, "tri_band_mid_kernel", 3))
        ll = _lowpass(x, low_kernel)
        mid_full = _lowpass(x, mid_kernel)
        mid = mid_full - ll  # edge band
        hh = x - mid_full     # texture band
        return ll, mid, hh

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
        elif self.training_target_projection_mode == "tri_band_wavelet":
            # === FC-SB v2 Scheme A: Three-band decomposition ===
            # LL (structure): locked to content's broad lowpass
            # Mid (edges): α-blend between content and target edges (preserves content contours)
            # HH (texture): fully from target (free style diffusion)
            c_ll, c_mid, c_hh = self._split_tri_band(content)
            t_ll, t_mid, t_hh = self._split_tri_band(target)
            edge_alpha = max(0.0, min(1.0, float(getattr(self.bridge_cfg, "tri_band_edge_preserve_alpha", 0.5))))
            projected = c_ll + (edge_alpha * c_mid + (1.0 - edge_alpha) * t_mid) + t_hh
            metrics["training_target_projection_mode_pure_vertical_flow_wavelet"] = content.new_tensor(1.0)
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

        if self.bridge_path_mode == "spherical_vp":
            # Spherical (VP-Flow) interpolation: constant variance path
            # x_t = cos(pi*t/2) * x_0 + sin(pi*t/2) * x_1
            # v_t = d/dt[x_t] = -(pi/2)*sin(pi*t/2)*x_0 + (pi/2)*cos(pi*t/2)*x_1
            half_pi = 3.141592653589793 / 2.0
            angle = half_pi * t4
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            # Velocity scaling factor: pi/2
            vel_scale = content.new_tensor(half_pi)

            if self.low_mode == "target_linear":
                x_low = cos_a * c_low + sin_a * p_low
                target_low_velocity = vel_scale * (-sin_a * c_low + cos_a * p_low)
            else:
                x_low = low_anchor * c_low + (1.0 - low_anchor) * p_low
                target_low_velocity = torch.zeros_like(c_low)
                if self.low_mode == "channel_mean":
                    c_mean = c_low.mean(dim=(2, 3), keepdim=True)
                    p_mean = p_low.mean(dim=(2, 3), keepdim=True)
                    x_low = x_low + sin_a * (p_mean - c_mean)
                    target_low_velocity = vel_scale * (cos_a * (p_mean - c_mean))
            x_t = x_low + cos_a * c_high + sin_a * p_high
            target_velocity = vel_scale * (-sin_a * c_high + cos_a * p_high) + target_low_velocity
        else:
            # Original linear interpolation
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

        # --- CFG dropout: randomly replace style with null tokens ---
        use_uncond = (self.cfg_dropout_prob > 0.0 and model.training
                      and random.random() < self.cfg_dropout_prob)
        if use_uncond:
            B, C, H, W = target_style.shape
            null_style = torch.randn(B, C, H, W, device=target_style.device,
                                     dtype=target_style.dtype) * self.cfg_null_token_init_std
            target_style_for_model = null_style
        else:
            target_style_for_model = target_style

        import inspect
        _model_fwd_sig = inspect.signature(model.forward)
        _model_fwd_params = set(_model_fwd_sig.parameters.keys())
        _call_kwargs = {
            "x": x_t, "x_t": x_t, "t": t,
            "style_id": target_style_id,
            "style_dino_patches": style_patches,
            "style_dino_cls": style_cls,
            "content_dino_patches": content_patches,
            "style_latent": target_style_for_model,
            "style_text_tokens": style_text_tokens,
        }
        # Only pass params that model.forward() actually accepts
        _filtered_kwargs = {k: v for k, v in _call_kwargs.items() if k in _model_fwd_params}
        pred_velocity = model(**_filtered_kwargs)
        z_hat1 = x_t + (1.0 - t).view(-1, 1, 1, 1).to(dtype=x_t.dtype) * pred_velocity
        # === FC-SB: 训练期高通 SDE 噪声注入 ===
        sde_noise_metrics = {}
        if self.bridge_sigma > 0 and self.training:
            sde_noise = torch.randn_like(target_velocity)
            sde_noise_hp = sde_noise - _lowpass(sde_noise, self.lowpass_kernel)
            target_velocity = target_velocity + self.bridge_sigma * sde_noise_hp
            sde_noise_metrics["training_sde_noise_lp_rms"] = _lowpass(sde_noise).std().item()
            sde_noise_metrics["training_sde_noise_hp_rms"] = sde_noise_hp.std().item()
        loss_type = getattr(self.bridge_cfg, 'loss_type', 'mse').lower().strip()
        if loss_type in ('huber', 'smooth_l1', 'smoothl1'):
            fm = F.smooth_l1_loss(pred_velocity.float(), target_velocity.float())
        elif loss_type == 'huber_delta':
            delta = float(getattr(self.bridge_cfg, 'huber_delta', 1.0))
            fm = F.huber_loss(pred_velocity.float(), target_velocity.float(), delta=delta, reduction='mean')
        else:
            fm = F.mse_loss(pred_velocity.float(), target_velocity.float())

        # === D3: per-subband FM loss（频域解耦）===
        # spectral_w_ll>0 时用 4 子带加权 loss 替代全频带 fm；否则保持原 fm（向后兼容）
        if self.spectral_fm_enabled:
            from spectral620 import dwt2_haar
            # velocity 形状 (B, C, H, W)，dwt2_haar 自动处理奇数空间维度
            pred_ll, pred_lh, pred_hl, pred_hh = dwt2_haar(pred_velocity.float())
            tgt_ll, tgt_lh, tgt_hl, tgt_hh = dwt2_haar(target_velocity.float())
            if loss_type in ('huber', 'smooth_l1', 'smoothl1'):
                loss_ll = F.smooth_l1_loss(pred_ll, tgt_ll)
                loss_lh = F.smooth_l1_loss(pred_lh, tgt_lh)
                loss_hl = F.smooth_l1_loss(pred_hl, tgt_hl)
                loss_hh = F.smooth_l1_loss(pred_hh, tgt_hh)
            elif loss_type == 'huber_delta':
                # delta 已在上方 huber_delta 分支定义
                loss_ll = F.huber_loss(pred_ll, tgt_ll, delta=delta, reduction='mean')
                loss_lh = F.huber_loss(pred_lh, tgt_lh, delta=delta, reduction='mean')
                loss_hl = F.huber_loss(pred_hl, tgt_hl, delta=delta, reduction='mean')
                loss_hh = F.huber_loss(pred_hh, tgt_hh, delta=delta, reduction='mean')
            else:
                loss_ll = F.mse_loss(pred_ll, tgt_ll)
                loss_lh = F.mse_loss(pred_lh, tgt_lh)
                loss_hl = F.mse_loss(pred_hl, tgt_hl)
                loss_hh = F.mse_loss(pred_hh, tgt_hh)
            fm = (self.spectral_w_ll * loss_ll + self.spectral_w_lh * loss_lh
                  + self.spectral_w_hl * loss_hl + self.spectral_w_hh * loss_hh)

        # Directional cosine loss with frequency split
        dir_cosine_loss = content.new_tensor(0.0)
        _clip_dir_val = 0.0
        _clip_fm_low_val = 0.0
        if self.w_directional_cosine > 0:
            if self.w_freq_split_cosine > 0:
                # === Frequency-split mode: low-freq MSE + high-freq Cosine ===
                _lp_kernel = getattr(self, 'lowpass_kernel', 5)

                def _lowpass_vel(x, k=_lp_kernel):
                    return F.avg_pool2d(x.float(), k, stride=1, padding=k // 2) if x.dim() == 4 else x

                v_pred_lp = _lowpass_vel(pred_velocity)
                v_tgt_lp = _lowpass_vel(target_velocity)
                v_pred_hp = pred_velocity.float() - v_pred_lp
                v_tgt_hp = target_velocity.float() - v_tgt_lp

                # Low-frequency: strict MSE (preserve structure)
                fm_low_freq = F.mse_loss(v_pred_lp, v_tgt_lp)

                # High-frequency: directional cosine (preserve style strokes)
                v_pred_hp_flat = v_pred_hp.reshape(v_pred_hp.shape[0], -1)
                v_tgt_hp_flat = v_tgt_hp.reshape(v_tgt_hp.shape[0], -1)
                v_pred_hp_n = F.normalize(v_pred_hp_flat, dim=-1)
                v_tgt_hp_n = F.normalize(v_tgt_hp_flat, dim=-1)
                cos_sim_hp = (v_pred_hp_n * v_tgt_hp_n).sum(dim=-1).mean()
                dir_loss_hp = (1.0 - cos_sim_hp).clamp(min=0.0)

                # Combined: MSE dominant + high-freq direction as auxiliary constraint
                dir_cosine_loss = fm_low_freq * 0.5 + dir_loss_hp
                fm = fm + self.w_directional_cosine * dir_cosine_loss
                _clip_dir_val = dir_loss_hp.item()
                _clip_fm_low_val = fm_low_freq.item()
            else:
                # Original full-band cosine loss (E8 behavior)
                v_pred_n = F.normalize(pred_velocity.float().reshape(pred_velocity.shape[0], -1), dim=-1)
                v_tgt_n = F.normalize(target_velocity.float().reshape(target_velocity.shape[0], -1), dim=-1)
                cos_sim = (v_pred_n * v_tgt_n).sum(dim=-1).mean()
                dir_cosine_loss = (1.0 - cos_sim).clamp(min=0.0)
                fm = fm + self.w_directional_cosine * dir_cosine_loss
                _clip_dir_val = dir_cosine_loss.item()

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

        # ===== Anti-whitening losses =====
        contrast_loss = content.new_tensor(0.0)
        if self.w_contrast_preserve > 0:
            gen_std = z_hat1.float().std(dim=[1, 2, 3]).mean()
            tgt_std = projected_target.float().std(dim=[1, 2, 3]).mean()
            contrast_loss = F.relu(
                tgt_std * self.contrast_preserve_threshold - gen_std
            )

        ch_var_loss = content.new_tensor(0.0)
        if self.w_channel_variance > 0:
            gen_ch_var = z_hat1.float().var(dim=[2, 3])
            ch_var_loss = -gen_ch_var.clamp_min(1e-8).log().mean()

        hf_loss = content.new_tensor(0.0)
        if self.w_hf_energy > 0:
            gen_hf = z_hat1.float() - _lowpass(z_hat1.float())
            tgt_hf = projected_target.float() - _lowpass(projected_target.float())
            gen_hf_e = gen_hf.pow(2).mean()
            tgt_hf_e = tgt_hf.pow(2).mean()
            hf_loss = F.relu(tgt_hf_e * self.hf_energy_threshold - gen_hf_e)

        anti_whiten_total = (
            self.w_contrast_preserve * contrast_loss
            + self.w_channel_variance * ch_var_loss
            + self.w_hf_energy * hf_loss
        )

        # Velocity Magnitude Loss: 确保 v_pred 的幅度接近 v_target
        vel_mag_loss = content.new_tensor(0.0)
        v_pred_norm = content.new_tensor(0.0)
        v_target_norm = content.new_tensor(0.0)
        velocity_ratio = content.new_tensor(1.0)
        if self.w_velocity_magnitude > 0:
            v_pred_norm = pred_velocity.float().norm(p=2, dim=(1, 2, 3)).mean()
            v_target_norm = target_velocity.float().norm(p=2, dim=(1, 2, 3)).mean()
            # 归一化的幅度差异（相对于 target 的比例）
            velocity_ratio = v_pred_norm / v_target_norm.clamp_min(1e-8)
            # 惩罚偏离 1.0 的情况
            vel_mag_loss = (velocity_ratio - 1.0).pow(2)

        # Pixel-Space Color Preservation Loss (Per-Channel Matching)
        # 在 latent 空间做细粒度的 per-channel mean/std 匹配
        # 解决 R5 诊断：latent 全局统计量匹配好但解码后仍有雾化
        pixel_color_loss = content.new_tensor(0.0)
        gen_per_ch_mean = content.new_tensor(0.0)
        tgt_per_ch_mean = content.new_tensor(0.0)
        gen_per_ch_std = content.new_tensor(0.0)
        tgt_per_ch_std = content.new_tensor(0.0)
        if self.w_pixel_color_match > 0:
            # Per-channel mean matching (B, C) - 更细粒度 than global mean
            gen_per_ch_mean = z_hat1.float().mean(dim=[2, 3])  # (B, C)
            tgt_per_ch_mean = projected_target.float().mean(dim=[2, 3])
            ch_mean_loss = F.mse_loss(gen_per_ch_mean, tgt_per_ch_mean)

            # Per-channel std matching (使用 log 空间避免数值问题)
            gen_per_ch_std = z_hat1.float().std(dim=[2, 3])  # (B, C)
            tgt_per_ch_std = projected_target.float().std(dim=[2, 3])
            ch_std_loss = F.mse_loss(
                gen_per_ch_std.clamp_min(1e-6).log(),
                tgt_per_ch_std.clamp_min(1e-6).log()
            )

            # 组合：mean + std 匹配
            pixel_color_loss = ch_mean_loss + ch_std_loss

        # ===== Saturation Proxy Loss =====
        sat_loss = content.new_tensor(0.0)
        gen_ch_vars = content.new_tensor(0.0)
        if self.w_hsv_saturation > 0:
            # 各通道的空间方差 (B, C) — 代表每个"颜色维度"的活跃度
            gen_ch_vars_raw = z_hat1.float().var(dim=[2, 3])  # (B, C)
            tgt_ch_vars_raw = projected_target.float().var(dim=[2, 3])  # (B, C)

            # 归一化为概率分布
            gen_ch_var_sum = gen_ch_vars_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            tgt_ch_var_sum = tgt_ch_vars_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            gen_p = gen_ch_vars_raw / gen_ch_var_sum  # (B, C), sum to 1
            tgt_p = tgt_ch_vars_raw / tgt_ch_var_sum

            # 方向性 KL 散度：让生成的通道方差分布接近目标
            sat_loss = F.kl_div(gen_p.clamp_min(1e-8).log(), tgt_p.clamp_min(1e-8), reduction='batchmean')
            gen_ch_vars = gen_ch_vars_raw.detach()

        # ===== Style Contrastive Loss =====
        contrastive_loss = content.new_tensor(0.0)
        avg_cross_sim = content.new_tensor(0.0)
        if self.w_style_contrastive > 0:
            B = z_hat1.shape[0]
            if B >= 2:
                # 展平为向量计算余弦相似度
                z_flat = z_hat1.float().reshape(B, -1)  # (B, C*H*W)
                z_norm = F.normalize(z_flat, p=2, dim=-1)  # L2 normalize

                # 计算 pairwise cosine similarity matrix
                cos_sim = torch.mm(z_norm, z_norm.t())  # (B, B)

                # 创建 mask: 只对不同 pair 惩罚（排除自身）
                mask = 1.0 - torch.eye(B, device=z_hat1.device)
                num_pairs = mask.sum()

                if num_pairs > 0:
                    # Margin-based contrastive loss
                    # 惩罚 cos_sim > margin 的 pair
                    sim_off_diag = (cos_sim * mask).flatten()
                    # 使用 soft margin: relu 形式
                    contrastive_loss = F.relu(sim_off_diag - self.contrastive_margin).mean()

                    # debug metric: 平均非对角相似度
                    avg_cross_sim = (cos_sim * mask).sum() / num_pairs.clamp_min(1)
            else:
                # batch size 太小，用简单的方差损失作为替代
                diversity_loss = -z_hat1.float().var().log().clamp_min(-10)
                contrastive_loss = diversity_loss

        # ===== FC-SB Phase 3 W: 风格排斥 Loss =====
        # FC-SB Phase 3 deepfix: W loss debug counter
        if not hasattr(self, '_w_debug_counter'):
            self._w_debug_counter = 0
        self._w_debug_counter += 1
        _w_debug_print = (self._w_debug_counter % 50 == 1)
        fiber_repulsion_loss = content.new_tensor(0.0)
        anti_input_loss = content.new_tensor(0.0)
        style_disc_loss = content.new_tensor(0.0)

        # 复用 fiber 分量（已在 forward 中计算，但此处独立计算确保正确）
        z_low = _lowpass(z_hat1, self.lowpass_kernel)
        f_gen = z_hat1 - z_low  # 生成 fiber
        f_in = content - _lowpass(content, self.lowpass_kernel)  # 输入 content fiber

        # W1: Cross-style Fiber Repulsion（跨风格 fiber 排斥）
        if self.w_fiber_repulsion > 0:
            B = z_hat1.shape[0]
            if B >= 2:
                # 展平 fiber 为向量
                f_flat = f_gen.float().reshape(B, -1)  # (B, C*H*W)
                # pairwise L2 距离: dist[i,j] = ||f_i - f_j||_2
                dist = torch.cdist(f_flat, f_flat, p=2)  # (B, B)
                # mask 排除对角
                mask = 1.0 - torch.eye(B, device=z_hat1.device)
                num_pairs = mask.sum().clamp_min(1)
                # margin loss: relu(margin - dist)
                fiber_repulsion_loss = (F.relu(self.fiber_repulsion_margin - dist) * mask).sum() / num_pairs
                # FC-SB Phase 3 deepfix: debug print pairwise dist 分布
                if _w_debug_print:
                    off_diag = dist[mask.bool()]
                    print(f"[W1-debug] step={self._w_debug_counter} pairwise_dist: mean={off_diag.mean().item():.4f} min={off_diag.min().item():.4f} max={off_diag.max().item():.4f} margin={self.fiber_repulsion_margin:.4f} loss={fiber_repulsion_loss.item():.6f}", flush=True)

        # W2: Anti-input-style Repulsion（输入风格排斥）
        if self.w_anti_input_style > 0:
            # 每个样本的生成 fiber 与输入 content fiber 的距离
            diff = (f_gen.float() - f_in.float()).reshape(f_gen.shape[0], -1)
            dist_input = torch.norm(diff, p=2, dim=1)  # (B,)
            anti_input_loss = F.relu(self.anti_input_margin - dist_input).mean()
            # FC-SB Phase 3 deepfix: debug print dist_input 分布
            if _w_debug_print:
                print(f"[W2-debug] step={self._w_debug_counter} dist_input: mean={dist_input.mean().item():.4f} min={dist_input.min().item():.4f} max={dist_input.max().item():.4f} margin={self.anti_input_margin:.4f} loss={anti_input_loss.item():.6f}", flush=True)

        # W3: Style Discriminative Loss（风格判别）
        if self.w_style_disc > 0 and getattr(model, "style_disc_head", None) is not None and torch.is_tensor(target_style_id):
            # f_gen 全局池化 -> (B, C)
            f_gen_pool = f_gen.float().mean(dim=[2, 3])
            logits = model.style_disc_head(f_gen_pool)  # (B, num_styles)
            style_disc_loss = F.cross_entropy(logits, target_style_id.long())

        # === FC-SB Phase 4 A4: Output Variance Matching (W 方向重生) ===
        # 理论: 白化根因是输出 fiber 方差被洗掉. 约束输出 fiber 的 per-channel std 对齐 target.
        # 与 W2 hinge 不同: 方差是连续值, 不会一步归零; 约束 output 而非 input-target 距离.
        output_variance_loss = content.new_tensor(0.0)
        if self.w_output_variance > 0:
            f_target = projected_target - _lowpass(projected_target, self.lowpass_kernel)  # target fiber
            if self.output_variance_band == "hh":
                # Haar HH 频带方差匹配（与推理期 N1 的 hh 一致）
                def _haar_hh(x):
                    return (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] - x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / 2.0
                f_gen_hh = _haar_hh(f_gen.float())
                f_target_hh = _haar_hh(f_target.float())
                gen_std = f_gen_hh.std(dim=[2, 3], keepdim=False)
                target_std = f_target_hh.std(dim=[2, 3], keepdim=False)
            elif self.output_variance_band == "mid":
                def _haar_mid(x):
                    lh = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] - x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / 2.0
                    hl = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] + x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / 2.0
                    return lh + hl
                gen_std = _haar_mid(f_gen.float()).std(dim=[2, 3], keepdim=False)
                target_std = _haar_mid(f_target.float()).std(dim=[2, 3], keepdim=False)
            else:  # "all"
                gen_std = f_gen.float().std(dim=[2, 3], keepdim=False)
                target_std = f_target.float().std(dim=[2, 3], keepdim=False)
            # L2 距离 between per-channel stds
            output_variance_loss = ((gen_std - target_std) ** 2).mean()
            if _w_debug_print:
                print(f"[A4-debug] step={self._w_debug_counter} band={self.output_variance_band} gen_std_mean={gen_std.mean().item():.4f} target_std_mean={target_std.mean().item():.4f} loss={output_variance_loss.item():.6f}", flush=True)

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
                + self.w_endpoint_velocity_reg * self.w_flow_scale * loss_endpoint_vel_reg
                + self.source_endpoint_aux_weight * source_endpoint_aux
                + self.endpoint_energy_band_weight * endpoint_energy_band
                + style_strength_loss
                + anti_whiten_total
                + self.w_velocity_magnitude * vel_mag_loss
                + self.w_pixel_color_match * pixel_color_loss
                + self.w_hsv_saturation * sat_loss
                + self.w_style_contrastive * contrastive_loss
                + self.w_fiber_repulsion * fiber_repulsion_loss
                + self.w_anti_input_style * anti_input_loss
                + self.w_style_disc * style_disc_loss
                + self.w_output_variance * output_variance_loss
            )
        else:
            loss = (
                self.fm_weight * self.w_flow_scale * fm
                + self.single_step_swd_weight * swd_ss
                + self.single_step_edge_weight * edge_ss
                + self.endpoint_lowfreq_weight * endpoint_lowfreq
                + self.source_endpoint_aux_weight * source_endpoint_aux
                + self.endpoint_energy_band_weight * endpoint_energy_band
                + style_strength_loss
                + anti_whiten_total
                + self.w_velocity_magnitude * vel_mag_loss
                + self.w_pixel_color_match * pixel_color_loss
                + self.w_hsv_saturation * sat_loss
                + self.w_style_contrastive * contrastive_loss
                + self.w_fiber_repulsion * fiber_repulsion_loss
                + self.w_anti_input_style * anti_input_loss
                + self.w_style_disc * style_disc_loss
                + self.w_output_variance * output_variance_loss
            )

        entropy_loss = content.new_tensor(0.0)
        if self.w_attn_entropy_reg > 0.0 and getattr(model, "last_cross_attn_entropy", None) is not None:
            entropy_loss = self.w_attn_entropy_reg * model.last_cross_attn_entropy
            loss = loss + entropy_loss
        # === FC-SB Phase 4 B4: Fiber-MoE Load Balancing ===
        # 理论: MoE router 需负载均衡以避免 expert 坍缩 (所有样本路由到同一 expert).
        # aux_loss = -H(p) = sum(p_i * log(p_i)), 最大化熵 = 鼓励均匀分布.
        # 注意: B4 MoE 当前位于 integrate_transport (推理路径), 训练时 probs 可能未设置.
        #       此时 aux_loss = 0 (no-op). 未来将 MoE 移至训练路径后自动激活.
        b4_moe_aux_loss = content.new_tensor(0.0)
        b4_router_probs = getattr(model, "last_debug", {}).get("b4_moe_router_probs")
        if b4_router_probs is not None and self.fiber_moe_load_balance_weight > 0.0:
            avg_probs = b4_router_probs.mean(dim=0)  # (num_experts,)
            # 熵: H(p) = -sum(p_i * log(p_i)); aux_loss = -H(p) (最小化 = 最大化熵)
            b4_moe_aux_loss = (avg_probs * torch.log(avg_probs + 1e-8)).sum()
            loss = loss + self.fiber_moe_load_balance_weight * b4_moe_aux_loss
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
            "loss_type": content.new_tensor(0.0),  # placeholder, set below
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
            "loss_contrast_preserve": contrast_loss.detach(),
            "loss_channel_variance": ch_var_loss.detach(),
            "loss_hf_energy": hf_loss.detach(),
            "anti_whiten_total": anti_whiten_total.detach(),
            "gen_global_std": z_hat1.float().std(dim=[1,2,3]).mean().detach(),
            "target_global_std": projected_target.float().std(dim=[1,2,3]).mean().detach(),
            "gen_hf_energy": (z_hat1.float() - _lowpass(z_hat1.float())).pow(2).mean().detach(),
            "target_hf_energy": (projected_target.float() - _lowpass(projected_target.float())).pow(2).mean().detach(),
            "loss_velocity_magnitude": vel_mag_loss.detach(),
            "v_pred_norm": v_pred_norm.detach(),
            "v_target_norm": v_target_norm.detach(),
            "velocity_ratio": velocity_ratio.detach(),  # key metric: should approach 1.0
            "loss_pixel_color_match": pixel_color_loss.detach(),
            "gen_per_ch_mean": gen_per_ch_mean.detach().mean() if torch.is_tensor(gen_per_ch_mean) and gen_per_ch_mean.numel() > 1 else gen_per_ch_mean.detach(),
            "tgt_per_ch_mean": tgt_per_ch_mean.detach().mean() if torch.is_tensor(tgt_per_ch_mean) and tgt_per_ch_mean.numel() > 1 else tgt_per_ch_mean.detach(),
            "gen_per_ch_std": gen_per_ch_std.detach().mean() if torch.is_tensor(gen_per_ch_std) and gen_per_ch_std.numel() > 1 else gen_per_ch_std.detach(),
            "tgt_per_ch_std": tgt_per_ch_std.detach().mean() if torch.is_tensor(tgt_per_ch_std) and tgt_per_ch_std.numel() > 1 else tgt_per_ch_std.detach(),
            "loss_saturation_proxy": sat_loss.detach(),
            "gen_ch_var_max_ratio": (gen_ch_vars.max(dim=-1)[0] / (gen_ch_vars.mean(dim=-1).clamp_min(1e-8))).detach().mean() if torch.is_tensor(gen_ch_vars) and gen_ch_vars.numel() > 1 else content.new_tensor(0.0),
            "flow_scaled_weight": content.new_tensor(self.fm_weight * self.w_flow_scale),
            "loss_style_contrastive": contrastive_loss.detach(),
            "loss_fiber_repulsion": fiber_repulsion_loss.detach() if isinstance(fiber_repulsion_loss, torch.Tensor) else fiber_repulsion_loss,
            "loss_anti_input": anti_input_loss.detach() if isinstance(anti_input_loss, torch.Tensor) else anti_input_loss,
            "loss_style_disc": style_disc_loss.detach() if isinstance(style_disc_loss, torch.Tensor) else style_disc_loss,
            "loss_output_variance": output_variance_loss.detach() if isinstance(output_variance_loss, torch.Tensor) else output_variance_loss,
            "style_cross_sim_mean": avg_cross_sim.detach(),
            "loss_directional_cosine": dir_cosine_loss.detach(),
            "clip_dir": content.new_tensor(_clip_dir_val),
            "clip_fm_low": content.new_tensor(_clip_fm_low_val),
            "cfg_uncond_active": content.new_tensor(1.0 if use_uncond else 0.0),
            "cfg_dropout_prob": content.new_tensor(self.cfg_dropout_prob),
            "loss_b4_moe_load_balance": b4_moe_aux_loss.detach() if torch.is_tensor(b4_moe_aux_loss) else content.new_tensor(float(b4_moe_aux_loss)),
            "b4_moe_router_entropy": content.new_tensor(float(getattr(model, "last_debug", {}).get("b4_moe_router_entropy", 0.0))),
            "b4_moe_router_max_prob": content.new_tensor(float(getattr(model, "last_debug", {}).get("b4_moe_router_max_prob", 0.0))),
            **{k: content.new_tensor(float(v)) for k, v in sde_noise_metrics.items()},
        }
        # Record actual loss_type used
        _lt = getattr(self.bridge_cfg, 'loss_type', 'mse').lower().strip()
        metrics["loss_type"] = content.new_tensor(1.0 if _lt in ('huber', 'smooth_l1', 'smoothl1', 'huber_delta') else 0.0)
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
