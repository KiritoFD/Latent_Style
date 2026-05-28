from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn

from config_schema import ModelConfig
from lancet_backbone import LatentAdaCUT, count_parameters
from utils.diffeomorphic import apply_texture_aligned_diffeomorphic_stroke, build_diffeomorphic_guide


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half <= 0:
        return t.unsqueeze(-1)
    scale = math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -scale)
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TimeConditionedLANCETBridge(LatentAdaCUT):
    def __init__(self, config: ModelConfig) -> None:
        bridge_config = config.validated()
        super().__init__(bridge_config)
        self.time_dim = int(bridge_config.time_dim)
        self.velocity_head_mode = str(bridge_config.velocity_head_mode).strip().lower()
        self.velocity_tanh_limit = max(1e-3, float(bridge_config.velocity_tanh_limit))
        self.bridge_style_dim = int(self.style_emb.embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.bridge_style_dim),
            nn.SiLU(),
            nn.Linear(self.bridge_style_dim, self.bridge_style_dim),
        )

    def _compute_delta(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        style_code = getattr(self, "_current_style_code_for_head", None)
        style_tokens = getattr(self, "_last_style_token_fields", None)
        raw_delta = self._decode_output_raw(h, style_code=style_code, style_tokens=style_tokens)
        self.last_raw_diffeomorphic = raw_delta
        if bool(getattr(self, "use_diffeomorphic_stroke", False)):
            if x is None:
                raise ValueError("diffeomorphic stroke mode requires input x.")
            stroked = apply_texture_aligned_diffeomorphic_stroke(
                x,
                raw_delta,
                guide=build_diffeomorphic_guide(
                    x,
                    mode=str(getattr(self, "diffeomorphic_guide_mode", "mean")),
                    channel=int(getattr(self, "diffeomorphic_guide_channel", 2)),
                    weights=getattr(self, "diffeomorphic_guide_weights", None),
                ),
                color_strength=float(getattr(self, "diffeomorphic_color_strength", 0.85)),
                warp_strength=float(getattr(self, "diffeomorphic_warp_strength", 0.08)),
                gate_strength=float(getattr(self, "diffeomorphic_texture_gate_strength", 8.0)),
                normal_leak=float(getattr(self, "diffeomorphic_normal_leak", 0.0)),
                color_lowpass_kernel=int(getattr(self, "diffeomorphic_color_lowpass_kernel", 1)),
                color_edge_gamma=float(getattr(self, "diffeomorphic_color_edge_gamma", 0.0)),
                head_mode=str(getattr(self, "diffeomorphic_head_mode", "standard")),
                amp_strength=float(getattr(self, "diffeomorphic_amp_strength", 0.5)),
                factorized_enable_color=bool(getattr(self, "diffeomorphic_factorized_enable_color", True)),
                factorized_enable_amp=bool(getattr(self, "diffeomorphic_factorized_enable_amp", True)),
                joint_bilateral_kernel=int(getattr(self, "diffeomorphic_joint_bilateral_kernel", 1)),
                joint_bilateral_range_sigma=float(getattr(self, "diffeomorphic_joint_bilateral_range_sigma", 0.5)),
                divergence_free_warp=bool(getattr(self, "diffeomorphic_divergence_free_warp", False)),
                metric_anchor=getattr(self, "_integration_anchor_x", None) if bool(getattr(self, "diffeomorphic_metric_mask_use_z0", False)) else x,
                metric_mask_gamma=float(getattr(self, "diffeomorphic_metric_mask_gamma", 0.0)),
                metric_mask_smooth_kernel=int(getattr(self, "diffeomorphic_metric_mask_smooth_kernel", 3)),
            )
            return stroked - x.float()
        if self.velocity_head_mode == "tanh":
            raw_delta = torch.tanh(raw_delta) * self.velocity_tanh_limit
        return raw_delta * self.latent_scale_factor * self.residual_gain

    def _resolve_t_input(self, x: torch.Tensor, t: torch.Tensor | float | None) -> torch.Tensor:
        if t is None:
            t = 1.0
        if not torch.is_tensor(t):
            return torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            return t.to(device=x.device, dtype=x.dtype).expand(x.shape[0])
        t = t.to(device=x.device, dtype=x.dtype).view(-1)
        if t.shape[0] == 1 and x.shape[0] > 1:
            return t.expand(x.shape[0])
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"time batch mismatch: expected {x.shape[0]} or 1, got {t.shape[0]}")
        return t

    def _compute_style_code(
        self,
        *,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        t: torch.Tensor,
        style_code_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if style_code_override is None:
            style_code = self.encode_style_id(style_id)
        else:
            self._last_style_token_fields = None
            style_code = style_code_override
            if style_code.ndim == 1:
                style_code = style_code.unsqueeze(0)
            style_code = style_code.to(device=x.device, dtype=x.dtype)
        if style_code.shape[0] == 1 and x.shape[0] > 1:
            style_code = style_code.expand(x.shape[0], -1)
        elif style_code.shape[0] != x.shape[0]:
            raise ValueError(f"style code batch mismatch: expected {x.shape[0]} or 1, got {style_code.shape[0]}")

        time_code = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim).to(dtype=style_code.dtype))
        return style_code + time_code

    def _resolve_integration_horizon(self, *, step_size: float, style_strength: float | None) -> float:
        strength = self._resolve_style_strength(style_strength)
        horizon = max(0.0, float(step_size)) * strength
        return max(0.0, min(1.0, horizon))

    @property
    def last_semantic_attn(self) -> torch.Tensor | None:
        for block in reversed(self.body_blocks):
            attn = getattr(block, "last_attn", None)
            if attn is not None:
                return attn
        return None

    @property
    def last_semantic_k(self) -> torch.Tensor | None:
        for block in reversed(self.body_blocks):
            k_matrix = getattr(block, "last_k", None)
            if k_matrix is not None:
                return k_matrix
        return None

    @torch.no_grad()
    def endpoint_map(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        *,
        step_size: float = 1.0,
        style_strength: float | None = None,
        style_code_override: torch.Tensor | None = None,
        target_style_latent: torch.Tensor | None = None,
        override_palette: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if style_id is None:
            raise ValueError("style_id is required for endpoint map.")
        horizon = self._resolve_integration_horizon(step_size=step_size, style_strength=style_strength)
        if horizon <= 0.0:
            return x
        t_fixed = torch.full((x.shape[0],), 1.0, device=x.device, dtype=x.dtype)
        velocity = self.forward(
            x,
            t=t_fixed,
            style_id=style_id,
            target_style_latent=target_style_latent,
            style_code_override=style_code_override,
            override_palette=override_palette,
        )
        return x + velocity * horizon

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
        style_id: torch.Tensor | int | None = None,
        step_size: float = 1.0,
        style_strength: float | None = None,
        target_style_latent: torch.Tensor | None = None,
        style_code_override: torch.Tensor | None = None,
        override_palette: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del source, step_size, style_strength
        if style_id is None and style_code_override is None:
            raise ValueError("style_id or style_code_override is required.")
        t_tensor = self._resolve_t_input(x, t)
        style_code = self._compute_style_code(
            x=x,
            style_id=style_id,
            t=t_tensor,
            style_code_override=style_code_override,
        )
        if style_id is None:
            raise ValueError("style_id is required for bridge spatial conditioning.")
        style_maps = self._prepare_style_maps(style_id=style_id)
        return self._predict_delta_from_context(
            x,
            style_code=style_code,
            style_maps=style_maps,
            override_palette=override_palette,
            strength=1.0,
            target_style_latent=target_style_latent,
        )

    @torch.no_grad()
    def integrate(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        num_steps: int = 16,
        step_size: float = 1.0,
        style_strength: float | None = None,
        target_style_latent: torch.Tensor | None = None,
        style_code_override: torch.Tensor | None = None,
        override_palette: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if style_id is None:
            raise ValueError("style_id is required for bridge integration.")
        steps = max(1, int(num_steps))
        horizon = self._resolve_integration_horizon(step_size=step_size, style_strength=style_strength)
        if horizon <= 0.0:
            return x
        x = self._apply_pre_integrate_moment_match(x, target_style_latent)
        x = self._inject_flat_highfreq_canvas(x, target_style_latent)
        dt = horizon / float(steps)
        h = x
        self._integration_anchor_x = x
        for idx in range(steps):
            t = horizon * ((idx + 0.5) / float(steps))
            velocity = self.forward(
                h,
                t=t,
                style_id=style_id,
                target_style_latent=target_style_latent,
                style_code_override=style_code_override,
                override_palette=override_palette,
            )
            h = h + velocity * dt
        self._integration_anchor_x = None
        return h

    def _apply_pre_integrate_moment_match(
        self,
        x: torch.Tensor,
        target_style_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        if (not self.pre_integrate_moment_match) or target_style_latent is None:
            return x
        ref = target_style_latent
        if ref.shape != x.shape:
            raise ValueError(
                "target_style_latent shape must match model input shape, "
                f"got x={tuple(x.shape)} ref={tuple(ref.shape)}"
            )
        ref = ref.to(device=x.device, dtype=x.dtype)
        eps = self.output_moment_match_eps
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        x_std = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
        ref_mean = ref.mean(dim=(2, 3), keepdim=True)
        ref_std = ref.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
        mapped = ((x - x_mean) / x_std) * ref_std + ref_mean
        return x.lerp(mapped, self.pre_integrate_moment_blend)

    def _inject_flat_highfreq_canvas(
        self,
        x: torch.Tensor,
        target_style_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        strength = float(getattr(self, "latent_canvas_strength", 0.0))
        if strength <= 0.0:
            return x
        ref = target_style_latent
        if ref is None or ref.shape != x.shape:
            noise = torch.randn_like(x)
        else:
            ref = ref.to(device=x.device, dtype=x.dtype)
            kernel = max(1, int(getattr(self, "latent_canvas_highpass_kernel", 5)))
            if kernel % 2 == 0:
                kernel += 1
            noise = ref - torch.nn.functional.avg_pool2d(ref.float(), kernel_size=kernel, stride=1, padding=kernel // 2).to(dtype=ref.dtype)
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        gx = torch.nn.functional.pad(gx.float(), (0, 1, 0, 0))
        gy = torch.nn.functional.pad(gy.float(), (0, 0, 0, 1))
        edge = torch.sqrt(gx.square() + gy.square() + 1e-12).mean(dim=1, keepdim=True)
        flat_mask = torch.exp(-float(getattr(self, "latent_canvas_edge_gamma", 4.0)) * edge)
        return x + noise * flat_mask.to(dtype=x.dtype) * strength


def _normalize_skip_routing_mode(config: ModelConfig) -> ModelConfig:
    model_cfg = config.validated()
    skip_mode = str(model_cfg.skip_routing_mode).strip().lower()
    if skip_mode not in {"none", "naive", "adaptive", "normalized"}:
        if bool(model_cfg.extra.get("skip_frequency_gated", True)):
            skip_mode = "normalized"
        else:
            skip_mode = "naive"
    model_cfg.skip_routing_mode = skip_mode
    return model_cfg


def build_model_from_config(
    model_cfg: ModelConfig | Mapping[str, object],
    *,
    use_checkpointing: bool = False,
) -> TimeConditionedLANCETBridge:
    config = model_cfg if isinstance(model_cfg, ModelConfig) else ModelConfig.from_mapping(model_cfg)
    config = _normalize_skip_routing_mode(config)
    config.use_checkpointing = bool(use_checkpointing)
    return TimeConditionedLANCETBridge(config)


__all__ = [
    "TimeConditionedLANCETBridge",
    "build_model_from_config",
    "count_parameters",
]
