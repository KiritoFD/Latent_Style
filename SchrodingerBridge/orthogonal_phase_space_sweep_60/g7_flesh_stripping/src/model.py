from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import warnings

from lancet_backbone import LatentAdaCUT


_MODEL_CONFIG_DEFAULTS = {
    "latent_channels": 4,
    "num_styles": 5,
    "style_dim": 160,
    "time_dim": 256,
    "base_dim": 96,
    "lift_channels": 128,
    "num_hires_blocks": 2,
    "num_res_blocks": 2,
    "num_decoder_blocks": 2,
    "num_groups": 4,
    "latent_scale_factor": 0.18215,
    "residual_gain": 1.0,
    "style_spatial_pre_gain_16": 0.35,
    "style_strength_default": 1.0,
    "style_strength_step_curve": "linear",
    "upsample_mode": "nearest",
    "style_id_spatial_jitter_px": 0,
    "upsample_blur": True,
    "upsample_blur_kernel": "box3",
    "style_attn_num_tokens": 64,
    "style_attn_num_heads": 4,
    "style_attn_sharpen_scale": 2.5,
    "style_attn_temperature": 0.08,
    "hires_block_type": "conv",
    "body_block_type": "global_attn",
    "decoder_block_type": "conv",
    "semantic_attn_temperature": 0.08,
    "feature_attn_num_heads": 4,
    "window_attn_window_size": 8,
    "skip_fusion_mode": "add_proj",
    "skip_routing_mode": "none",
    "skip_naive_gain": 1.0,
    "style_skip_content_retention_boost": 0.0,
    "input_anchor_noise_std": 0.0,
    "input_anchor_noise_eval": False,
    "ablation_skip_clean": True,
    "ablation_skip_blur": True,
    "ablation_no_residual": False,
    "ablation_no_residual_gain": 1.0,
    "ablation_disable_spatial_prior": False,
    "output_moment_match": False,
    "output_moment_match_eps": 1e-6,
    "output_moment_match_train_only": False,
    "use_style_blender": False,
}


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
    """
    Reuses the LANCET feature backbone but reinterprets it as a time-conditioned
    vector-field estimator v_theta(z_t, t, style).
    """

    def __init__(
        self,
        *,
        time_dim: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.time_dim = int(time_dim)
        self.bridge_style_dim = int(self.style_emb.embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.bridge_style_dim),
            nn.SiLU(),
            nn.Linear(self.bridge_style_dim, self.bridge_style_dim),
        )

    def _compute_delta(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        # In bridge mode, the backbone predicts the instantaneous velocity field
        # directly instead of a bounded residual-to-anchor delta.
        return self.dec_out(h) * self.latent_scale_factor * self.residual_gain

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

    def _resolve_integration_horizon(
        self,
        *,
        step_size: float,
        style_strength: float | None,
    ) -> float:
        # In bridge mode, "style_strength" is interpreted as how far along the
        # learned [0, 1] probability path we integrate, instead of heuristically
        # scaling the style embedding. This keeps inference aligned with the ODE
        # semantics learned during flow matching.
        strength = self._resolve_style_strength(style_strength)
        horizon = max(0.0, float(step_size)) * strength
        return max(0.0, min(1.0, horizon))

    @property
    def last_semantic_attn(self) -> torch.Tensor | None:
        """
        Extract the semantic routing table from the deepest semantic cross-attn block.
        Returns a tensor shaped [B, HW_query, HW_key] when available.
        """
        for block in reversed(self.body_blocks):
            attn = getattr(block, "last_attn", None)
            if attn is not None:
                return attn
        return None

    @property
    def last_semantic_k(self) -> torch.Tensor | None:
        """
        Extract the normalized semantic key matrix from the deepest cross-attn block.
        Returns a tensor shaped [B, C, HW_key] when available.
        """
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
            style_code_override=style_code_override,
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
        del source
        del step_size
        del target_style_latent
        del override_palette
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
            override_palette=None,
            strength=1.0,
            target_style_latent=None,
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
        del target_style_latent
        del override_palette
        if style_id is None:
            raise ValueError("style_id is required for bridge integration.")
        steps = max(1, int(num_steps))
        horizon = self._resolve_integration_horizon(step_size=step_size, style_strength=style_strength)
        if horizon <= 0.0:
            return x
        dt = horizon / float(steps)
        h = x
        for idx in range(steps):
            t = horizon * ((idx + 0.5) / float(steps))
            velocity = self.forward(
                h,
                t=t,
                style_id=style_id,
                style_code_override=style_code_override,
            )
            h = h + velocity * dt
        return h


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model_from_config(
    model_cfg: dict,
    *,
    use_checkpointing: bool = False,
) -> TimeConditionedLANCETBridge:
    unknown_keys = sorted(k for k in model_cfg.keys() if k not in _MODEL_CONFIG_DEFAULTS)
    if unknown_keys:
        warnings.warn(
            "Unknown model config key(s): " + ", ".join(unknown_keys),
            category=UserWarning,
            stacklevel=2,
        )
    return TimeConditionedLANCETBridge(
        latent_channels=int(model_cfg.get("latent_channels", _MODEL_CONFIG_DEFAULTS["latent_channels"])),
        num_styles=int(model_cfg.get("num_styles", _MODEL_CONFIG_DEFAULTS["num_styles"])),
        style_dim=int(model_cfg.get("style_dim", _MODEL_CONFIG_DEFAULTS["style_dim"])),
        base_dim=int(model_cfg.get("base_dim", _MODEL_CONFIG_DEFAULTS["base_dim"])),
        lift_channels=int(model_cfg.get("lift_channels", _MODEL_CONFIG_DEFAULTS["lift_channels"])),
        num_hires_blocks=int(model_cfg.get("num_hires_blocks", _MODEL_CONFIG_DEFAULTS["num_hires_blocks"])),
        num_res_blocks=int(model_cfg.get("num_res_blocks", _MODEL_CONFIG_DEFAULTS["num_res_blocks"])),
        num_decoder_blocks=int(model_cfg.get("num_decoder_blocks", _MODEL_CONFIG_DEFAULTS["num_decoder_blocks"])),
        num_groups=int(model_cfg.get("num_groups", _MODEL_CONFIG_DEFAULTS["num_groups"])),
        use_checkpointing=bool(use_checkpointing),
        latent_scale_factor=float(model_cfg.get("latent_scale_factor", _MODEL_CONFIG_DEFAULTS["latent_scale_factor"])),
        residual_gain=float(model_cfg.get("residual_gain", _MODEL_CONFIG_DEFAULTS["residual_gain"])),
        style_spatial_pre_gain_16=float(model_cfg.get("style_spatial_pre_gain_16", _MODEL_CONFIG_DEFAULTS["style_spatial_pre_gain_16"])),
        style_strength_default=float(model_cfg.get("style_strength_default", _MODEL_CONFIG_DEFAULTS["style_strength_default"])),
        style_strength_step_curve=str(model_cfg.get("style_strength_step_curve", _MODEL_CONFIG_DEFAULTS["style_strength_step_curve"])),
        upsample_mode=str(model_cfg.get("upsample_mode", _MODEL_CONFIG_DEFAULTS["upsample_mode"])),
        style_id_spatial_jitter_px=int(model_cfg.get("style_id_spatial_jitter_px", _MODEL_CONFIG_DEFAULTS["style_id_spatial_jitter_px"])),
        upsample_blur=bool(model_cfg.get("upsample_blur", _MODEL_CONFIG_DEFAULTS["upsample_blur"])),
        upsample_blur_kernel=str(model_cfg.get("upsample_blur_kernel", _MODEL_CONFIG_DEFAULTS["upsample_blur_kernel"])),
        style_attn_num_tokens=int(model_cfg.get("style_attn_num_tokens", _MODEL_CONFIG_DEFAULTS["style_attn_num_tokens"])),
        style_attn_num_heads=int(model_cfg.get("style_attn_num_heads", _MODEL_CONFIG_DEFAULTS["style_attn_num_heads"])),
        style_attn_sharpen_scale=float(model_cfg.get("style_attn_sharpen_scale", _MODEL_CONFIG_DEFAULTS["style_attn_sharpen_scale"])),
        style_attn_temperature=float(model_cfg.get("style_attn_temperature", _MODEL_CONFIG_DEFAULTS["style_attn_temperature"])),
        hires_block_type=str(model_cfg.get("hires_block_type", _MODEL_CONFIG_DEFAULTS["hires_block_type"])),
        body_block_type=str(model_cfg.get("body_block_type", _MODEL_CONFIG_DEFAULTS["body_block_type"])),
        decoder_block_type=str(model_cfg.get("decoder_block_type", _MODEL_CONFIG_DEFAULTS["decoder_block_type"])),
        semantic_attn_temperature=float(model_cfg.get("semantic_attn_temperature", _MODEL_CONFIG_DEFAULTS["semantic_attn_temperature"])),
        feature_attn_num_heads=int(model_cfg.get("feature_attn_num_heads", _MODEL_CONFIG_DEFAULTS["feature_attn_num_heads"])),
        window_attn_window_size=int(model_cfg.get("window_attn_window_size", _MODEL_CONFIG_DEFAULTS["window_attn_window_size"])),
        skip_fusion_mode=str(model_cfg.get("skip_fusion_mode", _MODEL_CONFIG_DEFAULTS["skip_fusion_mode"])),
        skip_routing_mode=str(model_cfg.get("skip_routing_mode", _MODEL_CONFIG_DEFAULTS["skip_routing_mode"])),
        skip_naive_gain=float(model_cfg.get("skip_naive_gain", _MODEL_CONFIG_DEFAULTS["skip_naive_gain"])),
        style_skip_content_retention_boost=float(model_cfg.get("style_skip_content_retention_boost", _MODEL_CONFIG_DEFAULTS["style_skip_content_retention_boost"])),
        input_anchor_noise_std=float(model_cfg.get("input_anchor_noise_std", _MODEL_CONFIG_DEFAULTS["input_anchor_noise_std"])),
        input_anchor_noise_eval=bool(model_cfg.get("input_anchor_noise_eval", _MODEL_CONFIG_DEFAULTS["input_anchor_noise_eval"])),
        ablation_skip_clean=bool(model_cfg.get("ablation_skip_clean", _MODEL_CONFIG_DEFAULTS["ablation_skip_clean"])),
        ablation_skip_blur=bool(model_cfg.get("ablation_skip_blur", _MODEL_CONFIG_DEFAULTS["ablation_skip_blur"])),
        ablation_no_residual=bool(model_cfg.get("ablation_no_residual", _MODEL_CONFIG_DEFAULTS["ablation_no_residual"])),
        ablation_no_residual_gain=float(model_cfg.get("ablation_no_residual_gain", _MODEL_CONFIG_DEFAULTS["ablation_no_residual_gain"])),
        ablation_disable_spatial_prior=bool(model_cfg.get("ablation_disable_spatial_prior", _MODEL_CONFIG_DEFAULTS["ablation_disable_spatial_prior"])),
        output_moment_match=bool(model_cfg.get("output_moment_match", _MODEL_CONFIG_DEFAULTS["output_moment_match"])),
        output_moment_match_eps=float(model_cfg.get("output_moment_match_eps", _MODEL_CONFIG_DEFAULTS["output_moment_match_eps"])),
        output_moment_match_train_only=bool(model_cfg.get("output_moment_match_train_only", _MODEL_CONFIG_DEFAULTS["output_moment_match_train_only"])),
        use_style_blender=bool(model_cfg.get("use_style_blender", _MODEL_CONFIG_DEFAULTS["use_style_blender"])),
        time_dim=int(model_cfg.get("time_dim", _MODEL_CONFIG_DEFAULTS["time_dim"])),
    )
