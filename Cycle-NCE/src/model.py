from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


class AdaGN(nn.Module):
    """Adaptive GroupNorm conditioned by style code."""

    def __init__(self, dim: int, style_dim: int, num_groups: int = 8) -> None:
        super().__init__()
        groups = max(1, min(num_groups, dim))
        while dim % groups != 0 and groups > 1:
            groups -= 1

        self.norm = nn.GroupNorm(groups, dim, affine=False, eps=1e-6)
        self.proj = nn.Linear(style_dim, dim * 2)

        # Identity init: scale=1, shift=0.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:dim] = 1.0

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        h = self.norm(x)
        params = self.proj(style_code).unsqueeze(-1).unsqueeze(-1)
        scale, shift = params.chunk(2, dim=1)
        adagn = h * scale + shift

        if torch.is_tensor(gate):
            g = gate.to(device=h.device, dtype=h.dtype)
            if g.ndim == 0:
                g = g.view(1, 1, 1, 1)
            elif g.ndim == 1:
                g = g.view(-1, 1, 1, 1)
            g = torch.nan_to_num(g, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            return h + g * (adagn - h)

        g = float(gate)
        if not math.isfinite(g):
            g = 0.0
        g = max(0.0, min(1.0, g))
        return h + g * (adagn - h)


class ResBlock(nn.Module):
    def __init__(self, dim: int, style_dim: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm1 = AdaGN(dim, style_dim, num_groups=num_groups)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = AdaGN(dim, style_dim, num_groups=num_groups)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        h = self.act(self.norm1(x, style_code, gate=gate))
        h = self.conv1(h)
        h = self.act(self.norm2(h, style_code, gate=gate))
        h = self.conv2(h)
        return x + h


class LatentAdaCUT(nn.Module):
    """Compact latent-space style transfer model (style-id conditioned)."""

    def __init__(
        self,
        latent_channels: int = 4,
        num_styles: int = 3,
        style_dim: int = 256,
        base_dim: int = 64,
        lift_channels: int | None = None,
        num_hires_blocks: int = 2,
        num_res_blocks: int = 4,
        num_groups: int = 8,
        use_checkpointing: bool = False,
        latent_scale_factor: float = 0.18215,
        residual_gain: float = 0.1,
        style_spatial_pre_gain_16: float = 0.35,  # kept for config compatibility
        use_decoder_adagn: bool = True,
        inject_gate_hires: float = 0.0,
        inject_gate_body: float = 1.0,
        inject_gate_decoder: float = 1.0,
        style_strength_default: float = 1.0,
        style_strength_step_curve: str = "linear",
        upsample_mode: str = "nearest",
        style_id_spatial_jitter_px: int = 0,  # kept for config compatibility
        loss_projector_use: bool = False,  # kept for config compatibility
        loss_projector_channels: int = 64,  # kept for config compatibility
        upsample_blur: bool = True,
        upsample_blur_kernel: str = "box3",
    ) -> None:
        super().__init__()
        del style_spatial_pre_gain_16, style_id_spatial_jitter_px, loss_projector_use, loss_projector_channels

        self.latent_channels = int(latent_channels)
        self.num_styles = int(num_styles)
        self.use_checkpointing = bool(use_checkpointing)
        self.latent_scale_factor = float(latent_scale_factor)
        self.residual_gain = float(residual_gain)

        self.lift_channels = int(lift_channels) if lift_channels is not None else int(base_dim)
        self.body_channels = int(base_dim * 2)

        self.inject_gate_hires = max(0.0, min(1.0, float(inject_gate_hires)))
        self.inject_gate_body = max(0.0, min(1.0, float(inject_gate_body)))
        self.inject_gate_decoder = max(0.0, min(1.0, float(inject_gate_decoder)))

        self.style_strength_default = max(0.0, min(1.0, float(style_strength_default)))
        self.style_strength_step_curve = str(style_strength_step_curve).lower()
        if self.style_strength_step_curve not in {"linear", "smoothstep", "sqrt"}:
            self.style_strength_step_curve = "linear"

        self.upsample_mode = str(upsample_mode)
        self.upsample_blur = bool(upsample_blur)
        self.upsample_blur_kernel = str(upsample_blur_kernel).lower()
        if self.upsample_blur_kernel not in {"box3", "gaussian3"}:
            self.upsample_blur_kernel = "box3"

        self.style_emb = nn.Embedding(self.num_styles, style_dim)
        nn.init.normal_(self.style_emb.weight, mean=0.0, std=0.02)

        self.enc_in = nn.Conv2d(latent_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        self.enc_in_act = nn.SiLU()

        self.hires_body = nn.ModuleList(
            [ResBlock(self.lift_channels, style_dim, num_groups=num_groups) for _ in range(max(0, int(num_hires_blocks)))]
        )
        self.down = nn.Conv2d(self.lift_channels, self.body_channels, kernel_size=4, stride=2, padding=1)

        self.body = nn.ModuleList(
            [ResBlock(self.body_channels, style_dim, num_groups=num_groups) for _ in range(max(1, int(num_res_blocks)))]
        )

        out_groups = max(1, min(num_groups, self.lift_channels))
        while self.lift_channels % out_groups != 0 and out_groups > 1:
            out_groups -= 1

        upsample_kwargs = {"scale_factor": 2, "mode": self.upsample_mode}
        if self.upsample_mode in {"bilinear", "bicubic"}:
            upsample_kwargs["align_corners"] = False
        self.dec_up = nn.Upsample(**upsample_kwargs)
        self.dec_conv = nn.Conv2d(self.body_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        self.dec_norm = AdaGN(self.lift_channels, style_dim, num_groups=out_groups) if use_decoder_adagn else nn.GroupNorm(
            out_groups, self.lift_channels, eps=1e-6
        )
        self.dec_act = nn.SiLU()
        self.dec_out = nn.Conv2d(self.lift_channels, latent_channels, kernel_size=3, stride=1, padding=1)

        if self.upsample_blur:
            if self.upsample_blur_kernel == "gaussian3":
                k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32) / 16.0
            else:
                k = torch.ones((3, 3), dtype=torch.float32) / 9.0
            self.register_buffer("_upsample_blur_kernel", k.view(1, 1, 3, 3), persistent=False)
        else:
            self.register_buffer("_upsample_blur_kernel", torch.empty(0), persistent=False)

    def _normalize_style_id_input(self, style_id: torch.Tensor | int | None, *, batch: int, device: torch.device) -> torch.Tensor:
        if style_id is None:
            return torch.zeros(batch, device=device, dtype=torch.long)
        if isinstance(style_id, int):
            style_id = torch.tensor([style_id], device=device, dtype=torch.long)
        style_id = style_id.long().view(-1)
        if style_id.device != device:
            style_id = style_id.to(device)
        if style_id.numel() == 1 and batch > 1:
            style_id = style_id.expand(batch)
        return style_id.clamp_min(0).clamp_max(max(1, self.num_styles) - 1)

    def _run_block(self, block: ResBlock, h: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            gate_in = gate if torch.is_tensor(gate) else h.new_tensor(float(gate))
            return ckpt.checkpoint(lambda _h, _s, _g, _blk=block: _blk(_h, _s, _g), h, style_code, gate_in, use_reentrant=False)
        return block(h, style_code, gate=gate)

    def _run_style_blocks(self, h: torch.Tensor, blocks: nn.ModuleList, style_code: torch.Tensor, gate: float | torch.Tensor) -> torch.Tensor:
        for block in blocks:
            h = self._run_block(block, h, style_code, gate=gate)
        return h

    def _resolve_style_strength(self, style_strength: float | None) -> float:
        if style_strength is None:
            return self.style_strength_default
        return max(0.0, min(1.0, float(style_strength)))

    def _style_strength_step_scale(self, style_strength: float) -> float:
        s = max(0.0, min(1.0, float(style_strength)))
        if self.style_strength_step_curve == "sqrt":
            return math.sqrt(s)
        if self.style_strength_step_curve == "smoothstep":
            return s * s * (3.0 - 2.0 * s)
        return s

    def _apply_upsample_blur(self, h: torch.Tensor) -> torch.Tensor:
        if (not self.upsample_blur) or self._upsample_blur_kernel.numel() == 0:
            return h
        c = h.shape[1]
        k = self._upsample_blur_kernel.to(device=h.device, dtype=h.dtype).repeat(c, 1, 1, 1)
        return F.conv2d(h, k, stride=1, padding=1, groups=c)

    def _compute_delta(self, h: torch.Tensor) -> torch.Tensor:
        out = self.dec_out(h)
        if self.residual_gain != 1.0:
            out = out * self.residual_gain
        if self.latent_scale_factor != 0.0:
            out = out * self.latent_scale_factor
        return out

    def _predict_delta(self, x: torch.Tensor, style_id: torch.Tensor | int | None, style_strength: float | None) -> torch.Tensor:
        strength = self._resolve_style_strength(style_strength)
        style_ids = self._normalize_style_id_input(style_id, batch=x.shape[0], device=x.device)
        style_code = self.style_emb(style_ids)

        gate_hires = self.inject_gate_hires * strength
        gate_body = self.inject_gate_body * strength
        gate_decoder = self.inject_gate_decoder * strength

        feat = x / max(self.latent_scale_factor, 1e-8)
        h = self.enc_in_act(self.enc_in(feat))
        h = self._run_style_blocks(h, self.hires_body, style_code, gate=gate_hires)
        h = self.down(h)
        h = self._run_style_blocks(h, self.body, style_code, gate=gate_body)

        h = self.dec_up(h)
        h = self._apply_upsample_blur(h)
        h = self.dec_conv(h)
        if isinstance(self.dec_norm, AdaGN):
            h = self.dec_norm(h, style_code, gate=gate_decoder)
        else:
            h = self.dec_norm(h)
        h = self.dec_act(h)
        return self._compute_delta(h)

    def integrate(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
        style_mix_alpha: float | torch.Tensor | None = None,
        num_steps: int = 1,
        step_size: float = 1.0,
        style_strength: float | None = None,
    ) -> torch.Tensor:
        del style_ref, style_mix_alpha
        steps = max(1, int(num_steps))
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        per_step = 1.0 / float(steps)

        h = x
        for _ in range(steps):
            delta = self._predict_delta(h, style_id=style_id, style_strength=strength)
            h = h + delta * float(step_size) * step_scale * per_step
        return h

    def forward(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
        style_mix_alpha: float | torch.Tensor | None = None,
        step_size: float = 1.0,
        style_strength: float | None = None,
    ) -> torch.Tensor:
        del style_ref, style_mix_alpha
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        delta = self._predict_delta(x, style_id=style_id, style_strength=strength)
        return x + delta * float(step_size) * step_scale


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model_from_config(model_cfg: dict, *, use_checkpointing: bool = False) -> LatentAdaCUT:
    known_keys = {
        "latent_channels",
        "num_styles",
        "style_dim",
        "base_dim",
        "lift_channels",
        "num_hires_blocks",
        "num_res_blocks",
        "num_groups",
        "latent_scale_factor",
        "residual_gain",
        "style_spatial_pre_gain_16",
        "use_decoder_adagn",
        "inject_gate_hires",
        "inject_gate_body",
        "inject_gate_decoder",
        "style_strength_default",
        "style_strength_step_curve",
        "upsample_mode",
        "style_id_spatial_jitter_px",
        "loss_projector_use",
        "loss_projector_channels",
        "upsample_blur",
        "upsample_blur_kernel",
    }
    unknown_keys = sorted(
        k
        for k in model_cfg.keys()
        if (k not in known_keys) and (not str(k).startswith("__comment"))
    )
    if unknown_keys:
        warnings.warn("Unknown model config key(s): " + ", ".join(unknown_keys), category=UserWarning, stacklevel=2)

    return LatentAdaCUT(
        latent_channels=int(model_cfg.get("latent_channels", 4)),
        num_styles=int(model_cfg.get("num_styles", 3)),
        style_dim=int(model_cfg.get("style_dim", 256)),
        base_dim=int(model_cfg.get("base_dim", 64)),
        lift_channels=int(model_cfg.get("lift_channels", model_cfg.get("base_dim", 64))),
        num_hires_blocks=int(model_cfg.get("num_hires_blocks", 2)),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 4)),
        num_groups=int(model_cfg.get("num_groups", 8)),
        use_checkpointing=bool(use_checkpointing),
        latent_scale_factor=float(model_cfg.get("latent_scale_factor", 0.18215)),
        residual_gain=float(model_cfg.get("residual_gain", 0.1)),
        style_spatial_pre_gain_16=float(model_cfg.get("style_spatial_pre_gain_16", 0.35)),
        use_decoder_adagn=bool(model_cfg.get("use_decoder_adagn", True)),
        inject_gate_hires=float(model_cfg.get("inject_gate_hires", 0.0)),
        inject_gate_body=float(model_cfg.get("inject_gate_body", 1.0)),
        inject_gate_decoder=float(model_cfg.get("inject_gate_decoder", 1.0)),
        style_strength_default=float(model_cfg.get("style_strength_default", 1.0)),
        style_strength_step_curve=str(model_cfg.get("style_strength_step_curve", "linear")),
        upsample_mode=str(model_cfg.get("upsample_mode", "nearest")),
        style_id_spatial_jitter_px=int(model_cfg.get("style_id_spatial_jitter_px", 0)),
        loss_projector_use=bool(model_cfg.get("loss_projector_use", False)),
        loss_projector_channels=int(model_cfg.get("loss_projector_channels", 64)),
        upsample_blur=bool(model_cfg.get("upsample_blur", True)),
        upsample_blur_kernel=str(model_cfg.get("upsample_blur_kernel", "box3")),
    )
