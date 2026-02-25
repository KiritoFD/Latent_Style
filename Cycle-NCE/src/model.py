from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


class SpatiallyAdaptiveAdaMixGN(nn.Module):
    def __init__(self, dim: int, style_dim: int, num_groups: int = 4, rank: int = 16) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, dim, affine=False)
        self.style_shift = nn.Linear(style_dim, dim)
        nn.init.zeros_(self.style_shift.weight)
        nn.init.zeros_(self.style_shift.bias)
        self.rank = int(rank)
        self.style_U = nn.Linear(style_dim, dim * self.rank)
        self.style_V = nn.Linear(style_dim, self.rank * dim)
        nn.init.normal_(self.style_U.weight, std=0.01)
        nn.init.zeros_(self.style_U.bias)
        # Safety lock: keep W = U @ V exactly zero at initialization.
        nn.init.zeros_(self.style_V.weight)
        nn.init.zeros_(self.style_V.bias)
        hidden_dim = max(32, dim // 2)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.spatial_gate[-2].bias, 0.0)

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        b, c, h_dim, w_dim = x.shape
        normalized = self.norm(x)
        shift = self.style_shift(style_code).view(b, c, 1, 1)
        u = self.style_U(style_code).view(b, c, self.rank)
        v = self.style_V(style_code).view(b, self.rank, c)
        w = torch.bmm(u, v)
        norm_flat = normalized.view(b, c, h_dim * w_dim)
        mixed_flat = torch.bmm(w, norm_flat)
        mixed = mixed_flat.view(b, c, h_dim, w_dim)
        mask = self.spatial_gate(x)
        adagn = normalized + mixed + shift
        final_gate = mask * gate if isinstance(gate, float) else mask * gate.to(dtype=normalized.dtype)
        return normalized + final_gate * (adagn - normalized)

# Backward compatibility for older checkpoints/code paths.
SpatiallyAdaptiveAdaGN = SpatiallyAdaptiveAdaMixGN


class ResBlock(nn.Module):
    def __init__(self, dim: int, style_dim: int, num_groups: int = 8, ada_mix_rank: int = 16) -> None:
        super().__init__()
        self.norm1 = SpatiallyAdaptiveAdaMixGN(dim, style_dim, num_groups=num_groups, rank=ada_mix_rank)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = SpatiallyAdaptiveAdaMixGN(dim, style_dim, num_groups=num_groups, rank=ada_mix_rank)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        h = self.act(self.norm1(x, style_code, gate=gate))
        h = self.conv1(h)
        h = self.act(self.norm2(h, style_code, gate=gate))
        h = self.conv2(h)
        return x + h


@dataclass
class StyleMaps:
    map_16: torch.Tensor | None = None


class LatentAdaCUT(nn.Module):
    """
    Micro U-Net with flexible high-resolution skip fusion.
    Input/Output latent shape: [B, 4, 32, 32]
    """

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
        style_spatial_pre_gain_16: float = 0.35,
        use_decoder_adagn: bool = True,
        inject_gate_hires: float = 0.0,
        inject_gate_body: float = 1.0,
        inject_gate_decoder: float = 1.0,
        style_strength_default: float = 1.0,
        style_strength_step_curve: str = "linear",
        upsample_mode: str = "nearest",
        style_id_spatial_jitter_px: int = 0,
        upsample_blur: bool = True,
        upsample_blur_kernel: str = "box3",
        ada_mix_rank: int = 16,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.num_styles = int(num_styles)
        self.use_checkpointing = bool(use_checkpointing)
        self.latent_scale_factor = float(latent_scale_factor)
        self.residual_gain = float(residual_gain)
        self.lift_channels = int(lift_channels) if lift_channels is not None else int(base_dim)
        self.body_channels = int(base_dim * 2)
        self.style_spatial_pre_gain_16 = float(style_spatial_pre_gain_16)
        # Pruned paths: keep only pre-inject + main AdaGN style route.
        self.use_decoder_adagn = bool(use_decoder_adagn)
        self.inject_gate_hires = max(0.0, min(1.0, float(inject_gate_hires)))
        self.inject_gate_body = max(0.0, min(1.0, float(inject_gate_body)))
        self.inject_gate_decoder = max(0.0, min(1.0, float(inject_gate_decoder)))
        self.style_strength_default = max(0.0, min(1.0, float(style_strength_default)))
        self.style_strength_step_curve = str(style_strength_step_curve).lower()
        if self.style_strength_step_curve not in {"linear", "smoothstep", "sqrt"}:
            self.style_strength_step_curve = "linear"
        self.upsample_mode = str(upsample_mode)
        self.style_id_spatial_jitter_px = max(0, int(style_id_spatial_jitter_px))
        self.upsample_blur = bool(upsample_blur)
        self.upsample_blur_kernel = str(upsample_blur_kernel).lower()
        self.ada_mix_rank = max(1, int(ada_mix_rank))
        if self.upsample_blur_kernel not in {"box3", "gaussian3"}:
            self.upsample_blur_kernel = "box3"

        self.style_emb = nn.Embedding(self.num_styles, style_dim)
        nn.init.normal_(self.style_emb.weight, mean=0.0, std=0.02)

        # Learnable style-id spatial priors for inference without reference image.
        self.style_spatial_id_16 = nn.Parameter(torch.zeros(self.num_styles, self.body_channels, 16, 16))
        nn.init.normal_(self.style_spatial_id_16, mean=0.0, std=0.02)

        # 32x32 lift stage before downsampling.
        self.enc_in = nn.Conv2d(latent_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        self.enc_in_act = nn.SiLU()
        self.hires_body = nn.ModuleList(
            [
                ResBlock(self.lift_channels, style_dim, num_groups=num_groups, ada_mix_rank=self.ada_mix_rank)
                for _ in range(max(0, int(num_hires_blocks)))
            ]
        )
        self.down = nn.Conv2d(self.lift_channels, self.body_channels, kernel_size=4, stride=2, padding=1)

        self.body = nn.ModuleList(
            [
                ResBlock(self.body_channels, style_dim, num_groups=num_groups, ada_mix_rank=self.ada_mix_rank)
                for _ in range(num_res_blocks)
            ]
        )

        out_groups = max(1, min(num_groups, self.lift_channels))
        while self.lift_channels % out_groups != 0 and out_groups > 1:
            out_groups -= 1

        # Decoder: 16 -> 32
        upsample_kwargs = {"scale_factor": 2, "mode": self.upsample_mode}
        if self.upsample_mode in {"bilinear", "bicubic"}:
            upsample_kwargs["align_corners"] = False
        self.dec_up = nn.Upsample(**upsample_kwargs)
        self.skip_fusion = nn.Sequential(
            nn.Conv2d(self.body_channels + self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.dec_conv = nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        if self.use_decoder_adagn:
            self.dec_norm = SpatiallyAdaptiveAdaMixGN(
                self.lift_channels,
                style_dim,
                num_groups=out_groups,
                rank=self.ada_mix_rank,
            )
        else:
            self.dec_norm = nn.GroupNorm(out_groups, self.lift_channels, eps=1e-6)
        self.dec_act = nn.SiLU()
        self.dec_out = nn.Conv2d(self.lift_channels, latent_channels, kernel_size=3, stride=1, padding=1)

        if self.upsample_blur:
            if self.upsample_blur_kernel == "gaussian3":
                k = torch.tensor(
                    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                    dtype=torch.float32,
                ) / 16.0
            else:
                k = torch.ones((3, 3), dtype=torch.float32) / 9.0
            self.register_buffer("_upsample_blur_kernel", k.view(1, 1, 3, 3), persistent=False)
        else:
            self.register_buffer("_upsample_blur_kernel", torch.empty(0), persistent=False)

    def _normalize_style_id_input(
        self,
        style_id: torch.Tensor | int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(style_id, int):
            style_id = torch.tensor([style_id], device=device, dtype=torch.long)
        style_id = style_id.long().view(-1)
        if style_id.device != device:
            style_id = style_id.to(device)
        return style_id.clamp_min(0).clamp_max(max(1, self.num_styles) - 1)

    def _run_block(
        self,
        block: ResBlock,
        h: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            gate_in = gate.to(device=h.device, dtype=h.dtype) if torch.is_tensor(gate) else h.new_tensor(float(gate))
            return ckpt.checkpoint(
                lambda _h, _s, _g, _blk=block: _blk(_h, _s, _g),
                h,
                style_code,
                gate_in,
                use_reentrant=False,
            )
        return block(h, style_code, gate=gate)

    def _run_style_blocks(
        self,
        h: torch.Tensor,
        blocks: nn.ModuleList,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        out = h
        for block in blocks:
            out = self._run_block(block, out, style_code, gate=gate)
        return out

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

    def _decode_core(
        self,
        h: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        h = self.dec_conv(h)
        if self.use_decoder_adagn:
            h = self.dec_norm(h, style_code, gate=gate)
        else:
            h = self.dec_norm(h)
        h = self.dec_act(h)
        return h

    def _run_decoder(
        self,
        h: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            gate_in = gate.to(device=h.device, dtype=h.dtype) if torch.is_tensor(gate) else h.new_tensor(float(gate))
            return ckpt.checkpoint(
                lambda _h, _s, _g: self._decode_core(_h, _s, _g),
                h,
                style_code,
                gate_in,
                use_reentrant=False,
            )
        return self._decode_core(h, style_code, gate=gate)

    def _prepare_style_maps(
        self,
        style_id: torch.Tensor | int,
    ) -> StyleMaps:
        return StyleMaps(
            map_16=self.encode_style_spatial_id(style_id).get(16),
        )

    def _prepare_spatial_map(self, style_map: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
        return self._match_style_map(style_map, target)

    def _prepare_style_context(
        self,
        *,
        style_id: torch.Tensor | int,
    ) -> tuple[torch.Tensor, StyleMaps]:
        style_code = self.encode_style_id(style_id)
        style_maps = self._prepare_style_maps(style_id=style_id)
        return style_code, style_maps

    def _apply_upsample_blur(self, h: torch.Tensor) -> torch.Tensor:
        if not self.upsample_blur or self._upsample_blur_kernel.numel() == 0:
            return h
        b, c, _, _ = h.shape
        if c <= 0 or b <= 0:
            return h
        kernel = self._upsample_blur_kernel.to(device=h.device, dtype=torch.float32).repeat(c, 1, 1, 1).contiguous()
        h_dtype = h.dtype
        if h.device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=False):
                out = F.conv2d(h.float(), kernel, stride=1, padding=1, groups=c)
        else:
            out = F.conv2d(h.float(), kernel, stride=1, padding=1, groups=c)
        return out.to(dtype=h_dtype)

    def _compute_delta(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.dec_out(h) * self.latent_scale_factor * self.residual_gain
        return delta


    def encode_style_id(self, style_id: torch.Tensor | int | None) -> torch.Tensor:
        if style_id is None:
            raise ValueError("style_id is required.")
        emb_device = self.style_emb.weight.device
        style_id = self._normalize_style_id_input(style_id, device=emb_device)
        return self.style_emb(style_id)

    @staticmethod
    def _normalize_style_map(feat: torch.Tensor) -> torch.Tensor:
        feat = feat - feat.mean(dim=(2, 3), keepdim=True)
        return feat / (feat.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)

    def encode_style_spatial_id(self, style_id: torch.Tensor | int) -> dict[int, torch.Tensor]:
        spatial_device = self.style_spatial_id_16.device
        style_id = self._normalize_style_id_input(style_id, device=spatial_device)
        maps = {16: self.style_spatial_id_16.index_select(0, style_id)}
        if self.training and self.style_id_spatial_jitter_px > 0:
            max_jit = self.style_id_spatial_jitter_px
            shifts_y = torch.randint(
                low=-max_jit,
                high=max_jit + 1,
                size=(style_id.shape[0],),
                device=style_id.device,
            )
            shifts_x = torch.randint(
                low=-max_jit,
                high=max_jit + 1,
                size=(style_id.shape[0],),
                device=style_id.device,
            )

            def _jitter_batch(feat: torch.Tensor) -> torch.Tensor:
                if max_jit <= 0:
                    return feat
                padded = F.pad(feat, (max_jit, max_jit, max_jit, max_jit), mode="reflect")
                b, c, _, wp = padded.shape
                h, w = feat.shape[-2], feat.shape[-1]
                # Fully tensorized crop with per-sample offsets to keep torch.compile graph intact.
                y_idx = (
                    torch.arange(h, device=feat.device, dtype=torch.long).view(1, h, 1)
                    + (max_jit + shifts_y).view(-1, 1, 1)
                )
                x_idx = (
                    torch.arange(w, device=feat.device, dtype=torch.long).view(1, 1, w)
                    + (max_jit + shifts_x).view(-1, 1, 1)
                )
                y_gather = y_idx.unsqueeze(1).expand(b, c, h, wp)
                cropped_h = padded.gather(dim=2, index=y_gather)
                x_gather = x_idx.unsqueeze(1).expand(b, c, h, w)
                return cropped_h.gather(dim=3, index=x_gather)

            maps[16] = _jitter_batch(maps[16])
        maps[16] = self._normalize_style_map(maps[16])
        return maps

    @staticmethod
    def _match_style_map(style_map: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
        if style_map is None:
            return None
        if style_map.shape[-2:] != target.shape[-2:]:
            style_map = F.interpolate(
                style_map,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if style_map.dtype != target.dtype:
            style_map = style_map.to(dtype=target.dtype)
        if style_map.device != target.device:
            style_map = style_map.to(device=target.device)
        return style_map

    def _predict_delta_from_context(
        self,
        x: torch.Tensor,
        *,
        style_code: torch.Tensor,
        style_maps: StyleMaps,
        strength: float,
    ) -> torch.Tensor:
        gate_hires = self.inject_gate_hires * strength
        gate_body = self.inject_gate_body * strength
        gate_decoder = self.inject_gate_decoder * strength

        feat = x / max(self.latent_scale_factor, 1e-8)
        h = self.enc_in_act(self.enc_in(feat))
        h = self._run_style_blocks(
            h,
            blocks=self.hires_body,
            style_code=style_code,
            gate=gate_hires,
        )
        # Preserve shallow structure features for flexible decoder-side fusion.
        skip_32 = h

        h = self.down(h)
        style_spatial_16 = self._prepare_spatial_map(style_maps.map_16, h)
        if style_spatial_16 is not None:
            h = h + (self.style_spatial_pre_gain_16 * strength) * torch.tanh(style_spatial_16)
        h = self._run_style_blocks(
            h,
            blocks=self.body,
            style_code=style_code,
            gate=gate_body,
        )

        h = self.dec_up(h)
        h = self._apply_upsample_blur(h)
        h = self.skip_fusion(torch.cat([h, skip_32], dim=1))
        h = self._run_decoder(
            h,
            style_code=style_code,
            gate=gate_decoder,
        )
        return self._compute_delta(h)

    def _predict_delta(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int,
        style_strength: float | None = None,
    ) -> torch.Tensor:
        strength = self._resolve_style_strength(style_strength)
        style_code, style_maps = self._prepare_style_context(
            style_id=style_id,
        )
        return self._predict_delta_from_context(
            x,
            style_code=style_code,
            style_maps=style_maps,
            strength=strength,
        )

    def integrate(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int,
        num_steps: int = 1,
        step_size: float = 1.0,
        style_strength: float | None = None,
    ) -> torch.Tensor:
        steps = max(1, int(num_steps))
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        per_step = 1.0 / float(steps)
        style_code, style_maps = self._prepare_style_context(
            style_id=style_id,
        )
        h = x
        for _ in range(steps):
            delta = self._predict_delta_from_context(
                h,
                style_code=style_code,
                style_maps=style_maps,
                strength=strength,
            )
            h = h + delta * float(step_size) * step_scale * per_step
        return h

    def forward(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int,
        step_size: float = 1.0,
        style_strength: float | None = None,
    ) -> torch.Tensor:
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        style_code, style_maps = self._prepare_style_context(
            style_id=style_id,
        )
        delta = self._predict_delta_from_context(
            x,
            style_code=style_code,
            style_maps=style_maps,
            strength=strength,
        )
        return x + delta * float(step_size) * step_scale

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model_from_config(
    model_cfg: dict,
    *,
    use_checkpointing: bool = False,
) -> LatentAdaCUT:
    """
    Single model-construction entrypoint used by both training and inference.
    Keeps config parsing consistent and prevents train/eval drift bugs.
    """
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
        "upsample_blur",
        "upsample_blur_kernel",
        "ada_mix_rank",
    }
    unknown_keys = sorted(k for k in model_cfg.keys() if k not in known_keys)
    if unknown_keys:
        warnings.warn(
            "Unknown model config key(s): " + ", ".join(unknown_keys),
            category=UserWarning,
            stacklevel=2,
        )

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
        upsample_blur=bool(model_cfg.get("upsample_blur", True)),
        upsample_blur_kernel=str(model_cfg.get("upsample_blur_kernel", "box3")),
        ada_mix_rank=int(model_cfg.get("ada_mix_rank", 16)),
    )
