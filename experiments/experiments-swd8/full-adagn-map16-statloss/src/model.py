from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


class AdaGN(nn.Module):
    """
    Adaptive GroupNorm with style-conditioned scale/shift.
    """

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
                g = g.reshape(1, 1, 1, 1)
            elif g.ndim == 1:
                g = g.view(-1, 1, 1, 1)
            g = torch.nan_to_num(g, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            return h + g * (adagn - h)
        else:
            g = float(gate)
            if not math.isfinite(g):
                g = 0.0
            g = max(0.0, min(1.0, g))
            return h + (adagn - h) * g


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


@dataclass
class StyleMaps:
    map_16: torch.Tensor | None = None
    map_8: torch.Tensor | None = None


class LatentAdaCUT(nn.Module):
    """
    Micro U-Net without skip-connections.
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
        style_spatial_pre_gain_8: float = 0.6,
        use_decoder_adagn: bool = True,
        inject_gate_hires: float = 0.0,
        inject_gate_body: float = 1.0,
        inject_gate_decoder: float = 1.0,
        style_strength_default: float = 1.0,
        style_strength_step_curve: str = "linear",
        upsample_mode: str = "nearest",
        style_id_spatial_jitter_px: int = 0,
        loss_projector_use: bool = True,
        loss_projector_channels: int = 64,
        upsample_blur: bool = True,
        upsample_blur_kernel: str = "box3",
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
        self.style_spatial_pre_gain_8 = float(style_spatial_pre_gain_8)
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
        self.loss_projector_use = bool(loss_projector_use)
        self.loss_projector_channels = max(8, int(loss_projector_channels))
        self.upsample_blur = bool(upsample_blur)
        self.upsample_blur_kernel = str(upsample_blur_kernel).lower()
        if self.upsample_blur_kernel not in {"box3", "gaussian3"}:
            self.upsample_blur_kernel = "box3"

        self.style_enc = nn.Sequential(
            nn.Conv2d(latent_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.lift_channels, base_dim * 2, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.SiLU(),
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.style_proj = nn.Linear(base_dim * 4, style_dim)
        self.style_emb = nn.Embedding(self.num_styles, style_dim)
        nn.init.normal_(self.style_emb.weight, mean=0.0, std=0.02)
        self.style_map8_proj = nn.Conv2d(base_dim * 4, self.body_channels, kernel_size=1, stride=1, padding=0)

        # Learnable style-id spatial priors for inference without reference image.
        self.style_spatial_id_16 = nn.Parameter(torch.zeros(self.num_styles, self.body_channels, 16, 16))
        nn.init.normal_(self.style_spatial_id_16, mean=0.0, std=0.02)
        self.style_spatial_id_8 = nn.Parameter(torch.zeros(self.num_styles, self.body_channels, 8, 8))
        nn.init.normal_(self.style_spatial_id_8, mean=0.0, std=0.02)

        # 32x32 lift stage before downsampling.
        self.enc_in = nn.Conv2d(latent_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        self.enc_in_act = nn.SiLU()
        self.hires_body = nn.ModuleList(
            [ResBlock(self.lift_channels, style_dim, num_groups=num_groups) for _ in range(max(0, int(num_hires_blocks)))]
        )
        self.down = nn.Conv2d(self.lift_channels, self.body_channels, kernel_size=4, stride=2, padding=1)

        self.body = nn.ModuleList(
            [ResBlock(self.body_channels, style_dim, num_groups=num_groups) for _ in range(num_res_blocks)]
        )
        upsample_kwargs = {"scale_factor": 2, "mode": self.upsample_mode}
        if self.upsample_mode in {"bilinear", "bicubic"}:
            upsample_kwargs["align_corners"] = False
        self.down2 = nn.Conv2d(self.body_channels, self.body_channels, kernel_size=4, stride=2, padding=1)
        self.body8 = nn.ModuleList(
            [ResBlock(self.body_channels, style_dim, num_groups=num_groups) for _ in range(max(1, num_res_blocks))]
        )
        self.up_from8 = nn.Upsample(**upsample_kwargs)
        self.up_from8_conv = nn.Conv2d(self.body_channels, self.body_channels, kernel_size=3, stride=1, padding=1)

        out_groups = max(1, min(num_groups, self.lift_channels))
        while self.lift_channels % out_groups != 0 and out_groups > 1:
            out_groups -= 1

        # Decoder: 16 -> 32
        self.dec_up = nn.Upsample(**upsample_kwargs)
        self.dec_conv = nn.Conv2d(self.body_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        if self.use_decoder_adagn:
            self.dec_norm = AdaGN(self.lift_channels, style_dim, num_groups=out_groups)
        else:
            self.dec_norm = nn.GroupNorm(out_groups, self.lift_channels, eps=1e-6)
        self.dec_act = nn.SiLU()
        self.dec_out = nn.Conv2d(self.lift_channels, latent_channels, kernel_size=3, stride=1, padding=1)

        if self.loss_projector_use:
            c = self.loss_projector_channels
            self.loss_projector = nn.Sequential(
                nn.Conv2d(self.latent_channels, c, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.loss_projector = nn.Identity()

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

    def _style_code(
        self,
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
        style_mix_alpha: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        code_id = self.encode_style_id(style_id) if style_id is not None else None
        code_ref = self.encode_style(style_ref) if style_ref is not None else None

        if code_id is None and code_ref is None:
            raise ValueError("Either style_id or style_ref must be provided.")
        if code_id is None:
            return code_ref
        if code_ref is None:
            return code_id

        alpha = self._resolve_alpha(style_mix_alpha, batch=code_id.shape[0], device=code_id.device, dtype=code_id.dtype)
        return alpha * code_ref + (1.0 - alpha) * code_id

    def _run_block(
        self,
        block: ResBlock,
        h: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            if torch.is_tensor(gate):
                gate_in = gate.to(device=h.device, dtype=h.dtype)
            else:
                gate_val = float(gate)
                if not math.isfinite(gate_val):
                    gate_val = 0.0
                gate_in = h.new_tensor(max(0.0, min(1.0, gate_val)))
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
            if torch.is_tensor(gate):
                gate_in = gate.to(device=h.device, dtype=h.dtype)
            else:
                gate_val = float(gate)
                if not math.isfinite(gate_val):
                    gate_val = 0.0
                gate_in = h.new_tensor(max(0.0, min(1.0, gate_val)))
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
        style_id: torch.Tensor | None,
        style_ref: torch.Tensor | None,
        style_mix_alpha: float | torch.Tensor | None,
        batch: int,
        device: torch.device,
    ) -> StyleMaps:
        mixed = self._blend_style_maps(
            maps_id=self.encode_style_spatial_id(style_id),
            maps_ref=self.encode_style_spatial_ref(style_ref),
            style_mix_alpha=style_mix_alpha,
            batch=batch,
            device=device,
        )
        return StyleMaps(
            map_16=mixed.get(16),
            map_8=mixed.get(8),
        )

    def _prepare_spatial_map(self, style_map: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
        return self._match_style_map(style_map, target)

    def _prepare_style_context(
        self,
        *,
        style_id: torch.Tensor | None,
        style_ref: torch.Tensor | None,
        style_mix_alpha: float | torch.Tensor | None,
        batch: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, StyleMaps]:
        style_code = self._style_code(style_id=style_id, style_ref=style_ref, style_mix_alpha=style_mix_alpha)
        style_maps = self._prepare_style_maps(
            style_id=style_id,
            style_ref=style_ref,
            style_mix_alpha=style_mix_alpha,
            batch=batch,
            device=device,
        )
        return style_code, style_maps

    def project_loss_features(self, z: torch.Tensor) -> torch.Tensor:
        if not self.loss_projector_use:
            return z / max(self.latent_scale_factor, 1e-8)
        x = z / max(self.latent_scale_factor, 1e-8)
        return self.loss_projector(x)

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


    def encode_style(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode style reference latent into style code space.
        """
        z = z / max(self.latent_scale_factor, 1e-8)
        h = self.style_enc(z).flatten(1)
        return self.style_proj(h)

    def encode_style_id(self, style_id: torch.Tensor | int | None) -> torch.Tensor:
        if style_id is None:
            raise ValueError("style_id is required.")
        emb_device = self.style_emb.weight.device
        style_id = self._normalize_style_id_input(style_id, device=emb_device)
        return self.style_emb(style_id)

    def encode_style_feats(self, z: torch.Tensor) -> list[torch.Tensor]:
        """
        Return multi-scale spatial style features before global pooling.
        Features are taken after each SiLU in style_enc backbone.
        """
        z = z / max(self.latent_scale_factor, 1e-8)
        feats: list[torch.Tensor] = []
        h = z
        backbone = self.style_enc[:-1]
        for layer in backbone:
            h = layer(h)
            if isinstance(layer, nn.SiLU):
                feats.append(h)
        return feats

    @staticmethod
    def _normalize_style_map(feat: torch.Tensor) -> torch.Tensor:
        # Equivalent per-sample per-channel spatial standardization in one fused op.
        return F.instance_norm(feat, running_mean=None, running_var=None, weight=None, bias=None, eps=1e-6)

    @classmethod
    def _extract_style_spatial_maps(cls, style_feats: list[torch.Tensor]) -> dict[int, torch.Tensor]:
        maps: dict[int, torch.Tensor] = {}
        if len(style_feats) >= 2:
            maps[16] = style_feats[1]
        elif len(style_feats) >= 1:
            maps[16] = F.interpolate(style_feats[0], size=(16, 16), mode="bilinear", align_corners=False)
        if len(style_feats) >= 3:
            maps[8] = style_feats[2]
        elif len(style_feats) >= 2:
            maps[8] = F.interpolate(style_feats[1], size=(8, 8), mode="bilinear", align_corners=False)
        return maps

    def encode_style_spatial_ref(self, style_ref: torch.Tensor | None) -> dict[int, torch.Tensor]:
        if style_ref is None:
            return {}
        maps = self._extract_style_spatial_maps(self.encode_style_feats(style_ref))
        if 8 in maps and maps[8].shape[1] != self.body_channels:
            maps[8] = self.style_map8_proj(maps[8])
        maps = {k: self._normalize_style_map(v) for k, v in maps.items()}
        return maps

    def encode_style_spatial_id(self, style_id: torch.Tensor | int | None) -> dict[int, torch.Tensor]:
        if style_id is None:
            return {}
        spatial_device = self.style_spatial_id_16.device
        style_id = self._normalize_style_id_input(style_id, device=spatial_device)
        maps = {
            16: self.style_spatial_id_16.index_select(0, style_id),
            8: self.style_spatial_id_8.index_select(0, style_id),
        }
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
            maps[8] = _jitter_batch(maps[8])
        maps[16] = self._normalize_style_map(maps[16])
        maps[8] = self._normalize_style_map(maps[8])
        return maps

    @staticmethod
    def _resolve_alpha(
        style_mix_alpha: float | torch.Tensor | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if style_mix_alpha is None:
            alpha = torch.ones(batch, 1, device=device, dtype=dtype)
        elif torch.is_tensor(style_mix_alpha):
            alpha = style_mix_alpha.to(device=device, dtype=dtype)
            if alpha.ndim == 0:
                alpha = alpha.expand(batch).reshape(batch, 1)
            elif alpha.ndim == 1:
                alpha = alpha.reshape(batch, 1)
            else:
                alpha = alpha.reshape(batch, 1)
        else:
            alpha = torch.full((batch, 1), float(style_mix_alpha), device=device, dtype=dtype)
        return alpha.clamp_(0.0, 1.0)

    def _blend_style_maps(
        self,
        maps_id: dict[int, torch.Tensor],
        maps_ref: dict[int, torch.Tensor],
        style_mix_alpha: float | torch.Tensor | None,
        batch: int,
        device: torch.device,
    ) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        keys = set(maps_id.keys()) | set(maps_ref.keys())
        if not keys:
            return out
        alpha = self._resolve_alpha(style_mix_alpha, batch=batch, device=device, dtype=torch.float32).view(batch, 1, 1, 1)
        for k in keys:
            m_id = maps_id.get(k)
            m_ref = maps_ref.get(k)
            if m_id is None:
                out[k] = m_ref
            elif m_ref is None:
                out[k] = m_id
            else:
                out[k] = alpha * m_ref + (1.0 - alpha) * m_id
        return out

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
        return_bottleneck: bool = False,
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

        h = self.down2(h)
        style_spatial_8 = self._prepare_spatial_map(style_maps.map_8, h)
        if style_spatial_8 is not None:
            h = h + (self.style_spatial_pre_gain_8 * strength) * torch.tanh(style_spatial_8)
        h = self._run_style_blocks(
            h,
            blocks=self.body8,
            style_code=style_code,
            gate=gate_body,
        )
        h8 = h

        h = self.up_from8(h)
        h = self._apply_upsample_blur(h)
        h = self.up_from8_conv(h)
        if return_bottleneck:
            return h8

        h = self.dec_up(h)
        h = self._apply_upsample_blur(h)
        h = self._run_decoder(
            h,
            style_code=style_code,
            gate=gate_decoder,
        )
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
        steps = max(1, int(num_steps))
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        per_step = 1.0 / float(steps)
        style_code, style_maps = self._prepare_style_context(
            style_id=style_id,
            style_ref=style_ref,
            style_mix_alpha=style_mix_alpha,
            batch=x.shape[0],
            device=x.device,
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
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
        style_mix_alpha: float | torch.Tensor | None = None,
        step_size: float = 1.0,
        style_strength: float | None = None,
    ) -> torch.Tensor:
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        style_code, style_maps = self._prepare_style_context(
            style_id=style_id,
            style_ref=style_ref,
            style_mix_alpha=style_mix_alpha,
            batch=x.shape[0],
            device=x.device,
        )
        delta = self._predict_delta_from_context(
            x,
            style_code=style_code,
            style_maps=style_maps,
            strength=strength,
        )
        return x + delta * float(step_size) * step_scale

    def extract_bottleneck_feature(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
        style_mix_alpha: float | torch.Tensor | None = None,
        style_strength: float | None = None,
    ) -> torch.Tensor:
        strength = self._resolve_style_strength(style_strength)
        style_code, style_maps = self._prepare_style_context(
            style_id=style_id,
            style_ref=style_ref,
            style_mix_alpha=style_mix_alpha,
            batch=x.shape[0],
            device=x.device,
        )
        return self._predict_delta_from_context(
            x,
            style_code=style_code,
            style_maps=style_maps,
            strength=strength,
            return_bottleneck=True,
        )

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
        "style_spatial_pre_gain_8",
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
        style_spatial_pre_gain_8=float(model_cfg.get("style_spatial_pre_gain_8", 0.6)),
        use_decoder_adagn=bool(model_cfg.get("use_decoder_adagn", True)),
        inject_gate_hires=float(model_cfg.get("inject_gate_hires", 0.0)),
        inject_gate_body=float(model_cfg.get("inject_gate_body", 1.0)),
        inject_gate_decoder=float(model_cfg.get("inject_gate_decoder", 1.0)),
        style_strength_default=float(model_cfg.get("style_strength_default", 1.0)),
        style_strength_step_curve=str(model_cfg.get("style_strength_step_curve", "linear")),
        upsample_mode=str(model_cfg.get("upsample_mode", "nearest")),
        style_id_spatial_jitter_px=int(model_cfg.get("style_id_spatial_jitter_px", 0)),
        loss_projector_use=bool(model_cfg.get("loss_projector_use", True)),
        loss_projector_channels=int(model_cfg.get("loss_projector_channels", 64)),
        upsample_blur=bool(model_cfg.get("upsample_blur", True)),
        upsample_blur_kernel=str(model_cfg.get("upsample_blur_kernel", "box3")),
    )
