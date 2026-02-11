from __future__ import annotations

from dataclasses import dataclass

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

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        params = self.proj(style_code).unsqueeze(-1).unsqueeze(-1)
        scale, shift = params.chunk(2, dim=1)
        return h * scale + shift


class ResBlock(nn.Module):
    def __init__(self, dim: int, style_dim: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm1 = AdaGN(dim, style_dim, num_groups=num_groups)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = AdaGN(dim, style_dim, num_groups=num_groups)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x, style_code))
        h = self.conv1(h)
        h = self.act(self.norm2(h, style_code))
        h = self.conv2(h)
        return x + h


@dataclass
class StyleMaps:
    map_32: torch.Tensor | None = None
    map_16: torch.Tensor | None = None


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
        projector_dim: int = 256,
        use_checkpointing: bool = False,
        latent_scale_factor: float = 0.18215,
        residual_gain: float = 0.1,
        style_ref_gain: float = 1.0,
        style_spatial_pre_gain_32: float = 0.25,
        style_spatial_block_gain_32: float = 0.10,
        style_spatial_pre_gain_16: float = 0.35,
        style_spatial_block_gain_16: float = 0.15,
        use_decoder_spatial_inject: bool = True,
        style_spatial_dec_gain_32: float = 0.18,
        style_spatial_dec_gain_out: float = 0.08,
        use_style_texture_head: bool = True,
        style_texture_gain: float = 0.12,
        use_style_delta_gate: bool = True,
        use_decoder_adagn: bool = True,
        use_delta_highpass_bias: bool = True,
        style_delta_lowfreq_gain: float = 0.35,
        use_style_spatial_highpass: bool = False,
        normalize_style_spatial_maps: bool = True,
        use_output_style_affine: bool = True,
        use_style_force_path: bool = True,
        style_force_gain: float = 1.0,
        style_gate_floor: float = 0.85,
        style_texture_ignore_residual_gain: bool = True,
        use_style_spatial_blur: bool = False,
        use_downsample_blur: bool = False,
        upsample_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.num_styles = int(num_styles)
        self.use_checkpointing = bool(use_checkpointing)
        self.latent_scale_factor = float(latent_scale_factor)
        self.residual_gain = float(residual_gain)
        self.style_ref_gain = float(style_ref_gain)
        self.lift_channels = int(lift_channels) if lift_channels is not None else int(base_dim)
        self.body_channels = int(base_dim * 2)
        self.style_spatial_pre_gain_32 = float(style_spatial_pre_gain_32)
        self.style_spatial_block_gain_32 = float(style_spatial_block_gain_32)
        self.style_spatial_pre_gain_16 = float(style_spatial_pre_gain_16)
        self.style_spatial_block_gain_16 = float(style_spatial_block_gain_16)
        self.use_decoder_spatial_inject = bool(use_decoder_spatial_inject)
        self.style_spatial_dec_gain_32 = float(style_spatial_dec_gain_32)
        self.style_spatial_dec_gain_out = float(style_spatial_dec_gain_out)
        self.use_style_texture_head = bool(use_style_texture_head)
        self.style_texture_gain = float(style_texture_gain)
        self.use_style_delta_gate = bool(use_style_delta_gate)
        self.use_decoder_adagn = bool(use_decoder_adagn)
        self.use_delta_highpass_bias = bool(use_delta_highpass_bias)
        self.style_delta_lowfreq_gain = float(style_delta_lowfreq_gain)
        self.use_style_spatial_highpass = bool(use_style_spatial_highpass)
        self.normalize_style_spatial_maps = bool(normalize_style_spatial_maps)
        self.use_output_style_affine = bool(use_output_style_affine)
        self.use_style_force_path = bool(use_style_force_path)
        self.style_force_gain = float(style_force_gain)
        self.style_gate_floor = float(style_gate_floor)
        self.style_texture_ignore_residual_gain = bool(style_texture_ignore_residual_gain)
        self.use_style_spatial_blur = bool(use_style_spatial_blur)
        self.use_downsample_blur = bool(use_downsample_blur)
        self.upsample_mode = str(upsample_mode)

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

        # Learnable style-id spatial priors for inference without reference image.
        self.style_spatial_id_32 = nn.Parameter(torch.zeros(self.num_styles, self.lift_channels, 32, 32))
        self.style_spatial_id_16 = nn.Parameter(torch.zeros(self.num_styles, self.body_channels, 16, 16))
        nn.init.normal_(self.style_spatial_id_32, mean=0.0, std=0.02)
        nn.init.normal_(self.style_spatial_id_16, mean=0.0, std=0.02)

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

        out_groups = max(1, min(num_groups, self.lift_channels))
        while self.lift_channels % out_groups != 0 and out_groups > 1:
            out_groups -= 1

        # Decoder: 16 -> 32
        upsample_kwargs = {"scale_factor": 2, "mode": self.upsample_mode}
        if self.upsample_mode in {"bilinear", "bicubic"}:
            upsample_kwargs["align_corners"] = False
        self.dec_up = nn.Upsample(**upsample_kwargs)
        self.dec_conv = nn.Conv2d(self.body_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        if self.use_decoder_adagn:
            self.dec_norm = AdaGN(self.lift_channels, style_dim, num_groups=out_groups)
        else:
            self.dec_norm = nn.GroupNorm(out_groups, self.lift_channels, eps=1e-6)
        self.dec_act = nn.SiLU()
        self.dec_out = nn.Conv2d(self.lift_channels, latent_channels, kernel_size=3, stride=1, padding=1)
        self.style_texture_head = nn.Sequential(
            nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.lift_channels, latent_channels, kernel_size=1, stride=1, padding=0),
        )
        nn.init.normal_(self.style_texture_head[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.style_texture_head[-1].bias)
        if self.use_style_force_path:
            self.style_force_head = nn.Sequential(
                nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(self.lift_channels, latent_channels, kernel_size=1, stride=1, padding=0),
            )
            nn.init.normal_(self.style_force_head[-1].weight, mean=0.0, std=0.05)
            nn.init.zeros_(self.style_force_head[-1].bias)

        # NCE projector.
        self.projector = nn.Sequential(
            nn.Linear(latent_channels, projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim),
        )

        if self.use_style_delta_gate:
            self.style_delta_gate = nn.Linear(style_dim, 1)
            nn.init.zeros_(self.style_delta_gate.weight)
            nn.init.constant_(self.style_delta_gate.bias, 0.0)
        if self.use_output_style_affine:
            self.output_style_affine = nn.Linear(style_dim, latent_channels * 2)
            nn.init.normal_(self.output_style_affine.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.output_style_affine.bias)
            with torch.no_grad():
                self.output_style_affine.bias[:latent_channels] = 1.0

    def _style_code(
        self,
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
        style_mix_alpha: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        code_id = self.encode_style_id(style_id) if style_id is not None else None
        code_ref = (self.encode_style(style_ref) * self.style_ref_gain) if style_ref is not None else None

        if code_id is None and code_ref is None:
            raise ValueError("Either style_id or style_ref must be provided.")
        if code_id is None:
            return code_ref
        if code_ref is None:
            return code_id

        alpha = self._resolve_alpha(style_mix_alpha, batch=code_id.shape[0], device=code_id.device, dtype=code_id.dtype)
        return alpha * code_ref + (1.0 - alpha) * code_id

    def _run_block(self, block: ResBlock, h: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            return ckpt.checkpoint(
                lambda _h, _s, _blk=block: _blk(_h, _s),
                h,
                style_code,
                use_reentrant=False,
            )
        return block(h, style_code)

    def _run_style_blocks(
        self,
        h: torch.Tensor,
        blocks: nn.ModuleList,
        style_code: torch.Tensor,
        style_map: torch.Tensor | None,
        style_gain: float,
    ) -> torch.Tensor:
        out = h
        for block in blocks:
            out = self._run_block(block, out, style_code)
            if style_map is not None:
                out = out + (style_gain * style_map)
        return out

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
            map_32=mixed.get(32),
            map_16=mixed.get(16),
        )

    def _prepare_spatial_map(self, style_map: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
        style_map = self._match_style_map(style_map, target)
        if style_map is not None and self.use_style_spatial_blur:
            style_map = self._blur2d(style_map)
        return style_map

    def _compute_delta(
        self,
        h: torch.Tensor,
        style_code: torch.Tensor,
        style_spatial_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        delta = self.dec_out(h) * self.latent_scale_factor * self.residual_gain
        if self.use_output_style_affine:
            aff = self.output_style_affine(style_code).view(style_code.shape[0], 2 * self.latent_channels, 1, 1)
            d_scale, d_shift = aff.chunk(2, dim=1)
            delta = (delta * d_scale) + (d_shift * self.latent_scale_factor * self.residual_gain)
        if self.use_style_delta_gate:
            gate = 0.5 + torch.sigmoid(self.style_delta_gate(style_code)).view(-1, 1, 1, 1)
            if self.style_gate_floor > 0.0:
                floor = min(max(self.style_gate_floor, 0.0), 1.5)
                gate = torch.clamp_min(gate, floor)
            delta = delta * gate
        if self.use_style_texture_head and style_spatial_dec is not None:
            style_tex = self.style_texture_head(style_spatial_dec)
            if self.use_delta_highpass_bias:
                style_tex = self._bias_to_highfreq(style_tex, self.style_delta_lowfreq_gain)
            style_tex_scale = self.latent_scale_factor * self.style_texture_gain
            if not self.style_texture_ignore_residual_gain:
                style_tex_scale = style_tex_scale * self.residual_gain
            delta = delta + (style_tex * style_tex_scale)
        if self.use_style_force_path and style_spatial_dec is not None:
            style_force = self.style_force_head(style_spatial_dec)
            if self.use_delta_highpass_bias:
                style_force = self._bias_to_highfreq(style_force, self.style_delta_lowfreq_gain)
            delta = delta + (style_force * self.latent_scale_factor * self.style_force_gain)
        if self.use_delta_highpass_bias:
            delta = self._bias_to_highfreq(delta, self.style_delta_lowfreq_gain)
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
        if isinstance(style_id, int):
            style_id = torch.tensor([style_id], device=self.style_emb.weight.device, dtype=torch.long)
        style_id = style_id.long().view(-1).to(self.style_emb.weight.device)
        style_id = style_id.clamp_min(0).clamp_max(max(1, self.num_styles) - 1)
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
    def _style_highpass_map(feat: torch.Tensor) -> torch.Tensor:
        feat = feat - feat.mean(dim=(2, 3), keepdim=True)
        feat = feat / (feat.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)
        low = F.interpolate(
            F.avg_pool2d(feat, kernel_size=2, stride=2),
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.tanh(feat - low)

    @staticmethod
    def _normalize_style_map(feat: torch.Tensor) -> torch.Tensor:
        feat = feat - feat.mean(dim=(2, 3), keepdim=True)
        return feat / (feat.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)

    @staticmethod
    def _bias_to_highfreq(feat: torch.Tensor, lowfreq_gain: float) -> torch.Tensor:
        low = F.interpolate(
            F.avg_pool2d(feat, kernel_size=2, stride=2),
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high = feat - low
        return high + low * float(lowfreq_gain)

    @staticmethod
    def _blur2d(feat: torch.Tensor) -> torch.Tensor:
        if feat.numel() == 0:
            return feat
        kernel = feat.new_tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
        )
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3)
        kernel = kernel.repeat(feat.shape[1], 1, 1, 1)
        return F.conv2d(feat, kernel, padding=1, groups=feat.shape[1])

    @classmethod
    def _extract_style_spatial_maps(cls, style_feats: list[torch.Tensor]) -> dict[int, torch.Tensor]:
        maps: dict[int, torch.Tensor] = {}
        if len(style_feats) >= 1:
            maps[32] = style_feats[0]
        if len(style_feats) >= 2:
            maps[16] = style_feats[1]
        return maps

    def encode_style_spatial_ref(self, style_ref: torch.Tensor | None) -> dict[int, torch.Tensor]:
        if style_ref is None:
            return {}
        maps = self._extract_style_spatial_maps(self.encode_style_feats(style_ref))
        if self.use_style_spatial_highpass:
            maps = {k: self._style_highpass_map(v) for k, v in maps.items()}
        return maps

    def encode_style_spatial_id(self, style_id: torch.Tensor | int | None) -> dict[int, torch.Tensor]:
        if style_id is None:
            return {}
        if isinstance(style_id, int):
            style_id = torch.tensor([style_id], device=self.style_spatial_id_32.device, dtype=torch.long)
        style_id = style_id.long().view(-1).to(self.style_spatial_id_32.device)
        maps = {
            32: self.style_spatial_id_32.index_select(0, style_id),
            16: self.style_spatial_id_16.index_select(0, style_id),
        }
        if self.use_style_spatial_highpass:
            maps[32] = self._style_highpass_map(maps[32])
            maps[16] = self._style_highpass_map(maps[16])
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
            if self.normalize_style_spatial_maps and out[k] is not None:
                out[k] = self._normalize_style_map(out[k])
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

    def _predict_delta(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
        style_mix_alpha: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        style_code = self._style_code(style_id=style_id, style_ref=style_ref, style_mix_alpha=style_mix_alpha)
        style_maps = self._prepare_style_maps(
            style_id=style_id,
            style_ref=style_ref,
            style_mix_alpha=style_mix_alpha,
            batch=x.shape[0],
            device=x.device,
        )

        feat = x / max(self.latent_scale_factor, 1e-8)
        h = self.enc_in_act(self.enc_in(feat))
        style_spatial_32 = self._prepare_spatial_map(style_maps.map_32, h)
        if style_spatial_32 is not None:
            h = h + self.style_spatial_pre_gain_32 * style_spatial_32
        h = self._run_style_blocks(
            h,
            blocks=self.hires_body,
            style_code=style_code,
            style_map=style_spatial_32,
            style_gain=self.style_spatial_block_gain_32,
        )

        if self.use_downsample_blur:
            h = self._blur2d(h)
        h = self.down(h)
        style_spatial_16 = self._prepare_spatial_map(style_maps.map_16, h)
        if style_spatial_16 is not None:
            h = h + self.style_spatial_pre_gain_16 * style_spatial_16
        h = self._run_style_blocks(
            h,
            blocks=self.body,
            style_code=style_code,
            style_map=style_spatial_16,
            style_gain=self.style_spatial_block_gain_16,
        )

        h = self.dec_up(h)
        h = self.dec_conv(h)
        style_spatial_dec = self._prepare_spatial_map(style_spatial_32, h)
        if self.use_decoder_spatial_inject and style_spatial_dec is not None:
            h = h + self.style_spatial_dec_gain_32 * style_spatial_dec
        if self.use_decoder_adagn:
            h = self.dec_norm(h, style_code)
        else:
            h = self.dec_norm(h)
        h = self.dec_act(h)
        if self.use_decoder_spatial_inject and style_spatial_dec is not None:
            h = h + self.style_spatial_dec_gain_out * style_spatial_dec

        return self._compute_delta(h, style_code=style_code, style_spatial_dec=style_spatial_dec)

    def integrate(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
        style_mix_alpha: float | torch.Tensor | None = None,
        num_steps: int = 1,
        step_size: float = 1.0,
    ) -> torch.Tensor:
        steps = max(1, int(num_steps))
        h = x
        for _ in range(steps):
            delta = self._predict_delta(
                h,
                style_id=style_id,
                style_ref=style_ref,
                style_mix_alpha=style_mix_alpha,
            )
            h = h + delta * float(step_size)
        return h

    def forward(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
        style_mix_alpha: float | torch.Tensor | None = None,
        step_size: float = 1.0,
    ) -> torch.Tensor:
        delta = self._predict_delta(
            x,
            style_id=style_id,
            style_ref=style_ref,
            style_mix_alpha=style_mix_alpha,
        )
        return x + delta * float(step_size)

    def project_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        return self.projector(tokens)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
