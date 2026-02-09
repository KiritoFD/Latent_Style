from __future__ import annotations

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
        use_content_skip_fusion: bool = True,
        content_skip_gain: float = 0.6,
        use_style_skip_gate: bool = True,
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
        self.use_content_skip_fusion = bool(use_content_skip_fusion)
        self.content_skip_gain = float(content_skip_gain)
        self.use_style_skip_gate = bool(use_style_skip_gate)

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
        self.dec_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv = nn.Conv2d(self.body_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        self.content_skip_proj = nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=1, stride=1, padding=0)
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
        nn.init.zeros_(self.style_texture_head[-1].weight)
        nn.init.zeros_(self.style_texture_head[-1].bias)

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
        if self.use_style_skip_gate:
            self.style_skip_gate = nn.Linear(style_dim, self.lift_channels)
            nn.init.zeros_(self.style_skip_gate.weight)
            nn.init.zeros_(self.style_skip_gate.bias)

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
    def _bias_to_highfreq(feat: torch.Tensor, lowfreq_gain: float) -> torch.Tensor:
        """
        Keep high-frequency style detail while damping global low-frequency shifts.
        This reduces the "brightness-only" shortcut.
        """
        low = F.interpolate(
            F.avg_pool2d(feat, kernel_size=2, stride=2),
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high = feat - low
        return high + low * float(lowfreq_gain)

    @classmethod
    def _extract_style_spatial_maps(cls, style_feats: list[torch.Tensor]) -> dict[int, torch.Tensor]:
        """
        Build style maps for both 32x32 and 16x16 paths.
        """
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
        return out

    @staticmethod
    def _match_style_map(style_map: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
        """
        Resize/cast style map to match target feature map.
        This keeps style priors compatible with both 32x32 and 64x64 latent grids.
        """
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

    def forward(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
        style_mix_alpha: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        style_code = self._style_code(style_id=style_id, style_ref=style_ref, style_mix_alpha=style_mix_alpha)
        style_maps = self._blend_style_maps(
            maps_id=self.encode_style_spatial_id(style_id),
            maps_ref=self.encode_style_spatial_ref(style_ref),
            style_mix_alpha=style_mix_alpha,
            batch=x.shape[0],
            device=x.device,
        )
        style_spatial_32 = style_maps.get(32)
        style_spatial_16 = style_maps.get(16)

        # Normalize to VAE latent native scale before style transform.
        feat = x / max(self.latent_scale_factor, 1e-8)
        h = self.enc_in_act(self.enc_in(feat))
        style_spatial_32 = self._match_style_map(style_spatial_32, h)
        if style_spatial_32 is not None:
            h = h + self.style_spatial_pre_gain_32 * style_spatial_32
        for block in self.hires_body:
            if self.use_checkpointing and self.training:
                h = ckpt.checkpoint(
                    lambda _h, _s, _blk=block: _blk(_h, _s),
                    h,
                    style_code,
                    use_reentrant=False,
                )
            else:
                h = block(h, style_code)
            if style_spatial_32 is not None:
                h = h + self.style_spatial_block_gain_32 * style_spatial_32

        content_skip = h
        h = self.down(h)
        style_spatial_16 = self._match_style_map(style_spatial_16, h)
        if style_spatial_16 is not None:
            h = h + self.style_spatial_pre_gain_16 * style_spatial_16
        for block in self.body:
            if self.use_checkpointing and self.training:
                h = ckpt.checkpoint(
                    lambda _h, _s, _blk=block: _blk(_h, _s),
                    h,
                    style_code,
                    use_reentrant=False,
                )
            else:
                h = block(h, style_code)
            if style_spatial_16 is not None:
                h = h + self.style_spatial_block_gain_16 * style_spatial_16
        h = self.dec_up(h)
        h = self.dec_conv(h)
        if self.use_content_skip_fusion:
            skip = self.content_skip_proj(content_skip)
            if skip.shape[-2:] != h.shape[-2:]:
                skip = F.interpolate(skip, size=h.shape[-2:], mode="bilinear", align_corners=False)
            if self.use_style_skip_gate:
                skip_gate = torch.sigmoid(self.style_skip_gate(style_code)).view(style_code.shape[0], -1, 1, 1)
                skip = skip * skip_gate
            h = h + self.content_skip_gain * skip
        style_spatial_dec = self._match_style_map(style_spatial_32, h)
        if self.use_decoder_spatial_inject and style_spatial_dec is not None:
            h = h + self.style_spatial_dec_gain_32 * style_spatial_dec
        if self.use_decoder_adagn:
            h = self.dec_norm(h, style_code)
        else:
            h = self.dec_norm(h)
        h = self.dec_act(h)
        if self.use_decoder_spatial_inject and style_spatial_dec is not None:
            h = h + self.style_spatial_dec_gain_out * style_spatial_dec
        delta = self.dec_out(h) * self.latent_scale_factor * self.residual_gain
        if self.use_style_delta_gate:
            # Per-sample style-conditioned residual strength in [0.5, 1.5].
            gate = 0.5 + torch.sigmoid(self.style_delta_gate(style_code)).view(-1, 1, 1, 1)
            delta = delta * gate
        if self.use_style_texture_head and style_spatial_dec is not None:
            style_tex = self.style_texture_head(style_spatial_dec)
            if self.use_delta_highpass_bias:
                style_tex = self._bias_to_highfreq(style_tex, self.style_delta_lowfreq_gain)
            delta = delta + (style_tex * self.style_texture_gain * self.latent_scale_factor * self.residual_gain)
        if self.use_delta_highpass_bias:
            delta = self._bias_to_highfreq(delta, self.style_delta_lowfreq_gain)
        # Residual learning: preserve structure, only learn style shift.
        return x + delta

    def project_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W] -> token embeddings [B*H*W,D]
        """
        tokens = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        return self.projector(tokens)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
