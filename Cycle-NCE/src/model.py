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
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.num_styles = int(num_styles)
        self.use_checkpointing = bool(use_checkpointing)
        self.latent_scale_factor = float(latent_scale_factor)
        self.residual_gain = float(residual_gain)
        self.style_ref_gain = float(style_ref_gain)
        self.lift_channels = int(lift_channels) if lift_channels is not None else int(base_dim)
        self.style_spatial_pre_gain_32 = float(style_spatial_pre_gain_32)
        self.style_spatial_block_gain_32 = float(style_spatial_block_gain_32)
        self.style_spatial_pre_gain_16 = float(style_spatial_pre_gain_16)
        self.style_spatial_block_gain_16 = float(style_spatial_block_gain_16)

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

        # 32x32 lift stage before downsampling.
        self.enc_in = nn.Conv2d(latent_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        self.enc_in_act = nn.SiLU()
        self.hires_body = nn.ModuleList(
            [ResBlock(self.lift_channels, style_dim, num_groups=num_groups) for _ in range(max(0, int(num_hires_blocks)))]
        )
        self.down = nn.Conv2d(self.lift_channels, base_dim * 2, kernel_size=4, stride=2, padding=1)

        self.body = nn.ModuleList(
            [ResBlock(base_dim * 2, style_dim, num_groups=num_groups) for _ in range(num_res_blocks)]
        )

        out_groups = max(1, min(num_groups, self.lift_channels))
        while self.lift_channels % out_groups != 0 and out_groups > 1:
            out_groups -= 1

        # Decoder: 16 -> 32
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_dim * 2, self.lift_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_groups, self.lift_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(self.lift_channels, latent_channels, kernel_size=3, stride=1, padding=1),
        )

        # NCE projector.
        self.projector = nn.Sequential(
            nn.Linear(latent_channels, projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim),
        )

    def _style_code(
        self,
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del style_id
        if style_ref is None:
            raise ValueError("style_ref is required for style code in reference-conditioned mode.")
        return self.encode_style(style_ref) * self.style_ref_gain

    def encode_style(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode style reference latent into style code space.
        """
        z = z / max(self.latent_scale_factor, 1e-8)
        h = self.style_enc(z).flatten(1)
        return self.style_proj(h)

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

    @classmethod
    def _extract_style_spatial_maps(cls, style_feats: list[torch.Tensor]) -> dict[int, torch.Tensor]:
        """
        Build high-frequency style maps for both 32x32 and 16x16 paths.
        """
        maps: dict[int, torch.Tensor] = {}
        if len(style_feats) >= 1:
            maps[32] = cls._style_highpass_map(style_feats[0])
        if len(style_feats) >= 2:
            maps[16] = cls._style_highpass_map(style_feats[1])
        return maps

    def forward(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | None = None,
        style_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        style_code = self._style_code(style_id=style_id, style_ref=style_ref)
        style_maps: dict[int, torch.Tensor] = {}
        if style_ref is not None:
            style_maps = self._extract_style_spatial_maps(self.encode_style_feats(style_ref))
        style_spatial_32 = style_maps.get(32)
        style_spatial_16 = style_maps.get(16)

        # Normalize to VAE latent native scale before style transform.
        feat = x / max(self.latent_scale_factor, 1e-8)
        h = self.enc_in_act(self.enc_in(feat))
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

        h = self.down(h)
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
        delta = self.dec(h) * self.latent_scale_factor * self.residual_gain
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
