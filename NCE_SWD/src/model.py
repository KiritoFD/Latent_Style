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
        num_res_blocks: int = 4,
        num_groups: int = 8,
        projector_dim: int = 256,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.num_styles = int(num_styles)
        self.use_checkpointing = bool(use_checkpointing)

        self.style_emb = nn.Embedding(num_styles, style_dim)

        # Encoder: 32 -> 16
        self.enc = nn.Sequential(
            nn.Conv2d(latent_channels, base_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=4, stride=2, padding=1),
        )

        self.body = nn.ModuleList(
            [ResBlock(base_dim * 2, style_dim, num_groups=num_groups) for _ in range(num_res_blocks)]
        )

        out_groups = max(1, min(num_groups, base_dim))
        while base_dim % out_groups != 0 and out_groups > 1:
            out_groups -= 1

        # Decoder: 16 -> 32
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_groups, base_dim, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(base_dim, latent_channels, kernel_size=3, stride=1, padding=1),
        )

        # NCE projector.
        self.projector = nn.Sequential(
            nn.Linear(latent_channels, projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim),
        )

    def forward(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        if style_id.ndim == 0:
            style_id = style_id.unsqueeze(0)
        style_code = self.style_emb(style_id.long())

        h = self.enc(x)
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
        return self.dec(h)

    def project_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W] -> token embeddings [B*H*W,D]
        """
        tokens = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        return self.projector(tokens)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
