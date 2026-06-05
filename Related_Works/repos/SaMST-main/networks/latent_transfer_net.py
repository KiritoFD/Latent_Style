from __future__ import annotations

import torch
import torch.nn as nn

from networks.transfer_net import TransformerNet


class LatentTransformerNet(nn.Module):
    """Thin latent wrapper that preserves the original SaMST core."""

    def __init__(self, style_num: int, latent_channels: int = 4):
        super().__init__()
        self.input_adapter = nn.Conv2d(latent_channels, 3, kernel_size=1, stride=1, padding=0)
        self.core = TransformerNet(style_num=style_num)
        self.output_adapter = nn.Conv2d(3, latent_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, style_id):
        x_rgb = self.input_adapter(x)
        y_rgb, representation = self.core(x_rgb, style_id=style_id)
        y_latent = self.output_adapter(y_rgb)
        return y_latent, representation
