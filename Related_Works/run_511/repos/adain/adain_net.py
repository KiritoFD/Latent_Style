"""Minimal AdaIN (Huang & Belongie 2017) style transfer — self-contained.

Uses a VGG19 encoder up to relu4_1 and a symmetric decoder.
Supports two encoder backends:
  - vgg_normalised.pth (custom format, 256-ch relu4_1)
  - Standard VGG-19 ImageNet pretrained (512-ch relu4_1, default)
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import vgg19


# ---------------------------------------------------------------------------
# VGG-normalised encoder (for vgg_normalised.pth with 1x1 norm conv)
# relu4_1 = index 26, 256 channels
# ---------------------------------------------------------------------------

def _build_vgg_norm_encoder() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 3, 1, 1, 0), nn.ReLU(inplace=True),          # 0-1  norm
        nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),         # 2-3  relu1_1
        nn.MaxPool2d(2, 2),                                        # 4
        nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),        # 5-6  relu2_1
        nn.MaxPool2d(2, 2),                                        # 7
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),       # 8-9
        nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),      # 10-11 relu3_2
        nn.MaxPool2d(2, 2),                                        # 12
        nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),      # 13-14
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),      # 15-16
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),      # 17-18
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),      # 19-20
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),      # 21-22
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),      # 23-24
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),      # 25-26 relu4_7
    )


class VGGEncoder(nn.Module):
    """VGG-19 feature extractor (relu1_1 … relu4_1)."""

    def __init__(self, weights_path: str | None = None):
        super().__init__()
        use_norm = (
            weights_path is not None
            and Path(weights_path).exists()
            and "vgg_normalised" in Path(weights_path).name
        )
        if use_norm:
            vgg = _build_vgg_norm_encoder()
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            clean = {k.replace("module.", ""): v for k, v in state.items()}
            vgg.load_state_dict(clean, strict=False)
            children = list(vgg.children())
            self.enc_1 = nn.Sequential(*children[:4])
            self.enc_2 = nn.Sequential(*children[4:8])
            self.enc_3 = nn.Sequential(*children[8:12])
            self.enc_4 = nn.Sequential(*children[12:27])
            self._out_channels = 256
        else:
            vgg = vgg19(weights="IMAGENET1K_V1").features
            children = list(vgg.children())
            self.enc_1 = nn.Sequential(*children[:2])    # relu1_1
            self.enc_2 = nn.Sequential(*children[2:7])   # relu2_1
            self.enc_3 = nn.Sequential(*children[7:12])  # relu3_1
            self.enc_4 = nn.Sequential(*children[12:21]) # relu4_1
            self._out_channels = 512
        for p in self.parameters():
            p.requires_grad = False

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        return [h1, h2, h3, h4]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, 3, 1, 1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1, padding_mode="reflect"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# AdaIN core
# ---------------------------------------------------------------------------

def adaptive_instance_norm(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-5
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-5
    return s_std * (content_feat - c_mean) / c_std + s_mean


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class AdaINNet(nn.Module):
    def __init__(self, vgg_weights: str | None = None):
        super().__init__()
        self.encoder = VGGEncoder(vgg_weights)
        self.decoder = Decoder(in_channels=self.encoder.out_channels)

    def forward(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        c_feats = self.encoder(content)
        s_feats = self.encoder(style)
        t = adaptive_instance_norm(c_feats[-1], s_feats[-1])
        t = alpha * t + (1 - alpha) * c_feats[-1]
        return self.decoder(t)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(x)

    @staticmethod
    def calc_mean_std(feat: torch.Tensor):
        mean = feat.mean(dim=[2, 3], keepdim=True)
        std = feat.std(dim=[2, 3], keepdim=True) + 1e-5
        return mean, std

    def calc_content_loss(self, gen_feats: list[torch.Tensor], target_feats: list[torch.Tensor]) -> torch.Tensor:
        return sum(nn.functional.mse_loss(g, t.detach()) for g, t in zip(gen_feats, target_feats))

    def calc_style_loss(self, gen_feats: list[torch.Tensor], target_feats: list[torch.Tensor]) -> torch.Tensor:
        return sum(self._layer_style_loss(g, t.detach()) for g, t in zip(gen_feats, target_feats))

    @staticmethod
    def _layer_style_loss(gen: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        g_mean, g_std = AdaINNet.calc_mean_std(gen)
        t_mean, t_std = AdaINNet.calc_mean_std(target)
        return nn.functional.mse_loss(g_mean, t_mean) + nn.functional.mse_loss(g_std, t_std)
