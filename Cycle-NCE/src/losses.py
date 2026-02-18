from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:
    from model import LatentAdaCUT


class LeakyProjectedStyleLoss(nn.Module):
    """Style loss based on projected latent features and Gram statistics."""

    def __init__(self, in_channels: int = 4, hidden_dim: int = 64, scale_factor: float = 0.13025) -> None:
        super().__init__()
        self.scale_factor = float(scale_factor)

        self.projector = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.projector.weight)
        self.projector.eval()
        for p in self.projector.parameters():
            p.requires_grad_(False)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=hidden_dim, eps=1e-6)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.scales = [1, 2]

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pred_scaled = x_pred * self.scale_factor
        x_target_scaled = x_target * self.scale_factor

        mu_pred, std_pred = self._calc_moments(x_pred_scaled)
        mu_tgt, std_tgt = self._calc_moments(x_target_scaled)
        loss_moment = F.l1_loss(mu_pred, mu_tgt) + F.l1_loss(std_pred, std_tgt)

        with torch.amp.autocast("cuda", enabled=False):
            grams_target = self._extract_features(x_target_scaled.float())
            grams_pred = self._extract_features(x_pred_scaled.float())
            loss_style = F.mse_loss(grams_pred, grams_target)

        return loss_style, loss_moment

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.norm(self.act(self.projector(x)))
        grams = []
        for s in self.scales:
            if s >= feats.shape[-1] or s >= feats.shape[-2]:
                continue
            dx = feats[..., :, s:] - feats[..., :, :-s]
            dy = feats[..., s:, :] - feats[..., :-s, :]
            grams.append(self._gram(dx))
            grams.append(self._gram(dy))
        return torch.cat(grams, dim=1) if grams else x.new_zeros(x.shape[0], 1)

    @staticmethod
    def _gram(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = x.reshape(b, c, -1)
        g = torch.bmm(f, f.transpose(1, 2)) / float(c)
        idx = torch.triu_indices(c, c, device=x.device)
        return g[:, idx[0], idx[1]]

    @staticmethod
    def _calc_moments(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x.mean(dim=[2, 3]), x.std(dim=[2, 3])


class PatchNCELoss(nn.Module):
    """Latent PatchNCE with trainable projection head."""

    def __init__(self, in_channels: int = 4, hidden_dim: int = 256, out_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=True),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        feat_q = self.mlp(feat_q)
        with torch.no_grad():
            feat_k = self.mlp(feat_k).detach()

        feat_q = F.normalize(feat_q, dim=1)
        feat_k = F.normalize(feat_k, dim=1)

        batch_size, dim, h, w = feat_q.shape
        feat_q = feat_q.permute(0, 2, 3, 1).reshape(batch_size, -1, dim)
        feat_k = feat_k.permute(0, 2, 3, 1).reshape(batch_size, -1, dim)

        logits = torch.bmm(feat_q, feat_k.transpose(1, 2)) / self.temperature

        num_patches = h * w
        labels = torch.arange(num_patches, device=feat_q.device).unsqueeze(0).expand(batch_size, -1).reshape(-1)

        logits = logits.reshape(-1, num_patches)
        return self.cross_entropy(logits, labels)


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        model_cfg = config.get("model", {})

        self.w_style = float(loss_cfg.get("w_style", 100.0))
        self.w_moment = float(loss_cfg.get("w_moment", 2.0))
        self.w_identity = float(loss_cfg.get("w_identity", 10.0))
        self.w_structure = float(loss_cfg.get("w_structure", 20.0))

        self.latent_scale_factor = float(model_cfg.get("latent_scale_factor", 0.13025))

        self._style_loss_module: LeakyProjectedStyleLoss | None = None
        self._content_loss_module: PatchNCELoss | None = None
        self._device: torch.device | None = None

        self.train_num_steps = 1
        self.train_step_size = 1.0
        self.train_style_strength = 1.0

    def _ensure_modules(self, device: torch.device) -> None:
        if self._style_loss_module is None or self._device != device:
            self._style_loss_module = LeakyProjectedStyleLoss(scale_factor=self.latent_scale_factor).to(device)
            self._content_loss_module = PatchNCELoss(in_channels=4, hidden_dim=256, out_dim=128).to(device)
            self._device = device

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        debug_timing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del debug_timing

        self._ensure_modules(content.device)
        assert self._style_loss_module is not None
        assert self._content_loss_module is not None

        pred = model(
            content,
            style_id=target_style_id,
            step_size=self.train_step_size,
            style_strength=self.train_style_strength,
        )

        loss_style, loss_moment = self._style_loss_module(pred, target_style)
        loss_struct = self._content_loss_module(pred, content)

        loss_idt = torch.tensor(0.0, device=content.device)
        if self.w_identity > 0.0 and source_style_id is not None:
            id_mask = source_style_id == target_style_id
            if id_mask.any():
                diff = (pred - content).abs().mean(dim=(1, 2, 3))
                denom = id_mask.float().sum().clamp_min(1.0)
                loss_idt = (diff * id_mask.float()).sum() / denom

        total = (
            self.w_style * loss_style
            + self.w_moment * loss_moment
            + self.w_identity * loss_idt
            + self.w_structure * loss_struct
        )

        return {
            "loss": total,
            "style_swd": loss_style.detach(),
            "style_moment": loss_moment.detach(),
            "identity": loss_idt.detach(),
            "structure": loss_struct.detach(),
            "tv": torch.tensor(0.0, device=content.device),
            "train_num_steps": torch.tensor(1.0, device=content.device),
            "train_step_size": torch.tensor(1.0, device=content.device),
            "train_style_strength": torch.tensor(1.0, device=content.device),
        }
