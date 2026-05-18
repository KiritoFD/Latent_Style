from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class SWDTransportCost:
    """Minimal SWD oracle for batch-level OT pairing."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config.get("bridge", {}) if config else {}
        self.cost_mode = str(cfg.get("ot_cost_mode", "swd")).strip().lower()
        self.swd_patch_sizes = [int(p) for p in cfg.get("swd_patch_sizes", [1, 3, 5, 9])]
        self.swd_num_projections = int(cfg.get("swd_num_projections", 64))
        self._projection_cache: dict[tuple, torch.Tensor] = {}

    def _get_projection_bank(self, channels: int, device: torch.device) -> dict[int, torch.Tensor]:
        bank: dict[int, torch.Tensor] = {}
        for patch_size in self.swd_patch_sizes:
            key = (channels, patch_size, self.swd_num_projections, str(device))
            weights = self._projection_cache.get(key)
            if weights is None:
                weights = torch.randn(self.swd_num_projections, channels, patch_size, patch_size, device=device, dtype=torch.float32)
                weights = F.normalize(weights.view(self.swd_num_projections, -1), p=2, dim=1).view_as(weights)
                self._projection_cache[key] = weights
            bank[patch_size] = weights
        return bank

    def _batch_pairwise_cost(self, x_proj: torch.Tensor, y_proj: torch.Tensor) -> torch.Tensor:
        x_sorted, _ = torch.sort(x_proj.float(), dim=-1)
        y_sorted, _ = torch.sort(y_proj.float(), dim=-1)
        return (x_sorted.unsqueeze(1) - y_sorted.unsqueeze(0)).abs().mean(dim=(2, 3))

    def _batch_aligned_cost(self, x_proj: torch.Tensor, y_proj: torch.Tensor) -> torch.Tensor:
        x_sorted, _ = torch.sort(x_proj.float(), dim=-1)
        y_sorted, _ = torch.sort(y_proj.float(), dim=-1)
        return (x_sorted - y_sorted).abs().mean()

    def _project(self, feat: torch.Tensor, patch_size: int, weights: torch.Tensor) -> torch.Tensor:
        return F.conv2d(feat, weights, padding=patch_size // 2).view(feat.shape[0], self.swd_num_projections, -1)

    def pairwise_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.cost_mode == "l2":
            return torch.cdist(x.flatten(1).float(), y.flatten(1).float(), p=2).pow(2)
        if not self.swd_patch_sizes:
            return torch.zeros((x.shape[0], y.shape[0]), device=x.device, dtype=torch.float32)
        x_norm = F.instance_norm(x.float().contiguous(), eps=1e-3)
        y_norm = F.instance_norm(y.float().contiguous(), eps=1e-3)
        bank = self._get_projection_bank(int(x.shape[1]), device=x.device)
        total = torch.zeros((x.shape[0], y.shape[0]), device=x.device, dtype=torch.float32)
        for ps in self.swd_patch_sizes:
            xp = self._project(x_norm, ps, bank[ps])
            yp = self._project(y_norm, ps, bank[ps])
            total = total + self._batch_pairwise_cost(xp, yp)
        return total / max(1, len(self.swd_patch_sizes))

    def aligned_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.cost_mode == "l2":
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"aligned L2 expects equal batch size, got {x.shape[0]} vs {y.shape[0]}")
            return (x.flatten(1).float() - y.flatten(1).float()).pow(2).mean()
        if not self.swd_patch_sizes:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"aligned SWD expects equal batch size, got {x.shape[0]} vs {y.shape[0]}")
        x_norm = F.instance_norm(x.float().contiguous(), eps=1e-3)
        y_norm = F.instance_norm(y.float().contiguous(), eps=1e-3)
        bank = self._get_projection_bank(int(x.shape[1]), device=x.device)
        total = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        for ps in self.swd_patch_sizes:
            xp = self._project(x_norm, ps, bank[ps])
            yp = self._project(y_norm, ps, bank[ps])
            total = total + self._batch_aligned_cost(xp, yp)
        return total / max(1, len(self.swd_patch_sizes))
