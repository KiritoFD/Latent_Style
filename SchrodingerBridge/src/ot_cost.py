from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class SWDTransportCost:
    """
    Pairwise OT cost oracle using full-band random projection SWD.
    """

    def __init__(self, config: Dict) -> None:
        bridge_cfg = config.get("bridge", {})
        self.cost_mode = str(bridge_cfg.get("ot_cost_mode", "swd")).strip().lower()
        self.swd_patch_sizes = [int(p) for p in bridge_cfg.get("swd_patch_sizes", [1, 3, 5, 9])]
        self.swd_num_projections = int(bridge_cfg.get("swd_num_projections", 64))
        self.swd_projection_chunk_size = int(bridge_cfg.get("swd_projection_chunk_size", 32))
        self.swd_distance_mode = str(bridge_cfg.get("swd_distance_mode", "cdf")).lower()
        self.swd_cdf_num_bins = int(bridge_cfg.get("swd_cdf_num_bins", 32))
        self.swd_cdf_tau = float(bridge_cfg.get("swd_cdf_tau", 0.01))
        self.swd_cdf_sample_size = int(bridge_cfg.get("swd_cdf_sample_size", 256))
        self.swd_cdf_bin_chunk_size = int(bridge_cfg.get("swd_cdf_bin_chunk_size", 4))
        self.swd_cdf_sample_chunk_size = int(bridge_cfg.get("swd_cdf_sample_chunk_size", 128))
        self.swd_deterministic_subsample = bool(bridge_cfg.get("swd_deterministic_subsample", True))
        self._projection_cache: Dict[tuple[int, int, int, str, str], torch.Tensor] = {}
        self._sample_idx_cache: Dict[tuple[int, int, str], torch.Tensor] = {}

    def _get_projection_bank(
        self,
        channels: int,
        *,
        device: torch.device,
        mask_mode: str = "none",
    ) -> Dict[int, torch.Tensor]:
        bank: Dict[int, torch.Tensor] = {}
        mode = str(mask_mode).strip().lower()
        for patch_size in self.swd_patch_sizes:
            key = (int(channels), int(patch_size), int(self.swd_num_projections), str(device), mode)
            weights = self._projection_cache.get(key)
            if weights is None:
                with torch.no_grad():
                    weights = torch.randn(
                        self.swd_num_projections,
                        channels,
                        patch_size,
                        patch_size,
                        device=device,
                        dtype=torch.float32,
                    )
                    if mode == "luma_chroma_masked" and channels >= 2:
                        luma_count = max(1, min(self.swd_num_projections - 1, int(self.swd_num_projections * 0.6)))
                        weights[:luma_count, 1:, :, :] = 0.0
                        weights[luma_count:, 0:1, :, :] = 0.0
                    weights = F.normalize(weights.view(self.swd_num_projections, -1), p=2, dim=1).view_as(weights)
                self._projection_cache[key] = weights
            bank[patch_size] = weights
        return bank

    def _select_sample_indices(self, n_pts: int, *, device: torch.device) -> torch.Tensor | None:
        sample_size = max(32, int(self.swd_cdf_sample_size))
        if n_pts <= sample_size:
            return None
        key = (int(n_pts), int(sample_size), str(device))
        cached = self._sample_idx_cache.get(key)
        if cached is not None:
            return cached
        if self.swd_deterministic_subsample:
            idx = (torch.arange(sample_size, device=device, dtype=torch.long) * n_pts) // sample_size
        else:
            idx = torch.randint(0, n_pts, (sample_size,), device=device)
        self._sample_idx_cache[key] = idx
        return idx

    def _pairwise_from_projected(self, x_proj: torch.Tensor, y_proj: torch.Tensor) -> torch.Tensor:
        x_proj = x_proj.float()
        y_proj = y_proj.float()
        mode = str(self.swd_distance_mode).lower()
        use_cdf = mode in {"cdf", "softcdf", "cdf_soft"}
        n_pts = int(x_proj.shape[-1])
        sample_idx = self._select_sample_indices(n_pts, device=x_proj.device)
        if sample_idx is not None:
            x_proj = x_proj.index_select(2, sample_idx)
            y_proj = y_proj.index_select(2, sample_idx)
            n_pts = int(x_proj.shape[-1])

        if not use_cdf:
            x_sorted, _ = torch.sort(x_proj, dim=2)
            y_sorted, _ = torch.sort(y_proj, dim=2)
            return (x_sorted.unsqueeze(1) - y_sorted.unsqueeze(0)).abs().mean(dim=(2, 3))

        bins = max(8, int(self.swd_cdf_num_bins))
        tau = max(1e-5, float(self.swd_cdf_tau))
        bin_chunk = max(1, int(self.swd_cdf_bin_chunk_size))
        sample_chunk = max(32, int(self.swd_cdf_sample_chunk_size))
        min_val = float(torch.minimum(x_proj.amin().detach(), y_proj.amin().detach()).item())
        max_val = float(torch.maximum(x_proj.amax().detach(), y_proj.amax().detach()).item())
        span = max(max_val - min_val, 1e-6)
        dx = span / float(bins - 1)
        grid = torch.linspace(min_val, max_val, bins, device=x_proj.device, dtype=torch.float32)
        bx, n_proj, _ = x_proj.shape
        by = int(y_proj.shape[0])
        acc_x = torch.zeros((bx, n_proj, bins), device=x_proj.device, dtype=torch.float32)
        acc_y = torch.zeros((by, n_proj, bins), device=y_proj.device, dtype=torch.float32)
        for b0 in range(0, bins, bin_chunk):
            b1 = min(bins, b0 + bin_chunk)
            g = grid[b0:b1].view(1, 1, 1, b1 - b0)
            bin_x = torch.zeros((bx, n_proj, b1 - b0), device=x_proj.device, dtype=torch.float32)
            bin_y = torch.zeros((by, n_proj, b1 - b0), device=y_proj.device, dtype=torch.float32)
            for n0 in range(0, n_pts, sample_chunk):
                n1 = min(n_pts, n0 + sample_chunk)
                bin_x = bin_x + torch.sigmoid((g - x_proj[:, :, n0:n1].unsqueeze(-1)) / tau).sum(dim=2)
                bin_y = bin_y + torch.sigmoid((g - y_proj[:, :, n0:n1].unsqueeze(-1)) / tau).sum(dim=2)
            acc_x[:, :, b0:b1] = bin_x
            acc_y[:, :, b0:b1] = bin_y
        cdf_x = acc_x / float(n_pts)
        cdf_y = acc_y / float(n_pts)
        return (cdf_x.unsqueeze(1) - cdf_y.unsqueeze(0)).abs().sum(dim=-1).mean(dim=-1) * dx

    def _aligned_from_projected(self, x_proj: torch.Tensor, y_proj: torch.Tensor) -> torch.Tensor:
        x_proj = x_proj.float()
        y_proj = y_proj.float()
        if x_proj.shape[0] != y_proj.shape[0]:
            raise ValueError(f"aligned SWD expects equal batch size, got {x_proj.shape[0]} vs {y_proj.shape[0]}")
        mode = str(self.swd_distance_mode).lower()
        use_cdf = mode in {"cdf", "softcdf", "cdf_soft"}
        n_pts = int(x_proj.shape[-1])
        sample_idx = self._select_sample_indices(n_pts, device=x_proj.device)
        if sample_idx is not None:
            x_proj = x_proj.index_select(2, sample_idx)
            y_proj = y_proj.index_select(2, sample_idx)
            n_pts = int(x_proj.shape[-1])

        if not use_cdf:
            x_sorted, _ = torch.sort(x_proj, dim=2)
            y_sorted, _ = torch.sort(y_proj, dim=2)
            return (x_sorted - y_sorted).abs().mean()

        bins = max(8, int(self.swd_cdf_num_bins))
        tau = max(1e-5, float(self.swd_cdf_tau))
        bin_chunk = max(1, int(self.swd_cdf_bin_chunk_size))
        sample_chunk = max(32, int(self.swd_cdf_sample_chunk_size))
        min_val = float(torch.minimum(x_proj.amin().detach(), y_proj.amin().detach()).item())
        max_val = float(torch.maximum(x_proj.amax().detach(), y_proj.amax().detach()).item())
        span = max(max_val - min_val, 1e-6)
        dx = span / float(bins - 1)
        grid = torch.linspace(min_val, max_val, bins, device=x_proj.device, dtype=torch.float32)
        bsz, n_proj, _ = x_proj.shape
        acc_x = torch.zeros((bsz, n_proj, bins), device=x_proj.device, dtype=torch.float32)
        acc_y = torch.zeros((bsz, n_proj, bins), device=y_proj.device, dtype=torch.float32)
        for b0 in range(0, bins, bin_chunk):
            b1 = min(bins, b0 + bin_chunk)
            g = grid[b0:b1].view(1, 1, 1, b1 - b0)
            bin_x = torch.zeros((bsz, n_proj, b1 - b0), device=x_proj.device, dtype=torch.float32)
            bin_y = torch.zeros((bsz, n_proj, b1 - b0), device=y_proj.device, dtype=torch.float32)
            for n0 in range(0, n_pts, sample_chunk):
                n1 = min(n_pts, n0 + sample_chunk)
                bin_x = bin_x + torch.sigmoid((g - x_proj[:, :, n0:n1].unsqueeze(-1)) / tau).sum(dim=2)
                bin_y = bin_y + torch.sigmoid((g - y_proj[:, :, n0:n1].unsqueeze(-1)) / tau).sum(dim=2)
            acc_x[:, :, b0:b1] = bin_x
            acc_y[:, :, b0:b1] = bin_y
        cdf_x = acc_x / float(n_pts)
        cdf_y = acc_y / float(n_pts)
        return (cdf_x - cdf_y).abs().sum(dim=-1).mean() * dx

    def _branch_pairwise_cost(
        self,
        x_feat: torch.Tensor,
        y_feat: torch.Tensor,
        *,
        patch_sizes: list[int],
        mask_mode: str = "none",
    ) -> torch.Tensor:
        if not patch_sizes:
            return torch.zeros((x_feat.shape[0], y_feat.shape[0]), device=x_feat.device, dtype=torch.float32)
        bank = self._get_projection_bank(int(x_feat.shape[1]), device=x_feat.device, mask_mode=mask_mode)
        total = torch.zeros((x_feat.shape[0], y_feat.shape[0]), device=x_feat.device, dtype=torch.float32)
        denom = max(1, len(patch_sizes))
        chunk = int(self.swd_projection_chunk_size)
        if chunk <= 0 or chunk >= self.swd_num_projections:
            chunk = self.swd_num_projections

        for patch_size in patch_sizes:
            weights = bank[patch_size]
            patch_cost = torch.zeros_like(total)
            for start in range(0, self.swd_num_projections, chunk):
                end = min(self.swd_num_projections, start + chunk)
                w = weights[start:end]
                x_proj = F.conv2d(x_feat, w, padding=patch_size // 2).view(x_feat.shape[0], end - start, -1)
                y_proj = F.conv2d(y_feat, w, padding=patch_size // 2).view(y_feat.shape[0], end - start, -1)
                patch_cost = patch_cost + self._pairwise_from_projected(x_proj, y_proj) * (
                    (end - start) / float(self.swd_num_projections)
                )
            total = total + patch_cost
        return total / float(denom)

    def _branch_aligned_cost(
        self,
        x_feat: torch.Tensor,
        y_feat: torch.Tensor,
        *,
        patch_sizes: list[int],
        mask_mode: str = "none",
    ) -> torch.Tensor:
        if not patch_sizes:
            return torch.tensor(0.0, device=x_feat.device, dtype=torch.float32)
        if x_feat.shape[0] != y_feat.shape[0]:
            raise ValueError(f"aligned SWD expects equal batch size, got {x_feat.shape[0]} vs {y_feat.shape[0]}")
        bank = self._get_projection_bank(int(x_feat.shape[1]), device=x_feat.device, mask_mode=mask_mode)
        total = torch.tensor(0.0, device=x_feat.device, dtype=torch.float32)
        denom = max(1, len(patch_sizes))
        chunk = int(self.swd_projection_chunk_size)
        if chunk <= 0 or chunk >= self.swd_num_projections:
            chunk = self.swd_num_projections

        for patch_size in patch_sizes:
            weights = bank[patch_size]
            patch_cost = torch.tensor(0.0, device=x_feat.device, dtype=torch.float32)
            for start in range(0, self.swd_num_projections, chunk):
                end = min(self.swd_num_projections, start + chunk)
                w = weights[start:end]
                x_proj = F.conv2d(x_feat, w, padding=patch_size // 2).view(x_feat.shape[0], end - start, -1)
                y_proj = F.conv2d(y_feat, w, padding=patch_size // 2).view(y_feat.shape[0], end - start, -1)
                patch_cost = patch_cost + self._aligned_from_projected(x_proj, y_proj) * (
                    (end - start) / float(self.swd_num_projections)
                )
            total = total + patch_cost
        return total / float(denom)

    def pairwise_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.cost_mode == "l2":
            return torch.cdist(x.flatten(1).float(), y.flatten(1).float(), p=2).pow(2)

        x_f32 = x.float().contiguous()
        y_f32 = y.float().contiguous()
        x_norm = F.instance_norm(x_f32, eps=1e-3)
        y_norm = F.instance_norm(y_f32, eps=1e-3)

        return self._branch_pairwise_cost(
            x_norm,
            y_norm,
            patch_sizes=self.swd_patch_sizes,
            mask_mode="none",
        )

    def aligned_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.cost_mode == "l2":
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"aligned L2 expects equal batch size, got {x.shape[0]} vs {y.shape[0]}")
            return (x.flatten(1).float() - y.flatten(1).float()).pow(2).mean()

        x_f32 = x.float().contiguous()
        y_f32 = y.float().contiguous()
        x_norm = F.instance_norm(x_f32, eps=1e-3)
        y_norm = F.instance_norm(y_f32, eps=1e-3)

        return self._branch_aligned_cost(
            x_norm,
            y_norm,
            patch_sizes=self.swd_patch_sizes,
            mask_mode="none",
        )
