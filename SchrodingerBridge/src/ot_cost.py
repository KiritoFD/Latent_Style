from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from config_schema import BridgeConfig, ExperimentConfig


class SWDTransportCost:
    """
    Pairwise OT cost oracle built from the original SWD machinery.

    Compared with the earlier simplified port, this version keeps the old
    micro/macro decomposition and computes the coupling cost in float32 so the
    Hungarian plan is driven by a stable Wasserstein proxy instead of AMP noise.
    """

    def __init__(self, config: Dict | ExperimentConfig | BridgeConfig) -> None:
        if isinstance(config, ExperimentConfig):
            bridge_cfg = config.bridge
        elif isinstance(config, BridgeConfig):
            bridge_cfg = config
        else:
            bridge_cfg = BridgeConfig.from_mapping(config.get("bridge", {}))
        self.cost_mode = str(bridge_cfg.ot_cost_mode).strip().lower()
        self.swd_patch_sizes = [int(p) for p in bridge_cfg.swd_patch_sizes]
        self.swd_num_projections = int(bridge_cfg.swd_num_projections)
        self.swd_projection_chunk_size = int(bridge_cfg.swd_projection_chunk_size)
        self.swd_distance_mode = str(bridge_cfg.swd_distance_mode).strip().lower()
        self.swd_cdf_num_bins = max(4, int(bridge_cfg.swd_cdf_num_bins))
        self.swd_cdf_tau = max(1e-5, float(bridge_cfg.swd_cdf_tau))
        self.swd_use_high_freq = bool(bridge_cfg.swd_use_high_freq)
        self.swd_hf_weight_ratio = max(0.0, float(bridge_cfg.swd_hf_weight_ratio))
        self.swd_micro_patch_max = int(bridge_cfg.swd_micro_patch_max)
        default_macro_min = min(
            (p for p in self.swd_patch_sizes if p > self.swd_micro_patch_max),
            default=self.swd_micro_patch_max + 2,
        )
        self.swd_macro_patch_min = int(bridge_cfg.swd_macro_patch_min or default_macro_min)
        self.swd_micro_weight = max(0.0, float(bridge_cfg.swd_micro_weight))
        self.swd_macro_weight = max(0.0, float(bridge_cfg.swd_macro_weight))
        self.swd_use_dilated_projections = bool(bridge_cfg.swd_use_dilated_projections)
        self.swd_projection_dilation = max(1, int(bridge_cfg.swd_projection_dilation))
        self._projection_cache: Dict[tuple[int, int, int, str, str], torch.Tensor] = {}
        self._sobel_kernel_cache: Dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}

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

    def _get_sobel_kernels(
        self,
        channels: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(channels), str(device))
        cached = self._sobel_kernel_cache.get(key)
        if cached is not None:
            return cached
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        wx = kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        wy = ky.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        self._sobel_kernel_cache[key] = (wx, wy)
        return wx, wy

    def _compute_fused_hf_feature(self, z: torch.Tensor) -> torch.Tensor:
        wx, wy = self._get_sobel_kernels(int(z.shape[1]), device=z.device)
        gx = F.conv2d(z, wx, padding=1, groups=int(z.shape[1]))
        gy = F.conv2d(z, wy, padding=1, groups=int(z.shape[1]))
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
        return mag / (mag.mean(dim=(2, 3), keepdim=True) + 1e-5)

    def _prepare_micro_features(self, z_norm: torch.Tensor) -> torch.Tensor:
        if z_norm.shape[1] >= 2:
            base = z_norm[:, :2, :, :]
        else:
            base = z_norm
        high_pass = base - F.avg_pool2d(base, kernel_size=5, stride=1, padding=2)
        if not self.swd_use_high_freq:
            return high_pass
        hf = self._compute_fused_hf_feature(high_pass)
        return torch.cat([high_pass, hf * self.swd_hf_weight_ratio], dim=1)

    def _prepare_macro_features(self, z_norm: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(z_norm, kernel_size=5, stride=1, padding=2)

    def _pairwise_from_projected(self, x_proj: torch.Tensor, y_proj: torch.Tensor) -> torch.Tensor:
        x_proj = x_proj.float()
        y_proj = y_proj.float()
        if self.swd_distance_mode in {"cdf", "soft_cdf", "soft-cdf"}:
            return self._pairwise_from_projected_cdf(x_proj, y_proj)
        x_sorted, _ = torch.sort(x_proj, dim=2)
        y_sorted, _ = torch.sort(y_proj, dim=2)
        return (x_sorted.unsqueeze(1) - y_sorted.unsqueeze(0)).abs().mean(dim=(2, 3))

    def _aligned_from_projected(self, x_proj: torch.Tensor, y_proj: torch.Tensor) -> torch.Tensor:
        x_proj = x_proj.float()
        y_proj = y_proj.float()
        if x_proj.shape[0] != y_proj.shape[0]:
            raise ValueError(f"aligned SWD expects equal batch size, got {x_proj.shape[0]} vs {y_proj.shape[0]}")
        if self.swd_distance_mode in {"cdf", "soft_cdf", "soft-cdf"}:
            return self._aligned_from_projected_cdf(x_proj, y_proj)
        x_sorted, _ = torch.sort(x_proj, dim=2)
        y_sorted, _ = torch.sort(y_proj, dim=2)
        return (x_sorted - y_sorted).abs().mean()

    def _cdf_grid(self, x_proj: torch.Tensor, y_proj: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            lo = torch.minimum(x_proj.amin(dim=(0, 2)), y_proj.amin(dim=(0, 2)))
            hi = torch.maximum(x_proj.amax(dim=(0, 2)), y_proj.amax(dim=(0, 2)))
            pad = (hi - lo).clamp_min(1e-4) * 0.05
            lo = lo - pad
            hi = hi + pad
            steps = torch.linspace(0.0, 1.0, self.swd_cdf_num_bins, device=x_proj.device, dtype=torch.float32)
            return lo[:, None] + (hi - lo)[:, None] * steps[None, :]

    def _soft_cdf(self, proj: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # proj: [B, P, N], grid: [P, M] -> [B, P, M]
        tau = proj.detach().std(dim=2, unbiased=False).mean(dim=0).clamp_min(1e-4)
        tau = (tau * self.swd_cdf_tau).clamp_min(1e-5)
        logits = (grid[None, :, :, None] - proj[:, :, None, :]) / tau[None, :, None, None]
        return torch.sigmoid(logits).mean(dim=-1)

    def _pairwise_from_projected_cdf(self, x_proj: torch.Tensor, y_proj: torch.Tensor) -> torch.Tensor:
        grid = self._cdf_grid(x_proj, y_proj)
        x_cdf = self._soft_cdf(x_proj, grid)
        y_cdf = self._soft_cdf(y_proj, grid)
        return (x_cdf.unsqueeze(1) - y_cdf.unsqueeze(0)).abs().mean(dim=(2, 3))

    def _aligned_from_projected_cdf(self, x_proj: torch.Tensor, y_proj: torch.Tensor) -> torch.Tensor:
        grid = self._cdf_grid(x_proj, y_proj)
        x_cdf = self._soft_cdf(x_proj, grid)
        y_cdf = self._soft_cdf(y_proj, grid)
        return (x_cdf - y_cdf).abs().mean()

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
            dilation = self.swd_projection_dilation if self.swd_use_dilated_projections else 1
            padding = (patch_size // 2) * dilation
            for start in range(0, self.swd_num_projections, chunk):
                end = min(self.swd_num_projections, start + chunk)
                w = weights[start:end]
                x_proj = F.conv2d(x_feat, w, padding=padding, dilation=dilation).view(x_feat.shape[0], end - start, -1)
                y_proj = F.conv2d(y_feat, w, padding=padding, dilation=dilation).view(y_feat.shape[0], end - start, -1)
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
            dilation = self.swd_projection_dilation if self.swd_use_dilated_projections else 1
            padding = (patch_size // 2) * dilation
            for start in range(0, self.swd_num_projections, chunk):
                end = min(self.swd_num_projections, start + chunk)
                w = weights[start:end]
                x_proj = F.conv2d(x_feat, w, padding=padding, dilation=dilation).view(x_feat.shape[0], end - start, -1)
                y_proj = F.conv2d(y_feat, w, padding=padding, dilation=dilation).view(y_feat.shape[0], end - start, -1)
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

        if self.cost_mode in {"swd_unified", "swd_full", "unified"}:
            return self._branch_pairwise_cost(
                x_norm,
                y_norm,
                patch_sizes=self.swd_patch_sizes,
                mask_mode="luma_chroma_masked",
            )

        micro_patches = [p for p in self.swd_patch_sizes if p <= self.swd_micro_patch_max]
        macro_patches = [p for p in self.swd_patch_sizes if p >= self.swd_macro_patch_min]
        if not micro_patches and not macro_patches:
            return self._branch_pairwise_cost(
                x_norm,
                y_norm,
                patch_sizes=self.swd_patch_sizes,
                mask_mode="none",
            )

        total = torch.zeros((x.shape[0], y.shape[0]), device=x.device, dtype=torch.float32)
        weight_sum = 0.0
        if micro_patches and self.swd_micro_weight > 0.0:
            total = total + (
                self._branch_pairwise_cost(
                    self._prepare_micro_features(x_norm),
                    self._prepare_micro_features(y_norm),
                    patch_sizes=micro_patches,
                    mask_mode="none",
                )
                * self.swd_micro_weight
            )
            weight_sum += self.swd_micro_weight
        if macro_patches and self.swd_macro_weight > 0.0:
            total = total + (
                self._branch_pairwise_cost(
                    self._prepare_macro_features(x_norm),
                    self._prepare_macro_features(y_norm),
                    patch_sizes=macro_patches,
                    mask_mode="none",
                )
                * self.swd_macro_weight
            )
            weight_sum += self.swd_macro_weight
        if weight_sum <= 0.0:
            return self._branch_pairwise_cost(
                x_norm,
                y_norm,
                patch_sizes=self.swd_patch_sizes,
                mask_mode="none",
            )
        return total / float(weight_sum)

    def aligned_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.cost_mode == "l2":
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"aligned L2 expects equal batch size, got {x.shape[0]} vs {y.shape[0]}")
            return (x.flatten(1).float() - y.flatten(1).float()).pow(2).mean()

        x_f32 = x.float().contiguous()
        y_f32 = y.float().contiguous()
        x_norm = F.instance_norm(x_f32, eps=1e-3)
        y_norm = F.instance_norm(y_f32, eps=1e-3)

        if self.cost_mode in {"swd_unified", "swd_full", "unified"}:
            return self._branch_aligned_cost(
                x_norm,
                y_norm,
                patch_sizes=self.swd_patch_sizes,
                mask_mode="luma_chroma_masked",
            )

        micro_patches = [p for p in self.swd_patch_sizes if p <= self.swd_micro_patch_max]
        macro_patches = [p for p in self.swd_patch_sizes if p >= self.swd_macro_patch_min]
        if not micro_patches and not macro_patches:
            return self._branch_aligned_cost(
                x_norm,
                y_norm,
                patch_sizes=self.swd_patch_sizes,
                mask_mode="none",
            )

        total = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        weight_sum = 0.0
        if micro_patches and self.swd_micro_weight > 0.0:
            total = total + (
                self._branch_aligned_cost(
                    self._prepare_micro_features(x_norm),
                    self._prepare_micro_features(y_norm),
                    patch_sizes=micro_patches,
                    mask_mode="none",
                )
                * self.swd_micro_weight
            )
            weight_sum += self.swd_micro_weight
        if macro_patches and self.swd_macro_weight > 0.0:
            total = total + (
                self._branch_aligned_cost(
                    self._prepare_macro_features(x_norm),
                    self._prepare_macro_features(y_norm),
                    patch_sizes=macro_patches,
                    mask_mode="none",
                )
                * self.swd_macro_weight
            )
            weight_sum += self.swd_macro_weight
        if weight_sum <= 0.0:
            return self._branch_aligned_cost(
                x_norm,
                y_norm,
                patch_sizes=self.swd_patch_sizes,
                mask_mode="none",
            )
        return total / float(weight_sum)
