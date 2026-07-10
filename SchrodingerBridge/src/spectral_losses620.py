"""FC-SB Phase 4 B2: Spectral ODE training objective.

3 per-subband FM losses (LL/LH/HL) + Sliced Wasserstein Distance (SWD) loss.

Spectral FM: per-subband flow matching, weights w_ll/w_lh/w_hl.
  Theory: w_ll≈0 (lock low-freq for LPIPS), w_lh/w_hl transfer mid-freq style.

SWD: endpoint distribution constraint on the predicted target z_hat1.
  Uses attention-weighted SWD when model provides pixel_entropy, enabling
  content-adaptive style transfer (cross-attn guided SWD).
  This is critical for distribution-level style matching: without SWD, the model
  only does point-wise FM matching without distribution-level constraints,
  leading to unnatural-looking outputs.

630 additions: SWD loss re-integrated from SpatialBridgeObjective620.
"""
from __future__ import annotations

import random
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from config_schema import ExperimentConfig
from spectral620 import dwt2_haar, idwt2_haar


def _lowpass(x: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    return F.avg_pool2d(x.float(), kernel_size=k, stride=1, padding=k // 2).to(dtype=x.dtype)


def _sliced_wasserstein(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    dirs: torch.Tensor,
    noise_sigma: float = 0.0,
    sample_weight: torch.Tensor | None = None,
    sample_size: int = 0,
) -> torch.Tensor:
    bsz, c, h, w = a.shape
    a_spatial = a.float().reshape(bsz, c, -1).transpose(1, 2)
    b_spatial = b.float().reshape(bsz, c, -1).transpose(1, 2)
    if sample_weight is not None:
        # Treat cross-attention guidance as a local empirical mass, not a feature
        # amplitude. This keeps latent values intact while focusing SWD where the
        # routing module actually edits content.
        flat_weight = sample_weight.detach().float()
        if flat_weight.ndim == 4:
            flat_weight = flat_weight.mean(dim=1).reshape(bsz, -1)
        else:
            flat_weight = flat_weight.reshape(bsz, -1)
        n = a_spatial.shape[1]
        if flat_weight.shape[0] == bsz and flat_weight.shape[1] == n:
            probs = flat_weight.clamp_min(1e-8)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            take = n if sample_size <= 0 else min(n, max(1, int(sample_size)))
            cdf = probs.cumsum(dim=1).contiguous()
            cdf[:, -1] = 1.0
            q = (torch.arange(take, device=a.device, dtype=cdf.dtype) + 0.5) / float(take)
            q = q.unsqueeze(0).expand(bsz, -1).contiguous()
            idx = torch.searchsorted(cdf, q, right=False).clamp_max(n - 1)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, c)
            a_spatial = a_spatial.gather(dim=1, index=gather_idx)
            b_spatial = b_spatial.gather(dim=1, index=gather_idx)
    proj_a = a_spatial @ dirs.t()
    proj_b = b_spatial @ dirs.t()
    if noise_sigma > 0.0:
        proj_a = proj_a + noise_sigma * torch.randn_like(proj_a)
        proj_b = proj_b + noise_sigma * torch.randn_like(proj_b)
    proj_a_sorted = torch.sort(proj_a, dim=1).values
    proj_b_sorted = torch.sort(proj_b, dim=1).values
    return (proj_a_sorted - proj_b_sorted).abs().mean()


def _kmeans_labels(feat: torch.Tensor, k: int, iters: int = 4) -> torch.Tensor:
    """Per-sample mini k-means on spatial features. feat: [B, N, D] -> labels [B, N].

    Cheap (K small, few iters, no grad) segmentation used to define semantic regions.
    Centroids are seeded by farthest-point-ish spread (evenly spaced sorted-by-norm picks)
    for determinism and stability across the batch.
    """
    bsz, n, d = feat.shape
    with torch.no_grad():
        # Seed: pick k anchors spread by feature norm ordering (stable, no RNG divergence).
        order = feat.norm(dim=2).argsort(dim=1)  # [B, N]
        pick = (torch.arange(k, device=feat.device) * (n - 1) // max(1, k - 1)).clamp_max(n - 1)
        seed_idx = order.gather(1, pick.unsqueeze(0).expand(bsz, -1))  # [B, k]
        centroids = feat.gather(1, seed_idx.unsqueeze(-1).expand(-1, -1, d)).clone()  # [B, k, D]
        labels = torch.zeros(bsz, n, device=feat.device, dtype=torch.long)
        for _ in range(max(1, iters)):
            # Assign: nearest centroid by squared distance.
            dist = torch.cdist(feat, centroids)  # [B, N, k]
            labels = dist.argmin(dim=2)  # [B, N]
            # Update: mean of assigned points (empty clusters keep old centroid).
            onehot = F.one_hot(labels, num_classes=k).to(feat.dtype)  # [B, N, k]
            counts = onehot.sum(dim=1).clamp_min(1.0)  # [B, k]
            new_c = torch.einsum("bnk,bnd->bkd", onehot, feat) / counts.unsqueeze(-1)
            empty = (onehot.sum(dim=1) < 0.5).unsqueeze(-1)
            centroids = torch.where(empty, centroids, new_c)
    return labels


def _semantic_region_swd(
    gen: torch.Tensor,
    target: torch.Tensor,
    *,
    seg_feat: torch.Tensor,
    num_regions: int,
    num_projections: int,
    kmeans_iters: int = 4,
    noise_sigma: float = 0.0,
) -> torch.Tensor:
    """Semantic region-matched SWD (vectorized: no per-sample Python loops).

    Groups spatial locations by content similarity (k-means on ``seg_feat``, the content
    latent) into ``num_regions`` regions. Each generated region is matched to the target
    region of best appearance correspondence: target locations are clustered independently
    and the two label sets are aligned by sorting centroids on their mean projection, so the
    k-th smoothest/darkest content region maps to the k-th target region. Matching within
    content-coherent regions keeps per-region statistics internally consistent, avoiding the
    muddy blend a single global marginal produces when incompatible regions share one match.

    Vectorization: loops only over K regions (not K×B). All pixels are pre-projected once
    (outside the region loop). Per region, per-batch masks are computed in parallel; region
    pixels are isolated via masked-sort (non-region set to +inf), then Q=256 fixed quantile
    positions are gathered (nearest-neighbor quantile matching). This eliminates ALL .item()
    GPU→CPU syncs and the Python batch loop while preserving the deterministic quantile
    matching semantics (stochastic multinomial sampling was verified to regress MUSIQ by
    ~10 points).

    All tensors [B, C, H, W]; seg_feat [B, C, H, W] (content latent, aligned to gen).
    """
    bsz, c, h, w = gen.shape
    n = h * w

    g_flat = gen.float().reshape(bsz, c, n).transpose(1, 2)      # [B, N, C]
    t_flat = target.float().reshape(bsz, c, n).transpose(1, 2)
    s_flat = seg_feat.float().reshape(bsz, seg_feat.shape[1], n).transpose(1, 2)

    g_labels = _kmeans_labels(s_flat, num_regions, iters=kmeans_iters)   # [B, N] content-defined
    t_labels = _kmeans_labels(t_flat, num_regions, iters=kmeans_iters)   # [B, N] appearance

    # Align region indices by centroid mean-projection order (shared appearance ordering).
    dirs = F.normalize(torch.randn(num_projections, c, device=gen.device, dtype=torch.float32), dim=1)
    with torch.no_grad():
        def _order(flat, labels):
            oh = F.one_hot(labels, num_regions).float()               # [B, N, K]
            cnt = oh.sum(1).clamp_min(1.0)                            # [B, K]
            cent = torch.einsum("bnk,bnc->bkc", oh, flat) / cnt.unsqueeze(-1)  # [B, K, C]
            score = cent.mean(dim=2)                                  # [B, K] mean-channel proxy
            return score.argsort(dim=1)                               # [B, K] region ids sorted
        g_ord = _order(g_flat, g_labels)   # [B, K] gen region id at rank r
        t_ord = _order(t_flat, t_labels)

    # Pre-project all pixels once (outside region loop) — avoids K*B per-region matmuls.
    g_proj = g_flat @ dirs.t()  # [B, N, P]
    t_proj = t_flat @ dirs.t()  # [B, N, P]
    if noise_sigma > 0.0:
        g_proj = g_proj + noise_sigma * torch.randn_like(g_proj)
        t_proj = t_proj + noise_sigma * torch.randn_like(t_proj)

    # Fixed quantile resolution for batched resampling (replaces per-batch F.interpolate
    # to variable max(ng, nt)). Nearest-neighbor quantile gather is statistically
    # equivalent for SWD and eliminates ALL .item() GPU→CPU syncs.
    Q = min(n, 256)
    q_pos = (torch.arange(Q, device=gen.device, dtype=torch.float32) + 0.5) / Q  # (0, 1)

    swd = gen.new_tensor(0.0)
    active = gen.new_tensor(0.0)
    # Only K iterations in Python; all per-batch work is fully vectorized (no B loop, no .item()).
    for r in range(num_regions):
        gk = g_ord[:, r]  # [B] region index for rank r
        tk = t_ord[:, r]  # [B]

        g_mask = g_labels == gk.unsqueeze(1)  # [B, N]
        t_mask = t_labels == tk.unsqueeze(1)  # [B, N]
        g_cnt = g_mask.sum(dim=1)             # [B]
        t_cnt = t_mask.sum(dim=1)             # [B]
        valid = (g_cnt >= 2) & (t_cnt >= 2)   # [B]

        g_cnt_safe = g_cnt.clamp_min(1)  # avoid div-by-zero for empty regions
        t_cnt_safe = t_cnt.clamp_min(1)

        # Masked sort: region pixels sorted ascending at front; non-region as +inf at back.
        g_fill = g_proj.masked_fill(~g_mask.unsqueeze(-1), float('inf'))  # [B, N, P]
        t_fill = t_proj.masked_fill(~t_mask.unsqueeze(-1), float('inf'))
        g_sorted = torch.sort(g_fill, dim=1).values  # [B, N, P]
        t_sorted = torch.sort(t_fill, dim=1).values  # [B, N, P]

        # Quantile gather: q_pos maps to index q_pos * (cnt - 1), clamped to [0, N-1].
        g_idx = (q_pos.unsqueeze(0) * (g_cnt_safe.float() - 1).unsqueeze(1)).long().clamp(max=n - 1)  # [B, Q]
        t_idx = (q_pos.unsqueeze(0) * (t_cnt_safe.float() - 1).unsqueeze(1)).long().clamp(max=n - 1)  # [B, Q]
        g_q = g_sorted.gather(1, g_idx.unsqueeze(-1).expand(-1, -1, num_projections))  # [B, Q, P]
        t_q = t_sorted.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, num_projections))  # [B, Q, P]

        # Zero out inf (empty-region positions) before L1; invalid batch items masked out.
        g_q = torch.where(torch.isinf(g_q), torch.zeros_like(g_q), g_q)
        t_q = torch.where(torch.isinf(t_q), torch.zeros_like(t_q), t_q)
        diff = (g_q - t_q).abs().mean(dim=(1, 2))  # [B]
        diff = diff * valid.float()
        swd = swd + diff.sum()
        active = active + valid.float()

    # No branch → no GPU sync. When active=0, swd=0/1=0.
    swd = swd / active.sum().clamp_min(1.0)
    return swd
























class SpectralODEObjective620:
    """Spectral ODE objective: per-subband FM losses + SWD distribution constraint."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.training = True
        self.bridge_cfg = config.bridge
        self.t_min = float(getattr(self.bridge_cfg, "t_min", 0.0))
        self.t_max = float(getattr(self.bridge_cfg, "t_max", 1.0))
        self.w_ll = float(getattr(self.bridge_cfg, "spectral_w_ll", 0.0))
        self.w_lh = float(getattr(self.bridge_cfg, "spectral_w_lh", 1.0))
        self.w_hl = float(getattr(self.bridge_cfg, "spectral_w_hl", 1.0))
        # HH velocity head supervision (only used when the model exposes v_dict["hh"],
        # i.e. model.enable_hh_head=True). HH is the finest diagonal high-frequency band —
        # the texture detail MUSIQ rewards most. It was previously frozen to content HH.
        self.w_hh = float(getattr(self.bridge_cfg, "spectral_w_hh", 2.0))
        self.loss_type = str(getattr(self.bridge_cfg, "loss_type", "mse")).lower().strip()
        self.structure_aligned_target = bool(getattr(self.bridge_cfg, "structure_aligned_target", False))

        # SWD parameters (from SpatialBridgeObjective620)
        self.single_step_swd_weight = float(getattr(self.bridge_cfg, "single_step_swd_weight", 8.0))
        self.single_step_edge_weight = float(getattr(self.bridge_cfg, "single_step_edge_weight", 0.1))
        self.w_endpoint_content = float(getattr(self.bridge_cfg, "w_endpoint_content", 1.0))
        self.w_endpoint_style = float(getattr(self.bridge_cfg, "w_endpoint_style", 8.0))
        self.w_pixel_color_match = float(getattr(self.bridge_cfg, "w_pixel_color_match", 0.0))
        self.w_channel_variance = float(getattr(self.bridge_cfg, "w_channel_variance", 0.0))
        # D1: Gram matrix style loss — captures inter-channel correlations (texture/brushstroke)
        # that SWD's marginal matching discards. Applied only on HF bands to avoid content damage.
        # w_gram_hf>0 enables it; w_gram_ll controls LL band (default 0 to protect content).
        self.w_gram_hf = float(getattr(self.bridge_cfg, "w_gram_hf", 0.0))
        self.w_gram_ll = float(getattr(self.bridge_cfg, "w_gram_ll", 0.0))
        # D2: High-order moment matching — per-channel skewness (3rd) and kurtosis (4th).
        # Gram captures 2nd-order inter-channel correlations; moments capture intra-channel
        # distribution SHAPE (asymmetry, tailedness) that both SWD marginals and Gram miss.
        self.w_moment_hf = float(getattr(self.bridge_cfg, "w_moment_hf", 0.0))
        self.w_moment_ll = float(getattr(self.bridge_cfg, "w_moment_ll", 0.0))
        # D6: Intrinsic style consistency loss — encode z_hat1 through model's own
        # intrinsic_style_cnn and match the global style vector to the reference's.
        # Creates direct style gradient without external pretrained models.
        self.w_style_consistency = float(getattr(self.bridge_cfg, "w_style_consistency", 0.0))
        self.terminal_swd_weight = float(getattr(self.bridge_cfg, "terminal_swd_weight", 0.1))
        self.semantic_supervision_family = str(
            getattr(self.bridge_cfg, "semantic_supervision_family", "legacy_terminal_swd")
        ).strip().lower()
        self.num_projections = int(getattr(self.bridge_cfg, "semantic_swd_num_projections", 64))
        self.lowpass_kernel = int(getattr(self.bridge_cfg, "training_target_projection_kernel", 5))
        self.training_target_projection_mode = str(
            getattr(self.bridge_cfg, "training_target_projection_mode", "legacy")
        ).strip().lower()
        self.low_anchor = float(getattr(self.bridge_cfg, "training_target_projection_low_anchor", 1.0))
        self.low_mode = str(getattr(self.bridge_cfg, "training_target_projection_low_mode", "all")).strip().lower()
        self.swd_scale_mode = str(getattr(self.bridge_cfg, "swd_scale_mode", "global")).strip().lower()
        self.swd_noise_sigma = float(getattr(self.bridge_cfg, "swd_noise_sigma", 0.0))
        self.swd_guidance_source = str(getattr(self.bridge_cfg, "swd_guidance_source", "style_delta")).strip().lower()
        self.swd_guidance_floor = max(0.0, min(1.0, float(getattr(self.bridge_cfg, "swd_guidance_floor", 0.25))))
        self.swd_guidance_power = max(1e-3, float(getattr(self.bridge_cfg, "swd_guidance_power", 1.0)))
        self.swd_guidance_sample_size = int(
            getattr(self.bridge_cfg, "swd_guidance_sample_size", getattr(self.bridge_cfg, "swd_cdf_sample_size", 256))
        )
        # Optional hard-region SWD partitions the content latent with deterministic
        # k-means and blends region-wise matching with the default global SWD.
        self.swd_semantic_mode = str(getattr(self.bridge_cfg, "swd_semantic_mode", "off")).strip().lower()
        if self.swd_semantic_mode not in {"off", "region"}:
            raise ValueError(
                f"Unsupported swd_semantic_mode={self.swd_semantic_mode!r}; "
                "supported modes are 'off' and 'region'"
            )
        self.swd_semantic_regions = max(2, int(getattr(self.bridge_cfg, "swd_semantic_regions", 4)))
        self.swd_semantic_kmeans_iters = max(1, int(getattr(self.bridge_cfg, "swd_semantic_kmeans_iters", 4)))
        self.swd_semantic_blend = max(0.0, min(1.0, float(getattr(self.bridge_cfg, "swd_semantic_blend", 0.5))))
        self.w_style_strength_reg = float(getattr(self.bridge_cfg, "w_style_strength_reg", 0.0))
        self.bridge_sigma = float(getattr(self.bridge_cfg, "bridge_sigma", 0.0))
        self._base_bridge_sigma = self.bridge_sigma
        self.training_sde_noise_mode = str(
            getattr(self.bridge_cfg, "training_sde_noise_mode", "subtractive")
        ).strip().lower()
        self.training_objective_mode = str(
            getattr(self.bridge_cfg, "training_objective_mode", "velocity")
        ).strip().lower()

        self.last_debug: dict = {}

    def _sample_t(self, content: torch.Tensor) -> torch.Tensor:
        lo = max(0.0, min(1.0, self.t_min))
        hi = max(lo + 1e-4, min(1.0, self.t_max))
        return torch.rand(content.shape[0], device=content.device, dtype=content.dtype) * (hi - lo) + lo

    def _fm_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type in ("huber", "smooth_l1", "smoothl1"):
            return F.smooth_l1_loss(pred.float(), target.float())
        return F.mse_loss(pred.float(), target.float())

    def _gram_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """D1: Gram matrix style loss.

        Computes per-sample channel self-correlation Gram matrix (C×C) for pred and target,
        returns L1 distance. Captures inter-channel texture correlations that SWD's marginal
        matching discards. Normalized by C to keep scale stable across channel counts.

        Args:
            pred: (B, C, H, W) predicted features (e.g. z_hat1 or a DWT subband)
            target: (B, C, H, W) target features
        Returns:
            scalar L1 distance between mean-normalized Gram matrices.
        """
        B, C, H, W = pred.shape
        pred_f = pred.float().reshape(B, C, H * W)
        tgt_f = target.float().reshape(B, C, H * W)
        # Normalize features per-sample to prevent scale drift
        pred_f = pred_f / (pred_f.std(dim=2, keepdim=True).clamp_min(eps))
        tgt_f = tgt_f / (tgt_f.std(dim=2, keepdim=True).clamp_min(eps))
        # Gram = F F^T / N  -> (B, C, C)
        gram_pred = torch.bmm(pred_f, pred_f.transpose(1, 2)) / (H * W)
        gram_tgt = torch.bmm(tgt_f, tgt_f.transpose(1, 2)) / (H * W)
        return (gram_pred - gram_tgt).abs().mean() / max(C, 1)

    def _moment_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """D2: High-order moment matching loss (skewness + kurtosis).

        Per-channel 3rd (skewness) and 4th (kurtosis) standardized moments capture
        distribution SHAPE — asymmetry and tailedness — that SWD per-pixel marginals
        and Gram inter-channel correlations both miss. Applied per-sample, normalized
        by std to isolate shape from scale.

        Returns mean L1 distance over (skewness, kurtosis) pairs.
        """
        B, C, H, W = pred.shape
        pred_f = pred.float().reshape(B, C, H * W)
        tgt_f = target.float().reshape(B, C, H * W)
        # Standardize per-channel (zero mean, unit std) to isolate shape
        pred_mean = pred_f.mean(dim=2, keepdim=True)
        tgt_mean = tgt_f.mean(dim=2, keepdim=True)
        pred_std = pred_f.std(dim=2, keepdim=True).clamp_min(eps)
        tgt_std = tgt_f.std(dim=2, keepdim=True).clamp_min(eps)
        pred_norm = (pred_f - pred_mean) / pred_std
        tgt_norm = (tgt_f - tgt_mean) / tgt_std
        N = max(1, H * W)
        # Skewness: E[X^3] (3rd standardized moment)
        pred_skew = (pred_norm ** 3).mean(dim=2)  # (B, C)
        tgt_skew = (tgt_norm ** 3).mean(dim=2)
        # Kurtosis: E[X^4] - 3 (excess kurtosis, 4th standardized moment)
        pred_kurt = (pred_norm ** 4).mean(dim=2) - 3.0
        tgt_kurt = (tgt_norm ** 4).mean(dim=2) - 3.0
        skew_diff = (pred_skew - tgt_skew).abs().mean()
        kurt_diff = (pred_kurt - tgt_kurt).abs().mean()
        return (skew_diff + kurt_diff) * 0.5

    def _projection_dirs(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fresh random projection directions for Sliced Wasserstein Distance.

        SWD approximates the Wasserstein distance by averaging over RANDOM 1D
        projections. Caching/deterministic dirs breaks the theoretical guarantee
        and lets the model overfit to fixed marginals — empirically this makes
        SWD harmful (removing it improves CLIP-S). Each call MUST return fresh
        random dirs.
        """
        c = tensor.shape[1]
        device = tensor.device
        n = self.num_projections
        dirs = torch.randn(n, c, device=device, dtype=torch.float32)
        return F.normalize(dirs, dim=1)

    def _dwt_energy_weight(self, content: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Content-adaptive SWD weight from DWT high-frequency energy.

        Computes |LH|+|HL|+|HH| of the content latent, upsamples to full
        resolution, and normalizes to mean=1. This guides SWD to focus on
        texture-rich regions (edges, details) — exactly the regions MUSIQ
        rewards for sharpness/naturalness. Smooth regions (sky, flat areas)
        get low weight, avoiding over-matching there.
        """
        _, lh, hl, hh = dwt2_haar(content.detach().float())
        energy = (lh.abs() + hl.abs() + hh.abs()).mean(dim=1, keepdim=True)  # [B,1,H/2,W/2]
        weight = F.interpolate(energy, size=like.shape[-2:], mode="bilinear", align_corners=False)
        weight = weight.to(dtype=like.dtype, device=like.device)
        weight = weight.clamp_min(1e-8)
        weight = weight / weight.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        if self.swd_guidance_power != 1.0:
            weight = weight.pow(self.swd_guidance_power)
            weight = weight / weight.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        if self.swd_guidance_floor > 0.0:
            weight = self.swd_guidance_floor + (1.0 - self.swd_guidance_floor) * weight
        return weight

    def _cross_attn_swd_weight(self, model, like: torch.Tensor, content: torch.Tensor | None = None) -> torch.Tensor | None:
        # DWT-energy guidance: content-adaptive, focuses SWD on texture-rich regions
        if self.swd_guidance_source in {"dwt_energy", "dwt-energy"} and content is not None:
            return self._dwt_energy_weight(content, like)
        # Combined: cross-attn entropy × DWT energy (element-wise product, renormalized)
        if self.swd_guidance_source in {"cross_attn_plus_dwt", "cross-attn-plus-dwt"} and content is not None:
            dwt_w = self._dwt_energy_weight(content, like)
            if self.swd_guidance_source in {"entropy", "pixel_entropy", "attention_entropy"}:
                guidance = getattr(model, "last_pixel_entropy", None)
            else:
                guidance = getattr(model, "last_cross_attn_guidance", None)
                if guidance is None:
                    guidance = getattr(model, "last_pixel_entropy", None)
            if guidance is None or not torch.is_tensor(guidance):
                return dwt_w  # fallback to DWT-only if cross-attn unavailable
            attn_w = guidance.detach().to(device=like.device, dtype=like.dtype)
            if attn_w.shape[-2:] != like.shape[-2:]:
                attn_w = F.interpolate(attn_w, size=like.shape[-2:], mode="bilinear", align_corners=False)
            attn_w = attn_w.float().abs().clamp_min(1e-8)
            attn_w = attn_w / attn_w.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            combined = (dwt_w.float() * attn_w).clamp_min(1e-8)
            combined = combined / combined.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            return combined.to(dtype=like.dtype)
        if self.swd_guidance_source in {"entropy", "pixel_entropy", "attention_entropy"}:
            guidance = getattr(model, "last_pixel_entropy", None)
        else:
            guidance = getattr(model, "last_cross_attn_guidance", None)
            if guidance is None:
                guidance = getattr(model, "last_pixel_entropy", None)
        if guidance is None or not torch.is_tensor(guidance):
            return None

        weight = guidance.detach().to(device=like.device, dtype=like.dtype)
        if weight.shape[-2:] != like.shape[-2:]:
            weight = F.interpolate(weight, size=like.shape[-2:], mode="bilinear", align_corners=False)
        weight = weight.float().abs().clamp_min(1e-8)
        weight = weight / weight.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        if self.swd_guidance_power != 1.0:
            weight = weight.pow(self.swd_guidance_power)
            weight = weight / weight.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        if self.swd_guidance_floor > 0.0:
            weight = self.swd_guidance_floor + (1.0 - self.swd_guidance_floor) * weight
        return weight.to(dtype=like.dtype)

    def _target_projection(self, content: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply training target projection (legacy: low-freq from content, high-freq from target)."""
        if self.training_target_projection_mode in ("legacy",):
            if self.low_anchor < 1.0:
                return self.low_anchor * _lowpass(content, self.lowpass_kernel) + (
                    1.0 - self.low_anchor
                ) * (target - _lowpass(target, self.lowpass_kernel))
            else:
                if self.low_mode == "all":
                    return target
                elif self.low_mode == "channel_mean":
                    cm = content.float().mean(dim=[2, 3], keepdim=True)
                    return cm + (target.float() - target.float().mean(dim=[2, 3], keepdim=True))
                else:
                    return target
        # Default: no projection
        return target

    def _compute_swd(
        self,
        z_hat1: torch.Tensor,
        projected_target: torch.Tensor,
        model,
        content: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute global SWD, optionally blended with hard content regions."""
        swd_guidance_active = z_hat1.new_tensor(0.0)
        swd_guidance_mean = z_hat1.new_tensor(0.0)
        swd_guidance_std = z_hat1.new_tensor(0.0)
        guided = self.swd_scale_mode in {
            "cross-attn-guided", "cross_attn_guided", "crossattn-guided", "crossattn_guided"
        }
        weight = self._cross_attn_swd_weight(model, z_hat1, content=content) if guided else None
        if weight is not None:
            swd_guidance_active = z_hat1.new_tensor(1.0)
            swd_guidance_mean = weight.detach().float().mean()
            swd_guidance_std = weight.detach().float().std()

        global_swd = _sliced_wasserstein(
            z_hat1,
            projected_target,
            dirs=self._projection_dirs(z_hat1),
            noise_sigma=self.swd_noise_sigma,
            sample_weight=weight,
            sample_size=self.swd_guidance_sample_size,
        )
        swd = global_swd
        if self.swd_semantic_mode == "region" and content is not None:
            region_swd = _semantic_region_swd(
                z_hat1,
                projected_target,
                seg_feat=content,
                num_regions=self.swd_semantic_regions,
                num_projections=self.num_projections,
                kmeans_iters=self.swd_semantic_kmeans_iters,
                noise_sigma=self.swd_noise_sigma,
            )
            beta = self.swd_semantic_blend
            swd = (1.0 - beta) * global_swd + beta * region_swd

        edge = F.l1_loss(
            (z_hat1 - _lowpass(z_hat1, self.lowpass_kernel)).float(),
            (projected_target - _lowpass(projected_target, self.lowpass_kernel)).float(),
        )
        return swd, edge, swd_guidance_active, swd_guidance_mean, swd_guidance_std

    def compute(
        self, model, *, content, target_style, target_style_id,
        source_style_id=None, aux_target_style=None, aux_target_valid=None,
        conditioning=None, **_,
    ) -> Dict[str, torch.Tensor]:
        del source_style_id, aux_target_style, aux_target_valid
        conditioning = conditioning or {}
        style_text_tokens = conditioning.get("target_style_text_tokens")
        style_latent = conditioning.get("target_style_latent")
        if not torch.is_tensor(style_text_tokens):
            style_text_tokens = None
        if not torch.is_tensor(style_latent):
            # Packed latent training batches expose the reference image as
            # target_style. Use it as the default local style reference so
            # guided SWD is driven by the sampled target, not only style_id.
            style_latent = target_style

        target = target_style
        # Structure-aligned target: x₁* = IDWT(LL_content, LH_style, HL_style, HH_style)
        if self.structure_aligned_target:
            ll_c, _, _, _ = dwt2_haar(content)
            _, lh_t, hl_t, hh_t = dwt2_haar(target)
            target = idwt2_haar(ll_c, lh_t, hl_t, hh_t)

        t = self._sample_t(content)
        t_view = t.view(-1, 1, 1, 1).to(dtype=content.dtype)

        # Bridge noise
        if self.bridge_sigma > 0.0:
            noise = torch.randn_like(content) * self.bridge_sigma
            if self.training_sde_noise_mode == "subtractive":
                x_t = (1.0 - t_view) * content + t_view * target - noise * (t_view * (1.0 - t_view)).sqrt()
            else:
                x_t = (1.0 - t_view) * content + t_view * target + noise * (t_view * (1.0 - t_view)).sqrt()
        else:
            x_t = (1.0 - t_view) * content + t_view * target

        target_delta = target - content
        target_ll, target_lh, target_hl, target_hh = dwt2_haar(target_delta)

        v_dict = model(
            x_t, t=t, style_id=target_style_id,
            style_latent=style_latent,
            style_text_tokens=style_text_tokens,
        )

        # Spectral FM losses
        loss_ll = self._fm_loss(v_dict["ll"], target_ll)
        loss_lh = self._fm_loss(v_dict["lh"], target_lh)
        loss_hl = self._fm_loss(v_dict["hl"], target_hl)
        loss_fm = self.w_ll * loss_ll + self.w_lh * loss_lh + self.w_hl * loss_hl
        # HH velocity supervision (only when the model exposes an HH head). HH is the finest
        # diagonal high-frequency band — the texture/detail MUSIQ rewards most — which was
        # frozen to content while no head predicted it. w_hh defaults to the config's highest
        # band weight, so re-enabling it routes learning capacity to that band.
        loss_hh = content.new_tensor(0.0)
        if "hh" in v_dict:
            loss_hh = self._fm_loss(v_dict["hh"], target_hh)
            loss_fm = loss_fm + self.w_hh * loss_hh

        # Endpoint prediction: z_hat1 = content + IDWT(v_ll, v_lh, v_hl, v_hh)
        # For spectral ODE, reconstruct from model output
        v_hh = v_dict.get("hh", torch.zeros_like(target_ll))
        z_hat1 = content + idwt2_haar(v_dict["ll"], v_dict["lh"], v_dict["hl"], v_hh)

        # Projected target for SWD
        projected_target = self._target_projection(content, target)

        # SWD loss (with attention-weighting support)
        swd_ss, edge_ss, swd_guidance_active, swd_guidance_mean, swd_guidance_std = self._compute_swd(z_hat1, projected_target, model, content=content)

        # Endpoint content loss (low-freq LPIPS anchor)
        loss_endpoint_content = F.mse_loss(
            _lowpass(z_hat1, self.lowpass_kernel).float(),
            _lowpass(projected_target, self.lowpass_kernel).float(),
        )

        # Pixel color match (per-channel mean/std alignment)
        loss_pixel_color = content.new_tensor(0.0)
        if self.w_pixel_color_match > 0.0:
            gen_mean = z_hat1.float().mean(dim=[2, 3])
            gen_std = z_hat1.float().std(dim=[2, 3])
            tgt_mean = projected_target.float().mean(dim=[2, 3])
            tgt_std = projected_target.float().std(dim=[2, 3])
            loss_pixel_color = F.mse_loss(gen_mean, tgt_mean) + F.mse_loss(gen_std, tgt_std)

        # Channel variance matching
        loss_channel_var = content.new_tensor(0.0)
        if self.w_channel_variance > 0.0:
            gen_var = z_hat1.float().var(dim=[2, 3])
            tgt_var = projected_target.float().var(dim=[2, 3])
            loss_channel_var = F.mse_loss(gen_var, tgt_var)

        # D1+D2: Gram matrix + high-order moment losses on DWT subbands.
        # DWT decomposition computed once and shared between D1 (Gram, 2nd-order inter-channel)
        # and D2 (moments, 3rd/4th-order intra-channel distribution shape).
        loss_gram = content.new_tensor(0.0)
        loss_moment = content.new_tensor(0.0)
        need_dwt = (self.w_gram_hf > 0.0 or self.w_gram_ll > 0.0
                    or self.w_moment_hf > 0.0 or self.w_moment_ll > 0.0)
        if need_dwt:
            pred_delta = z_hat1 - content  # IDWT(v) = predicted delta
            pred_ll, pred_lh, pred_hl, pred_hh = dwt2_haar(pred_delta)
            # D1: Gram matrix
            if self.w_gram_hf > 0.0 or self.w_gram_ll > 0.0:
                gram_terms = []
                if self.w_gram_hf > 0.0:
                    gram_terms.append(self.w_gram_hf * (
                        self._gram_loss(pred_lh, target_lh)
                        + self._gram_loss(pred_hl, target_hl)
                        + self._gram_loss(pred_hh, target_hh)
                    ))
                if self.w_gram_ll > 0.0:
                    gram_terms.append(self.w_gram_ll * self._gram_loss(pred_ll, target_ll))
                loss_gram = sum(gram_terms)
            # D2: High-order moments (skewness + kurtosis)
            if self.w_moment_hf > 0.0 or self.w_moment_ll > 0.0:
                moment_terms = []
                if self.w_moment_hf > 0.0:
                    moment_terms.append(self.w_moment_hf * (
                        self._moment_loss(pred_lh, target_lh)
                        + self._moment_loss(pred_hl, target_hl)
                        + self._moment_loss(pred_hh, target_hh)
                    ))
                if self.w_moment_ll > 0.0:
                    moment_terms.append(self.w_moment_ll * self._moment_loss(pred_ll, target_ll))
                loss_moment = sum(moment_terms)

        # D6: Intrinsic style consistency loss — encode predicted endpoint through the
        # model's own intrinsic_style_cnn and match its global style vector to the
        # reference's. This creates a direct style gradient signal that SWD's marginal
        # distribution matching does not provide. Uses only model-internal features
        # (no external pretrained models → no prior contamination).
        loss_style_consist = content.new_tensor(0.0)
        if (self.w_style_consistency > 0.0
                and hasattr(model, "intrinsic_style_cnn")
                and model.intrinsic_style_cnn is not None
                and model.intrinsic_style_pool is not None
                and model.intrinsic_style_global is not None
                and "style_global" in v_dict):
            ref_style_global = v_dict["style_global"].detach()
            gen_feat = model.intrinsic_style_cnn(z_hat1.float())
            gen_feat = model.intrinsic_style_pool(gen_feat)
            gen_global = model.intrinsic_style_global(gen_feat.mean(dim=[2, 3]).float())
            gen_global = gen_global.to(dtype=ref_style_global.dtype)
            loss_style_consist = (
                1.0 - F.cosine_similarity(gen_global, ref_style_global, dim=-1)
            ).mean()

        # Total loss
        loss = (
            loss_fm
            + self.single_step_swd_weight * swd_ss
            + self.single_step_edge_weight * edge_ss
            + self.w_endpoint_content * loss_endpoint_content
            + self.w_pixel_color_match * loss_pixel_color
            + self.w_channel_variance * loss_channel_var
            + loss_gram
            + loss_moment
            + self.w_style_consistency * loss_style_consist
        )

        zero = content.new_tensor(0.0)
        metrics: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_fm_spectral_ll": loss_ll.detach(),
            "loss_fm_spectral_lh": loss_lh.detach(),
            "loss_fm_spectral_hl": loss_hl.detach(),
            "loss_swd": swd_ss.detach(),
            "loss_swd_ss": swd_ss.detach(),
            "loss_edge": edge_ss.detach(),
            "loss_endpoint_content": loss_endpoint_content.detach(),
            "loss_pixel_color": loss_pixel_color.detach() if isinstance(loss_pixel_color, torch.Tensor) else zero,
            "loss_gram": loss_gram.detach() if isinstance(loss_gram, torch.Tensor) else zero,
            "loss_moment": loss_moment.detach() if isinstance(loss_moment, torch.Tensor) else zero,
            "loss_style_consist": loss_style_consist.detach() if isinstance(loss_style_consist, torch.Tensor) else zero,
            "swd_guidance_active": swd_guidance_active.detach(),
            "swd_guidance_mean": swd_guidance_mean.detach(),
            "swd_guidance_std": swd_guidance_std.detach(),
            "t_mean": t.detach().float().mean(),
            "flow": loss_fm.detach(),
            "terminal_swd": swd_ss.detach(),
            "single_step_swd": (swd_ss * self.single_step_swd_weight).detach(),
            "single_step_edge": (edge_ss * self.single_step_edge_weight).detach(),
            "ot_cost": zero,
            "kinetic_energy": zero,
            "curvature": zero,
        }
        self.last_debug = {
            key: value.detach().float() if value.numel() == 1 else 0.0
            for key, value in metrics.items()
        }
        return metrics

    def update_weights_for_epoch(self, epoch: int, num_epochs: int = 3) -> dict:
        return {
            "stage": 0, "bridge_sigma": self._base_bridge_sigma,
            "w_endpoint_content": self.w_endpoint_content,
            "w_endpoint_style": self.w_endpoint_style,
            "w_style_strength_reg": self.w_style_strength_reg,
        }
