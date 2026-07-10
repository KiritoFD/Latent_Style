"""FC-SB Phase 4 B2: Spectral ODE training objective.

3 per-subband FM losses (LL/LH/HL) + Sliced Wasserstein Distance (SWD) loss.

Spectral FM: per-subband flow matching, weights w_ll/w_lh/w_hl.
  Theory: w_ll≈0 (lock low-freq for LPIPS), w_lh/w_hl transfer mid-freq style.

SWD: endpoint distribution constraint on the predicted target z_hat1.
  Uses attention-weighted SWD when model provides pixel_entropy, enabling
  content-adaptive style transfer (cross-attn guided SWD).
  This is critical for perceptual quality (MUSIQ): without SWD, the model
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


def _semantic_region_swd_softmask(
    gen: torch.Tensor,
    target: torch.Tensor,
    *,
    seg_feat: torch.Tensor,
    num_regions: int,
    num_projections: int,
    kmeans_iters: int = 4,
    noise_sigma: float = 0.0,
    softmax_temp: float = 1.0,
    min_weight: float = 0.05,
) -> torch.Tensor:
    """Soft-mask semantic region SWD.

    Mechanism change vs _semantic_region_swd:
      - Hard k-means argmax labels → soft softmax membership probabilities.
      - Each pixel contributes to ALL regions with weight p_k (clamped to min_weight
        to avoid degenerate zero-contribution).
      - Within-region SWD uses probability-weighted sliced Wasserstein instead of
        boolean-masked gather + sort + F.interpolate.

    This breaks the hard partition boundary: a pixel "between sky and skin" now
    contributes fractionally to both regions, preserving transition information
    that the hard mask destroys. The coupling set Π_soft is a relaxation of Π_r
    that allows weak cross-region transport, interpolating between global OT (Π)
    and strict region OT (Π_r).

    Args:
        softmax_temp: temperature for membership softmax. temp→0 recovers hard
            k-means; temp→∞ makes all memberships uniform (degenerates to global).
        min_weight: floor on per-pixel per-region membership weight, so every
            pixel contributes at least weakly to every region's statistics.

    All tensors [B, C, H, W]; seg_feat [B, C', H, W].
    """
    bsz, c, h, w = gen.shape
    n = h * w

    g_flat = gen.float().reshape(bsz, c, n).transpose(1, 2)      # [B, N, C]
    t_flat = target.float().reshape(bsz, c, n).transpose(1, 2)
    s_flat = seg_feat.float().reshape(bsz, seg_feat.shape[1], n).transpose(1, 2)

    # Run k-means to get centroids (reuses _kmeans_labels for centroid init),
    # but use softmax membership instead of argmax labels.
    with torch.no_grad():
        # Seed centroids same as hard version (stable, deterministic).
        order = s_flat.norm(dim=2).argsort(dim=1)
        pick = (torch.arange(num_regions, device=s_flat.device) * (n - 1) // max(1, num_regions - 1)).clamp_max(n - 1)
        seed_idx = order.gather(1, pick.unsqueeze(0).expand(bsz, -1))
        centroids = s_flat.gather(1, seed_idx.unsqueeze(-1).expand(-1, -1, s_flat.shape[-1])).clone()
        for _ in range(max(1, kmeans_iters)):
            dist = torch.cdist(s_flat, centroids)  # [B, N, K]
            labels = dist.argmin(dim=2)
            onehot = F.one_hot(labels, num_classes=num_regions).to(s_flat.dtype)
            counts = onehot.sum(dim=1).clamp_min(1.0)
            new_c = torch.einsum("bnk,bnd->bkd", onehot, s_flat) / counts.unsqueeze(-1)
            empty = (onehot.sum(dim=1) < 0.5).unsqueeze(-1)
            centroids = torch.where(empty, centroids, new_c)

    # Soft membership: softmax over -dist / temp. [B, N, K]
    s_dist = torch.cdist(s_flat, centroids)  # [B, N, K]
    membership = F.softmax(-s_dist / max(1e-6, softmax_temp), dim=2)  # [B, N, K]
    membership = membership.clamp_min(min_weight)  # floor to avoid degenerate zeros
    # Re-normalize after clamping so each pixel's memberships sum to 1.
    membership = membership / membership.sum(dim=2, keepdim=True).clamp_min(1e-8)

    # Target-side membership: cluster target by its own appearance, build soft membership
    # the same way. We then align region indices by centroid mean-projection ordering
    # (same scheme as the hard version) so region k has a stable appearance-rank meaning.
    with torch.no_grad():
        t_order = t_flat.norm(dim=2).argsort(dim=1)
        t_seed = t_order.gather(1, pick.unsqueeze(0).expand(bsz, -1))
        t_centroids = t_flat.gather(1, t_seed.unsqueeze(-1).expand(-1, -1, t_flat.shape[-1])).clone()
        for _ in range(max(1, kmeans_iters)):
            t_dist = torch.cdist(t_flat, t_centroids)
            t_labels = t_dist.argmin(dim=2)
            t_onehot = F.one_hot(t_labels, num_classes=num_regions).to(t_flat.dtype)
            t_counts = t_onehot.sum(dim=1).clamp_min(1.0)
            t_new_c = torch.einsum("bnk,bnd->bkd", t_onehot, t_flat) / t_counts.unsqueeze(-1)
            t_empty = (t_onehot.sum(dim=1) < 0.5).unsqueeze(-1)
            t_centroids = torch.where(t_empty, t_centroids, t_new_c)
    t_dist_full = torch.cdist(t_flat, t_centroids)  # [B, N, K]
    t_membership = F.softmax(-t_dist_full / max(1e-6, softmax_temp), dim=2)
    t_membership = t_membership.clamp_min(min_weight)
    t_membership = t_membership / t_membership.sum(dim=2, keepdim=True).clamp_min(1e-8)

    # Align region indices by centroid mean-projection order (shared appearance ordering).
    with torch.no_grad():
        def _order(cent):
            return cent.mean(dim=2).argsort(dim=1)  # [B, K]
        g_ord = _order(centroids)
        t_ord = _order(t_centroids)

    dirs = F.normalize(torch.randn(num_projections, c, device=gen.device, dtype=torch.float32), dim=1)

    # For each region rank r, compute probability-weighted sliced Wasserstein.
    # Weighted SWD: project pixels onto dirs, then within each region compute
    # weighted-empirical-CDF L1 distance. This is the soft generalization of
    # sort+interpolate: we sort by projection value and compare weighted CDFs.
    # Vectorized over P projections: avoid the per-projection Python loop by
    # batching the sort + cumsum over the P axis.
    swd = gen.new_tensor(0.0)
    active = 0
    grid = torch.linspace(0.0, 1.0, max(2, num_projections), device=gen.device)
    for r in range(num_regions):
        gk = g_ord[:, r]  # [B] region index for rank r
        tk = t_ord[:, r]  # [B]

        for bi in range(bsz):
            g_w = membership[bi, :, gk[bi].item()]  # [N] weights for gen region r
            t_w = t_membership[bi, :, tk[bi].item()]  # [N] weights for target region r
            # Skip if region has negligible mass.
            if g_w.sum() < 1e-4 or t_w.sum() < 1e-4:
                continue
            gp = g_flat[bi] @ dirs.t()  # [N, P]
            tp = t_flat[bi] @ dirs.t()  # [N, P]
            if noise_sigma > 0.0:
                gp = gp + noise_sigma * torch.randn_like(gp)
                tp = tp + noise_sigma * torch.randn_like(tp)
            # Vectorized over P: sort each column, build weighted CDF per column.
            g_sort_idx = gp.argsort(dim=0)  # [N, P]
            t_sort_idx = tp.argsort(dim=0)  # [N, P]
            g_sorted = gp.gather(0, g_sort_idx)  # [N, P]
            t_sorted = tp.gather(0, t_sort_idx)  # [N, P]
            g_w_sorted = g_w.unsqueeze(1).expand(-1, num_projections).gather(0, g_sort_idx)  # [N, P]
            t_w_sorted = t_w.unsqueeze(1).expand(-1, num_projections).gather(0, t_sort_idx)  # [N, P]
            g_cdf = g_w_sorted.cumsum(0)
            t_cdf = t_w_sorted.cumsum(0)
            g_cdf = g_cdf / g_cdf[-1:].clamp_min(1e-8)
            t_cdf = t_cdf / t_cdf[-1:].clamp_min(1e-8)
            # Inverse CDF at grid levels, per projection column.
            g_q = _weighted_quantile_from_sorted_vec(g_sorted, g_cdf, grid)  # [G, P]
            t_q = _weighted_quantile_from_sorted_vec(t_sorted, t_cdf, grid)  # [G, P]
            swd = swd + (g_q - t_q).abs().mean()
            active += 1
    if active > 0:
        swd = swd / active
    return swd


def _weighted_quantile_from_sorted_vec(
    sorted_vals: torch.Tensor,
    sorted_cdf: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Vectorized inverse CDF for sorted, weighted empirical distributions.

    Args:
        sorted_vals: [N, P] ascending values per projection column.
        sorted_cdf: [N, P] ascending CDF values in [0,1] per column.
        grid: [G] quantile levels in [0,1].

    Returns: [G, P] interpolated values per projection column.
    """
    n, p = sorted_vals.shape
    g = grid.shape[0]
    # torch.searchsorted supports per-row independent search when both tensors
    # share the leading dimension. Transpose to [P, N] (boundaries) and
    # [P, G] (values) so each column's CDF is searched independently.
    cdf_t = sorted_cdf.t().contiguous()  # [P, N]
    grid_t = grid.unsqueeze(0).expand(p, g).contiguous()  # [P, G]
    idx = torch.searchsorted(cdf_t, grid_t, right=False).clamp_max(n - 1)  # [P, G]
    idx_lo = (idx - 1).clamp_min(0)
    # Transpose sorted_vals to [P, N] to match idx layout.
    vals_t = sorted_vals.t().contiguous()  # [P, N]
    cdf_hi = cdf_t.gather(1, idx)
    cdf_lo = cdf_t.gather(1, idx_lo)
    val_hi = vals_t.gather(1, idx)
    val_lo = vals_t.gather(1, idx_lo)
    denom = (cdf_hi - cdf_lo).clamp_min(1e-8)
    alpha = ((grid_t - cdf_lo) / denom).clamp(0.0, 1.0)
    out_t = val_lo + alpha * (val_hi - val_lo)  # [P, G]
    return out_t.t().contiguous()  # [G, P]


def _sinkhorn_1d_batched(
    a_vals: torch.Tensor,  # [P, N] source projections (P projections batched)
    b_vals: torch.Tensor,  # [P, M] target projections
    a_mass: torch.Tensor | None = None,  # [P, N] or None
    b_mass: torch.Tensor | None = None,  # [P, M] or None
    epsilon: float = 0.1,
    n_iters: int = 20,
) -> torch.Tensor:
    """Batched entropic-regularized 1D OT across P projections (architecture-level fix).

    Two architecture changes vs the original per-projection _sinkhorn_1d:

    1. **Envelope theorem (detach dual variables).** The Sinkhorn dual variables
       f, g are Lagrange multipliers at the optimum. By the envelope theorem,
       grad of the optimal cost w.r.t. inputs = partial derivative holding
       duals fixed. So iterations run under ``torch.no_grad()`` and the final
       transport plan T is detached — gradients flow ONLY through the cost
       matrix C = (a - b)^2 in the final ``<T*, C>`` product. This eliminates
       the deep autograd graph through n_iters iterations (the OOM root cause)
       while giving the correct gradient at convergence. This is standard
       practice in OT libraries (POT, geomloss use the same principle).

    2. **Projection vectorization.** All P projections are processed in
       parallel via batched [P, N, M] tensors, eliminating the innermost
       ``for p in range(num_projections)`` Python loop. Combined with the
       detach fix, memory is O(P·N·M) for one cost matrix instead of
       O(P·N·M·n_iters) for the unrolled graph.

    Mechanism motivation: Sinkhorn OT with entropic regularizer ε smooths the
    transport plan, trading a small bias for large variance reduction. This
    should reduce the sample-size sensitivity that causes the tswd artifact at
    large K (K=16 tswd drop), while preserving per-region sharpness better
    than the soft-mask mechanism.

    Args:
        a_vals, b_vals: [P, N] and [P, M] — P projection values for source/target.
        epsilon: entropic regularization. ε→0 recovers exact OT.
        n_iters: Sinkhorn iterations (convergence typically reached in 10-50).

    Returns: scalar = mean over P of <T*_p, C_p> (average per-projection OT cost).
    """
    P, N = a_vals.shape
    _, M = b_vals.shape
    if N < 2 or M < 2:
        return a_vals.new_tensor(0.0)
    if a_mass is None:
        a_mass = torch.full_like(a_vals, 1.0 / N)
    if b_mass is None:
        b_mass = torch.full_like(b_vals, 1.0 / M)
    a_mass = a_mass.clamp_min(1e-8)
    b_mass = b_mass.clamp_min(1e-8)
    a_mass = a_mass / a_mass.sum(dim=1, keepdim=True)
    b_mass = b_mass / b_mass.sum(dim=1, keepdim=True)

    # Cost matrix: [P, N, M]. Gradients flow through here (a_vals has grad
    # from the generator; b_vals is the detached target).
    c = (a_vals.unsqueeze(2) - b_vals.unsqueeze(1)) ** 2

    # Dual variable iterations under no_grad (envelope theorem).
    with torch.no_grad():
        log_a = a_mass.clamp_min(1e-30).log()  # [P, N]
        log_b = b_mass.clamp_min(1e-30).log()  # [P, M]
        log_K = -c.detach() / max(1e-6, epsilon)  # [P, N, M]
        f = torch.zeros_like(log_a)  # [P, N]
        g = torch.zeros_like(log_b)  # [P, M]
        for _ in range(max(1, n_iters)):
            # f-update: f[p,i] = log_a[p,i] - logsumexp_j(log_K[p,i,j] + g[p,j])
            f = log_a - torch.logsumexp(log_K + g.unsqueeze(1), dim=2)
            # g-update: g[p,j] = log_b[p,j] - logsumexp_i(log_K[p,i,j] + f[p,i])
            g = log_b - torch.logsumexp(log_K + f.unsqueeze(2), dim=1)
        # Transport plan at convergence (detached — envelope theorem).
        log_T = log_K + f.unsqueeze(2) + g.unsqueeze(1)  # [P, N, M]
        T = log_T.exp().detach()

    # Transport cost = <T*, C>. T is detached; grad flows through C → a_vals.
    # Mean over P projections to match the caller's averaging convention.
    return (T * c).sum(dim=(1, 2)).mean()


def _semantic_region_swd_sinkhorn(
    gen: torch.Tensor,
    target: torch.Tensor,
    *,
    seg_feat: torch.Tensor,
    num_regions: int,
    num_projections: int,
    kmeans_iters: int = 4,
    noise_sigma: float = 0.0,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iters: int = 10,
) -> torch.Tensor:
    """Semantic region SWD with Sinkhorn (entropic-regularized) OT per region.

    Mechanism: identical region partitioning to _semantic_region_swd (hard
    k-means, region-aligned by centroid mean-projection). The difference is
    that within each region, instead of F.interpolate quantile matching
    (exact 1D OT), we compute Sinkhorn OT with entropic regularization ε.

    Theoretical motivation: Theorem 1 (upper bound) is loose at large K
    because empirical 1D OT on small samples is biased downward (variance
    O(1/√n_k)). Sinkhorn OT explicitly controls this bias-variance tradeoff
    via ε, making the transport cost a more reliable proxy across K values.

    Args:
        sinkhorn_epsilon: entropic regularization. ε→0 recovers exact OT
            (and the K=16 tswd artifact); larger ε smooths the transport
            plan and reduces sample-size sensitivity.
        sinkhorn_iters: Sinkhorn iterations (10-50 typical).
    """
    bsz, c, h, w = gen.shape
    n = h * w

    g_flat = gen.float().reshape(bsz, c, n).transpose(1, 2)
    t_flat = target.float().reshape(bsz, c, n).transpose(1, 2)
    s_flat = seg_feat.float().reshape(bsz, seg_feat.shape[1], n).transpose(1, 2)

    g_labels = _kmeans_labels(s_flat, num_regions, iters=kmeans_iters)
    t_labels = _kmeans_labels(t_flat, num_regions, iters=kmeans_iters)

    # Align region indices by centroid mean-projection order.
    with torch.no_grad():
        def _order(flat, labels):
            oh = F.one_hot(labels, num_regions).float()
            cnt = oh.sum(1).clamp_min(1.0)
            cent = torch.einsum("bnk,bnc->bkc", oh, flat) / cnt.unsqueeze(-1)
            return cent.mean(dim=2).argsort(dim=1)
        g_ord = _order(g_flat, g_labels)
        t_ord = _order(t_flat, t_labels)

    dirs = F.normalize(torch.randn(num_projections, c, device=gen.device, dtype=torch.float32), dim=1)
    # Architecture fix for soft-OOM: chunked checkpoint with graph retention.
    # The naive approach (96 checkpointed calls summed into one autograd graph)
    # retains 96 scalar-cost nodes + their gp/tp leaf tensors simultaneously,
    # causing PyTorch caching allocator fragmentation on 8GB GPUs.
    #
    # Fix: process loop iterations in CHUNKS. Each chunk sums its costs into a
    # single scalar (still part of the autograd graph), then we DEL the per-pair
    # gp/tp tensors and call empty_cache() before the next chunk. The chunk sum
    # nodes remain in the graph (they're cheap scalars), but the expensive [P,N,M]
    # intermediates from checkpoint are already discarded in forward, and the
    # gp/tp leaf tensors (which carry the autograd graph back to gen) are released
    # per-chunk. The final total_cost carries grad to gen via the retained scalar
    # chain; backward recomputes each checkpoint's [P,N,M] on demand.
    #
    # IMPORTANT: we do NOT call .backward() here — the returned total_cost must
    # remain connected to gen so the outer trainer's loss.backward() works.
    # Manual backward would break the outer autograd graph.
    eps = float(sinkhorn_epsilon)
    iters = int(sinkhorn_iters)
    CHUNK = 8  # process 8 (region, batch) pairs per chunk → ~6 chunks total

    def _sinkhorn_fn(a, b):
        return _sinkhorn_1d_batched(a, b, epsilon=eps, n_iters=iters)

    # Collect all (gp, tp) projection pairs first (small tensors, cheap).
    pairs = []
    for r in range(num_regions):
        gk = g_ord[:, r]
        tk = t_ord[:, r]
        for bi in range(bsz):
            g_mask = g_labels[bi] == gk[bi]
            t_mask = t_labels[bi] == tk[bi]
            ng = int(g_mask.sum().item())
            nt = int(t_mask.sum().item())
            if ng < 2 or nt < 2:
                continue
            gp = g_flat[bi][g_mask] @ dirs.t()  # [ng, P]
            tp = t_flat[bi][t_mask] @ dirs.t()  # [nt, P]
            if noise_sigma > 0.0:
                gp = gp + noise_sigma * torch.randn_like(gp)
                tp = tp + noise_sigma * torch.randn_like(tp)
            pairs.append((gp, tp))

    total_cost = gen.new_tensor(0.0)
    n_pairs = len(pairs)
    # Process in chunks: forward only (grad flows through checkpoint). Per-chunk
    # release of gp/tp reduces allocator pressure between chunks.
    for start in range(0, n_pairs, CHUNK):
        end = min(start + CHUNK, n_pairs)
        chunk_pairs = pairs[start:end]
        for gp, tp in chunk_pairs:
            # Checkpointed Sinkhorn: cost matrix [P, ng, nt] discarded in forward;
            # recomputed in backward. gp.t() → [P, ng], tp.t() → [P, nt].
            cost = checkpoint(
                _sinkhorn_fn,
                gp.t().contiguous(), tp.t().contiguous(),
                use_reentrant=False,
            )
            total_cost = total_cost + cost
        # Release chunk's gp/tp references so the allocator can defrag.
        # The cost scalars remain in the autograd graph (they're tiny).
        del chunk_pairs
        torch.cuda.empty_cache()

    if n_pairs > 0:
        total_cost = total_cost / n_pairs
    return total_cost


def _semantic_region_swd_hier(
    gen: torch.Tensor,
    target: torch.Tensor,
    *,
    seg_feat: torch.Tensor,
    num_regions_coarse: int,
    num_regions_fine: int,
    num_projections: int,
    kmeans_iters: int = 4,
    noise_sigma: float = 0.0,
    fine_weight: float = 0.5,
) -> torch.Tensor:
    """Hierarchical (coarse + fine) semantic region SWD.

    Mechanism: run two independent k-means partitions at different granularities
    (K_coarse < K_fine), compute region SWD at both levels, and blend them.
    Coarse level captures structural content categories (sky, skin, background);
    fine level captures texture-level subregions within each coarse category.
    A single-K partition cannot simultaneously represent both scales.

    Theoretical motivation: a single K trades structural coherence (small K)
    against texture discrimination (large K). The hierarchical blend gives the
    model both signals: coarse-level SWD enforces structural consistency, while
    fine-level SWD enforces texture-level style match. This is a two-level
    discretization of the hierarchical OT problem.

    Args:
        num_regions_coarse: K_coarse (e.g. 4) for structural partition.
        num_regions_fine: K_fine (e.g. 16) for texture partition.
        fine_weight: blend weight in [0,1]. 0=pure coarse, 1=pure fine.
    """
    coarse_swd = _semantic_region_swd(
        gen, target,
        seg_feat=seg_feat,
        num_regions=num_regions_coarse,
        num_projections=num_projections,
        kmeans_iters=kmeans_iters,
        noise_sigma=noise_sigma,
    )
    fine_swd = _semantic_region_swd(
        gen, target,
        seg_feat=seg_feat,
        num_regions=num_regions_fine,
        num_projections=num_projections,
        kmeans_iters=kmeans_iters,
        noise_sigma=noise_sigma,
    )
    return (1.0 - fine_weight) * coarse_swd + fine_weight * fine_swd


def _kmeans_inertia(feat: torch.Tensor, centroids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of squared distances from each point to its assigned centroid (per sample).

    feat: [B, N, D], centroids: [B, K, D], labels: [B, N].
    Returns: [B] inertia per sample.
    """
    bsz, n, d = feat.shape
    # Gather assigned centroid for each point.
    cent_per_pt = torch.gather(
        centroids, 1, labels.unsqueeze(-1).expand(-1, -1, d)
    )  # [B, N, D]
    sq_dist = ((feat - cent_per_pt) ** 2).sum(dim=-1)  # [B, N]
    return sq_dist.sum(dim=1)  # [B]


def _kmeans_labels_with_inertia(
    feat: torch.Tensor, k: int, iters: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns labels, centroids, and inertia (per-sample sum of squared distances).

    Same algorithm as _kmeans_labels but also returns centroids and inertia
    so callers can decide whether more clusters are justified.
    """
    bsz, n, d = feat.shape
    with torch.no_grad():
        order = feat.norm(dim=2).argsort(dim=1)
        pick = (torch.arange(k, device=feat.device) * (n - 1) // max(1, k - 1)).clamp_max(n - 1)
        seed_idx = order.gather(1, pick.unsqueeze(0).expand(bsz, -1))
        centroids = feat.gather(1, seed_idx.unsqueeze(-1).expand(-1, -1, d)).clone()
        labels = torch.zeros(bsz, n, device=feat.device, dtype=torch.long)
        for _ in range(max(1, iters)):
            dist = torch.cdist(feat, centroids)
            labels = dist.argmin(dim=2)
            onehot = F.one_hot(labels, num_classes=k).to(feat.dtype)
            counts = onehot.sum(dim=1).clamp_min(1.0)
            new_c = torch.einsum("bnk,bnd->bkd", onehot, feat) / counts.unsqueeze(-1)
            empty = (onehot.sum(dim=1) < 0.5).unsqueeze(-1)
            centroids = torch.where(empty, centroids, new_c)
        # Final inertia.
        cent_per_pt = torch.gather(centroids, 1, labels.unsqueeze(-1).expand(-1, -1, d))
        inertia = ((feat - cent_per_pt) ** 2).sum(dim=-1).sum(dim=1)  # [B]
    return labels, centroids, inertia


def _semantic_region_swd_adaptive_k(
    gen: torch.Tensor,
    target: torch.Tensor,
    *,
    seg_feat: torch.Tensor,
    k_candidates: tuple[int, ...] = (4, 8, 16),
    inertia_threshold: float = 0.1,
    num_projections: int,
    kmeans_iters: int = 4,
    noise_sigma: float = 0.0,
) -> torch.Tensor:
    """Content-adaptive-K semantic region SWD.

    Mechanism: instead of a single global K, pick K per-sample by measuring
    how much k-means inertia drops when going from K to 2K. If the relative
    drop is below inertia_threshold, more clusters are not justified and we
    use the smaller K. This gives each sample a K appropriate to its content
    complexity (simple images get small K, complex images get large K).

    Theoretical motivation: the inverted-U peak K* depends on content
    complexity. A fixed K=8 is suboptimal for both simple (K*≈4) and complex
    (K*≈16) images. Adaptive-K tracks the per-sample peak directly via the
    inertia-elbow criterion.

    Args:
        k_candidates: K values to try, in increasing order.
        inertia_threshold: if inertia(K) / inertia(K/2) > 1 - threshold,
            K/2 is deemed sufficient (diminishing returns).
    """
    bsz, c, h, w = gen.shape
    n = h * w
    s_flat = seg_feat.float().reshape(bsz, seg_feat.shape[1], n).transpose(1, 2)

    # Run k-means at all candidate K values, pick best per sample.
    k_to_label_cent_inertia: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for k in k_candidates:
        labels, cent, inertia = _kmeans_labels_with_inertia(s_flat, k, iters=kmeans_iters)
        k_to_label_cent_inertia[k] = (labels, cent, inertia)

    # Per-sample K selection: smallest K whose inertia drop vs previous K
    # is below threshold (elbow). First sample always uses smallest K.
    sorted_ks = sorted(k_candidates)
    chosen_k = torch.full((bsz,), sorted_ks[0], device=gen.device, dtype=torch.long)
    for i in range(1, len(sorted_ks)):
        k_prev = sorted_ks[i - 1]
        k_cur = sorted_ks[i]
        _, _, inertia_prev = k_to_label_cent_inertia[k_prev]
        _, _, inertia_cur = k_to_label_cent_inertia[k_cur]
        # Relative drop: (inertia_prev - inertia_cur) / inertia_prev.
        # If drop < threshold, K_prev is sufficient — keep it.
        # Else, upgrade to K_cur.
        rel_drop = (inertia_prev - inertia_cur) / inertia_prev.clamp_min(1e-8)
        upgrade = rel_drop > inertia_threshold
        chosen_k = torch.where(upgrade, torch.full_like(chosen_k, k_cur), chosen_k)

    # Compute SWD per sample with its chosen K. Since _semantic_region_swd is
    # vectorized over batch with a single K, we loop over unique K values and
    # process the corresponding samples. This is still much cheaper than
    # per-sample Python loops.
    swd = gen.new_tensor(0.0)
    active = 0
    for k in sorted_ks:
        mask = (chosen_k == k)
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        # Gather subset and run _semantic_region_swd on it.
        g_sub = gen[idx]
        t_sub = target[idx]
        s_sub = seg_feat[idx]
        sub_swd = _semantic_region_swd(
            g_sub, t_sub,
            seg_feat=s_sub,
            num_regions=int(k),
            num_projections=num_projections,
            kmeans_iters=kmeans_iters,
            noise_sigma=noise_sigma,
        )
        swd = swd + sub_swd * len(idx)
        active += len(idx)
    if active > 0:
        swd = swd / active
    return swd


def _semantic_region_swd_spectral(
    gen: torch.Tensor,
    target: torch.Tensor,
    *,
    seg_feat: torch.Tensor,
    num_regions: int,
    num_projections: int,
    kmeans_iters: int = 4,
    noise_sigma: float = 0.0,
    ll_weight: float = 1.0,
    hf_weight: float = 2.0,
) -> torch.Tensor:
    """Spectral-decoupled semantic region SWD (Mechanism 5: frequency-band division).

    Architecture change vs _semantic_region_swd:
      - Current region SWD operates in the spatial domain: one partition covers
        all frequencies. But style and content live in different bands — style
        (brushstrokes, texture) is high-frequency; content (structure) is
        low-frequency. A single region partition conflates them.
      - This mechanism DWT-decomposes gen and target into LL (low-freq) and
        LH/HL/HH (high-freq). Low-freq gets GLOBAL SWD (preserve structure,
        no region matching needed). High-freq gets REGION SWD (match texture
        statistics within content-coherent regions).
      - Theoretical motivation: the inverted-U peak K* depends on frequency.
        High-freq texture needs fine regions (large K); low-freq structure
        needs global matching (K=1). Splitting by frequency lets each band
        use its optimal K without compromise.

    This is a deeper architecture change than the 4 mechanism variants above:
    it changes WHERE SWD operates (frequency domain) rather than HOW region
    partitioning works. It naturally integrates with the existing DWT route
    architecture (cross_attn_dwt_route).

    Args:
        ll_weight: SWD weight for LL (low-freq) band, global SWD.
        hf_weight: SWD weight for LH+HL+HH (high-freq) bands, region SWD.
    """
    # DWT decomposition: 1-level Haar.
    g_ll, g_lh, g_hl, g_hh = dwt2_haar(gen.float())
    t_ll, t_lh, t_hl, t_hh = dwt2_haar(target.float())
    # seg_feat also needs downsampling to match subband resolution.
    s_ll = F.avg_pool2d(seg_feat.float(), kernel_size=2, stride=2)

    dirs_ll = F.normalize(
        torch.randn(num_projections, g_ll.shape[1], device=gen.device, dtype=torch.float32), dim=1
    )
    # Low-freq: GLOBAL SWD (no region matching, K=1 effectively).
    g_ll_flat = g_ll.reshape(g_ll.shape[0], g_ll.shape[1], -1).transpose(1, 2)  # [B, N, C]
    t_ll_flat = t_ll.reshape(t_ll.shape[0], t_ll.shape[1], -1).transpose(1, 2)
    proj_g_ll = g_ll_flat @ dirs_ll.t()
    proj_t_ll = t_ll_flat @ dirs_ll.t()
    if noise_sigma > 0.0:
        proj_g_ll = proj_g_ll + noise_sigma * torch.randn_like(proj_g_ll)
        proj_t_ll = proj_t_ll + noise_sigma * torch.randn_like(proj_t_ll)
    pg_sorted = torch.sort(proj_g_ll, dim=1).values
    pt_sorted = torch.sort(proj_t_ll, dim=1).values
    # Quantile match via F.interpolate to common size.
    ng, nt = pg_sorted.shape[1], pt_sorted.shape[1]
    m_ll = max(ng, nt)
    if ng != m_ll:
        pg_sorted = F.interpolate(pg_sorted.transpose(1, 2), size=m_ll, mode="linear", align_corners=True).transpose(1, 2)
    if nt != m_ll:
        pt_sorted = F.interpolate(pt_sorted.transpose(1, 2), size=m_ll, mode="linear", align_corners=True).transpose(1, 2)
    ll_swd = (pg_sorted - pt_sorted).abs().mean()

    # High-freq: REGION SWD per subband (LH, HL, HH), summed.
    hf_swd = gen.new_tensor(0.0)
    for g_hf, t_hf in [(g_lh, t_lh), (g_hl, t_hl), (g_hh, t_hh)]:
        hf_swd = hf_swd + _semantic_region_swd(
            g_hf, t_hf,
            seg_feat=s_ll,
            num_regions=num_regions,
            num_projections=num_projections,
            kmeans_iters=kmeans_iters,
            noise_sigma=noise_sigma,
        )
    hf_swd = hf_swd / 3.0  # average over 3 high-freq subbands

    # Blend: low-freq global + high-freq region.
    total_w = ll_weight + hf_weight
    return (ll_weight * ll_swd + hf_weight * hf_swd) / total_w


def _semantic_region_swd_attn(
    gen: torch.Tensor,
    target: torch.Tensor,
    *,
    attn_weight: torch.Tensor,
    num_projections: int,
    noise_sigma: float = 0.0,
    num_regions: int = 0,
) -> torch.Tensor:
    """Attention-guided dynamic region SWD (Mechanism 6: style-conditional regions).

    Architecture change vs _semantic_region_swd:
      - Current region SWD uses k-means on CONTENT latent to define regions.
        These regions are style-AGNOSTIC: the same content point belongs to the
        same region regardless of the target style. But semantically, a "sky"
        region in a landscape should be matched differently to Impressionism
        (brushstroke sky) vs Ukiyo-e (flat color sky).
      - This mechanism uses the cross-attention map directly as the region
        definition. attn_weight is [B, N, S] (S = style tokens), already
        style-conditional. Each style token defines a "soft region" — locations
        attending strongly to that token form one region.
      - Within each attention-defined region, we match gen to target via SWD.
        The regions are style-conditional: the same content location may belong
        to different regions for different target styles.

    This is a deeper change than k-means region: it replaces the region
    DEFINITION (content k-means → attention map) rather than the region
    MATCHING (hard/soft/Sinkhorn). It makes regions style-adaptive.

    Args:
        attn_weight: [B, N, S] cross-attention weights (style-conditional regions).
            If [B, C, H, W], will be reshaped and treated as region features.
        num_regions: ignored if attn_weight is 3D [B,N,S] (S defines regions).
    """
    bsz, c, h, w = gen.shape
    n = h * w

    g_flat = gen.float().reshape(bsz, c, n).transpose(1, 2)  # [B, N, C]
    t_flat = target.float().reshape(bsz, c, n).transpose(1, 2)

    # Normalize attention weights into region membership.
    # attn_weight: [B, N, S] → soft membership per style token.
    if attn_weight.ndim == 4:
        # [B, C', H, W] → [B, N, C']; treat channels as region features.
        aw = attn_weight.float().reshape(bsz, attn_weight.shape[1], n).transpose(1, 2)
    else:
        aw = attn_weight.float()  # [B, N, S]
    # Normalize: each location's membership sums to 1 across regions.
    membership = aw.clamp_min(1e-8)
    membership = membership / membership.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    # membership: [B, N, S], S = number of attention regions.
    S = membership.shape[-1]

    dirs = F.normalize(torch.randn(num_projections, c, device=gen.device, dtype=torch.float32), dim=1)
    swd = gen.new_tensor(0.0)
    active = 0
    # For each attention region s, extract top contributing locations and match.
    # We use hard top-k (top 50% of locations per region) to preserve sharpness
    # (lesson from soft-mask: hard boundaries are a feature).
    keep = max(2, n // S)
    for s in range(S):
        mem_s = membership[:, :, s]  # [B, N]
        for bi in range(bsz):
            # Top-k locations for this region in this batch item.
            vals, idx = mem_s[bi].topk(min(keep, n), largest=True)
            # Only keep locations with meaningful membership (above mean).
            thr = vals.mean().clamp_min(1e-8)
            idx = idx[vals >= thr]
            ng = idx.shape[0]
            if ng < 2:
                continue
            gp = g_flat[bi][idx] @ dirs.t()  # [ng, P]
            # Target: match against ALL target pixels (global target pool for
            # this region — since attention is style-conditional, target doesn't
            # have a natural region decomposition).
            tp = t_flat[bi] @ dirs.t()  # [n, P]
            if noise_sigma > 0.0:
                gp = gp + noise_sigma * torch.randn_like(gp)
                tp = tp + noise_sigma * torch.randn_like(tp)
            # Quantile match.
            gs = torch.sort(gp, dim=0).values
            ts = torch.sort(tp, dim=0).values
            m = max(ng, n)
            if ng != m:
                gs = F.interpolate(gs.t().unsqueeze(0), size=m, mode="linear", align_corners=True).squeeze(0).t()
            if n != m:
                ts = F.interpolate(ts.t().unsqueeze(0), size=m, mode="linear", align_corners=True).squeeze(0).t()
            swd = swd + (gs - ts).abs().mean()
            active += 1
    if active > 0:
        swd = swd / active
    return swd


def _patch_swd(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    patch: int,
    num_projections: int,
    noise_sigma: float = 0.0,
    sample_weight: torch.Tensor | None = None,
    sample_size: int = 0,
) -> torch.Tensor:
    """Sliced Wasserstein over k×k local patches.

    Each spatial location becomes a (C·patch²)-dim texture vector via im2col (unfold).
    Projecting these onto random directions and matching sorted quantiles constrains the
    local texture distribution, not just the per-pixel color marginal. patch=1 reduces to
    the pixel-marginal _sliced_wasserstein. The cross-attention guidance map is reused as
    empirical sampling mass, downsampled to the unfolded grid so guided sampling still
    focuses where the routing module edits content.
    """
    if patch <= 1:
        dirs = F.normalize(torch.randn(num_projections, a.shape[1], device=a.device, dtype=a.dtype), dim=1)
        return _sliced_wasserstein(
            a, b, dirs=dirs, noise_sigma=noise_sigma,
            sample_weight=sample_weight, sample_size=sample_size,
        )
    bsz, c, h, w = a.shape
    pad = patch // 2
    # Unfold into patches: [B, C·patch², L] where L = number of spatial locations
    a_unf = F.unfold(a.float(), kernel_size=patch, padding=pad)
    b_unf = F.unfold(b.float(), kernel_size=patch, padding=pad)
    feat_dim = a_unf.shape[1]
    a_spatial = a_unf.transpose(1, 2)  # [B, L, C·patch²]
    b_spatial = b_unf.transpose(1, 2)
    n = a_spatial.shape[1]
    if sample_weight is not None:
        flat_weight = sample_weight.detach().float()
        if flat_weight.ndim == 4:
            # Downsample guidance map to the unfolded spatial grid (L locations).
            gh = int(round(n ** 0.5))
            if gh * gh == n and flat_weight.shape[-2:] != (gh, gh):
                flat_weight = F.interpolate(flat_weight, size=(gh, gh), mode="bilinear", align_corners=False)
            flat_weight = flat_weight.mean(dim=1).reshape(bsz, -1)
        else:
            flat_weight = flat_weight.reshape(bsz, -1)
        if flat_weight.shape[0] == bsz and flat_weight.shape[1] == n:
            probs = flat_weight.clamp_min(1e-8)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            take = n if sample_size <= 0 else min(n, max(1, int(sample_size)))
            cdf = probs.cumsum(dim=1).contiguous()
            cdf[:, -1] = 1.0
            q = (torch.arange(take, device=a.device, dtype=cdf.dtype) + 0.5) / float(take)
            q = q.unsqueeze(0).expand(bsz, -1).contiguous()
            idx = torch.searchsorted(cdf, q, right=False).clamp_max(n - 1)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, feat_dim)
            a_spatial = a_spatial.gather(dim=1, index=gather_idx)
            b_spatial = b_spatial.gather(dim=1, index=gather_idx)
    dirs = F.normalize(torch.randn(num_projections, feat_dim, device=a.device, dtype=a_spatial.dtype), dim=1)
    proj_a = a_spatial @ dirs.t()
    proj_b = b_spatial @ dirs.t()
    if noise_sigma > 0.0:
        proj_a = proj_a + noise_sigma * torch.randn_like(proj_a)
        proj_b = proj_b + noise_sigma * torch.randn_like(proj_b)
    proj_a_sorted = torch.sort(proj_a, dim=1).values
    proj_b_sorted = torch.sort(proj_b, dim=1).values
    return (proj_a_sorted - proj_b_sorted).abs().mean()


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
        # 631: Patch-based SWD — project k×k local patches instead of single pixels.
        # Single-pixel SWD only matches the latent color/tone marginal (sorted quantiles
        # of 4-dim channel vectors), discarding all local texture arrangement. MUSIQ is a
        # no-reference metric that rewards texture naturalness/sharpness, so pixel-marginal
        # SWD cannot target it. Patch SWD lifts each sample to a 4·k²-dim texture vector so
        # sliced projections carry local structure, aligning the terminal constraint with
        # perceptual quality. swd_patch_mode: "off" (legacy pixel) | "multi" (multi-scale).
        self.swd_patch_mode = str(getattr(self.bridge_cfg, "swd_patch_mode", "off")).strip().lower()
        raw_patch_sizes = getattr(self.bridge_cfg, "swd_patch_sizes", [1, 3, 5, 9])
        self.swd_patch_sizes = [int(p) for p in raw_patch_sizes if int(p) >= 1] or [1]
        raw_patch_weights = getattr(self.bridge_cfg, "swd_patch_weights", None)
        if raw_patch_weights and len(raw_patch_weights) == len(self.swd_patch_sizes):
            self.swd_patch_weights = [float(w) for w in raw_patch_weights]
        else:
            self.swd_patch_weights = [1.0 for _ in self.swd_patch_sizes]
        # Spectral band-split SWD: match latent distributions per DWT sub-band instead of on
        # the full latent. The full-latent SWD is dominated by low-frequency (structure/color)
        # energy, so it barely constrains the high-frequency band that MUSIQ actually rewards
        # (edge/texture sharpness). Since the model is already a wavelet-domain ODE, matching
        # per-band distributions with a high-frequency emphasis aligns the terminal constraint
        # with perceptual quality using the model's own decomposition — not a global scalar.
        # swd_band_mode: "off" (full-latent) | "split" (per-band LL/LH/HL/HH).
        self.swd_band_mode = str(getattr(self.bridge_cfg, "swd_band_mode", "off")).strip().lower()
        self.swd_band_w_ll = float(getattr(self.bridge_cfg, "swd_band_w_ll", 0.25))
        self.swd_band_w_lh = float(getattr(self.bridge_cfg, "swd_band_w_lh", 1.0))
        self.swd_band_w_hl = float(getattr(self.bridge_cfg, "swd_band_w_hl", 1.0))
        self.swd_band_w_hh = float(getattr(self.bridge_cfg, "swd_band_w_hh", 1.5))
        # Semantic region SWD (true semantic SWD): partition the content latent into
        # content-similar regions (k-means) and match each region's endpoint distribution
        # to its appearance-corresponding target region, instead of pooling all pixels into
        # one global marginal. The global match forces a smooth region's pixels partway
        # toward a textured target region's statistics, producing a muddy blend that lowers
        # MUSIQ; region-coherent matching keeps per-region statistics internally consistent.
        # swd_semantic_mode: "off" | "region". Blended with the global SWD via swd_semantic_blend.
        self.swd_semantic_mode = str(getattr(self.bridge_cfg, "swd_semantic_mode", "off")).strip().lower()
        self.swd_semantic_regions = max(2, int(getattr(self.bridge_cfg, "swd_semantic_regions", 4)))
        self.swd_semantic_kmeans_iters = max(1, int(getattr(self.bridge_cfg, "swd_semantic_kmeans_iters", 4)))
        self.swd_semantic_blend = max(0.0, min(1.0, float(getattr(self.bridge_cfg, "swd_semantic_blend", 0.5))))
        # Soft-mask region SWD parameters (mechanism-exploration variant).
        # swd_semantic_mode == "region_soft" activates _semantic_region_swd_softmask.
        self.swd_semantic_softmax_temp = float(getattr(self.bridge_cfg, "swd_semantic_softmax_temp", 1.0))
        self.swd_semantic_min_weight = float(getattr(self.bridge_cfg, "swd_semantic_min_weight", 0.05))
        # Sinkhorn (entropic-regularized) OT parameters (mechanism-exploration variant).
        # swd_semantic_mode == "region_sinkhorn" activates _semantic_region_swd_sinkhorn.
        self.swd_semantic_sinkhorn_epsilon = float(getattr(self.bridge_cfg, "swd_semantic_sinkhorn_epsilon", 0.1))
        self.swd_semantic_sinkhorn_iters = int(getattr(self.bridge_cfg, "swd_semantic_sinkhorn_iters", 10))
        # Hierarchical (coarse + fine) region SWD parameters.
        # swd_semantic_mode == "region_hier" activates _semantic_region_swd_hier.
        self.swd_semantic_hier_coarse = int(getattr(self.bridge_cfg, "swd_semantic_hier_coarse", 4))
        self.swd_semantic_hier_fine = int(getattr(self.bridge_cfg, "swd_semantic_hier_fine", 16))
        self.swd_semantic_hier_fine_weight = float(getattr(self.bridge_cfg, "swd_semantic_hier_fine_weight", 0.5))
        # Content-adaptive-K region SWD parameters.
        # swd_semantic_mode == "region_adaptive_k" activates _semantic_region_swd_adaptive_k.
        self.swd_semantic_adaptive_k_candidates = tuple(
            int(x) for x in getattr(self.bridge_cfg, "swd_semantic_adaptive_k_candidates", (4, 8, 16))
        )
        self.swd_semantic_adaptive_k_threshold = float(getattr(self.bridge_cfg, "swd_semantic_adaptive_k_threshold", 0.1))
        # Spectral-decoupled region SWD (Mechanism 5: frequency-band division).
        # swd_semantic_mode == "region_spectral" activates _semantic_region_swd_spectral.
        self.swd_semantic_spectral_ll_weight = float(getattr(self.bridge_cfg, "swd_semantic_spectral_ll_weight", 1.0))
        self.swd_semantic_spectral_hf_weight = float(getattr(self.bridge_cfg, "swd_semantic_spectral_hf_weight", 2.0))
        # Attention-guided region SWD (Mechanism 6: style-conditional regions).
        # swd_semantic_mode == "region_attn" activates _semantic_region_swd_attn.
        # Uses cross-attention map as region definition instead of k-means.
        # NOTE: The previous projection cap (min(num_projections, 16) for sinkhorn
        # mode) was a parameter-level hack. The OOM root cause is now fixed at the
        # architecture level: _sinkhorn_1d_batched uses the envelope theorem
        # (detached dual variables) + projection vectorization, so the full
        # num_projections=64 can be used without OOM.
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
        """Compute SWD loss with attention-weighting support."""
        swd_guidance_active = z_hat1.new_tensor(0.0)
        swd_guidance_mean = z_hat1.new_tensor(0.0)
        swd_guidance_std = z_hat1.new_tensor(0.0)
        # Semantic region SWD (true semantic SWD): match distributions within content-coherent
        # regions rather than one global marginal. seg_feat is the content latent, so regions
        # are defined by content similarity ("内容相近的部分"); each is matched to its
        # appearance-corresponding target region. Blended with the global SWD so the global
        # distribution constraint (which drives MUSIQ via the reference artwork's statistics)
        # is preserved while region coherence cleans up incompatible cross-region blending.
        # swd_semantic_mode: "off" | "region" (hard k-means) | "region_soft" (softmax membership)
        #                   | "region_sinkhorn" (entropic OT per region)
        #                   | "region_hier" (coarse + fine hierarchical)
        #                   | "region_adaptive_k" (content-adaptive K via inertia elbow)
        #                   | "region_spectral" (Mechanism 5: DWT-decoupled LL global + HF region)
        #                   | "region_attn" (Mechanism 6: attention-guided style-conditional regions).
        if self.swd_semantic_mode in {
            "region", "region_soft", "region_sinkhorn", "region_hier", "region_adaptive_k",
            "region_spectral", "region_attn",
        } and content is not None:
            guided = self.swd_scale_mode in {
                "cross-attn-guided", "cross_attn_guided", "crossattn-guided", "crossattn_guided"
            }
            weight = self._cross_attn_swd_weight(model, z_hat1, content=content) if guided else None
            if weight is not None:
                swd_guidance_active = z_hat1.new_tensor(1.0)
                swd_guidance_mean = weight.detach().float().mean()
                swd_guidance_std = weight.detach().float().std()
            swd_dirs = self._projection_dirs(z_hat1)
            global_swd = _sliced_wasserstein(
                z_hat1, projected_target,
                dirs=swd_dirs,
                noise_sigma=self.swd_noise_sigma,
                sample_weight=weight, sample_size=self.swd_guidance_sample_size,
            )
            if self.swd_semantic_mode == "region":
                region_swd = _semantic_region_swd(
                    z_hat1, projected_target,
                    seg_feat=content,
                    num_regions=self.swd_semantic_regions,
                    num_projections=self.num_projections,
                    kmeans_iters=self.swd_semantic_kmeans_iters,
                    noise_sigma=self.swd_noise_sigma,
                )
            elif self.swd_semantic_mode == "region_soft":
                region_swd = _semantic_region_swd_softmask(
                    z_hat1, projected_target,
                    seg_feat=content,
                    num_regions=self.swd_semantic_regions,
                    num_projections=self.num_projections,
                    kmeans_iters=self.swd_semantic_kmeans_iters,
                    noise_sigma=self.swd_noise_sigma,
                    softmax_temp=self.swd_semantic_softmax_temp,
                    min_weight=self.swd_semantic_min_weight,
                )
            elif self.swd_semantic_mode == "region_sinkhorn":
                region_swd = _semantic_region_swd_sinkhorn(
                    z_hat1, projected_target,
                    seg_feat=content,
                    num_regions=self.swd_semantic_regions,
                    num_projections=self.num_projections,
                    kmeans_iters=self.swd_semantic_kmeans_iters,
                    noise_sigma=self.swd_noise_sigma,
                    sinkhorn_epsilon=self.swd_semantic_sinkhorn_epsilon,
                    sinkhorn_iters=self.swd_semantic_sinkhorn_iters,
                )
            elif self.swd_semantic_mode == "region_hier":
                region_swd = _semantic_region_swd_hier(
                    z_hat1, projected_target,
                    seg_feat=content,
                    num_regions_coarse=self.swd_semantic_hier_coarse,
                    num_regions_fine=self.swd_semantic_hier_fine,
                    num_projections=self.num_projections,
                    kmeans_iters=self.swd_semantic_kmeans_iters,
                    noise_sigma=self.swd_noise_sigma,
                    fine_weight=self.swd_semantic_hier_fine_weight,
                )
            elif self.swd_semantic_mode == "region_adaptive_k":
                region_swd = _semantic_region_swd_adaptive_k(
                    z_hat1, projected_target,
                    seg_feat=content,
                    k_candidates=self.swd_semantic_adaptive_k_candidates,
                    inertia_threshold=self.swd_semantic_adaptive_k_threshold,
                    num_projections=self.num_projections,
                    kmeans_iters=self.swd_semantic_kmeans_iters,
                    noise_sigma=self.swd_noise_sigma,
                )
            elif self.swd_semantic_mode == "region_spectral":
                # Mechanism 5: frequency-band division. LL→global SWD, HF→region SWD.
                # Returns blended cost directly; global_swd above is still computed
                # and blended via swd_semantic_blend for consistency.
                region_swd = _semantic_region_swd_spectral(
                    z_hat1, projected_target,
                    seg_feat=content,
                    num_regions=self.swd_semantic_regions,
                    num_projections=self.num_projections,
                    kmeans_iters=self.swd_semantic_kmeans_iters,
                    noise_sigma=self.swd_noise_sigma,
                    ll_weight=self.swd_semantic_spectral_ll_weight,
                    hf_weight=self.swd_semantic_spectral_hf_weight,
                )
            elif self.swd_semantic_mode == "region_attn":
                # Mechanism 6: attention-guided style-conditional regions.
                # Use cross-attention guidance as region definition (style-conditional).
                # If no attention weight available, fall back to hard k-means.
                if weight is not None:
                    region_swd = _semantic_region_swd_attn(
                        z_hat1, projected_target,
                        attn_weight=weight,
                        num_projections=self.num_projections,
                        noise_sigma=self.swd_noise_sigma,
                    )
                else:
                    region_swd = _semantic_region_swd(
                        z_hat1, projected_target,
                        seg_feat=content,
                        num_regions=self.swd_semantic_regions,
                        num_projections=self.num_projections,
                        kmeans_iters=self.swd_semantic_kmeans_iters,
                        noise_sigma=self.swd_noise_sigma,
                    )
            else:
                # Should not reach here given the outer guard, but keep a safe fallback.
                region_swd = _semantic_region_swd(
                    z_hat1, projected_target,
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
        # Simple global SWD (default path): single sliced Wasserstein on full latent.
        swd = _sliced_wasserstein(
            z_hat1, projected_target,
            dirs=self._projection_dirs(z_hat1),
            noise_sigma=self.swd_noise_sigma,
        )
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
        self.last_debug = {k: v.detach().float().cpu().item() if v.numel() == 1 else 0.0 for k, v in metrics.items()}
        return metrics

    def update_weights_for_epoch(self, epoch: int, num_epochs: int = 3) -> dict:
        return {
            "stage": 0, "bridge_sigma": self._base_bridge_sigma,
            "w_endpoint_content": self.w_endpoint_content,
            "w_endpoint_style": self.w_endpoint_style,
            "w_style_strength_reg": self.w_style_strength_reg,
        }
