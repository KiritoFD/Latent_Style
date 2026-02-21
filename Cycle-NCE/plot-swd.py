#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent-only SWD 鍒嗘瀽锛堜笉璁粌锛?
- data_root/
    - styleA/*.pt
    - styleB/*.pt
    - ...
姣忎釜 .pt 榛樿淇濆瓨涓€涓?Tensor锛堟垨 dict 閲屽寘鍚?Tensor锛夛紝褰㈢姸鏀寔锛?
  [C,H,W] 鎴?[1,C,H,W] 鎴?[B,C,H,W]锛堝彇绗竴涓牱鏈級
杈撳嚭锛?
  1) SWD 鍚岄鏍?寮傞鏍艰窛绂诲垎甯?+ AUC / d'
  2) patch_size 脳 k_channels 鐨?AUC 鐑姏鍥撅紙鍚ǔ瀹氭€э級
  3) style 鍘熷瀷璺濈鐭╅樀 heatmap + MDS 2D 鍙鍖?
  4) 绌洪棿椋庢牸鎵胯浇鐑姏鍥撅紙Fisher map锛岀敤浜庣洿瑙傜湅鈥滄爲閭ｅ潡鏇撮鏍煎寲鈥濓級

渚濊禆锛歵orch, numpy, matplotlib, scikit-learn, tqdm
"""

import os
import re
import math
import json
import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.manifold import MDS


# -----------------------------
# Utils
# -----------------------------

def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def ensure_chw(x: torch.Tensor) -> torch.Tensor:
    """Return [C,H,W]."""
    if x.ndim == 4:
        # [B,C,H,W] -> take first
        return x[0]
    if x.ndim == 3:
        return x
    raise ValueError(f"Unsupported latent ndim={x.ndim}, shape={tuple(x.shape)}")


def load_latent(path: str, key: Optional[str] = None, map_location: str = "cpu") -> torch.Tensor:
    """
    Load .pt:
      - Tensor directly
      - dict containing a Tensor
    If dict and key is None, heuristically pick the first Tensor value.
    """
    obj = torch.load(path, map_location=map_location)

    if torch.is_tensor(obj):
        return ensure_chw(obj).float()

    if isinstance(obj, dict):
        if key is not None:
            if key not in obj:
                raise KeyError(f"Key '{key}' not found in dict. Keys={list(obj.keys())[:20]}")
            v = obj[key]
            if not torch.is_tensor(v):
                raise TypeError(f"obj['{key}'] is not a Tensor: {type(v)}")
            return ensure_chw(v).float()

        # heuristic
        for k, v in obj.items():
            if torch.is_tensor(v):
                return ensure_chw(v).float()

        # nested heuristic
        for k, v in obj.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        return ensure_chw(vv).float()

        raise TypeError(f"No Tensor found in dict: {path}")

    raise TypeError(f"Unsupported .pt content type: {type(obj)} in {path}")


def list_style_files(data_root: str, exts=(".pt", ".pth")) -> Dict[str, List[str]]:
    """style_name -> list of file paths (one level under data_root)."""
    styles = {}
    for name in sorted(os.listdir(data_root)):
        d = os.path.join(data_root, name)
        if not os.path.isdir(d):
            continue
        paths = []
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(exts):
                paths.append(os.path.join(d, fn))
        if paths:
            styles[name] = paths
    if not styles:
        raise FileNotFoundError(f"No style subfolders with {exts} found under: {data_root}")
    return styles


def basename_id(path: str) -> str:
    """Match same content across styles by filename stem."""
    bn = os.path.basename(path)
    stem = re.sub(r"\.(pt|pth)$", "", bn, flags=re.IGNORECASE)
    return stem


def pair_intersection(styles: Dict[str, List[str]], base_style: str) -> Dict[str, Dict[str, str]]:
    """
    Build mapping:
      common_id -> {style: path}
    Only keep ids present in all styles AND in base_style.
    """
    id2path = {}
    for s, paths in styles.items():
        tmp = {}
        for p in paths:
            tmp[basename_id(p)] = p
        id2path[s] = tmp

    common_ids = set(id2path[base_style].keys())
    for s in styles.keys():
        common_ids &= set(id2path[s].keys())

    common = {}
    for cid in sorted(common_ids):
        common[cid] = {s: id2path[s][cid] for s in styles.keys()}
    return common


# -----------------------------
# SWD core
# -----------------------------

@dataclass
class SWDConfig:
    patch: int
    stride: int = 1
    num_projections: int = 128
    max_patches: int = 2048
    patch_norm: str = "none"   # "none" | "l2" | "standardize"
    orthogonal_proj: bool = True
    padding_mode: str = "same"  # "same" | "valid"


def make_projections(dim: int, num_projections: int, seed: int, device: torch.device,
                     orthogonal: bool = True) -> torch.Tensor:
    """
    Return [dim, num_projections] normalized projections.
    If orthogonal=True and num_projections <= dim: QR for lower variance.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # generate on cpu for determinism then move
    W = torch.randn(dim, num_projections, generator=g, dtype=torch.float32)
    if orthogonal and num_projections <= dim:
        # QR on (dim x num_proj) -> Q is (dim x num_proj)
        # For numerical stability do QR on (dim x num_proj) directly
        Q, _ = torch.linalg.qr(W, mode="reduced")
        W = Q[:, :num_projections]
    # normalize columns
    W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
    return W.to(device)


def extract_patches(x_chw: torch.Tensor,
                    patch: int,
                    stride: int,
                    channels: Optional[torch.Tensor],
                    max_patches: int,
                    patch_norm: str,
                    padding_mode: str,
                    device: torch.device,
                    seed: int) -> torch.Tensor:
    """
    x_chw: [C,H,W] on CPU or GPU
    return: [N, dim] patches on device
    """
    x = x_chw.unsqueeze(0).to(device)  # [1,C,H,W]
    if channels is not None:
        x = x[:, channels, :, :]

    if padding_mode == "same":
        padding = patch // 2
    elif padding_mode == "valid":
        padding = 0
    else:
        raise ValueError(f"Unsupported padding_mode={padding_mode}, expected same|valid")
    # [1, C*patch*patch, L], where L = number of patches
    unfolded = F.unfold(x, kernel_size=patch, stride=stride, padding=padding)
    # [L, dim]
    pts = unfolded.squeeze(0).transpose(0, 1).contiguous()

    # subsample patches
    L = pts.shape[0]
    if max_patches is not None and L > max_patches:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        idx = torch.randperm(L, generator=g, device=device)[:max_patches]
        pts = pts[idx]

    # patch normalization
    if patch_norm == "l2":
        pts = pts / (pts.norm(dim=1, keepdim=True) + 1e-12)
    elif patch_norm == "standardize":
        m = pts.mean(dim=1, keepdim=True)
        v = pts.var(dim=1, unbiased=False, keepdim=True)
        pts = (pts - m) / (v.sqrt() + 1e-6)

    return pts


@torch.no_grad()
def swd_distance_from_patches(Pa: torch.Tensor, Pb: torch.Tensor, projections: torch.Tensor) -> torch.Tensor:
    """
    Pa, Pb: [N, dim]
    projections: [dim, num_projections]
    return scalar tensor
    """
    dev = projections.device
    dt = projections.dtype
    if Pa.device != dev or Pa.dtype != dt:
        Pa = Pa.to(dev, dtype=dt, non_blocking=True)
    if Pb.device != dev or Pb.dtype != dt:
        Pb = Pb.to(dev, dtype=dt, non_blocking=True)
    # project: [N, num_proj]
    xa = Pa @ projections
    xb = Pb @ projections
    # sort along N
    xa_sorted, _ = torch.sort(xa, dim=0)
    xb_sorted, _ = torch.sort(xb, dim=0)
    # mean absolute difference
    return (xa_sorted - xb_sorted).abs().mean()


# -----------------------------
# Channel ranking (Top-k)
# -----------------------------

@torch.no_grad()
def channel_fisher_scores(latents: List[torch.Tensor], labels: np.ndarray) -> np.ndarray:
    """
    Score each channel by Fisher ratio using pooled features:
      mean_c and std_c across spatial -> 2 features per channel, sum their fisher.
    Returns: [C] scores (higher = more style-discriminative)
    """
    # latents: list of [C,H,W] tensors (CPU ok)
    C = latents[0].shape[0]
    X_mean = []
    X_std = []
    for z in latents:
        z = z.float()
        X_mean.append(z.mean(dim=(1, 2)).cpu().numpy())  # [C]
        X_std.append(z.std(dim=(1, 2)).cpu().numpy())    # [C]
    X_mean = np.stack(X_mean, axis=0)  # [N,C]
    X_std = np.stack(X_std, axis=0)    # [N,C]

    scores = np.zeros((C,), dtype=np.float64)
    classes = np.unique(labels)

    def fisher_1d(x: np.ndarray) -> float:
        # x: [N]
        mu = x.mean()
        num = 0.0
        den = 0.0
        for c in classes:
            xc = x[labels == c]
            if len(xc) < 2:
                continue
            muc = xc.mean()
            varc = xc.var()
            num += len(xc) * (muc - mu) ** 2
            den += len(xc) * varc
        return float(num / (den + 1e-12))

    for c in range(C):
        scores[c] = fisher_1d(X_mean[:, c]) + fisher_1d(X_std[:, c])

    return scores


# -----------------------------
# Evaluation
# -----------------------------

def sample_pairs(style_to_indices: Dict[int, List[int]],
                 n_pairs: int,
                 rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pairs indices and labels:
      y=1: intra-style
      y=0: inter-style
    Returns:
      pairs: [2*n_pairs, 2] indices
      y:     [2*n_pairs]
    """
    styles = list(style_to_indices.keys())
    pairs = []
    y = []

    # intra
    for _ in range(n_pairs):
        s = rng.choice(styles)
        idxs = style_to_indices[s]
        if len(idxs) < 2:
            continue
        a, b = rng.choice(idxs, size=2, replace=False)
        pairs.append((a, b))
        y.append(1)

    # inter
    for _ in range(n_pairs):
        s1, s2 = rng.choice(styles, size=2, replace=False)
        a = rng.choice(style_to_indices[s1])
        b = rng.choice(style_to_indices[s2])
        pairs.append((a, b))
        y.append(0)

    return np.array(pairs, dtype=np.int64), np.array(y, dtype=np.int64)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size between distributions a (intra) and b (inter): (mean_b - mean_a)/pooled_std."""
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(0.5 * (va + vb) + 1e-12)
    return float((mb - ma) / pooled)


@torch.no_grad()
def fit_subspace_from_patch_cache(
    patch_cache: Dict[int, torch.Tensor],
    sample_labels: Dict[int, int],
    mode: str,
    subspace_dim: int,
    max_fit_patches: int,
    seed: int,
    device: torch.device,
) -> Dict:
    """
    Fit subspace transform on sampled patch vectors from current seed.
    Returns a dict with fields: mode, mean, W, out_dim.
    """
    if mode == "none":
        any_pts = next(iter(patch_cache.values()))
        return {"mode": "none", "mean": None, "W": None, "out_dim": int(any_pts.shape[1])}

    rng = np.random.RandomState(seed)
    n_samples = len(patch_cache)
    if n_samples == 0:
        raise ValueError("Empty patch_cache for subspace fitting.")

    per_sample_cap = max(1, int(max_fit_patches) // max(1, n_samples))
    xs = []
    ys = []
    for idx, pts in patch_cache.items():
        n = int(pts.shape[0])
        take = min(n, per_sample_cap)
        if take <= 0:
            continue
        if take < n:
            sel = rng.choice(n, size=take, replace=False)
            sel_t = torch.from_numpy(sel).to(pts.device, dtype=torch.long)
            x = pts.index_select(0, sel_t)
        else:
            x = pts
        xs.append(x)
        ys.append(torch.full((x.shape[0],), int(sample_labels[int(idx)]), device=pts.device, dtype=torch.long))

    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    n, d = int(X.shape[0]), int(X.shape[1])
    if n < 4:
        return {"mode": "none", "mean": None, "W": None, "out_dim": d}

    if n > int(max_fit_patches):
        sel = rng.choice(n, size=int(max_fit_patches), replace=False)
        sel_t = torch.from_numpy(sel).to(X.device, dtype=torch.long)
        X = X.index_select(0, sel_t)
        y = y.index_select(0, sel_t)
        n = int(X.shape[0])

    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu

    if mode in ("pca", "zca"):
        cov = (Xc.T @ Xc) / float(max(1, n - 1))
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        dout = d if int(subspace_dim) <= 0 else min(d, int(subspace_dim))
        V = eigvecs[:, :dout]
        if mode == "pca":
            W = V
        else:
            # PCA-whitened coordinates (often referred to as ZCA/PCA whitening in practice).
            scale = torch.rsqrt(eigvals[:dout].clamp_min(1e-8))
            W = V * scale.unsqueeze(0)
        return {"mode": mode, "mean": mu.to(device), "W": W.to(device), "out_dim": int(dout)}

    if mode == "lda":
        classes = torch.unique(y)
        if int(classes.numel()) < 2:
            return {"mode": "none", "mean": None, "W": None, "out_dim": d}
        mu_all = X.mean(dim=0, keepdim=True)
        Sw = torch.zeros((d, d), dtype=X.dtype, device=X.device)
        Sb = torch.zeros((d, d), dtype=X.dtype, device=X.device)
        for c in classes:
            m = (y == c)
            Xc_cls = X[m]
            if Xc_cls.shape[0] < 2:
                continue
            muc = Xc_cls.mean(dim=0, keepdim=True)
            Xcw = Xc_cls - muc
            Sw = Sw + (Xcw.T @ Xcw)
            dm = (muc - mu_all)
            Sb = Sb + float(Xc_cls.shape[0]) * (dm.T @ dm)
        reg = 1e-4 * (Sw.trace() / float(max(1, d)) + 1e-8)
        Sw = Sw + reg * torch.eye(d, device=X.device, dtype=X.dtype)
        ew, Uw = torch.linalg.eigh(Sw)
        inv_sqrt = Uw @ torch.diag(torch.rsqrt(ew.clamp_min(1e-8))) @ Uw.T
        M = inv_sqrt @ Sb @ inv_sqrt
        em, Um = torch.linalg.eigh(M)
        max_lda = int(classes.numel()) - 1
        dout = min(d, max_lda)
        if int(subspace_dim) > 0:
            dout = min(dout, int(subspace_dim))
        if dout <= 0:
            return {"mode": "none", "mean": None, "W": None, "out_dim": d}
        order = torch.argsort(em, descending=True)
        P = Um[:, order[:dout]]
        W = inv_sqrt @ P
        return {"mode": "lda", "mean": mu_all.to(device), "W": W.to(device), "out_dim": int(dout)}

    raise ValueError(f"Unsupported subspace mode: {mode}")


@torch.no_grad()
def apply_subspace(pts: torch.Tensor, subspace: Dict) -> torch.Tensor:
    if subspace["mode"] == "none":
        return pts
    return (pts - subspace["mean"]) @ subspace["W"]


@torch.no_grad()
def eval_swd_separability(latents: List[torch.Tensor],
                          labels: np.ndarray,
                          cfg: SWDConfig,
                          k: int,
                          channel_order: Optional[np.ndarray],
                          seeds: List[int],
                          n_pairs: int,
                          device: torch.device,
                          cache_device: str = "cpu",
                          subspace: str = "none",
                          subspace_dim: int = 0,
                          subspace_fit_patches: int = 50000,
                          use_delta: bool = False) -> Dict:
    """
    Evaluate SWD as a distance-based classifier for "same style?".
    Returns metrics aggregated over seeds.
    """
    N = len(latents)
    C = latents[0].shape[0]
    k = min(k, C)

    if channel_order is None:
        ch_idx = None
    else:
        ch_idx = torch.tensor(channel_order[:k], dtype=torch.long)

    style_to_indices = {}
    for i, lab in enumerate(labels):
        style_to_indices.setdefault(int(lab), []).append(i)

    all_auc = []
    all_d = []
    all_cv = []
    example = None

    for sd in seeds:
        rng = np.random.RandomState(sd)
        pairs, y = sample_pairs(style_to_indices, n_pairs=n_pairs, rng=rng)
        if len(pairs) < 10:
            continue

        used = np.unique(pairs.reshape(-1))
        patch_cache = {}
        for idx in used:
            z = latents[int(idx)]
            pts = extract_patches(
                z,
                patch=cfg.patch,
                stride=cfg.stride,
                channels=ch_idx.to(device) if ch_idx is not None else None,
                max_patches=cfg.max_patches,
                patch_norm=cfg.patch_norm,
                padding_mode=cfg.padding_mode,
                device=device,
                seed=sd * 1000003 + int(idx),
            )
            patch_cache[int(idx)] = pts.cpu() if cache_device == "cpu" else pts

        subspace_fit = fit_subspace_from_patch_cache(
            patch_cache=patch_cache,
            sample_labels={int(i): int(labels[int(i)]) for i in used},
            mode=subspace,
            subspace_dim=int(subspace_dim),
            max_fit_patches=int(subspace_fit_patches),
            seed=sd,
            device=torch.device("cpu") if cache_device == "cpu" else device,
        )
        dim = int(subspace_fit["out_dim"])
        proj = make_projections(dim, cfg.num_projections, seed=sd, device=device, orthogonal=cfg.orthogonal_proj)
        patch_cache_tx = {idx: apply_subspace(pts, subspace_fit) for idx, pts in patch_cache.items()}

        dists = np.zeros((len(pairs),), dtype=np.float64)
        for j, (a, b) in enumerate(pairs):
            da = patch_cache_tx[int(a)]
            db = patch_cache_tx[int(b)]
            d = swd_distance_from_patches(da, db, proj)
            dists[j] = float(d.detach().cpu().item())

        scores = -dists
        auc = roc_auc_score(y, scores)

        intra = dists[y == 1]
        inter = dists[y == 0]
        d_eff = cohen_d(intra, inter)
        cv = float(dists.std() / (dists.mean() + 1e-12))

        all_auc.append(float(auc))
        all_d.append(float(d_eff))
        all_cv.append(float(cv))

        if example is None:
            example = {"pairs": pairs, "y": y, "dists": dists}

    if not all_auc:
        return {"ok": False, "reason": "Not enough pairs / seeds to evaluate."}

    return {
        "ok": True,
        "auc_mean": float(np.mean(all_auc)),
        "auc_std": float(np.std(all_auc)),
        "dprime_mean": float(np.mean(all_d)),
        "dprime_std": float(np.std(all_d)),
        "cv_mean": float(np.mean(all_cv)),
        "cv_std": float(np.std(all_cv)),
        "example": example,
        "C": int(C),
        "k_used": int(k),
        "patch": int(cfg.patch),
        "padding_mode": cfg.padding_mode,
        "subspace": subspace,
        "subspace_dim": int(subspace_dim),
    }


# -----------------------------
# Style prototype distances
# -----------------------------

@torch.no_grad()
def build_style_prototypes(latents: List[torch.Tensor],
                           labels: np.ndarray,
                           cfg: SWDConfig,
                           k: int,
                           channel_order: Optional[np.ndarray],
                           max_samples_per_style: int,
                           device: torch.device,
                           seed: int) -> Dict[int, torch.Tensor]:
    """
    For each style, sample up to max_samples_per_style latents, extract patches and concatenate,
    then subsample to cfg.max_patches to form a prototype patch set.
    """
    rng = np.random.RandomState(seed)
    C = latents[0].shape[0]
    k = min(k, C)
    ch_idx = None if channel_order is None else torch.tensor(channel_order[:k], dtype=torch.long).to(device)

    style_ids = np.unique(labels)
    protos = {}
    for s in style_ids:
        idxs = np.where(labels == s)[0]
        if len(idxs) == 0:
            continue
        take = idxs
        if len(idxs) > max_samples_per_style:
            take = rng.choice(idxs, size=max_samples_per_style, replace=False)

        pts_all = []
        for idx in take:
            z = latents[int(idx)]
            pts = extract_patches(
                z, patch=cfg.patch, stride=cfg.stride, channels=ch_idx,
                max_patches=cfg.max_patches, patch_norm=cfg.patch_norm,
                padding_mode=cfg.padding_mode,
                device=device, seed=seed * 1000003 + int(idx)
            )
            pts_all.append(pts.cpu())
        pts_cat = torch.cat(pts_all, dim=0)

        # subsample prototype patches
        if cfg.max_patches is not None and pts_cat.shape[0] > cfg.max_patches:
            g = torch.Generator(device="cpu")
            g.manual_seed(seed + int(s) * 1337)
            perm = torch.randperm(pts_cat.shape[0], generator=g, device="cpu")[:cfg.max_patches]
            pts_cat = pts_cat[perm]

        protos[int(s)] = pts_cat

    return protos


@torch.no_grad()
def style_distance_matrix(protos: Dict[int, torch.Tensor],
                          cfg: SWDConfig,
                          device: torch.device,
                          seed: int) -> Tuple[np.ndarray, List[int]]:
    """
    Compute SWD between style prototype patch sets.
    Return (SxS matrix, style_id_list)
    """
    style_ids = sorted(protos.keys())
    S = len(style_ids)
    if S < 2:
        raise ValueError("Need >=2 styles to compute distance matrix.")

    dim = protos[style_ids[0]].shape[1]
    proj = make_projections(dim, cfg.num_projections, seed=seed, device=device, orthogonal=cfg.orthogonal_proj)

    M = np.zeros((S, S), dtype=np.float64)
    for i, si in enumerate(style_ids):
        for j, sj in enumerate(style_ids):
            if j < i:
                continue
            d = swd_distance_from_patches(protos[si], protos[sj], proj)
            v = float(d.detach().cpu().item())
            M[i, j] = v
            M[j, i] = v
    return M, style_ids


# -----------------------------
# Fisher map (spatial) for visual explanation
# -----------------------------

@torch.no_grad()
def fisher_map(latents: List[torch.Tensor], labels: np.ndarray, step: int = 1) -> np.ndarray:
    """
    For each (h,w) (subsampled by step), compute Fisher ratio on vector z[:,h,w].
    Return heatmap [H,W] (upsampled back by nearest).
    """
    Z = torch.stack([z.float() for z in latents], dim=0)  # [N,C,H,W]
    N, C, H, W = Z.shape
    classes = np.unique(labels)
    labels_t = torch.tensor(labels, dtype=torch.long)

    hs = list(range(0, H, step))
    ws = list(range(0, W, step))
    out = torch.zeros((len(hs), len(ws)), dtype=torch.float32)

    # Precompute class masks
    masks = {int(c): (labels_t == int(c)) for c in classes}

    # For each position, compute fisher on magnitude across channels:
    # between-class scatter / within-class scatter using vector mean norms
    for i, h in enumerate(hs):
        for j, w in enumerate(ws):
            X = Z[:, :, h, w]  # [N,C]
            mu = X.mean(dim=0, keepdim=True)  # [1,C]
            # between / within
            num = 0.0
            den = 0.0
            for c in classes:
                m = masks[int(c)]
                if m.sum().item() < 2:
                    continue
                Xc = X[m]  # [Nc,C]
                muc = Xc.mean(dim=0, keepdim=True)
                varc = Xc.var(dim=0, unbiased=False).mean().item()
                num += float(Xc.shape[0]) * float(((muc - mu) ** 2).mean().item())
                den += float(Xc.shape[0]) * varc
            out[i, j] = num / (den + 1e-12)

    # upsample back
    out = out.unsqueeze(0).unsqueeze(0)  # [1,1,h',w']
    out_up = F.interpolate(out, size=(H, W), mode="nearest").squeeze().cpu().numpy()
    return out_up


# -----------------------------
# Plotting
# -----------------------------

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plot_hist(intra: np.ndarray, inter: np.ndarray, out_path: str, title: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(intra, bins=50, alpha=0.6, density=True, label="intra (same style)")
    plt.hist(inter, bins=50, alpha=0.6, density=True, label="inter (diff style)")
    plt.legend()
    plt.title(title)
    plt.xlabel("SWD distance")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(y: np.ndarray, scores: np.ndarray, out_path: str, title: str):
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title(title)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_heatmap(mat: np.ndarray, x_ticks: List[int], y_ticks: List[int],
                 out_path: str, title: str, xlabel: str, ylabel: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(x_ticks)), x_ticks)
    plt.yticks(range(len(y_ticks)), y_ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pair_scatter(pairs: np.ndarray,
                      dists: np.ndarray,
                      labels: np.ndarray,
                      style_names: List[str],
                      out_path: str,
                      max_points: int = 4000):
    import matplotlib.pyplot as plt

    M = len(dists)
    if M == 0:
        return
    idx = np.arange(M)
    if M > max_points:
        rng = np.random.RandomState(0)
        idx = rng.choice(idx, size=max_points, replace=False)

    pairs_s = pairs[idx]
    dists_s = dists[idx]

    keys = []
    intra_flags = []
    for a, b in pairs_s:
        sa = style_names[int(labels[int(a)])]
        sb = style_names[int(labels[int(b)])]
        if sa <= sb:
            keys.append(f"{sa}-{sb}")
        else:
            keys.append(f"{sb}-{sa}")
        intra_flags.append(sa == sb)

    uniq = sorted(set(keys))
    xmap = {k: i for i, k in enumerate(uniq)}
    xs = np.array([xmap[k] for k in keys], dtype=np.float32)
    rng = np.random.RandomState(1)
    xs = xs + (rng.rand(len(xs)) - 0.5) * 0.35
    colors = ["tab:blue" if intra else "tab:orange" for intra in intra_flags]

    plt.figure(figsize=(max(10.0, len(uniq) * 0.6), 5.0))
    plt.scatter(xs, dists_s, s=10, alpha=0.35, c=colors, linewidths=0)
    plt.xticks(range(len(uniq)), uniq, rotation=45, ha="right")
    plt.ylabel("SWD distance")
    plt.title("Pairwise SWD distances by style pair (blue=intra, orange=inter)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def compute_swd_distance_matrix_for_samples(latents: List[torch.Tensor],
                                            labels: np.ndarray,
                                            cfg: SWDConfig,
                                            k: int,
                                            channel_order: Optional[np.ndarray],
                                            device: torch.device,
                                            cache_device: str = "cpu",
                                            subspace: str = "none",
                                            subspace_dim: int = 0,
                                            subspace_fit_patches: int = 50000,
                                            sample_n: int = 300,
                                            seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    N = len(latents)
    all_idx = np.arange(N, dtype=np.int64)
    if sample_n is not None and sample_n > 0 and N > int(sample_n):
        idx = rng.choice(all_idx, size=int(sample_n), replace=False)
        idx = np.sort(idx)
    else:
        idx = all_idx

    C = latents[0].shape[0]
    k = min(int(k), int(C))
    ch_idx = None if channel_order is None else torch.tensor(channel_order[:k], dtype=torch.long).to(device)

    patch_cache: Dict[int, torch.Tensor] = {}
    for ii in idx:
        patch_cache[int(ii)] = extract_patches(
            latents[int(ii)],
            patch=cfg.patch,
            stride=cfg.stride,
            channels=ch_idx,
            max_patches=cfg.max_patches,
            patch_norm=cfg.patch_norm,
            padding_mode=cfg.padding_mode,
            device=device,
            seed=seed * 1000003 + int(ii),
        )
        if cache_device == "cpu":
            patch_cache[int(ii)] = patch_cache[int(ii)].cpu()

    sub = fit_subspace_from_patch_cache(
        patch_cache=patch_cache,
        sample_labels={int(i): int(labels[int(i)]) for i in idx},
        mode=subspace,
        subspace_dim=int(subspace_dim),
        max_fit_patches=int(subspace_fit_patches),
        seed=seed,
        device=torch.device("cpu") if cache_device == "cpu" else device,
    )
    patch_cache_tx = {ii: apply_subspace(pts, sub) for ii, pts in patch_cache.items()}
    dim = int(sub["out_dim"])
    proj = make_projections(dim, cfg.num_projections, seed=seed, device=device, orthogonal=cfg.orthogonal_proj)

    S = len(idx)
    D = np.zeros((S, S), dtype=np.float64)
    for a in range(S):
        ia = int(idx[a])
        for b in range(a + 1, S):
            ib = int(idx[b])
            d = swd_distance_from_patches(patch_cache_tx[ia], patch_cache_tx[ib], proj)
            v = float(d.detach().cpu().item())
            D[a, b] = v
            D[b, a] = v
    return D, idx


def plot_mds_samples(D: np.ndarray,
                     labs: np.ndarray,
                     style_names: List[str],
                     out_path: str,
                     title: str = "MDS on sample SWD distances"):
    import matplotlib.pyplot as plt

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    X = mds.fit_transform(D)
    plt.figure(figsize=(6, 6))
    for s in np.unique(labs):
        m = (labs == s)
        plt.scatter(X[m, 0], X[m, 1], s=14, alpha=0.65, label=style_names[int(s)])
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_patch_combinations(patch_sizes: List[int], max_combo_size: int = 0) -> List[Tuple[int, ...]]:
    """
    Build all unique patch-size combinations with size >= 2.
    max_combo_size <= 0 means no cap.
    """
    uniq = sorted(set(int(p) for p in patch_sizes))
    if len(uniq) < 2:
        return []
    upper = len(uniq) if max_combo_size <= 0 else min(len(uniq), int(max_combo_size))
    combos: List[Tuple[int, ...]] = []
    for r in range(2, upper + 1):
        combos.extend(list(itertools.combinations(uniq, r)))
    return combos


def resolve_subspace_modes(subspace_args: List[str]) -> List[str]:
    base = ["none", "pca", "zca", "lda"]
    out: List[str] = []
    for s in subspace_args:
        if s == "all":
            for x in base:
                if x not in out:
                    out.append(x)
        elif s in base and s not in out:
            out.append(s)
    if not out:
        out = base.copy()
    return out


@torch.no_grad()
def eval_swd_patch_combo(latents: List[torch.Tensor],
                         labels: np.ndarray,
                         patch_combo: Tuple[int, ...],
                         stride: int,
                         padding_mode: str,
                         num_projections: int,
                         max_patches: int,
                         patch_norm: str,
                         k: int,
                         channel_order: Optional[np.ndarray],
                         seeds: List[int],
                         n_pairs: int,
                         device: torch.device,
                         cache_device: str = "cpu",
                         subspace: str = "none",
                         subspace_dim: int = 0,
                         subspace_fit_patches: int = 50000) -> Dict:
    """
    Evaluate SWD separability using an average distance over multiple patch sizes.
    """
    if len(patch_combo) < 2:
        return {"ok": False, "reason": "patch_combo must contain at least 2 patch sizes."}

    N = len(latents)
    C = latents[0].shape[0]
    k = min(k, C)

    if channel_order is None:
        ch_idx = None
    else:
        ch_idx = torch.tensor(channel_order[:k], dtype=torch.long)

    style_to_indices = {}
    for i, lab in enumerate(labels):
        style_to_indices.setdefault(int(lab), []).append(i)

    all_auc = []
    all_d = []
    all_cv = []
    example = None

    for sd in seeds:
        rng = np.random.RandomState(sd)
        pairs, y = sample_pairs(style_to_indices, n_pairs=n_pairs, rng=rng)
        if len(pairs) < 10:
            continue

        used = np.unique(pairs.reshape(-1))

        patch_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        patch_subspace: Dict[int, Dict] = {}
        patch_cache_tx: Dict[Tuple[int, int], torch.Tensor] = {}
        proj_cache: Dict[int, torch.Tensor] = {}

        for idx in used:
            z = latents[int(idx)]
            for p in patch_combo:
                pts = extract_patches(
                    z,
                    patch=int(p),
                    stride=int(stride),
                    channels=ch_idx.to(device) if ch_idx is not None else None,
                    max_patches=max_patches,
                    patch_norm=patch_norm,
                    padding_mode=padding_mode,
                    device=device,
                    seed=sd * 1000003 + int(idx) * 97 + int(p),
                )
                patch_cache[(int(idx), int(p))] = pts.cpu() if cache_device == "cpu" else pts

        for p in patch_combo:
            single_cache = {int(idx): patch_cache[(int(idx), int(p))] for idx in used}
            sub = fit_subspace_from_patch_cache(
                patch_cache=single_cache,
                sample_labels={int(i): int(labels[int(i)]) for i in used},
                mode=subspace,
                subspace_dim=int(subspace_dim),
                max_fit_patches=int(subspace_fit_patches),
                seed=sd * 9973 + int(p),
                device=torch.device("cpu") if cache_device == "cpu" else device,
            )
            patch_subspace[int(p)] = sub
            dim = int(sub["out_dim"])
            proj_cache[int(p)] = make_projections(
                dim=dim,
                num_projections=num_projections,
                seed=sd * 9973 + int(p),
                device=device,
                orthogonal=True,
            )
            for idx in used:
                key = (int(idx), int(p))
                patch_cache_tx[key] = apply_subspace(patch_cache[key], sub)

        dists = np.zeros((len(pairs),), dtype=np.float64)
        for j, (a, b) in enumerate(pairs):
            per_patch = []
            for p in patch_combo:
                da = patch_cache_tx[(int(a), int(p))]
                db = patch_cache_tx[(int(b), int(p))]
                d = swd_distance_from_patches(da, db, proj_cache[int(p)])
                per_patch.append(float(d.detach().cpu().item()))
            dists[j] = float(np.mean(per_patch))

        scores = -dists
        auc = roc_auc_score(y, scores)
        intra = dists[y == 1]
        inter = dists[y == 0]
        d_eff = cohen_d(intra, inter)
        cv = float(dists.std() / (dists.mean() + 1e-12))

        all_auc.append(float(auc))
        all_d.append(float(d_eff))
        all_cv.append(float(cv))

        if example is None:
            example = {"pairs": pairs, "y": y, "dists": dists}

    if not all_auc:
        return {"ok": False, "reason": "Not enough pairs / seeds to evaluate."}

    return {
        "ok": True,
        "patch_combo": [int(p) for p in patch_combo],
        "patch_label": "+".join(str(int(p)) for p in patch_combo),
        "auc_mean": float(np.mean(all_auc)),
        "auc_std": float(np.std(all_auc)),
        "dprime_mean": float(np.mean(all_d)),
        "dprime_std": float(np.std(all_d)),
        "cv_mean": float(np.mean(all_cv)),
        "cv_std": float(np.std(all_cv)),
        "example": example,
        "C": int(C),
        "k_used": int(k),
        "padding_mode": padding_mode,
        "subspace": subspace,
        "subspace_dim": int(subspace_dim),
    }


def plot_matrix(M: np.ndarray, labels: List[str], out_path: str, title: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(M, aspect="equal")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mds(M: np.ndarray, labels: List[str], out_path: str, title: str):
    import matplotlib.pyplot as plt
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    X = mds.fit_transform(M)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    for i, lab in enumerate(labels):
        plt.text(X[i, 0], X[i, 1], str(lab))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_fisher_map(H: np.ndarray, out_path: str, title: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(H, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=False,default="../latent-256", help="root folder containing style subfolders")
    ap.add_argument("--out_dir", type=str, required=False,default="swd_results", help="folder to save results")
    ap.add_argument("--latent_key", type=str, default=None, help="if .pt is dict, use this key")
    ap.add_argument("--device", type=str, default="cuda",
                    help="compute device, default=cuda")
    ap.add_argument("--allow_cpu", action="store_true",
                    help="allow CPU fallback when CUDA is unavailable")

    ap.add_argument("--base_style", type=str, default=None,
                    help="optional: compute delta = z_style - z_base_style by matching filenames across styles")
    ap.add_argument("--max_per_style", type=int, default=2000, help="cap samples per style to control cost")

    ap.add_argument("--patch_sizes", type=int, nargs="+", default=[1, 3, 5, 7,9])
    ap.add_argument("--enable_patch_combos", action="store_true",
                    help="also evaluate combinations of patch sizes (size>=2)")
    ap.add_argument("--max_combo_size", type=int, default=0,
                    help="max number of patch sizes in a combo; <=0 means no cap")
    ap.add_argument("--k_list", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--padding_mode", type=str, default="same", choices=["same", "valid"])
    ap.add_argument("--num_projections", type=int, default=512)
    ap.add_argument("--max_patches", type=int, default=4096,
                    help="max patches per sample (reduced to 1/4 for VRAM)")
    ap.add_argument("--patch_norm", type=str, default="standardize", choices=["none", "l2", "standardize"])
    ap.add_argument("--subspace", type=str, nargs="+", default=["all"], choices=["all", "none", "pca", "zca", "lda"],
                    help="subspace mode(s); default runs all: none/pca/zca/lda")
    ap.add_argument("--subspace_dim", type=int, default=0, help="<=0 means auto/max allowed for selected subspace")
    ap.add_argument("--subspace_fit_patches", type=int, default=12500,
                    help="max sampled patches to fit subspace transform per seed (reduced to 1/4)")
    ap.add_argument("--cache_device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="where to cache extracted patches; cpu greatly reduces VRAM")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_pairs", type=int, default=5000)

    ap.add_argument("--proto_samples_per_style", type=int, default=30)
    ap.add_argument("--fisher_step", type=int, default=1)
    ap.add_argument("--pair_scatter_max_points", type=int, default=4000)
    ap.add_argument("--sample_mds_n", type=int, default=300,
                    help="number of samples for sample-level SWD MDS; <=1 disables")
    ap.add_argument("--sample_mds_seed", type=int, default=0)

    args = ap.parse_args()
    subspace_modes = resolve_subspace_modes(args.subspace)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        if args.allow_cpu:
            print("[WARN] CUDA unavailable, falling back to CPU because --allow_cpu is set.")
            args.device = "cpu"
        else:
            raise RuntimeError(
                "CUDA is unavailable, but --device is set to cuda. "
                "Install CUDA-enabled PyTorch or pass --allow_cpu to run on CPU."
            )
    device = torch.device(args.device)
    print(f"[INFO] device: {device}")
    print(f"[INFO] cache_device: {args.cache_device}")
    print(f"[INFO] subspace modes: {subspace_modes}")

    styles = list_style_files(args.data_root)
    style_names = sorted(styles.keys())
    print(f"[INFO] styles: {style_names}")
    for s in style_names:
        print(f"  - {s}: {len(styles[s])} files")

    # Build sample list (raw or delta)
    latents: List[torch.Tensor] = []
    labels: List[int] = []
    sample_meta: List[Dict] = []

    if args.base_style is not None:
        if args.base_style not in styles:
            raise ValueError(f"--base_style '{args.base_style}' not found. Available={style_names}")
        common = pair_intersection(styles, base_style=args.base_style)
        if not common:
            raise ValueError("No common filenames across styles. Cannot compute delta mode.")

        # limit by max_per_style via ids (uniformly)
        all_ids = list(common.keys())
        # if too many, subsample ids
        if args.max_per_style is not None:
            # ensure each style has at most max_per_style ids
            max_ids = args.max_per_style
            if len(all_ids) > max_ids:
                all_ids = all_ids[:max_ids]

        print(f"[INFO] delta-mode: using {len(all_ids)} matched ids across {len(style_names)} styles.")

        for cid in tqdm(all_ids, desc="load delta latents"):
            base_path = common[cid][args.base_style]
            z_base = load_latent(base_path, key=args.latent_key, map_location="cpu")

            for si, s in enumerate(style_names):
                if s == args.base_style:
                    continue
                z = load_latent(common[cid][s], key=args.latent_key, map_location="cpu")
                dz = (z - z_base).contiguous()
                latents.append(dz)
                labels.append(si)  # label by style index (including base in label space; ok)
                sample_meta.append({"id": cid, "style": s, "path": common[cid][s], "base": base_path})

        # relabel to consecutive ids excluding base if it was skipped
        # keep as-is for simplicity

    else:
        # raw-mode
        rng = np.random.RandomState(0)
        for si, s in enumerate(style_names):
            paths = styles[s]
            if args.max_per_style is not None and len(paths) > args.max_per_style:
                paths = list(rng.choice(paths, size=args.max_per_style, replace=False))
            for p in tqdm(paths, desc=f"load {s}"):
                z = load_latent(p, key=args.latent_key, map_location="cpu")
                latents.append(z)
                labels.append(si)
                sample_meta.append({"style": s, "path": p})

    labels_np = np.array(labels, dtype=np.int64)
    N = len(latents)
    if N < 10:
        raise ValueError(f"Too few samples loaded: {N}")

    # Check shapes consistent
    C, H, W = latents[0].shape
    for z in latents[1:]:
        if z.shape != (C, H, W):
            raise ValueError(f"Shape mismatch: expected {(C,H,W)} but got {tuple(z.shape)}")

    print(f"[INFO] Loaded {N} samples, shape=(C,H,W)=({C},{H},{W})")

    # Channel ranking
    scores = channel_fisher_scores(latents, labels_np)
    order = np.argsort(-scores)  # desc
    save_json(
        {"C": int(C),
         "top_channels": [{"ch": int(order[i]), "score": float(scores[order[i]])} for i in range(min(50, C))]},
        os.path.join(args.out_dir, "channel_ranking.json"),
    )

    # Sweep p x k x subspace
    patch_sizes = args.patch_sizes
    k_list = sorted({min(int(k), int(C)) for k in args.k_list if int(k) >= 1})
    if not k_list:
        k_list = [int(C)]
    if k_list[-1] != C:
        k_list.append(C)

    subspace_mats: Dict[str, Dict[str, np.ndarray]] = {}
    for sm in subspace_modes:
        subspace_mats[sm] = {
            "auc": np.zeros((len(patch_sizes), len(k_list)), dtype=np.float64),
            "auc_std": np.zeros((len(patch_sizes), len(k_list)), dtype=np.float64),
            "d": np.zeros((len(patch_sizes), len(k_list)), dtype=np.float64),
            "cv": np.zeros((len(patch_sizes), len(k_list)), dtype=np.float64),
        }

    summary = []
    best = None

    for sm in subspace_modes:
        print(f"[INFO] sweep subspace={sm}")
        for ip, p in enumerate(patch_sizes):
            cfg = SWDConfig(
                patch=p,
                stride=args.stride,
                num_projections=args.num_projections,
                max_patches=args.max_patches,
                patch_norm=args.patch_norm,
                orthogonal_proj=True,
                padding_mode=args.padding_mode,
            )
            for ik, k in enumerate(k_list):
                res = eval_swd_separability(
                    latents=latents,
                    labels=labels_np,
                    cfg=cfg,
                    k=k,
                    channel_order=order,
                    seeds=args.seeds,
                    n_pairs=args.n_pairs,
                    device=device,
                    cache_device=args.cache_device,
                    subspace=sm,
                    subspace_dim=args.subspace_dim,
                    subspace_fit_patches=args.subspace_fit_patches,
                )
                if not res["ok"]:
                    print(f"[WARN] skip subspace={sm} p={p} k={k}: {res.get('reason','')}")
                    continue

                subspace_mats[sm]["auc"][ip, ik] = res["auc_mean"]
                subspace_mats[sm]["auc_std"][ip, ik] = res["auc_std"]
                subspace_mats[sm]["d"][ip, ik] = res["dprime_mean"]
                subspace_mats[sm]["cv"][ip, ik] = res["cv_mean"]

                summary.append({
                    "subspace": sm,
                    "patch": p,
                    "k": int(res["k_used"]),
                    "auc_mean": res["auc_mean"],
                    "auc_std": res["auc_std"],
                    "dprime_mean": res["dprime_mean"],
                    "dprime_std": res["dprime_std"],
                    "cv_mean": res["cv_mean"],
                    "cv_std": res["cv_std"],
                    "patch_norm": args.patch_norm,
                    "num_projections": args.num_projections,
                    "max_patches": args.max_patches,
                    "stride": args.stride,
                    "padding_mode": args.padding_mode,
                    "subspace_dim": int(args.subspace_dim),
                    "cache_device": args.cache_device,
                })
                print(
                    f"[RES] subspace={sm} p={p} k={int(res['k_used'])} "
                    f"AUC={res['auc_mean']:.4f}±{res['auc_std']:.4f} "
                    f"d'={res['dprime_mean']:.4f}±{res['dprime_std']:.4f} "
                    f"CV={res['cv_mean']:.4f}±{res['cv_std']:.4f}"
                )

                auc = res["auc_mean"]
                cv = res["cv_mean"]
                if best is None or (auc > best["auc"]) or (abs(auc - best["auc"]) < 1e-6 and cv < best["cv"]):
                    best = {
                        "subspace": sm,
                        "patch": p,
                        "k": int(res["k_used"]),
                        "auc": float(auc),
                        "cv": float(cv),
                        "ip": ip,
                        "ik": ik,
                    }
            print(f"[INFO] finished subspace={sm} patch={p}")

    save_json(summary, os.path.join(args.out_dir, "sweep_summary.json"))

    # Optional: sweep patch-size combinations (average SWD across patches in combo)
    combo_summary = []
    if args.enable_patch_combos:
        patch_combos = build_patch_combinations(patch_sizes, max_combo_size=args.max_combo_size)
        if patch_combos:
            combo_labels = ["+".join(str(p) for p in combo) for combo in patch_combos]
            for sm in subspace_modes:
                combo_auc_mat = np.zeros((len(patch_combos), len(k_list)), dtype=np.float64)
                combo_auc_std_mat = np.zeros_like(combo_auc_mat)
                combo_d_mat = np.zeros_like(combo_auc_mat)
                combo_cv_mat = np.zeros_like(combo_auc_mat)

                for ic, combo in enumerate(patch_combos):
                    for ik, k in enumerate(k_list):
                        res = eval_swd_patch_combo(
                            latents=latents,
                            labels=labels_np,
                            patch_combo=combo,
                            stride=args.stride,
                            padding_mode=args.padding_mode,
                            num_projections=args.num_projections,
                            max_patches=args.max_patches,
                            patch_norm=args.patch_norm,
                            k=k,
                            channel_order=order,
                            seeds=args.seeds,
                            n_pairs=args.n_pairs,
                            device=device,
                            cache_device=args.cache_device,
                            subspace=sm,
                            subspace_dim=args.subspace_dim,
                            subspace_fit_patches=args.subspace_fit_patches,
                        )
                        if not res["ok"]:
                            print(f"[WARN] skip subspace={sm} combo={combo} k={k}: {res.get('reason','')}")
                            continue

                        combo_auc_mat[ic, ik] = res["auc_mean"]
                        combo_auc_std_mat[ic, ik] = res["auc_std"]
                        combo_d_mat[ic, ik] = res["dprime_mean"]
                        combo_cv_mat[ic, ik] = res["cv_mean"]

                        combo_summary.append({
                            "subspace": sm,
                            "patch_combo": res["patch_combo"],
                            "patch_label": res["patch_label"],
                            "k": int(res["k_used"]),
                            "auc_mean": res["auc_mean"],
                            "auc_std": res["auc_std"],
                            "dprime_mean": res["dprime_mean"],
                            "dprime_std": res["dprime_std"],
                            "cv_mean": res["cv_mean"],
                            "cv_std": res["cv_std"],
                            "patch_norm": args.patch_norm,
                            "num_projections": args.num_projections,
                            "max_patches": args.max_patches,
                            "stride": args.stride,
                            "padding_mode": args.padding_mode,
                            "subspace_dim": int(args.subspace_dim),
                            "cache_device": args.cache_device,
                            "distance_fusion": "mean",
                        })
                        print(
                            f"[RES] subspace={sm} combo={res['patch_label']} k={int(res['k_used'])} "
                            f"AUC={res['auc_mean']:.4f}±{res['auc_std']:.4f} "
                            f"d'={res['dprime_mean']:.4f}±{res['dprime_std']:.4f} "
                            f"CV={res['cv_mean']:.4f}±{res['cv_std']:.4f}"
                        )
                    print(f"[INFO] finished subspace={sm} patch_combo={combo}")

                suffix = "" if len(subspace_modes) == 1 else f"_{sm}"
                plot_heatmap(
                    combo_auc_mat, x_ticks=k_list, y_ticks=combo_labels,
                    out_path=os.path.join(args.out_dir, f"heatmap_combo_auc{suffix}.png"),
                    title=f"SWD separability (AUC): patch combos x k_channels [{sm}]",
                    xlabel="k_channels (Top-k by Fisher score)",
                    ylabel="patch combo",
                )
                plot_heatmap(
                    combo_auc_std_mat, x_ticks=k_list, y_ticks=combo_labels,
                    out_path=os.path.join(args.out_dir, f"heatmap_combo_auc_std{suffix}.png"),
                    title=f"SWD AUC std across seeds: patch combos x k_channels [{sm}]",
                    xlabel="k_channels",
                    ylabel="patch combo",
                )
                plot_heatmap(
                    combo_d_mat, x_ticks=k_list, y_ticks=combo_labels,
                    out_path=os.path.join(args.out_dir, f"heatmap_combo_dprime{suffix}.png"),
                    title=f"Effect size (d'): patch combos x k_channels [{sm}]",
                    xlabel="k_channels",
                    ylabel="patch combo",
                )
        else:
            print("[INFO] patch combo sweep skipped: need >=2 distinct patch_sizes.")
    save_json(combo_summary, os.path.join(args.out_dir, "combo_sweep_summary.json"))

    if best is None:
        raise RuntimeError("No valid SWD result found in sweep.")
    print(
        f"[BEST] subspace={best['subspace']} patch={best['patch']} "
        f"k={best['k']} AUC={best['auc']:.4f} CV={best['cv']:.4f}"
    )

    # Visualize heatmaps (per subspace)
    for sm in subspace_modes:
        suffix = "" if len(subspace_modes) == 1 else f"_{sm}"
        plot_heatmap(
            subspace_mats[sm]["auc"], x_ticks=k_list, y_ticks=patch_sizes,
            out_path=os.path.join(args.out_dir, f"heatmap_auc{suffix}.png"),
            title=f"SWD separability (AUC) heatmap: patch x k_channels [{sm}]",
            xlabel="k_channels (Top-k by Fisher score)",
            ylabel="patch size",
        )
        plot_heatmap(
            subspace_mats[sm]["auc_std"], x_ticks=k_list, y_ticks=patch_sizes,
            out_path=os.path.join(args.out_dir, f"heatmap_auc_std{suffix}.png"),
            title=f"SWD AUC std across seeds: patch x k_channels [{sm}]",
            xlabel="k_channels",
            ylabel="patch size",
        )
        plot_heatmap(
            subspace_mats[sm]["d"], x_ticks=k_list, y_ticks=patch_sizes,
            out_path=os.path.join(args.out_dir, f"heatmap_dprime{suffix}.png"),
            title=f"Effect size (d') heatmap: patch x k_channels [{sm}]",
            xlabel="k_channels",
            ylabel="patch size",
        )

    # Produce detailed plots for best config (distance hist + ROC)
    best_cfg = SWDConfig(
        patch=int(best["patch"]),
        stride=args.stride,
        num_projections=args.num_projections,
        max_patches=args.max_patches,
        patch_norm=args.patch_norm,
        orthogonal_proj=True,
        padding_mode=args.padding_mode,
    )
    best_eval = eval_swd_separability(
        latents=latents, labels=labels_np, cfg=best_cfg,
        k=int(best["k"]), channel_order=order,
        seeds=[args.seeds[0]], n_pairs=args.n_pairs,
        device=device,
        cache_device=args.cache_device,
        subspace=best["subspace"],
        subspace_dim=args.subspace_dim,
        subspace_fit_patches=args.subspace_fit_patches,
    )
    ex = best_eval["example"]
    y = ex["y"]
    dists = ex["dists"]
    plot_hist(
        intra=dists[y == 1], inter=dists[y == 0],
        out_path=os.path.join(args.out_dir, "best_hist.png"),
        title=f"Best config distance dist: subspace={best['subspace']} patch={best['patch']} k={best['k']} (seed={args.seeds[0]})"
    )
    plot_roc(
        y=y, scores=-dists,
        out_path=os.path.join(args.out_dir, "best_roc.png"),
        title=f"Best config ROC: subspace={best['subspace']} patch={best['patch']} k={best['k']} (seed={args.seeds[0]})"
    )
    plot_pair_scatter(
        pairs=ex["pairs"],
        dists=dists,
        labels=labels_np,
        style_names=style_names,
        out_path=os.path.join(args.out_dir, "best_pair_scatter.png"),
        max_points=int(args.pair_scatter_max_points),
    )

    if int(args.sample_mds_n) > 1:
        D_samples, idx_samples = compute_swd_distance_matrix_for_samples(
            latents=latents,
            labels=labels_np,
            cfg=best_cfg,
            k=int(best["k"]),
            channel_order=order,
            device=device,
            cache_device=args.cache_device,
            subspace=best["subspace"],
            subspace_dim=args.subspace_dim,
            subspace_fit_patches=args.subspace_fit_patches,
            sample_n=int(args.sample_mds_n),
            seed=int(args.sample_mds_seed),
        )
        np.save(os.path.join(args.out_dir, "sample_swd_distance_matrix.npy"), D_samples)
        np.save(os.path.join(args.out_dir, "sample_swd_indices.npy"), idx_samples)
        plot_mds_samples(
            D_samples,
            labels_np[idx_samples],
            style_names=style_names,
            out_path=os.path.join(args.out_dir, "sample_swd_mds.png"),
            title=f"Sample-level MDS by SWD (n={len(idx_samples)}, subspace={best['subspace']}, patch={best['patch']} k={best['k']})",
        )

    # Style prototype distance matrix + MDS for best config
    protos = build_style_prototypes(
        latents=latents, labels=labels_np, cfg=best_cfg,
        k=int(best["k"]), channel_order=order,
        max_samples_per_style=args.proto_samples_per_style,
        device=device, seed=0
    )
    M, style_ids = style_distance_matrix(protos, best_cfg, device=device, seed=0)
    style_labels = [style_names[i] if i < len(style_names) else f"style_{i}" for i in style_ids]
    plot_matrix(
        M, labels=style_labels,
        out_path=os.path.join(args.out_dir, "style_distance_matrix.png"),
        title=f"Style prototype SWD distance matrix (patch={best['patch']} k={best['k']})"
    )
    plot_mds(
        M, labels=style_labels,
        out_path=os.path.join(args.out_dir, "style_mds.png"),
        title=f"MDS of styles by SWD (patch={best['patch']} k={best['k']})"
    )

    # Fisher map (spatial) for intuitive visualization
    fmap = fisher_map(latents, labels_np, step=max(1, int(args.fisher_step)))
    # normalize for visualization
    fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-12)
    np.save(os.path.join(args.out_dir, "fisher_map.npy"), fmap_norm)
    plot_fisher_map(
        fmap_norm,
        out_path=os.path.join(args.out_dir, "fisher_map.png"),
        title="Spatial style-carrying heatmap (Fisher ratio) 鈥?where style is most separable"
    )

    # Save final report
    report = {
        "data_root": args.data_root,
        "num_samples": int(N),
        "shape": [int(C), int(H), int(W)],
        "styles": style_names,
        "best": best,
        "patch_sizes": patch_sizes,
        "enable_patch_combos": bool(args.enable_patch_combos),
        "max_combo_size": int(args.max_combo_size),
        "num_patch_combos": int(len(build_patch_combinations(patch_sizes, max_combo_size=args.max_combo_size))),
        "k_list": k_list,
        "padding_mode": args.padding_mode,
        "subspace_modes": subspace_modes,
        "subspace_dim": int(args.subspace_dim),
        "subspace_fit_patches": int(args.subspace_fit_patches),
        "cache_device": args.cache_device,
        "pair_scatter_max_points": int(args.pair_scatter_max_points),
        "sample_mds_n": int(args.sample_mds_n),
        "sample_mds_seed": int(args.sample_mds_seed),
        "notes": {
            "Interpretation": [
                "Higher AUC means SWD better separates same-style vs different-style pairs.",
                "Higher d-prime indicates stronger separation between intra/inter distance distributions.",
                "Lower CV means distance estimates are more stable.",
                "Brighter fisher_map regions indicate spatial positions with stronger style discriminability."
            ]
        }
    }
    save_json(report, os.path.join(args.out_dir, "report.json"))
    print(f"[DONE] Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

