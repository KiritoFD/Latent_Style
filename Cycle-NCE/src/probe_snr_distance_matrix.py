from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from torchvision.models import VGG19_Weights, vgg19
except Exception:  # pragma: no cover
    VGG19_Weights = None
    vgg19 = None


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    src_dir = Path(__file__).resolve().parent
    candidates = [base_dir / p, src_dir / p, src_dir.parent / p, Path.cwd() / p]
    for c in candidates:
        rc = c.resolve()
        if rc.exists():
            return rc
    return candidates[0].resolve()


def _patch_sizes_from_optuna_use_flags(use_flags: Dict[str, int]) -> List[int]:
    patches: List[int] = []
    for k, v in use_flags.items():
        if not str(k).startswith("use_p"):
            continue
        if int(v) != 1:
            continue
        try:
            p = int(str(k).replace("use_p", ""))
        except ValueError:
            continue
        if p > 0:
            patches.append(p)
    return sorted(set(patches))


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_domain_latents(
    data_root: Path,
    subdir: str,
    max_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    style_dir = data_root / subdir
    files = sorted(style_dir.glob("*.pt")) + sorted(style_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No latent files found in {style_dir}")
    if max_samples > 0:
        files = files[: max(1, int(max_samples))]

    def _load_latent_file(path: Path) -> torch.Tensor:
        if path.suffix.lower() == ".pt":
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                obj = obj.get("latent", obj)
            latent = torch.as_tensor(obj).float()
        elif path.suffix.lower() == ".npy":
            latent = torch.from_numpy(np.load(path)).float()
        else:
            raise ValueError(f"Unsupported latent format: {path}")
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        if latent.ndim != 3:
            raise ValueError(
                f"Expected latent shape [C,H,W], got {tuple(latent.shape)} from {path}"
            )
        return latent

    latents = [_load_latent_file(p) for p in files]
    return (
        torch.stack(latents, dim=0)
        .to(device=device, dtype=dtype, non_blocking=False)
        .contiguous()
    )


def _load_rgb_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if int(image_size) > 0:
            rgb = rgb.resize((int(image_size), int(image_size)), Image.Resampling.BICUBIC)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_domain_images(
    image_root: Path,
    subdir: str,
    max_samples: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    style_dir = image_root / subdir
    files: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        files.extend(sorted(style_dir.glob(ext)))
    if not files:
        raise RuntimeError(f"No image files found in {style_dir}")
    if max_samples > 0:
        files = files[: max(1, int(max_samples))]
    imgs = [_load_rgb_image(p, image_size=image_size) for p in files]
    return (
        torch.stack(imgs, dim=0)
        .to(device=device, dtype=dtype, non_blocking=False)
        .contiguous()
    )


def _iter_progress(iterable, *, total: int | None = None, desc: str = ""):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _build_vgg19_style_blocks(device: torch.device) -> torch.nn.ModuleList:
    if vgg19 is None or VGG19_Weights is None:
        raise RuntimeError(
            "torchvision is not available in the current interpreter, so pixel_vgg_gram cannot run. "
            "Use the project venv / `uv run`."
        )
    features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
    blocks = torch.nn.ModuleList(
        [
            features[:2],
            features[2:7],
            features[7:12],
            features[12:21],
        ]
    )
    for p in blocks.parameters():
        p.requires_grad_(False)
    return blocks


@torch.no_grad()
def _extract_vgg_features(
    x: torch.Tensor,
    *,
    blocks: torch.nn.ModuleList,
    image_size: int,
    batch_size: int,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    n = int(x.shape[0])
    model_device = next(blocks.parameters()).device
    steps = len(range(0, n, batch_size))
    for s in _iter_progress(range(0, n, batch_size), total=steps, desc="vgg feat batches"):
        e = min(n, s + batch_size)
        xb = x[s:e].to(device=model_device, dtype=torch.float32, non_blocking=False)
        mean = xb.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = xb.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        if int(image_size) > 0 and (
            int(xb.shape[-1]) != int(image_size) or int(xb.shape[-2]) != int(image_size)
        ):
            xb = F.interpolate(
                xb,
                size=(int(image_size), int(image_size)),
                mode="bilinear",
                align_corners=False,
            )
        xb = (xb - mean) / std
        grams: List[torch.Tensor] = []
        feat = xb
        for block in blocks:
            feat = block(feat)
            b, c, h, w = feat.shape
            flat = feat.view(b, c, -1)
            gram = torch.bmm(flat, flat.transpose(1, 2)) / float(c * h * w)
            grams.append(gram.view(b, -1))
        outs.append(torch.cat(grams, dim=1).float().cpu())
    return torch.cat(outs, dim=0)


def _pairwise_mse_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    chunk: int,
    compute_device: torch.device,
) -> np.ndarray:
    nx = int(x.shape[0])
    ny = int(y.shape[0])
    out = np.empty((nx, ny), dtype=np.float32)
    same_tensor = x.data_ptr() == y.data_ptr()
    sx_range = range(0, nx, chunk)
    for sx in _iter_progress(sx_range, total=len(range(0, nx, chunk)), desc="mse blocks"):
        ex = min(nx, sx + chunk)
        xb = x[sx:ex].to(device=compute_device, dtype=torch.float32, non_blocking=False)
        xb = xb.reshape(xb.shape[0], -1)
        sy_start = sx if same_tensor else 0
        for sy in range(sy_start, ny, chunk):
            ey = min(ny, sy + chunk)
            yb = y[sy:ey].to(device=compute_device, dtype=torch.float32, non_blocking=False)
            yb = yb.reshape(yb.shape[0], -1)
            d = torch.cdist(xb, yb, p=2.0).pow(2)
            db = d.detach().cpu().numpy()
            out[sx:ex, sy:ey] = db
            if same_tensor and sy != sx:
                out[sy:ey, sx:ex] = db.T
    return out


def _build_projection_bank(
    channels: int,
    patch_sizes: Iterable[int],
    num_projections: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Dict[int, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    bank: Dict[int, torch.Tensor] = {}
    for p in sorted({int(v) for v in patch_sizes if int(v) > 0}):
        w = torch.randn(
            num_projections, channels, p, p, device=device, dtype=dtype, generator=gen
        )
        w = F.normalize(w.view(num_projections, -1), p=2, dim=1).view_as(w)
        bank[p] = w.contiguous()
    return bank


def _gram_features(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    flat = x.view(n, c, h * w)
    gram = torch.bmm(flat, flat.transpose(1, 2)) / float(max(h * w, 1))
    return gram * 100.0


def _pairwise_gram_distance(
    gram_x: torch.Tensor, gram_y: torch.Tensor, chunk: int
) -> torch.Tensor:
    nx = int(gram_x.shape[0])
    ny = int(gram_y.shape[0])
    out = torch.empty((nx, ny), dtype=torch.float32, device=gram_x.device)
    same_tensor = gram_x.data_ptr() == gram_y.data_ptr()
    sx_range = range(0, nx, chunk)
    for sx in _iter_progress(sx_range, total=len(range(0, nx, chunk)), desc="gram blocks"):
        ex = min(nx, sx + chunk)
        gx = gram_x[sx:ex]
        sy_start = sx if same_tensor else 0
        for sy in range(sy_start, ny, chunk):
            ey = min(ny, sy + chunk)
            gy = gram_y[sy:ey]
            db = (gx[:, None] - gy[None, :]).abs().mean(dim=(2, 3))
            out[sx:ex, sy:ey] = db
            if same_tensor and sy != sx:
                out[sy:ey, sx:ex] = db.transpose(0, 1)
    return out


def _project_patches(
    x: torch.Tensor, projection_bank: Dict[int, torch.Tensor]
) -> Dict[int, torch.Tensor]:
    projected: Dict[int, torch.Tensor] = {}
    for p, weights in projection_bank.items():
        proj = F.conv2d(x, weights, padding=p // 2).view(x.shape[0], weights.shape[0], -1)
        projected[p] = torch.sort(proj, dim=-1).values.contiguous()
    return projected


def _pairwise_projected_swd(
    feat_x: Dict[int, torch.Tensor],
    feat_y: Dict[int, torch.Tensor],
    patch_sizes: List[int],
    chunk: int,
) -> torch.Tensor:
    nx = int(next(iter(feat_x.values())).shape[0])
    ny = int(next(iter(feat_y.values())).shape[0])
    out = torch.zeros(
        (nx, ny), dtype=torch.float32, device=next(iter(feat_x.values())).device
    )
    for p in patch_sizes:
        px = feat_x[p]
        py = feat_y[p]
        sx_range = range(0, nx, chunk)
        for sx in _iter_progress(sx_range, total=len(range(0, nx, chunk)), desc=f"swd p={p}"):
            ex = min(nx, sx + chunk)
            vx = px[sx:ex]
            for sy in range(0, ny, chunk):
                ey = min(ny, sy + chunk)
                vy = py[sy:ey]
                out[sx:ex, sy:ey] += (vx[:, None] - vy[None, :]).abs().mean(dim=(2, 3))
    return out / float(max(len(patch_sizes), 1))


def _pairwise_projected_swd_streaming(
    x: torch.Tensor,
    y: torch.Tensor,
    projection_bank: Dict[int, torch.Tensor],
    patch_sizes: List[int],
    *,
    chunk: int,
    compute_device: torch.device,
    projection_chunk: int,
    distance_mode: str,
    cdf_num_bins: int,
    cdf_tau: float,
    cdf_sample_size: int,
    cdf_sample_chunk_size: int,
) -> np.ndarray:
    x = x.to(device=compute_device, dtype=torch.float32, non_blocking=False).contiguous()
    y = y.to(device=compute_device, dtype=torch.float32, non_blocking=False).contiguous()
    nx = int(x.shape[0])
    ny = int(y.shape[0])
    row_chunk = max(1, int(chunk))
    proj_chunk = max(1, int(projection_chunk))
    total = torch.zeros((nx, ny), dtype=torch.float32, device=compute_device)
    mode = str(distance_mode).strip().lower()
    use_cdf = mode in {"cdf", "softcdf", "cdf_soft"}
    cdf_bins = max(8, int(cdf_num_bins))
    tau = max(1e-5, float(cdf_tau))
    sample_size = max(32, int(cdf_sample_size))
    sample_chunk = max(32, int(cdf_sample_chunk_size))
    same_tensor = x.data_ptr() == y.data_ptr()

    def _pairwise_l1_from_features(
        xf: torch.Tensor,
        yf: torch.Tensor,
        denom: float,
        *,
        symmetric: bool,
    ) -> torch.Tensor:
        out = torch.empty((xf.shape[0], yf.shape[0]), dtype=torch.float32, device=compute_device)
        for sx in range(0, int(xf.shape[0]), row_chunk):
            ex = min(int(xf.shape[0]), sx + row_chunk)
            x_block = xf[sx:ex]
            sy_start = sx if symmetric else 0
            for sy in range(sy_start, int(yf.shape[0]), row_chunk):
                ey = min(int(yf.shape[0]), sy + row_chunk)
                y_block = yf[sy:ey]
                db = torch.cdist(x_block, y_block, p=1.0) / float(max(denom, 1.0))
                out[sx:ex, sy:ey] = db
                if symmetric and sy != sx:
                    out[sy:ey, sx:ex] = db.transpose(0, 1)
        return out

    for p in patch_sizes:
        weights = projection_bank[p].to(device=compute_device, dtype=torch.float32)
        num_proj = int(weights.shape[0])
        patch_acc = torch.zeros((nx, ny), dtype=torch.float32, device=compute_device)
        for sp in range(0, num_proj, proj_chunk):
            ep = min(num_proj, sp + proj_chunk)
            w = weights[sp:ep]
            pcount = int(ep - sp)
            x_proj = F.conv2d(x, w, padding=p // 2).view(nx, pcount, -1).contiguous()
            if same_tensor:
                y_proj = x_proj
            else:
                y_proj = F.conv2d(y, w, padding=p // 2).view(ny, pcount, -1).contiguous()

            if use_cdf:
                n_pts = int(x_proj.shape[-1])
                if n_pts > sample_size:
                    sample_idx = torch.randint(
                        0, n_pts, (sample_size,), device=compute_device, dtype=torch.long
                    )
                    x_proj = x_proj.index_select(2, sample_idx)
                    y_proj = y_proj.index_select(2, sample_idx)
                    n_pts = int(x_proj.shape[-1])

                min_val = torch.minimum(x_proj.amin().detach(), y_proj.amin().detach())
                max_val = torch.maximum(x_proj.amax().detach(), y_proj.amax().detach())
                span = (max_val - min_val).clamp_min(1e-6)
                dx = span / float(cdf_bins - 1)
                grid = torch.linspace(min_val, max_val, cdf_bins, device=compute_device, dtype=x_proj.dtype)
                g = grid.view(1, 1, 1, cdf_bins)

                acc_x = torch.zeros((nx, pcount, cdf_bins), device=compute_device, dtype=x_proj.dtype)
                acc_y = torch.zeros((ny, pcount, cdf_bins), device=compute_device, dtype=y_proj.dtype)
                for n0 in range(0, n_pts, sample_chunk):
                    n1 = min(n_pts, n0 + sample_chunk)
                    px = x_proj[:, :, n0:n1].unsqueeze(-1)
                    py = y_proj[:, :, n0:n1].unsqueeze(-1)
                    acc_x += torch.sigmoid((g - px) / tau).sum(dim=2)
                    acc_y += torch.sigmoid((g - py) / tau).sum(dim=2)

                cdf_x = (acc_x / float(max(n_pts, 1))).reshape(nx, -1)
                cdf_y = (acc_y / float(max(n_pts, 1))).reshape(ny, -1)
                chunk_dist = _pairwise_l1_from_features(
                    cdf_x,
                    cdf_y,
                    float(pcount),
                    symmetric=same_tensor,
                ) * float(dx)
            else:
                x_sorted = torch.sort(x_proj, dim=-1).values.reshape(nx, -1)
                y_sorted = torch.sort(y_proj, dim=-1).values.reshape(ny, -1)
                denom = float(max(pcount * int(x_proj.shape[-1]), 1))
                chunk_dist = _pairwise_l1_from_features(
                    x_sorted,
                    y_sorted,
                    denom,
                    symmetric=same_tensor,
                )

            patch_acc += chunk_dist * (float(pcount) / float(max(num_proj, 1)))

        total += patch_acc

    return (total / float(max(len(patch_sizes), 1))).detach().cpu().numpy()


def _vectorize_patches(
    x: torch.Tensor,
    patch_sizes: List[int],
    *,
    max_patches: int,
    seed: int,
) -> Dict[int, torch.Tensor]:
    n, c, _, _ = x.shape
    vecs: Dict[int, torch.Tensor] = {}
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    h = int(x.shape[2])
    w = int(x.shape[3])
    num_positions = h * w
    target_patches = num_positions if max_patches <= 0 else min(int(max_patches), num_positions)
    flat_positions = torch.randperm(num_positions, generator=gen, device=x.device)[:target_patches]
    flat_positions, _ = torch.sort(flat_positions)
    ys = torch.div(flat_positions, w, rounding_mode="floor")
    xs = torch.remainder(flat_positions, w)

    for p in patch_sizes:
        pad = p // 2
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        patches: List[torch.Tensor] = []
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            patch = padded[:, :, y0 : y0 + p, x0 : x0 + p].reshape(n, c * p * p)
            patches.append(patch)
        vecs[p] = torch.stack(patches, dim=1).contiguous()
    return vecs


def _encode_tree_histograms(
    x_vecs: torch.Tensor,
    *,
    num_trees: int,
    max_depth: int,
    seed: int,
) -> List[torch.Tensor]:
    x_vecs = x_vecs - x_vecs.mean(dim=1, keepdim=True)
    n_samples, n_pts, feat_dim = x_vecs.shape
    device = x_vecs.device
    dtype = x_vecs.dtype
    num_trees = max(1, int(num_trees))
    max_depth = max(1, int(max_depth))
    total_internal = (1 << max_depth) - 1

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    split_rules = torch.randn(
        num_trees, total_internal, feat_dim, device=device, dtype=dtype, generator=gen
    )
    split_rules = F.normalize(split_rules, dim=-1)

    tree_ids = (
        torch.arange(num_trees, device=device)
        .view(1, 1, num_trees)
        .expand(n_samples, n_pts, -1)
    )
    node_indices = torch.zeros((n_samples, n_pts, num_trees), device=device, dtype=torch.long)
    offset = 0
    for depth in range(max_depth):
        nodes_at_depth = 1 << depth
        level_rules = (
            split_rules[:, offset : offset + nodes_at_depth, :].permute(1, 0, 2).contiguous()
        )
        selected_rules = level_rules[node_indices, tree_ids]
        dots = (x_vecs.unsqueeze(2) * selected_rules).sum(dim=-1)
        node_indices = (node_indices * 2) + (dots > 0).long()
        offset += nodes_at_depth

    num_leaves = 1 << max_depth
    hist = torch.zeros((n_samples, num_trees, num_leaves), device=device, dtype=torch.float32)
    ones = torch.ones((n_samples, n_pts, num_trees), device=device, dtype=torch.float32)
    hist.scatter_add_(2, node_indices.transpose(1, 2), ones.transpose(1, 2))
    hist = hist / float(max(n_pts, 1))

    levels: List[torch.Tensor] = [hist]
    current = hist
    while current.shape[-1] > 1:
        current = current.view(n_samples, num_trees, -1, 2).sum(dim=-1)
        levels.append(current)
    return levels


def _fast_pairwise_tree_swd(
    cum_mass_x: List[torch.Tensor],
    cum_mass_y: List[torch.Tensor],
    *,
    chunk: int,
) -> torch.Tensor:
    n_x = int(cum_mass_x[0].shape[0])
    n_y = int(cum_mass_y[0].shape[0])
    device = cum_mass_x[0].device
    out = torch.zeros((n_x, n_y), device=device, dtype=torch.float32)

    sx_range = range(0, n_x, chunk)
    for sx in _iter_progress(sx_range, total=len(range(0, n_x, chunk)), desc="tree hist blocks"):
        ex = min(n_x, sx + chunk)
        block = torch.zeros((ex - sx, n_y), device=device, dtype=torch.float32)
        level_weight = 1.0
        for level_hist_x, level_hist_y in zip(cum_mass_x, cum_mass_y):
            hx = level_hist_x[sx:ex]
            diff = torch.abs(hx[:, None] - level_hist_y[None, :])
            block += diff.sum(dim=(2, 3)) * level_weight
            level_weight *= 0.5
        out[sx:ex] = block / float(max(int(cum_mass_x[0].shape[1]), 1))
    return out


def _pairwise_tree_swd(
    vec_x: Dict[int, torch.Tensor],
    vec_y: Dict[int, torch.Tensor],
    patch_sizes: List[int],
    chunk: int,
    *,
    num_trees: int,
    max_depth: int,
    seed: int,
) -> torch.Tensor:
    nx = int(next(iter(vec_x.values())).shape[0])
    ny = int(next(iter(vec_y.values())).shape[0])
    out = torch.zeros((nx, ny), dtype=torch.float32, device=next(iter(vec_x.values())).device)
    for p_idx, p in enumerate(patch_sizes):
        cum_mass_x = _encode_tree_histograms(
            vec_x[p],
            num_trees=num_trees,
            max_depth=max_depth,
            seed=seed + (p_idx * 1009),
        )
        same_tensor = vec_x[p].data_ptr() == vec_y[p].data_ptr()
        if same_tensor:
            cum_mass_y = cum_mass_x
        else:
            cum_mass_y = _encode_tree_histograms(
                vec_y[p],
                num_trees=num_trees,
                max_depth=max_depth,
                seed=seed + (p_idx * 1009),
            )
        out += _fast_pairwise_tree_swd(cum_mass_x, cum_mass_y, chunk=chunk)
    return out / float(max(len(patch_sizes), 1))


def _classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n_components), dtype=np.float64)
    d2 = d**2
    eye = np.eye(n, dtype=np.float64)
    ones = np.ones((n, n), dtype=np.float64) / float(n)
    j = eye - ones
    b = -0.5 * j @ d2 @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    pos = np.clip(eigvals[:n_components], a_min=0.0, a_max=None)
    return eigvecs[:, :n_components] * np.sqrt(pos[None, :])


def _density_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return np.zeros_like(grid)
    std = float(vals.std())
    bw = 1.06 * max(std, 1e-6) * (vals.size ** (-1.0 / 5.0))
    bw = max(bw, 1e-3)
    diff = (grid[:, None] - vals[None, :]) / bw
    dens = np.exp(-0.5 * diff * diff).mean(axis=1) / (bw * math.sqrt(2.0 * math.pi))
    return dens


def _write_svg(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def _save_bar_svg(path: Path, labels: List[str], values: List[float], title: str) -> None:
    width, height = 900, 520
    left, top, right, bottom = 80, 60, 30, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    vmax = max(max(values), 1e-6)
    bar_w = plot_w / max(len(labels), 1)
    colors = ["#2f6fed", "#ff7a00", "#1fa971", "#a046f2", "#d64545", "#0ea5e9"]
    lines = _svg_header(width, height)
    lines.append(
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>'
    )
    lines.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>'
    )
    lines.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>'
    )
    for idx, (label, value) in enumerate(zip(labels, values)):
        h = (value / vmax) * (plot_h - 10)
        x = left + idx * bar_w + bar_w * 0.15
        y = top + plot_h - h
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.7:.1f}" height="{h:.1f}" fill="{colors[idx % len(colors)]}"/>'
        )
        lines.append(
            f'<text x="{x + bar_w * 0.35:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="14" font-family="Arial">{value:.3f}</text>'
        )
        lines.append(
            f'<text x="{x + bar_w * 0.35:.1f}" y="{top + plot_h + 22:.1f}" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>'
        )
    lines.append("</svg>")
    _write_svg(path, "\n".join(lines))


def _save_scatter_svg(
    path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str,
) -> None:
    width, height = 900, 520
    left, top, right, bottom = 70, 50, 180, 50
    plot_w = width - left - right
    plot_h = height - top - bottom
    style_color_map = {
        "photo": "#7f7f7f",   # stone gray
        "hayao": "#17becf",   # sky blue
        "monet": "#2ca02c",   # muted green
        "vangogh": "#ff7f0e", # bright orange
        "cezanne": "#d62728", # burgundy red
    }
    fallback_colors = ["#2f6fed", "#a046f2", "#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b"]
    colors = [
        style_color_map.get(str(name).strip().lower(), fallback_colors[idx % len(fallback_colors)])
        for idx, name in enumerate(class_names)
    ]
    x = coords[:, 0]
    y = coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(coords[:, 0])
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    lines = _svg_header(width, height)
    lines.append(
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>'
    )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black"/>'
    )
    for idx in range(coords.shape[0]):
        px = left + ((x[idx] - x_min) / x_span) * plot_w
        py = top + plot_h - ((y[idx] - y_min) / y_span) * plot_h
        color = colors[int(labels[idx]) % len(colors)]
        lines.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.2" fill="{color}" fill-opacity="0.75"/>'
        )
    for idx, name in enumerate(class_names):
        ly = top + 20 + idx * 24
        color = colors[idx % len(colors)]
        lines.append(
            f'<rect x="{width - 150}" y="{ly - 10}" width="14" height="14" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{width - 130}" y="{ly + 2}" font-size="14" font-family="Arial">{name}</text>'
        )
    lines.append("</svg>")
    _write_svg(path, "\n".join(lines))


def _save_kde_svg(path: Path, intra: np.ndarray, inter: np.ndarray, title: str) -> None:
    width, height = 900, 520
    left, top, right, bottom = 80, 50, 30, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = (
        np.concatenate([intra, inter], axis=0)
        if intra.size and inter.size
        else (intra if intra.size else inter)
    )
    if values.size == 0:
        grid = np.linspace(0.0, 1.0, 200)
    else:
        grid = np.linspace(float(values.min()), float(values.max()), 240)
    intra_d = _density_curve(intra, grid)
    inter_d = _density_curve(inter, grid)
    ymax = max(float(intra_d.max()), float(inter_d.max()), 1e-6)
    lines = _svg_header(width, height)
    lines.append(
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>'
    )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black"/>'
    )

    def _polyline(density: np.ndarray, color: str) -> None:
        pts = []
        for xv, yv in zip(grid, density):
            px = left + ((xv - grid[0]) / max(grid[-1] - grid[0], 1e-6)) * plot_w
            py = top + plot_h - (yv / ymax) * plot_h
            pts.append(f"{px:.1f},{py:.1f}")
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(pts)}"/>'
        )

    _polyline(intra_d, "#2f6fed")
    _polyline(inter_d, "#d64545")
    lines.append(
        f'<text x="{width - 160}" y="{top + 20}" font-size="14" font-family="Arial" fill="#2f6fed">intra-class</text>'
    )
    lines.append(
        f'<text x="{width - 160}" y="{top + 42}" font-size="14" font-family="Arial" fill="#d64545">inter-class</text>'
    )
    lines.append("</svg>")
    _write_svg(path, "\n".join(lines))


def _save_heatmap_svg(path: Path, matrix: np.ndarray, title: str) -> None:
    n = matrix.shape[0]
    cell = max(3, min(12, 720 // max(n, 1)))
    width = 140 + cell * n
    height = 100 + cell * n
    left, top = 70, 50
    m = matrix.astype(np.float64)
    vmin = float(m.min())
    vmax = float(m.max())
    span = max(vmax - vmin, 1e-6)
    lines = _svg_header(width, height)
    lines.append(
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>'
    )
    for i in range(n):
        for j in range(n):
            val = (m[i, j] - vmin) / span
            shade = int(round(255 * (1.0 - val)))
            color = f"rgb(255,{shade},{shade})"
            lines.append(
                f'<rect x="{left + j * cell}" y="{top + i * cell}" width="{cell}" height="{cell}" fill="{color}" stroke="none"/>'
            )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{cell * n}" height="{cell * n}" fill="none" stroke="black"/>'
    )
    lines.append("</svg>")
    _write_svg(path, "\n".join(lines))


def _summarize_distances(distance_matrix: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    n = int(distance_matrix.shape[0])
    intra_vals: List[float] = []
    inter_vals: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            val = float(distance_matrix[i, j])
            if int(labels[i]) == int(labels[j]):
                intra_vals.append(val)
            else:
                inter_vals.append(val)
    intra = np.asarray(intra_vals, dtype=np.float64)
    inter = np.asarray(inter_vals, dtype=np.float64)
    mean_intra = float(intra.mean()) if intra.size else 0.0
    mean_inter = float(inter.mean()) if inter.size else 0.0
    snr = mean_inter / max(mean_intra, 1e-8)
    return {
        "snr": float(snr),
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
        "median_intra": float(np.median(intra)) if intra.size else 0.0,
        "median_inter": float(np.median(inter)) if inter.size else 0.0,
        "num_intra_pairs": int(intra.size),
        "num_inter_pairs": int(inter.size),
    }


def _silhouette_score_precomputed(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    d = np.asarray(distance_matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = int(d.shape[0])
    if n <= 1:
        return 0.0

    unique_labels = np.unique(y)
    if unique_labels.size <= 1:
        return 0.0

    scores = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        same_mask = y == y[i]
        same_mask[i] = False
        if same_mask.any():
            a_i = float(d[i, same_mask].mean())
        else:
            scores[i] = 0.0
            continue

        b_i = math.inf
        for other in unique_labels:
            if int(other) == int(y[i]):
                continue
            other_mask = y == other
            if other_mask.any():
                b_i = min(b_i, float(d[i, other_mask].mean()))

        if not math.isfinite(b_i):
            scores[i] = 0.0
            continue
        denom = max(a_i, b_i, 1e-12)
        scores[i] = (b_i - a_i) / denom
    return float(scores.mean())


def _stack_domains(
    pools: Dict[str, torch.Tensor], style_subdirs: List[str]
) -> Tuple[torch.Tensor, np.ndarray]:
    xs = [pools[name] for name in style_subdirs]
    labels = []
    for idx, name in enumerate(style_subdirs):
        labels.extend([idx] * int(pools[name].shape[0]))
    return torch.cat(xs, dim=0), np.asarray(labels, dtype=np.int64)


# =====================================================================
# Main
# =====================================================================
@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distance-matrix SNR probe for latent Gram/SWD/Tree-SWD."
    )
    parser.add_argument("--config", type=str, default="Layer-Norm.json")
    parser.add_argument("--output-dir", type=str, default="../probe_snr_distance_matrix")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--max-samples-per-domain", type=int, default=400)
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument("--num-projections", type=int, default=512)
    parser.add_argument("--swd-projection-chunk", type=int, default=32)
    parser.add_argument("--swd-row-chunk", type=int, default=128)
    parser.add_argument("--tree-num-trees", type=int, default=8)
    parser.add_argument("--tree-max-depth", type=int, default=6)
    parser.add_argument("--max-tree-patches", type=int, default=0)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--image-root", type=str, default="../../style_data/train")
    parser.add_argument("--pixel-size", type=int, default=64)
    parser.add_argument("--vgg-image-size", type=int, default=224)
    parser.add_argument("--vgg-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    requested_device = str(args.device).strip().lower()
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but no CUDA device is available.")
        compute_device = torch.device("cuda")
    elif requested_device == "cpu":
        compute_device = torch.device("cpu")
    else:
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if compute_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Keep full datasets on CPU to reduce peak VRAM.
    data_device = torch.device("cpu")

    print(f"[INFO] Compute device={compute_device}")
    print(f"[INFO] Data staging device={data_device}")

    config_path = Path(args.config).resolve()
    cfg = _load_json(config_path)
    config_dir = config_path.parent
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    data_root = _resolve_path(str(data_cfg.get("data_root", "")), config_dir)
    image_root = _resolve_path(str(args.image_root), config_dir)

    style_subdirs = list(data_cfg.get("style_subdirs", []))
    if not style_subdirs:
        raise ValueError("config.data.style_subdirs is empty")

    optuna_trial1_use_flags: Dict[str, int] = {
        "use_p1": 1,
        "use_p2": 0,
        "use_p3": 1,
        "use_p4": 1,
        "use_p5": 0,
        "use_p6": 1,
        "use_p7": 0,
        "use_p8": 0,
        "use_p9": 1,
        "use_p10": 1,
        "use_p11": 1,
        "use_p12": 0,
        "use_p13": 0,
        "use_p14": 1,
        "use_p15": 0,
        "use_p16": 0,
        "use_p17": 0,
        "use_p18": 0,
        "use_p19": 1,
        "use_p20": 0,
        "use_p21": 1,
        "use_p22": 0,
        "use_p23": 1,
        "use_p24": 1,
        "use_p25": 0,
    }
    default_optuna_patches = _patch_sizes_from_optuna_use_flags(optuna_trial1_use_flags)
    if args.patch_sizes is None:
        patch_sizes = default_optuna_patches
        patch_source = "optuna-trial1"
    else:
        patch_sizes = sorted({int(v) for v in args.patch_sizes if int(v) > 0})
        patch_source = "cli"
    if not patch_sizes:
        raise ValueError("No valid patch sizes resolved for SWD")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_out_dir = out_dir / f"images_{time.strftime('%Y%m%d_%H%M%S')}"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32

    print(f"[INFO] Data root={data_root}")
    print(f"[INFO] Image root={image_root}")
    print(f"[INFO] Styles={style_subdirs}")
    print(f"[INFO] Patch sizes={patch_sizes} (source={patch_source})")
    print(f"[INFO] Image outputs={image_out_dir}")

    swd_distance_mode = str(loss_cfg.get("swd_distance_mode", "cdf")).lower()
    swd_cdf_num_bins = int(loss_cfg.get("swd_cdf_num_bins", 32))
    swd_cdf_tau = float(loss_cfg.get("swd_cdf_tau", 0.01))
    swd_cdf_sample_size = int(loss_cfg.get("swd_cdf_sample_size", 256))
    swd_cdf_sample_chunk_size = int(loss_cfg.get("swd_cdf_sample_chunk_size", 256))
    print(
        "[INFO] SWD config "
        f"mode={swd_distance_mode}, bins={swd_cdf_num_bins}, tau={swd_cdf_tau}, "
        f"sample={swd_cdf_sample_size}, sample_chunk={swd_cdf_sample_chunk_size}"
    )

    pools: Dict[str, torch.Tensor] = {}
    pixel_pools: Dict[str, torch.Tensor] = {}
    load_times_ms: Dict[str, float] = {}
    style_iter = _iter_progress(style_subdirs, total=len(style_subdirs), desc="load domains")
    for name in style_iter:
        t0 = time.perf_counter()
        pools[name] = _load_domain_latents(
            data_root=data_root,
            subdir=name,
            max_samples=int(args.max_samples_per_domain),
            device=data_device,
            dtype=dtype,
        )
        pixel_pools[name] = _load_domain_images(
            image_root=image_root,
            subdir=name,
            max_samples=int(args.max_samples_per_domain),
            image_size=int(args.pixel_size),
            device=data_device,
            dtype=dtype,
        )
        load_times_ms[name] = (time.perf_counter() - t0) * 1000.0

    common_n = min(
        min(int(v.shape[0]) for v in pools.values()),
        min(int(v.shape[0]) for v in pixel_pools.values()),
    )
    if common_n < 2:
        raise ValueError("Need at least 2 samples per domain.")
    for name in style_subdirs:
        pools[name] = pools[name][:common_n].contiguous()
        pixel_pools[name] = pixel_pools[name][:common_n].contiguous()

    x_latent_all, labels = _stack_domains(pools, style_subdirs)
    x_pixel_all, pixel_labels = _stack_domains(pixel_pools, style_subdirs)
    print("[INFO] Using raw latent/pixel tensors (no latent preprocessing).")
    label_names = style_subdirs[:]
    np.save(out_dir / "labels.npy", labels)

    projection_bank = _build_projection_bank(
        channels=int(x_latent_all.shape[1]),
        patch_sizes=patch_sizes,
        num_projections=int(args.num_projections),
        device=data_device,
        dtype=dtype,
        seed=int(args.seed),
    )

    vgg_blocks = _build_vgg19_style_blocks(device=compute_device)
    x_vgg_features = _extract_vgg_features(
        x_pixel_all,
        blocks=vgg_blocks,
        image_size=int(args.vgg_image_size),
        batch_size=max(1, int(args.vgg_batch_size)),
    )

    pair_chunk = max(1, int(args.pair_batch_size))
    timed_results: Dict[str, np.ndarray] = {}
    compute_times_ms: Dict[str, float] = {}
    mode_order = [
        "latent_mse",
        "pixel_mse",
        "pixel_vgg_gram",
        "latent_gram",
        "latent_swd",
    ]

    for mode_name in _iter_progress(mode_order, total=len(mode_order), desc="metric modes"):
        t0 = time.perf_counter()
        print(f"[INFO] Computing: {mode_name}...")
        if mode_name == "latent_mse":
            matrix = _pairwise_mse_distance(
                x_latent_all,
                x_latent_all,
                chunk=pair_chunk,
                compute_device=compute_device,
            )
        elif mode_name == "pixel_mse":
            matrix = _pairwise_mse_distance(
                x_pixel_all,
                x_pixel_all,
                chunk=pair_chunk,
                compute_device=compute_device,
            )
        elif mode_name == "pixel_vgg_gram":
            matrix = torch.cdist(
                x_vgg_features.to(device=compute_device, dtype=torch.float32, non_blocking=False),
                x_vgg_features.to(device=compute_device, dtype=torch.float32, non_blocking=False),
                p=2.0,
            ).cpu().numpy()
        elif mode_name == "latent_gram":
            gram = _gram_features(x_latent_all)
            matrix = _pairwise_gram_distance(gram, gram, chunk=pair_chunk).cpu().numpy()
        elif mode_name == "latent_swd":
            matrix = _pairwise_projected_swd_streaming(
                x_latent_all,
                x_latent_all,
                projection_bank,
                patch_sizes=patch_sizes,
                chunk=max(1, int(args.swd_row_chunk)),
                compute_device=compute_device,
                projection_chunk=int(args.swd_projection_chunk),
                distance_mode=swd_distance_mode,
                cdf_num_bins=swd_cdf_num_bins,
                cdf_tau=swd_cdf_tau,
                cdf_sample_size=swd_cdf_sample_size,
                cdf_sample_chunk_size=swd_cdf_sample_chunk_size,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown mode: {mode_name}")
        timed_results[mode_name] = matrix
        compute_times_ms[mode_name] = (time.perf_counter() - t0) * 1000.0
        print(f"[INFO] ... done in {compute_times_ms[mode_name]:.1f}ms")
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows: List[Dict[str, Any]] = []
    for mode_name, matrix in timed_results.items():
        matrix_labels = pixel_labels if mode_name.startswith("pixel_") else labels
        stats = _summarize_distances(matrix, matrix_labels)
        stats["silhouette"] = _silhouette_score_precomputed(matrix, matrix_labels)
        stats["mode"] = mode_name
        stats["elapsed_ms"] = float(compute_times_ms.get(mode_name, 0.0))
        summary_rows.append(stats)

        coords = _classical_mds(matrix, n_components=2)
        _save_scatter_svg(
            image_out_dir / f"{mode_name}_mds.svg",
            coords,
            matrix_labels,
            label_names,
            title=f"MDS Scatter: {mode_name}",
        )

    summary_rows.sort(key=lambda row: float(row["snr"]), reverse=True)
    with open(out_dir / "snr_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "snr",
                "silhouette",
                "mean_intra",
                "mean_inter",
                "median_intra",
                "median_inter",
                "num_intra_pairs",
                "num_inter_pairs",
                "elapsed_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    (out_dir / "snr_summary.json").write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "compute_times_ms": compute_times_ms,
                "load_times_ms": load_times_ms,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("[DONE] Outputs written to:")
    print(f"  {out_dir}")
    print(f"  {image_out_dir}  (SVG scatter only)")
    for row in summary_rows:
        print(
            f"  - {row['mode']}: SNR={float(row['snr']):.4f} | "
            f"sil={float(row['silhouette']):.4f} | "
            f"inter={float(row['mean_inter']):.4f} | intra={float(row['mean_intra']):.4f} | "
            f"time={float(row['elapsed_ms']):.1f}ms"
        )


if __name__ == "__main__":
    main()
