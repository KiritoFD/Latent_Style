import argparse
import csv
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ParametricExtractor(nn.Module):
    def __init__(self, in_channels: int = 4, config: dict | None = None):
        super().__init__()
        self.cfg = config or {}

        dim = int(self.cfg.get("dim", 64))
        kernel = int(self.cfg.get("kernel", 1))
        padding = kernel // 2

        self.projector = nn.Conv2d(in_channels, dim, kernel_size=kernel, padding=padding, bias=False)
        init_mode = str(self.cfg.get("init", "orthogonal"))
        if init_mode == "kaiming":
            nn.init.kaiming_normal_(self.projector.weight, mode="fan_out", nonlinearity="relu")
        else:
            nn.init.orthogonal_(self.projector.weight)

        self.projector.eval()
        for p in self.projector.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * float(self.cfg.get("scale_factor", 0.13025))
        h = self.projector(x)

        act = str(self.cfg.get("act", "relu"))
        if act == "relu":
            h = F.relu(h)
        elif act == "leaky":
            h = F.leaky_relu(h, 0.2)
        elif act == "gelu":
            h = F.gelu(h)
        elif act == "silu":
            h = F.silu(h)
        elif act == "tanh":
            h = torch.tanh(h)

        norm = str(self.cfg.get("norm", "instance"))
        if norm == "instance":
            h = F.instance_norm(h)
        elif norm == "layer":
            h = F.layer_norm(h, h.shape[1:])
        elif norm == "group":
            groups = int(self.cfg.get("group_norm_groups", 4))
            groups = max(1, min(groups, h.shape[1]))
            while h.shape[1] % groups != 0 and groups > 1:
                groups -= 1
            h = F.group_norm(h, num_groups=groups)

        feature_mode = str(self.cfg.get("feature_mode", "diff"))
        scales = self.cfg.get("scales", [1, 2])
        grams = []

        if feature_mode in {"raw+diff", "raw"}:
            grams.append(self._gram(h))

        if feature_mode in {"raw+diff", "diff"}:
            for s in scales:
                s = int(s)
                if s >= h.shape[-1] or s >= h.shape[-2]:
                    continue
                dx = h[:, :, :, s:] - h[:, :, :, :-s]
                dy = h[:, :, s:, :] - h[:, :, :-s, :]
                grams.append(self._gram(dx))
                grams.append(self._gram(dy))

        if not grams:
            return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        return torch.cat(grams, dim=1)

    def _gram(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = x.reshape(b, c, -1)

        if bool(self.cfg.get("gram_center", False)):
            f = f - f.mean(dim=2, keepdim=True)

        g = torch.bmm(f, f.transpose(1, 2))
        gram_norm = str(self.cfg.get("gram_norm", "hw"))
        if gram_norm == "chw":
            g = g / float(max(c * h * w, 1))
        else:
            g = g / float(max(h * w, 1))

        idx = torch.triu_indices(c, c, device=x.device)
        return g[:, idx[0], idx[1]]


def parse_styles(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def load_latents(
    data_root: Path,
    styles: list[str],
    device: torch.device,
    max_per_style: int,
    latent_dtype: torch.dtype,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, list[str]]:
    all_latents = []
    all_labels = []
    used_styles = []

    print(f"Loading latents from: {data_root}")
    for style in styles:
        files = list((data_root / style).glob("*.pt"))
        if max_per_style > 0:
            files = files[:max_per_style]
        if not files:
            print(f"  - skip {style}: no files")
            continue

        used_styles.append(style)
        print(f"  - {style}: {len(files)}")
        chunk_size = 200
        for i in range(0, len(files), chunk_size):
            chunk = files[i : i + chunk_size]
            lats = torch.stack([torch.load(f, map_location=device, weights_only=True) for f in chunk]).to(dtype=latent_dtype)
            all_latents.append(lats)
            all_labels.extend([style] * len(chunk))

    if not all_latents:
        raise RuntimeError("No latent data loaded; check --data-root and style folders.")

    X_raw = torch.cat(all_latents, dim=0)
    labels_np = np.array(all_labels)

    style_to_id = {s: i for i, s in enumerate(used_styles)}
    labels_int = torch.tensor([style_to_id[s] for s in all_labels], device=device, dtype=torch.long)

    print(f"Loaded tensor: {tuple(X_raw.shape)} dtype={X_raw.dtype}")
    return X_raw, labels_np, labels_int, used_styles


def extract_features_gpu(extractor: nn.Module, X_raw: torch.Tensor, batch_size: int, out_dtype: torch.dtype) -> torch.Tensor:
    feats = []
    with torch.no_grad():
        for i in range(0, len(X_raw), batch_size):
            batch = X_raw[i : i + batch_size]
            f = extractor(batch).to(out_dtype)
            feats.append(f)
    return torch.cat(feats, dim=0)


def standardize_gpu(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(eps)
    return (X - mean) / std


def silhouette_score_torch(
    X_norm: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    chunk: int = 256,
) -> float:
    # X_norm: [N, D], labels: [N]
    N = X_norm.shape[0]
    if N < 2:
        return -1.0

    class_indices = [torch.nonzero(labels == c, as_tuple=False).flatten() for c in range(n_classes)]

    s_sum = X_norm.new_tensor(0.0)
    n_count = 0

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        xb = X_norm[start:end]  # [B, D]
        lb = labels[start:end]  # [B]
        B = xb.shape[0]

        dist = torch.cdist(xb, X_norm)  # [B, N]

        a = torch.zeros(B, device=X_norm.device, dtype=X_norm.dtype)
        b = torch.full((B,), float("inf"), device=X_norm.device, dtype=X_norm.dtype)

        for c, idx in enumerate(class_indices):
            if idx.numel() == 0:
                continue
            d_c = dist.index_select(dim=1, index=idx).mean(dim=1)  # [B]

            same = lb == c
            if same.any():
                count_c = idx.numel()
                if count_c > 1:
                    # exclude self (distance 0) for same-class mean
                    d_self = d_c[same] * (count_c / (count_c - 1.0))
                    a[same] = d_self
                else:
                    a[same] = 0.0

            b = torch.where((lb != c) & (d_c < b), d_c, b)

        denom = torch.maximum(a, b).clamp_min(1e-8)
        s = (b - a) / denom
        s_sum += s.sum()
        n_count += B

    return float((s_sum / max(n_count, 1)).item())


def maybe_subsample(
    X: torch.Tensor,
    labels_np: np.ndarray,
    labels_int: torch.Tensor,
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    if max_samples <= 0 or X.shape[0] <= max_samples:
        return X, labels_np, labels_int

    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g, device=X.device)[:max_samples]

    X_sub = X.index_select(0, idx)
    labels_int_sub = labels_int.index_select(0, idx)
    labels_np_sub = labels_np[idx.cpu().numpy()]
    return X_sub, labels_np_sub, labels_int_sub


def plot_best_pca_torch(
    X_feats: torch.Tensor,
    labels_np: np.ndarray,
    styles: list[str],
    score: float,
    cfg: dict,
    out_path: Path,
) -> None:
    X_norm = standardize_gpu(X_feats.float())
    X_centered = X_norm - X_norm.mean(dim=0, keepdim=True)

    # torch PCA on GPU
    _, _, V = torch.pca_lowrank(X_centered, q=2)
    pca_2d = (X_centered @ V[:, :2]).detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    for style in styles:
        mask = labels_np == style
        if mask.any():
            plt.scatter(pca_2d[mask, 0], pca_2d[mask, 1], s=14, alpha=0.65, label=style)

    title_cfg = (
        f"dim={cfg['dim']} k={cfg['kernel']} act={cfg['act']} norm={cfg['norm']} "
        f"mode={cfg['feature_mode']} scales={cfg['scales']} center={cfg['gram_center']} normg={cfg['gram_norm']}"
    )
    plt.title(f"Best Config PCA | silhouette={score:.5f}\\n{title_cfg}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_search_space() -> dict:
    return {
        # Mini-search: focused low-rank sweep.
        "dim": [16, 24, 32,48,64],
        "kernel": [1],
        "act": ["relu"],
        "norm": ["instance", "layer", "group"],
        "feature_mode": ["diff"],
        "scales": [[1, 2],[1],[1,3]],
        "gram_center": [False, True],
        "gram_norm": ["hw"],
        "init": ["orthogonal"],
        "scale_factor": [0.13025],
        "group_norm_groups": [2,4,8],
    }


def run_full_grid_search(args: argparse.Namespace) -> None:
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from sklearn.metrics import silhouette_score

    def _silhouette_worker(x_norm_np: np.ndarray, labels_np_sub: np.ndarray) -> float:
        try:
            return float(silhouette_score(x_norm_np, labels_np_sub))
        except Exception:
            return -1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    styles = parse_styles(args.styles)

    latent_dtype = torch.float16 if args.latents_fp16 and device.type == "cuda" else torch.float32
    X_raw, labels_np, labels_int, used_styles = load_latents(args.data_root, styles, device, args.max_per_style, latent_dtype)

    search_space = build_search_space()
    keys = list(search_space.keys())
    combinations = list(itertools.product(*[search_space[k] for k in keys]))

    print(f"\\nSearching {len(combinations)} configs on {device} ...")
    print("=" * 120)
    print(f"{'score':<10} | {'dim':<4} | {'k':<1} | {'act':<6} | {'norm':<8} | {'mode':<8} | {'scales':<10} | {'center':<6} | {'gnorm':<4}")
    print("-" * 120)

    results = []
    best = None
    pending: dict = {}
    max_pending = max(1, int(args.num_workers))

    def _flush(block: bool) -> None:
        nonlocal best
        if not pending:
            return
        futures = list(pending.keys())
        if block:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
        else:
            done, _ = wait(futures, timeout=0.0, return_when=FIRST_COMPLETED)
        for fut in done:
            cfg = pending.pop(fut)
            try:
                score = float(fut.result())
            except Exception:
                score = -1.0

            results.append({"score": score, "cfg": cfg})
            if best is None or score > best["score"]:
                best = {"score": score, "cfg": cfg}

            tqdm.write(
                f"{score: .5f} | {cfg['dim']:<4} | {cfg['kernel']} | {cfg['act']:<6} | {cfg['norm']:<8} | "
                f"{cfg['feature_mode']:<8} | {str(cfg['scales']):<10} | {str(cfg['gram_center']):<6} | {cfg['gram_norm']:<4}"
            )

    with ThreadPoolExecutor(max_workers=max_pending) as pool:
        for combo in tqdm(combinations, desc="Grid Search"):
            cfg = dict(zip(keys, combo))
            extractor = ParametricExtractor(in_channels=4, config=cfg).to(device)

            X_feats = None
            X_norm = None
            X_norm_np = None
            try:
                # GPU heavy path.
                X_feats = extract_features_gpu(extractor, X_raw, args.batch_size, out_dtype=torch.float32)
                X_feats, labels_np_sub, _ = maybe_subsample(
                    X_feats,
                    labels_np,
                    labels_int,
                    max_samples=args.silhouette_samples,
                    seed=args.seed,
                )
                X_norm = standardize_gpu(X_feats)
                # CPU-parallel scoring path.
                X_norm_np = X_norm.detach().cpu().numpy()
                fut = pool.submit(_silhouette_worker, X_norm_np, labels_np_sub)
                pending[fut] = cfg
            except Exception:
                results.append({"score": -1.0, "cfg": cfg})
                if best is None or -1.0 > best["score"]:
                    best = {"score": -1.0, "cfg": cfg}
                tqdm.write(
                    f"{-1.0: .5f} | {cfg['dim']:<4} | {cfg['kernel']} | {cfg['act']:<6} | {cfg['norm']:<8} | "
                    f"{cfg['feature_mode']:<8} | {str(cfg['scales']):<10} | {str(cfg['gram_center']):<6} | {cfg['gram_norm']:<4}"
                )
            finally:
                del extractor
                if X_feats is not None:
                    del X_feats
                if X_norm is not None:
                    del X_norm
                if X_norm_np is not None:
                    del X_norm_np
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if len(pending) >= max_pending:
                _flush(block=True)
            else:
                _flush(block=False)

        while pending:
            _flush(block=True)

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\\n" + "=" * 120)
    print(f"TOP {args.top_k} CONFIGURATIONS")
    print("=" * 120)
    for i, r in enumerate(results[: args.top_k], start=1):
        print(f"Rank {i:02d}: score={r['score']:.5f} cfg={r['cfg']}")

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "score",
            "dim",
            "kernel",
            "act",
            "norm",
            "feature_mode",
            "scales",
            "gram_center",
            "gram_norm",
            "init",
            "scale_factor",
        ])
        for i, r in enumerate(results, start=1):
            c = r["cfg"]
            writer.writerow([
                i,
                r["score"],
                c["dim"],
                c["kernel"],
                c["act"],
                c["norm"],
                c["feature_mode"],
                str(c["scales"]),
                c["gram_center"],
                c["gram_norm"],
                c["init"],
                c["scale_factor"],
            ])
    print(f"Saved full ranking CSV: {args.out_csv}")

    if best is not None:
        best_extractor = ParametricExtractor(in_channels=4, config=best["cfg"]).to(device)
        X_feats = extract_features_gpu(best_extractor, X_raw, args.batch_size, out_dtype=torch.float32)
        X_feats, labels_np_sub, _ = maybe_subsample(
            X_feats,
            labels_np,
            labels_int,
            max_samples=args.pca_samples,
            seed=args.seed,
        )
        plot_best_pca_torch(X_feats, labels_np_sub, used_styles, best["score"], best["cfg"], args.out_pca)
        print(f"Saved best-config PCA: {args.out_pca}")
        print(f"Best score={best['score']:.5f}, cfg={best['cfg']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Style separability grid search with best-config PCA (GPU-heavy mode).")
    p.add_argument("--data-root", type=Path, default=Path("../sdxl-256"))
    p.add_argument("--styles", type=str, default="photo,Hayao,vangogh,monet,cezanne")
    p.add_argument("--max-per-style", type=int, default=0, help="0 means all")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4, help="parallel silhouette workers (recommended 4)")
    p.add_argument("--silhouette-samples", type=int, default=2500, help="subsample size for silhouette to control O(N^2)")
    p.add_argument("--silhouette-chunk", type=int, default=256, help="cdist chunk size")
    p.add_argument("--pca-samples", type=int, default=5000, help="subsample size for PCA plot")
    p.add_argument("--latents-fp16", action="store_true", help="store loaded latents in fp16 to reduce VRAM")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out-csv", type=Path, default=Path("grid_search_results.csv"))
    p.add_argument("--out-pca", type=Path, default=Path("best_config_pca.png"))
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_full_grid_search(args)
