import argparse
import csv
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, calinski_harabasz_score, davies_bouldin_score, f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
    content_id_mode: str,
    content_id_sep: str,
    content_id_regex: str,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray, list[str]]:
    all_latents = []
    all_labels = []
    all_content_ids = []
    used_styles = []
    compiled_regex = re.compile(content_id_regex) if content_id_mode == "regex" and content_id_regex else None

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
            for f in chunk:
                stem = f.stem
                if content_id_mode == "none":
                    content_id = f"{style}::{stem}"
                elif content_id_mode == "stem":
                    content_id = stem
                elif content_id_mode == "prefix":
                    content_id = stem.split(content_id_sep)[0] if content_id_sep else stem
                elif content_id_mode == "regex":
                    if compiled_regex is None:
                        content_id = stem
                    else:
                        m = compiled_regex.search(stem)
                        if m is None:
                            content_id = stem
                        elif m.groups():
                            content_id = m.group(1)
                        else:
                            content_id = m.group(0)
                else:
                    content_id = stem
                all_content_ids.append(content_id)

    if not all_latents:
        raise RuntimeError("No latent data loaded; check --data-root and style folders.")

    X_raw = torch.cat(all_latents, dim=0)
    labels_np = np.array(all_labels)
    content_ids_np = np.array(all_content_ids)

    style_to_id = {s: i for i, s in enumerate(used_styles)}
    labels_int = torch.tensor([style_to_id[s] for s in all_labels], device=device, dtype=torch.long)

    print(f"Loaded tensor: {tuple(X_raw.shape)} dtype={X_raw.dtype}")
    if content_id_mode != "none":
        print(f"Detected content groups: {len(np.unique(content_ids_np))}")
    return X_raw, labels_np, labels_int, content_ids_np, used_styles


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
    groups_np: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    if max_samples <= 0 or X.shape[0] <= max_samples:
        return X, labels_np, labels_int, groups_np

    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g, device=X.device)[:max_samples]

    X_sub = X.index_select(0, idx)
    labels_int_sub = labels_int.index_select(0, idx)
    idx_np = idx.cpu().numpy()
    labels_np_sub = labels_np[idx_np]
    groups_np_sub = groups_np[idx_np]
    return X_sub, labels_np_sub, labels_int_sub, groups_np_sub


def parse_int_list(raw: str) -> list[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError(f"Invalid int list: {raw}")
    return vals


def fisher_ratio(X: np.ndarray, y: np.ndarray, ridge: float = 1e-4) -> tuple[float, float]:
    classes = np.unique(y)
    if len(classes) < 2:
        return -1.0, -1.0

    mu = X.mean(axis=0)
    sw = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    sb = np.zeros_like(sw)
    trace_sw = 0.0
    trace_sb = 0.0

    for cls in classes:
        Xk = X[y == cls]
        if Xk.shape[0] == 0:
            continue
        muk = Xk.mean(axis=0)
        centered = Xk - muk
        sw += centered.T @ centered
        trace_sw += float(np.trace(centered.T @ centered))
        diff = (muk - mu).reshape(-1, 1)
        sb += Xk.shape[0] * (diff @ diff.T)
        trace_sb += float(Xk.shape[0] * np.dot((muk - mu), (muk - mu)))

    if trace_sw <= 1e-12:
        j = -1.0
    else:
        j = trace_sb / trace_sw

    sw_reg = sw + ridge * np.eye(sw.shape[0], dtype=sw.dtype)
    try:
        j_star = float(np.trace(np.linalg.solve(sw_reg, sb)))
    except np.linalg.LinAlgError:
        j_star = -1.0
    return float(j), float(j_star)


def evaluate_cv_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    groups: np.ndarray | None,
    classifier: str,
    knn_k: int = 5,
) -> dict:
    acc_scores: list[float] = []
    f1_scores: list[float] = []
    classes = np.unique(y)

    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    else:
        n_group = len(np.unique(groups))
        split_n = min(n_splits, n_group)
        if split_n < 2:
            return {"acc": -1.0, "macro_f1": -1.0}

        try:
            from sklearn.model_selection import StratifiedGroupKFold

            splitter = StratifiedGroupKFold(n_splits=split_n, shuffle=True, random_state=seed)
            split_iter = splitter.split(X, y, groups=groups)
        except Exception:
            splitter = GroupKFold(n_splits=split_n)
            split_iter = splitter.split(X, y, groups=groups)

    for tr, te in split_iter:
        y_tr = y[tr]
        y_te = y[te]

        if len(np.unique(y_tr)) < 2:
            continue

        if classifier == "logreg":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
        elif classifier == "knn":
            clf = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=knn_k),
            )
        else:
            raise ValueError(f"Unsupported classifier={classifier}")

        clf.fit(X[tr], y_tr)
        pred = clf.predict(X[te])
        acc_scores.append(accuracy_score(y_te, pred))
        f1_scores.append(f1_score(y_te, pred, labels=classes, average="macro", zero_division=0))

    if not acc_scores:
        return {"acc": -1.0, "macro_f1": -1.0}
    return {"acc": float(np.mean(acc_scores)), "macro_f1": float(np.mean(f1_scores))}


def evaluate_distance_stats(X: np.ndarray, y: np.ndarray, max_pairs: int, seed: int) -> dict:
    n = X.shape[0]
    if n < 2:
        return {"dist_auc_same_vs_diff": -1.0, "inter_minus_intra": -1.0}

    rng = np.random.default_rng(seed)
    i_idx = rng.integers(0, n, size=max_pairs)
    j_idx = rng.integers(0, n, size=max_pairs)
    neq = i_idx != j_idx
    i_idx = i_idx[neq]
    j_idx = j_idx[neq]
    if i_idx.size == 0:
        return {"dist_auc_same_vs_diff": -1.0, "inter_minus_intra": -1.0}

    diff = X[i_idx] - X[j_idx]
    d = np.sqrt((diff * diff).sum(axis=1))
    same = (y[i_idx] == y[j_idx]).astype(np.int32)
    if same.min() == same.max():
        auc = -1.0
    else:
        auc = float(roc_auc_score(same, -d))

    intra = d[same == 1]
    inter = d[same == 0]
    if intra.size == 0 or inter.size == 0:
        margin = -1.0
    else:
        margin = float(inter.mean() - intra.mean())
    return {"dist_auc_same_vs_diff": auc, "inter_minus_intra": margin}


def evaluate_cluster_metrics(X: np.ndarray, y: np.ndarray) -> dict:
    uniq = np.unique(y)
    if len(uniq) < 2 or X.shape[0] < len(uniq):
        return {"silhouette": -1.0, "davies_bouldin": -1.0, "calinski_harabasz": -1.0}
    try:
        sil = float(silhouette_score(X, y))
    except Exception:
        sil = -1.0
    try:
        db = float(davies_bouldin_score(X, y))
    except Exception:
        db = -1.0
    try:
        ch = float(calinski_harabasz_score(X, y))
    except Exception:
        ch = -1.0
    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}


def evaluate_separability_metrics(
    X: np.ndarray,
    labels_np: np.ndarray,
    groups_np: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    out = {}
    out["probe_random"] = evaluate_cv_classifier(X, labels_np, args.cv_folds, args.seed, groups=None, classifier="logreg")

    if args.enable_group_eval and args.content_id_mode != "none":
        out["probe_group"] = evaluate_cv_classifier(
            X,
            labels_np,
            args.cv_folds,
            args.seed,
            groups=groups_np,
            classifier="logreg",
        )
    else:
        out["probe_group"] = {"acc": -1.0, "macro_f1": -1.0}

    out["knn"] = {}
    for k in parse_int_list(args.knn_k_list):
        out["knn"][k] = {
            "random": evaluate_cv_classifier(X, labels_np, args.cv_folds, args.seed, groups=None, classifier="knn", knn_k=k),
            "group": evaluate_cv_classifier(X, labels_np, args.cv_folds, args.seed, groups=groups_np if args.enable_group_eval and args.content_id_mode != "none" else None, classifier="knn", knn_k=k),
        }

    j, j_star = fisher_ratio(X, labels_np, ridge=args.fisher_ridge)
    out["fisher_j"] = j
    out["fisher_j_star"] = j_star
    out.update(evaluate_cluster_metrics(X, labels_np))
    out.update(evaluate_distance_stats(X, labels_np, max_pairs=args.max_dist_pairs, seed=args.seed))
    return out


def pick_primary_score(metrics: dict, priority: str) -> float:
    if priority == "probe_group_macro_f1":
        return float(metrics["probe_group"]["macro_f1"])
    if priority == "probe_random_macro_f1":
        return float(metrics["probe_random"]["macro_f1"])
    if priority == "fisher_j":
        return float(metrics["fisher_j"])
    if priority == "distance_auc":
        return float(metrics["dist_auc_same_vs_diff"])
    if priority == "silhouette":
        return float(metrics["silhouette"])
    raise ValueError(f"Unsupported score-priority={priority}")


def plot_best_pca_torch(
    X_feats: torch.Tensor,
    labels_np: np.ndarray,
    styles: list[str],
    score: float,
    score_name: str,
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
    plt.title(f"Best Config PCA | {score_name}={score:.5f}\\n{title_cfg}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_search_space() -> dict:
    return {
        # Focused sweep around top-ranked region from previous run.
        "dim": [40, 48, 56, 64],
        "kernel": [1],
        "act": ["leaky"],
        "norm": ["group"],
        "feature_mode": ["diff"],
        "scales": [[1], [1, 2]],
        "gram_center": [False, True],
        "gram_norm": ["chw"],
        "init": ["orthogonal"],
        "scale_factor": [0.13025],
        "group_norm_groups": [1, 2, 4],
    }


def run_full_grid_search(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    styles = parse_styles(args.styles)

    latent_dtype = torch.float16 if args.latents_fp16 and device.type == "cuda" else torch.float32
    X_raw, labels_np, labels_int, groups_np, used_styles = load_latents(
        args.data_root,
        styles,
        device,
        args.max_per_style,
        latent_dtype,
        content_id_mode=args.content_id_mode,
        content_id_sep=args.content_id_sep,
        content_id_regex=args.content_id_regex,
    )
    if not args.enable_group_eval:
        print("Warning: --enable-group-eval is OFF, so probe_group_* will be -1.")
    elif args.content_id_mode == "none":
        print("Warning: --content-id-mode is 'none', so probe_group_* will be -1. Set stem/prefix/regex.")
    elif len(np.unique(groups_np)) < 2:
        print("Warning: fewer than 2 unique content groups detected; group probe is invalid.")

    search_space = build_search_space()
    keys = list(search_space.keys())
    combinations = list(itertools.product(*[search_space[k] for k in keys]))

    score_priority = args.score_priority
    if score_priority == "auto":
        if args.enable_group_eval and args.content_id_mode != "none":
            score_priority = "probe_group_macro_f1"
        else:
            score_priority = "probe_random_macro_f1"

    print(f"\\nSearching {len(combinations)} configs on {device} ...")
    print(f"Ranking metric: {score_priority}")
    print("=" * 150)
    print(f"{'score':<10} | {'probe_r_f1':<10} | {'probe_g_f1':<10} | {'fisher_j':<10} | {'dist_auc':<9} | {'dim':<4} | {'act':<6} | {'norm':<8} | {'scales':<10}")
    print("-" * 150)

    results = []
    best = None

    for combo in tqdm(combinations, desc="Grid Search"):
        cfg = dict(zip(keys, combo))
        if cfg["norm"] == "group" and (int(cfg["dim"]) % int(cfg["group_norm_groups"]) != 0):
            continue
        extractor = ParametricExtractor(in_channels=4, config=cfg).to(device)

        X_feats = None
        X_eval = None
        try:
            X_feats = extract_features_gpu(extractor, X_raw, args.batch_size, out_dtype=torch.float32)
            X_eval, labels_np_sub, _, groups_np_sub = maybe_subsample(
                X_feats,
                labels_np,
                labels_int,
                groups_np,
                max_samples=args.eval_samples,
                seed=args.seed,
            )
            X_eval_np = standardize_gpu(X_eval).detach().cpu().numpy()
            metrics = evaluate_separability_metrics(X_eval_np, labels_np_sub, groups_np_sub, args)
            score = pick_primary_score(metrics, score_priority)
        except Exception as e:
            metrics = {
                "probe_random": {"acc": -1.0, "macro_f1": -1.0},
                "probe_group": {"acc": -1.0, "macro_f1": -1.0},
                "fisher_j": -1.0,
                "fisher_j_star": -1.0,
                "silhouette": -1.0,
                "davies_bouldin": -1.0,
                "calinski_harabasz": -1.0,
                "dist_auc_same_vs_diff": -1.0,
                "inter_minus_intra": -1.0,
                "knn": {},
                "error": str(e),
            }
            score = -1.0

        result = {"score": score, "cfg": cfg, "metrics": metrics}
        results.append(result)
        if best is None or score > best["score"]:
            best = result

        tqdm.write(
            f"{score: .5f} | {metrics['probe_random']['macro_f1']: .5f} | {metrics['probe_group']['macro_f1']: .5f} | "
            f"{metrics['fisher_j']: .5f} | {metrics['dist_auc_same_vs_diff']: .5f} | "
            f"{cfg['dim']:<4} | {cfg['act']:<6} | {cfg['norm']:<8} | {str(cfg['scales']):<10}"
        )

        del extractor
        if X_feats is not None:
            del X_feats
        if X_eval is not None:
            del X_eval
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
            "probe_random_acc",
            "probe_random_macro_f1",
            "probe_group_acc",
            "probe_group_macro_f1",
            "fisher_j",
            "fisher_j_star",
            "distance_auc_same_vs_diff",
            "distance_inter_minus_intra",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
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
            m = r["metrics"]
            writer.writerow([
                i,
                r["score"],
                m["probe_random"]["acc"],
                m["probe_random"]["macro_f1"],
                m["probe_group"]["acc"],
                m["probe_group"]["macro_f1"],
                m["fisher_j"],
                m["fisher_j_star"],
                m["dist_auc_same_vs_diff"],
                m["inter_minus_intra"],
                m["silhouette"],
                m["davies_bouldin"],
                m["calinski_harabasz"],
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
        X_feats, labels_np_sub, _, _ = maybe_subsample(
            X_feats,
            labels_np,
            labels_int,
            groups_np,
            max_samples=args.pca_samples,
            seed=args.seed,
        )
        plot_best_pca_torch(X_feats, labels_np_sub, used_styles, best["score"], score_priority, best["cfg"], args.out_pca)
        print(f"Saved best-config PCA: {args.out_pca}")
        print(f"Best score={best['score']:.5f}, cfg={best['cfg']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Style separability grid search with linear probe/group-split centric metrics.")
    p.add_argument("--data-root", type=Path, default=Path("../sdxl-fp32"))
    p.add_argument("--styles", type=str, default="photo,Hayao,vangogh,monet,cezanne")
    p.add_argument("--max-per-style", type=int, default=0, help="0 means all")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-samples", type=int, default=2500, help="subsample size for evaluation metrics")
    p.add_argument("--pca-samples", type=int, default=5000, help="subsample size for PCA plot")
    p.add_argument("--cv-folds", type=int, default=5, help="folds for random/group cross-validation")
    p.add_argument("--enable-group-eval", action="store_true", help="enable GroupKFold evaluation")
    p.add_argument("--content-id-mode", type=str, default="none", choices=["none", "stem", "prefix", "regex"])
    p.add_argument("--content-id-sep", type=str, default="_", help="separator for content-id-mode=prefix")
    p.add_argument("--content-id-regex", type=str, default="", help="regex for content-id-mode=regex; first capture group used if present")
    p.add_argument("--score-priority", type=str, default="auto", choices=["auto", "probe_group_macro_f1", "probe_random_macro_f1", "fisher_j", "distance_auc", "silhouette"])
    p.add_argument("--knn-k-list", type=str, default="1,5,10")
    p.add_argument("--fisher-ridge", type=float, default=1e-4)
    p.add_argument("--max-dist-pairs", type=int, default=20000)
    p.add_argument("--latents-fp16", action="store_true", help="store loaded latents in fp16 to reduce VRAM")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out-csv", type=Path, default=Path("grid_search_results.csv"))
    p.add_argument("--out-pca", type=Path, default=Path("best_config_pca-fp32.png"))
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_full_grid_search(args)

