#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.model import LatentAdaCUT, build_model_from_config


def list_style_files(data_root: str, exts: Tuple[str, ...] = (".pt", ".pth")) -> Dict[str, List[str]]:
    styles: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(data_root)):
        d = os.path.join(data_root, name)
        if not os.path.isdir(d):
            continue
        paths = [os.path.join(d, fn) for fn in sorted(os.listdir(d)) if fn.lower().endswith(exts)]
        if paths:
            styles[name] = paths
    if not styles:
        raise FileNotFoundError(f"No style subfolders with {exts} found under: {data_root}")
    return styles


def ensure_chw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expect [C,H,W], got shape={tuple(x.shape)}")
    return x


def load_latent(path: str, key: Optional[str] = None, map_location: str = "cpu") -> torch.Tensor:
    obj = torch.load(path, map_location=map_location)
    if torch.is_tensor(obj):
        return ensure_chw(obj).float()
    if isinstance(obj, dict):
        if key is not None:
            if key not in obj:
                raise KeyError(f"Key '{key}' not found in {path}")
            v = obj[key]
            if not torch.is_tensor(v):
                raise TypeError(f"obj['{key}'] is not a Tensor: {type(v)}")
            return ensure_chw(v).float()
        for v in obj.values():
            if torch.is_tensor(v):
                return ensure_chw(v).float()
        for v in obj.values():
            if isinstance(v, dict):
                for vv in v.values():
                    if torch.is_tensor(vv):
                        return ensure_chw(vv).float()
        raise TypeError(f"No Tensor found in dict: {path}")
    raise TypeError(f"Unsupported .pt content type: {type(obj)}")


def make_projections(dim: int, nproj: int, seed: int, device: torch.device, orth: bool = True) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    W = torch.randn(dim, nproj, generator=g, dtype=torch.float32)
    if orth and nproj <= dim:
        Q, _ = torch.linalg.qr(W, mode="reduced")
        W = Q[:, :nproj]
    W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
    return W.to(device)


def extract_patches(
    x_bchw: torch.Tensor,
    patch: int,
    stride: int,
    padding_mode: str,
    max_patches: int,
    seed: int,
) -> torch.Tensor:
    if padding_mode == "same":
        pad = patch // 2
    elif padding_mode == "valid":
        pad = 0
    else:
        raise ValueError(f"Unsupported padding_mode={padding_mode}")

    unfolded = F.unfold(x_bchw, kernel_size=patch, stride=stride, padding=pad)
    b, d, l = unfolded.shape
    pts = unfolded.transpose(1, 2).reshape(b * l, d).contiguous()

    if max_patches is not None and pts.shape[0] > max_patches:
        g = torch.Generator(device=pts.device)
        g.manual_seed(seed)
        idx = torch.randperm(pts.shape[0], generator=g, device=pts.device)[:max_patches]
        pts = pts[idx]

    m = pts.mean(dim=1, keepdim=True)
    v = pts.var(dim=1, keepdim=True, unbiased=False)
    pts = (pts - m) / (v.sqrt() + 1e-6)
    return pts


@torch.no_grad()
def swd_from_patches(Pa: torch.Tensor, Pb: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    dev = proj.device
    if Pa.device != dev:
        Pa = Pa.to(dev)
    if Pb.device != dev:
        Pb = Pb.to(dev)
    xa = Pa @ proj
    xb = Pb @ proj
    xa, _ = torch.sort(xa, dim=0)
    xb, _ = torch.sort(xb, dim=0)
    return (xa - xb).abs().mean()


def save_heatmap(mat: np.ndarray, labels: List[str], out_path: str, title: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, aspect="equal")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_gap(gap: np.ndarray, labels: List[str], out_path: str, title: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 3))
    plt.bar(range(len(labels)), gap)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def register_hooks(model: LatentAdaCUT):
    feats: Dict[str, torch.Tensor] = {}
    handles = []

    def save(name: str):
        def _hook(_m, _inp, out):
            feats[name] = out
        return _hook

    handles.append(model.enc_in_act.register_forward_hook(save("enc_in_act")))
    for i, blk in enumerate(model.hires_body):
        handles.append(blk.register_forward_hook(save(f"hires_{i}")))
    handles.append(model.down.register_forward_hook(save("down")))
    for i, blk in enumerate(model.body):
        handles.append(blk.register_forward_hook(save(f"body_{i}")))
    handles.append(model.dec_up.register_forward_hook(save("dec_up")))
    handles.append(model.dec_conv.register_forward_hook(save("dec_conv")))
    handles.append(model.dec_norm.register_forward_hook(save("dec_norm")))
    handles.append(model.dec_act.register_forward_hook(save("dec_act")))
    return feats, handles


def extract_state_dict(blob):
    if isinstance(blob, dict):
        for k in ["state_dict", "model", "ema", "net"]:
            v = blob.get(k, None)
            if isinstance(v, dict):
                return v
    return blob


def load_ckpt_robust(model: torch.nn.Module, ckpt_path: str):
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = extract_state_dict(raw)
    if not isinstance(sd, dict):
        raise TypeError(f"Unsupported checkpoint format at {ckpt_path}")

    cleaned = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    model_sd = model.state_dict()
    filtered = {}
    skipped_shape = []
    for k, v in cleaned.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        elif k in model_sd:
            skipped_shape.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "missing": missing,
        "unexpected": unexpected,
        "skipped_shape": skipped_shape,
    }


def merge_cap_pool(pool: Optional[torch.Tensor], pts: torch.Tensor, cap: int, seed: int) -> torch.Tensor:
    if pool is None:
        x = pts
    else:
        x = torch.cat([pool, pts], dim=0)
    if cap is not None and x.shape[0] > cap:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=g)[:cap]
        x = x[idx]
    return x


@torch.no_grad()
def score_from_style_pools(
    pools: List[Optional[torch.Tensor]],
    pools_a: List[Optional[torch.Tensor]],
    pools_b: List[Optional[torch.Tensor]],
    n_styles: int,
    device: torch.device,
    num_projections: int,
    seed: int,
    cache_device: str,
):
    def finalize_pool(pool: Optional[torch.Tensor], sid: int) -> torch.Tensor:
        if pool is None or pool.numel() == 0:
            raise RuntimeError(f"No feature patches collected for style_id={sid}")
        return pool.to(device) if cache_device == "cpu" else pool

    P = [finalize_pool(pools[s], s) for s in range(n_styles)]
    PA = [finalize_pool(pools_a[s] if pools_a[s] is not None else pools[s], s) for s in range(n_styles)]
    PB = [finalize_pool(pools_b[s] if pools_b[s] is not None else pools[s], s) for s in range(n_styles)]

    dim = int(P[0].shape[1])
    proj = make_projections(dim, num_projections, seed=seed, device=device, orth=True)

    W = np.zeros((n_styles,), dtype=np.float64)
    for s in range(n_styles):
        W[s] = float(swd_from_patches(PA[s], PB[s], proj).detach().cpu().item())

    B = np.zeros((n_styles, n_styles), dtype=np.float64)
    for i in range(n_styles):
        for j in range(i, n_styles):
            d = float(swd_from_patches(P[i], P[j], proj).detach().cpu().item())
            B[i, j] = d
            B[j, i] = d

    off = B[~np.eye(n_styles, dtype=bool)]
    R_mean = float(np.mean(off) / (np.mean(W) + 1e-12))
    R_worst = float(np.min(off) / (np.max(W) + 1e-12))
    gap = np.zeros((n_styles,), dtype=np.float64)
    for i in range(n_styles):
        nn = np.min([B[i, j] for j in range(n_styles) if j != i])
        gap[i] = float(nn - W[i])

    return {"W": W, "B": B, "R_mean": R_mean, "R_worst": R_worst, "gap": gap}


@torch.no_grad()
def build_pools_and_score(
    feats_iter: Iterator[Tuple[int, torch.Tensor]],
    n_styles: int,
    patch: int,
    device: torch.device,
    num_projections: int,
    max_patches_per_pool: int,
    max_patches_per_sample: int,
    padding_mode: str,
    cache_device: str,
    seed: int,
):
    rng = np.random.RandomState(seed)

    pools: List[Optional[torch.Tensor]] = [None for _ in range(n_styles)]
    pools_a: List[Optional[torch.Tensor]] = [None for _ in range(n_styles)]
    pools_b: List[Optional[torch.Tensor]] = [None for _ in range(n_styles)]

    for sid, feat in feats_iter:
        pts = extract_patches(
            feat,
            patch=patch,
            stride=1,
            padding_mode=padding_mode,
            max_patches=max_patches_per_sample,
            seed=seed * 1000003 + int(rng.randint(1_000_000_000)),
        )
        pts = pts.detach().cpu() if cache_device == "cpu" else pts.detach()
        merge_seed = seed * 1000003 + int(rng.randint(1_000_000_000))
        pools[sid] = merge_cap_pool(pools[sid], pts, max_patches_per_pool, merge_seed)
        if rng.rand() < 0.5:
            pools_a[sid] = merge_cap_pool(pools_a[sid], pts, max_patches_per_pool, merge_seed + 17)
        else:
            pools_b[sid] = merge_cap_pool(pools_b[sid], pts, max_patches_per_pool, merge_seed + 31)

    return score_from_style_pools(
        pools=pools,
        pools_a=pools_a,
        pools_b=pools_b,
        n_styles=n_styles,
        device=device,
        num_projections=num_projections,
        seed=seed,
        cache_device=cache_device,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="src/config.json")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--latent_key", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--allow_cpu", action="store_true")
    ap.add_argument("--max_per_style", type=int, default=1200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--style_condition", type=str, default="style_ref", choices=["style_ref", "style_id"])
    ap.add_argument("--style_strength", type=float, default=1.0)
    ap.add_argument("--step_size", type=float, default=1.0)

    ap.add_argument("--patch_sizes", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--padding_mode", type=str, default="valid", choices=["valid", "same"])
    ap.add_argument("--num_projections", type=int, default=256)
    ap.add_argument("--max_patches_per_pool", type=int, default=4000)
    ap.add_argument("--max_patches_per_sample", type=int, default=256)
    ap.add_argument("--cache_device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer_shard_count", type=int, default=1, help="split layers into N shards")
    ap.add_argument("--layer_shard_index", type=int, default=0, help="0-based shard index to run")
    ap.add_argument("--empty_cache_each_layer", action="store_true")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        if args.allow_cpu:
            args.device = "cpu"
        else:
            raise RuntimeError("CUDA unavailable. Use --allow_cpu to run on CPU.")
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    styles = list_style_files(args.data_root)
    style_names = sorted(styles.keys())
    n_styles = len(style_names)
    print(f"[INFO] device={device} cache_device={args.cache_device}")
    print(f"[INFO] styles({n_styles})={style_names}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_all = json.load(f)
    model_cfg = dict(cfg_all.get("model", {}))
    if args.style_condition == "style_id":
        model_cfg["num_styles"] = int(n_styles)
    model = build_model_from_config(model_cfg, use_checkpointing=False)

    if args.ckpt:
        info = load_ckpt_robust(model, args.ckpt)
        print(
            f"[CKPT] loaded={info['loaded']} "
            f"missing={len(info['missing'])} unexpected={len(info['unexpected'])} "
            f"skipped_shape={len(info['skipped_shape'])}"
        )
    else:
        print("[WARN] --ckpt not provided; random weights make scan less meaningful.")

    model = model.to(device).eval()
    hook_dict, handles = register_hooks(model)

    samples = []
    for sid, s in enumerate(style_names):
        paths = styles[s]
        if args.max_per_style is not None and len(paths) > args.max_per_style:
            paths = paths[: args.max_per_style]
        for p in paths:
            samples.append((sid, p))
    print(f"[INFO] total_samples={len(samples)}")

    def batch_iter():
        for i in range(0, len(samples), args.batch_size):
            batch = samples[i : i + args.batch_size]
            sids = [x[0] for x in batch]
            z_list = [load_latent(x[1], key=args.latent_key, map_location="cpu") for x in batch]
            z = torch.stack(z_list, dim=0).to(device)
            sid_t = torch.tensor(sids, dtype=torch.long, device=device)

            loss_feat = model.project_loss_features(z)
            style_feats = model.encode_style_feats(z)
            hook_dict.clear()
            if args.style_condition == "style_id":
                _ = model(z, style_id=sid_t, step_size=args.step_size, style_strength=args.style_strength)
            else:
                _ = model(z, style_ref=z, step_size=args.step_size, style_strength=args.style_strength)

            out = {
                "z_in": z,
                "loss_proj": loss_feat,
                "style_32": style_feats[0] if len(style_feats) > 0 else None,
                "style_16": style_feats[1] if len(style_feats) > 1 else None,
                "style_8": style_feats[2] if len(style_feats) > 2 else None,
            }
            for k, v in hook_dict.items():
                out[k] = v
            yield sid_t, out

    layer_names = [
        "z_in",
        "loss_proj",
        "style_32",
        "style_16",
        "style_8",
        "enc_in_act",
    ]
    layer_names += [f"hires_{i}" for i in range(len(model.hires_body))]
    layer_names += ["down"]
    layer_names += [f"body_{i}" for i in range(len(model.body))]
    layer_names += ["dec_up", "dec_conv", "dec_norm", "dec_act"]

    shard_n = max(1, int(args.layer_shard_count))
    shard_i = int(args.layer_shard_index)
    if shard_i < 0 or shard_i >= shard_n:
        raise ValueError(f"--layer_shard_index must be in [0, {shard_n - 1}]")
    layer_names = [ln for li, ln in enumerate(layer_names) if (li % shard_n) == shard_i]
    print(f"[INFO] layer shard {shard_i + 1}/{shard_n}, num_layers={len(layer_names)}")

    results = []
    for p in args.patch_sizes:
        print(f"[INFO] collecting pools for patch={p}")
        layer_pools: Dict[str, Dict[str, List[Optional[torch.Tensor]]]] = {}
        for lname in layer_names:
            layer_pools[lname] = {
                "p": [None for _ in range(n_styles)],
                "a": [None for _ in range(n_styles)],
                "b": [None for _ in range(n_styles)],
            }

        rng = np.random.RandomState(args.seed + p * 9973)
        for sid_t, out in batch_iter():
            for lname in layer_names:
                feat = out.get(lname, None)
                if feat is None:
                    continue
                h, w = feat.shape[-2], feat.shape[-1]
                if args.padding_mode == "valid" and p > min(h, w):
                    continue
                for b in range(feat.shape[0]):
                    sid = int(sid_t[b].item())
                    pts = extract_patches(
                        feat[b : b + 1],
                        patch=p,
                        stride=1,
                        padding_mode=args.padding_mode,
                        max_patches=args.max_patches_per_sample,
                        seed=args.seed * 1000003 + int(rng.randint(1_000_000_000)),
                    )
                    pts = pts.detach().cpu() if args.cache_device == "cpu" else pts.detach()
                    ms = args.seed * 1000003 + int(rng.randint(1_000_000_000))
                    layer_pools[lname]["p"][sid] = merge_cap_pool(
                        layer_pools[lname]["p"][sid], pts, args.max_patches_per_pool, ms
                    )
                    if rng.rand() < 0.5:
                        layer_pools[lname]["a"][sid] = merge_cap_pool(
                            layer_pools[lname]["a"][sid], pts, args.max_patches_per_pool, ms + 17
                        )
                    else:
                        layer_pools[lname]["b"][sid] = merge_cap_pool(
                            layer_pools[lname]["b"][sid], pts, args.max_patches_per_pool, ms + 31
                        )

        for lname in layer_names:
            try:
                sc = score_from_style_pools(
                    pools=layer_pools[lname]["p"],
                    pools_a=layer_pools[lname]["a"],
                    pools_b=layer_pools[lname]["b"],
                    n_styles=n_styles,
                    device=device,
                    num_projections=args.num_projections,
                    seed=args.seed + p * 9973,
                    cache_device=args.cache_device,
                )
            except RuntimeError as e:
                print(f"[WARN] skip layer={lname} patch={p}: {e}")
                continue

            results.append(
                {
                    "layer": lname,
                    "patch": int(p),
                    "R_mean": float(sc["R_mean"]),
                    "R_worst": float(sc["R_worst"]),
                    "W": sc["W"].tolist(),
                    "gap": sc["gap"].tolist(),
                }
            )

            tag = re.sub(r"[^a-zA-Z0-9_\\-]", "_", f"{lname}_p{p}")
            mat = sc["B"].copy()
            for si in range(n_styles):
                mat[si, si] = sc["W"][si]
            save_heatmap(
                mat,
                style_names,
                os.path.join(args.out_dir, f"within_between_{tag}.png"),
                title=f"{lname} p={p} (diag=within, off=between)",
            )
            save_gap(
                sc["gap"],
                style_names,
                os.path.join(args.out_dir, f"gap_{tag}.png"),
                title=f"{lname} p={p} gap=min_between-within",
            )
            print(
                f"[SCAN] layer={lname:12s} patch={p:2d} "
                f"R_worst={sc['R_worst']:.4f} R_mean={sc['R_mean']:.4f}"
            )

        del layer_pools
        if args.empty_cache_each_layer and device.type == "cuda":
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    results_sorted = sorted(results, key=lambda x: x["R_worst"], reverse=True)
    result_name = "layer_scan_results.json" if shard_n == 1 else f"layer_scan_results_shard{shard_i}_of_{shard_n}.json"
    with open(os.path.join(args.out_dir, result_name), "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)

    print("\n[TOP-10 by R_worst]")
    for r in results_sorted[:10]:
        print(
            f"layer={r['layer']:<12s} patch={r['patch']} "
            f"R_worst={r['R_worst']:.4f} R_mean={r['R_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
