from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


def _load_latent(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("latent", "z", "image", "tensor"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        raise TypeError(f"Unsupported latent payload in {path}")
    if payload.ndim == 4:
        payload = payload[0]
    if payload.ndim != 3:
        raise ValueError(f"Expected latent [C,H,W], got {tuple(payload.shape)} from {path}")
    return payload.float().contiguous()


def _style_paths(root: Path, styles: list[str], max_per_style: int) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for style in styles:
        paths = sorted((root / style).glob("*.pt"))
        if max_per_style > 0:
            paths = paths[:max_per_style]
        out[style] = paths
    return out


def _sobel_abs_mean(z: torch.Tensor) -> torch.Tensor:
    c = int(z.shape[0])
    x = z.unsqueeze(0)
    kx = z.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = z.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx.expand(c, 1, 3, 3), padding=1, groups=c)
    gy = F.conv2d(x, ky.expand(c, 1, 3, 3), padding=1, groups=c)
    return torch.sqrt(gx.square() + gy.square() + 1e-12).mean(dim=(0, 2, 3))


def _patch_snr(z: torch.Tensor, k: int) -> dict[str, float]:
    if z.ndim == 3:
        z = z.unsqueeze(0)
    _, c, h, w = z.shape
    if k > h or k > w:
        return {"num_patches": 0.0, "inter": float("nan"), "intra": float("nan"), "snr": float("nan")}
    stride = max(1, k // 2)
    patches = F.unfold(z.float(), kernel_size=k, stride=stride)
    n = int(patches.shape[-1])
    patches = patches.view(1, c, k * k, n).permute(0, 3, 1, 2)
    patch_means = patches.mean(dim=-1)
    patch_vars = patches.var(dim=-1, unbiased=False)
    inter = patch_means.var(dim=1, unbiased=False).mean()
    intra = patch_vars.mean()
    return {
        "num_patches": float(n),
        "inter": float(inter.item()),
        "intra": float(intra.item()),
        "snr": float((inter / (intra + 1e-8)).item()),
    }


def _feature_bank(z: torch.Tensor, k: int, max_patches: int) -> torch.Tensor:
    if z.ndim == 3:
        z = z.unsqueeze(0)
    _, _, h, w = z.shape
    if k > h or k > w:
        return torch.empty((0, 1), dtype=torch.float32)
    stride = max(1, k // 2)
    patches = F.unfold(z.float(), kernel_size=k, stride=stride).transpose(1, 2).squeeze(0)
    if max_patches > 0 and patches.shape[0] > max_patches:
        idx = torch.linspace(0, patches.shape[0] - 1, max_patches).round().long()
        patches = patches.index_select(0, idx)
    return F.normalize(patches, dim=-1, eps=1e-6)


def _transport_entropy(z_photo: torch.Tensor, z_style: torch.Tensor, k: int, max_patches: int, temperature: float) -> dict[str, float]:
    a = _feature_bank(z_photo, k, max_patches)
    b = _feature_bank(z_style, k, max_patches)
    if a.numel() == 0 or b.numel() == 0:
        return {"cost_mean": float("nan"), "cost_std": float("nan"), "entropy_norm": float("nan")}
    cost = 1.0 - a @ b.t()
    prob = torch.softmax(-cost / max(temperature, 1e-8), dim=-1)
    ent = -(prob * (prob + 1e-8).log()).sum(dim=-1).mean()
    return {
        "cost_mean": float(cost.mean().item()),
        "cost_std": float(cost.std(unbiased=False).item()),
        "entropy_norm": float((ent / math.log(max(2, prob.shape[-1]))).item()),
    }


def _new_acc() -> dict[str, float]:
    return {"n": 0.0, "sum": 0.0, "sumsq": 0.0, "min": float("inf"), "max": float("-inf")}


def _update(acc: dict[str, float], value: float) -> None:
    if not math.isfinite(float(value)):
        return
    v = float(value)
    acc["n"] += 1.0
    acc["sum"] += v
    acc["sumsq"] += v * v
    acc["min"] = min(acc["min"], v)
    acc["max"] = max(acc["max"], v)


def _finish(acc: dict[str, float]) -> dict[str, float | int | None]:
    n = int(acc["n"])
    if n <= 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = acc["sum"] / n
    var = max(0.0, acc["sumsq"] / n - mean * mean)
    return {"count": n, "mean": mean, "std": math.sqrt(var), "min": acc["min"], "max": acc["max"]}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _top_eigenvector(cov: torch.Tensor) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(cov.float())
    vec = vecs[:, int(torch.argmax(vals).item())].float()
    if vec.sum().item() < 0:
        vec = -vec
    return vec / (vec.norm() + 1e-8)


def _guide_maps(z: torch.Tensor, means: torch.Tensor, stds: torch.Tensor, grad_weights: torch.Tensor, pca_vec: torch.Tensor) -> dict[str, torch.Tensor]:
    z_std = (z - means[:, None, None]) / (stds[:, None, None] + 1e-6)
    max_grad_channel = int(torch.argmax(grad_weights).item())
    return {
        "raw_mean": z.mean(dim=0, keepdim=True),
        "whitened_mean": z_std.mean(dim=0, keepdim=True),
        "grad_weighted": (z_std * grad_weights[:, None, None]).sum(dim=0, keepdim=True),
        "pca1": (z_std * pca_vec[:, None, None]).sum(dim=0, keepdim=True),
        f"channel_{max_grad_channel}": z[max_grad_channel : max_grad_channel + 1],
    }


def _find_knee(rows: list[dict[str, Any]], key: str = "snr_mean") -> int | None:
    valid = [r for r in rows if r.get(key) is not None and float(r[key]) > 0]
    if len(valid) < 3:
        return None
    xs = [math.log(float(r["k"])) for r in valid]
    ys = [math.log(float(r[key]) + 1e-12) for r in valid]
    slopes = [(ys[i] - ys[i - 1]) / max(xs[i] - xs[i - 1], 1e-8) for i in range(1, len(valid))]
    best_i = max(range(1, len(slopes)), key=lambda i: abs(slopes[i] - slopes[i - 1]))
    return int(valid[best_i]["k"])


def diagnose_root(name: str, latent_root: Path, styles: list[str], k_sizes: list[int], args: argparse.Namespace) -> dict[str, Any]:
    paths_by_style = _style_paths(latent_root, styles, int(args.max_per_style))
    all_paths = [p for style in styles for p in paths_by_style[style]]
    if not all_paths:
        raise FileNotFoundError(f"No latent files found under {latent_root}")

    first = _load_latent(all_paths[0])
    channels = int(first.shape[0])
    sum_c = torch.zeros(channels)
    sumsq_c = torch.zeros(channels)
    cross = torch.zeros(channels, channels)
    grad_sum = torch.zeros(channels)
    count_pixels = 0
    count_files = 0
    channel_min = torch.full((channels,), float("inf"))
    channel_max = torch.full((channels,), float("-inf"))

    snr_acc: dict[tuple[str, str, int, str], dict[str, float]] = defaultdict(_new_acc)
    loaded_by_style: dict[str, list[tuple[Path, torch.Tensor]]] = {style: [] for style in styles}

    for style in styles:
        for path in paths_by_style[style]:
            z = _load_latent(path)
            loaded_by_style[style].append((path, z))
            flat = z.view(channels, -1)
            sum_c += flat.sum(dim=1)
            sumsq_c += flat.square().sum(dim=1)
            cross += flat @ flat.t()
            grad_sum += _sobel_abs_mean(z)
            channel_min = torch.minimum(channel_min, flat.min(dim=1).values)
            channel_max = torch.maximum(channel_max, flat.max(dim=1).values)
            count_pixels += int(flat.shape[1])
            count_files += 1

    means = sum_c / max(1, count_pixels)
    vars_c = (sumsq_c / max(1, count_pixels) - means.square()).clamp_min(1e-12)
    stds = torch.sqrt(vars_c)
    cov = cross / max(1, count_pixels) - torch.outer(means, means)
    corr = cov / (stds[:, None] * stds[None, :] + 1e-8)
    grad_mean = grad_sum / max(1, count_files)
    grad_weights = grad_mean / (grad_mean.sum() + 1e-8)
    pca1 = _top_eigenvector(cov)

    for style in styles:
        for _, z in loaded_by_style[style]:
            guides = _guide_maps(z, means, stds, grad_weights, pca1)
            sources: dict[str, torch.Tensor] = {"all_channels": z, **guides}
            for source_name, source_tensor in sources.items():
                for k in k_sizes:
                    stats = _patch_snr(source_tensor, k)
                    for metric, value in stats.items():
                        _update(snr_acc[(style, source_name, k, metric)], value)

    snr_rows: list[dict[str, Any]] = []
    global_acc: dict[tuple[str, int, str], dict[str, float]] = defaultdict(_new_acc)
    for style in styles:
        for source_name in ["all_channels", "raw_mean", "whitened_mean", "grad_weighted", "pca1", f"channel_{int(torch.argmax(grad_weights).item())}"]:
            for k in k_sizes:
                row = {"root": name, "style": style, "source": source_name, "k": k}
                for metric in ("num_patches", "inter", "intra", "snr"):
                    fin = _finish(snr_acc[(style, source_name, k, metric)])
                    row[f"{metric}_mean"] = fin["mean"]
                    row[f"{metric}_std"] = fin["std"]
                    if fin["mean"] is not None:
                        _update(global_acc[(source_name, k, metric)], float(fin["mean"]))
                snr_rows.append(row)

    global_rows: list[dict[str, Any]] = []
    for source_name in ["all_channels", "raw_mean", "whitened_mean", "grad_weighted", "pca1", f"channel_{int(torch.argmax(grad_weights).item())}"]:
        for k in k_sizes:
            row = {"root": name, "source": source_name, "k": k}
            for metric in ("num_patches", "inter", "intra", "snr"):
                row[f"{metric}_mean"] = _finish(global_acc[(source_name, k, metric)])["mean"]
            global_rows.append(row)

    transport_acc: dict[tuple[str, str, int, str], dict[str, float]] = defaultdict(_new_acc)
    photo_items = loaded_by_style.get("photo", [])
    for target_style in [s for s in styles if s != "photo"]:
        target_items = loaded_by_style.get(target_style, [])
        if not photo_items or not target_items:
            continue
        pair_count = min(len(photo_items), len(target_items))
        if int(args.max_transport_pairs_per_style) > 0:
            pair_count = min(pair_count, int(args.max_transport_pairs_per_style))
        for idx in range(pair_count):
            z_photo = photo_items[idx][1]
            z_style = target_items[idx][1]
            for k in k_sizes:
                stats = _transport_entropy(
                    z_photo,
                    z_style,
                    k,
                    int(args.transport_max_patches),
                    float(args.transport_temperature),
                )
                for metric, value in stats.items():
                    _update(transport_acc[(target_style, "all_channels", k, metric)], value)

    transport_rows: list[dict[str, Any]] = []
    for target_style in [s for s in styles if s != "photo"]:
        for k in k_sizes:
            row = {"root": name, "src_style": "photo", "tgt_style": target_style, "source": "all_channels", "k": k}
            for metric in ("cost_mean", "cost_std", "entropy_norm"):
                fin = _finish(transport_acc[(target_style, "all_channels", k, metric)])
                row[f"{metric}_mean"] = fin["mean"]
                row[f"{metric}_std"] = fin["std"]
            transport_rows.append(row)

    channel_rows = []
    for idx in range(channels):
        channel_rows.append(
            {
                "root": name,
                "channel": idx,
                "mean": float(means[idx].item()),
                "std": float(stds[idx].item()),
                "min": float(channel_min[idx].item()),
                "max": float(channel_max[idx].item()),
                "grad_abs_mean": float(grad_mean[idx].item()),
                "grad_weight": float(grad_weights[idx].item()),
                "pca1_weight": float(pca1[idx].item()),
            }
        )

    return {
        "root_name": name,
        "latent_root": str(latent_root),
        "files": count_files,
        "pixels_per_channel": count_pixels,
        "channels": channels,
        "channel_rows": channel_rows,
        "channel_correlation": corr.tolist(),
        "pca1": pca1.tolist(),
        "max_grad_channel": int(torch.argmax(grad_weights).item()),
        "snr_rows": snr_rows,
        "global_rows": global_rows,
        "transport_rows": transport_rows,
        "all_channel_knee": _find_knee([r for r in global_rows if r["source"] == "all_channels"]),
        "pca_guide_knee": _find_knee([r for r in global_rows if r["source"] == "pca1"]),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SDXL Channel and Patch Diagnostic",
        "",
        "## Why This Exists",
        "",
        "SDXL VAE latents are learned coordinates. The 4 channels do not carry stable RGB/luma/chroma labels, so any 6ch extension that uses a raw channel mean as structure must be treated as a VAE-specific hypothesis.",
        "",
        "## Roots",
        "",
    ]
    for root in payload["roots"]:
        lines.extend(
            [
                f"### {root['root_name']}",
                "",
                f"- latent root: `{root['latent_root']}`",
                f"- files: `{root['files']}`",
                f"- channels: `{root['channels']}`",
                f"- max-gradient channel: `{root['max_grad_channel']}`",
                f"- all-channel SNR knee: `K={root['all_channel_knee']}`",
                f"- PCA-guide SNR knee: `K={root['pca_guide_knee']}`",
                "",
                "| ch | mean | std | grad abs mean | grad weight | pca1 weight |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in root["channel_rows"]:
            lines.append(
                f"| {row['channel']} | {row['mean']:.6f} | {row['std']:.6f} | {row['grad_abs_mean']:.6f} | {row['grad_weight']:.6f} | {row['pca1_weight']:.6f} |"
            )
        lines.extend(["", "Patch knees are diagnostics, not final hyperparameters. Prefer patch sets that sit just after the SNR drop while keeping transport entropy selective.", ""])
    lines.extend(
        [
            "## Initial Engineering Consequences",
            "",
            "- Keep a plain 4ch residual SDXL control as the necessary floor.",
            "- Replace raw `mean(channel)` texture guide with a VAE-calibrated guide: whitened mean, gradient-weighted guide, PCA1, or the max-gradient channel.",
            "- Test SDXL 6ch as `4 residual + 2 warp`, but compute tangent/texture geometry from the calibrated guide.",
            "- Test SDXL 7ch factorized mode as `4 residual + 1 amplitude + 2 warp` only after the calibrated 6ch guide is stable.",
            "- Patch candidates should be chosen from this diagnostic instead of blindly copying SD15 `[3,5,7,15]`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose VAE-dependent latent channel geometry and SWD patch scale.")
    parser.add_argument(
        "--latent-roots",
        default=f"sd15:{(ROOT.parent / 'latent-256')},sdxl:{(ROOT.parent / 'latent-256-sdxl-fp32')}",
        help="Comma-separated name:path entries.",
    )
    parser.add_argument("--style-subdirs", default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--k-sizes", default="1,2,3,4,5,7,8,9,15,16,21,29,31")
    parser.add_argument("--max-per-style", type=int, default=0)
    parser.add_argument("--max-transport-pairs-per-style", type=int, default=0)
    parser.add_argument("--transport-max-patches", type=int, default=256)
    parser.add_argument("--transport-temperature", type=float, default=0.05)
    parser.add_argument("--output-root", type=Path, default=ROOT / "exp" / "vae_backend_256_status" / "sdxl_channel_patch_diagnostic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots: list[tuple[str, Path]] = []
    for item in str(args.latent_roots).split(","):
        if not item.strip():
            continue
        if ":" in item:
            name, path = item.split(":", 1)
        else:
            path = item
            name = Path(path).name
        roots.append((name.strip(), Path(path).expanduser().resolve()))
    styles = [s.strip() for s in str(args.style_subdirs).split(",") if s.strip()]
    k_sizes = [int(x.strip()) for x in str(args.k_sizes).split(",") if x.strip()]

    payload = {
        "style_subdirs": styles,
        "k_sizes": k_sizes,
        "max_per_style": int(args.max_per_style),
        "max_transport_pairs_per_style": int(args.max_transport_pairs_per_style),
        "roots": [],
    }
    all_channel_rows: list[dict[str, Any]] = []
    all_snr_rows: list[dict[str, Any]] = []
    all_global_rows: list[dict[str, Any]] = []
    all_transport_rows: list[dict[str, Any]] = []
    for name, root in roots:
        result = diagnose_root(name, root, styles, k_sizes, args)
        payload["roots"].append({k: v for k, v in result.items() if k in {
            "root_name",
            "latent_root",
            "files",
            "pixels_per_channel",
            "channels",
            "channel_rows",
            "channel_correlation",
            "pca1",
            "max_grad_channel",
            "all_channel_knee",
            "pca_guide_knee",
        }})
        all_channel_rows.extend(result["channel_rows"])
        all_snr_rows.extend(result["snr_rows"])
        all_global_rows.extend(result["global_rows"])
        all_transport_rows.extend(result["transport_rows"])

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(
        out_root / "channel_stats.csv",
        all_channel_rows,
        ["root", "channel", "mean", "std", "min", "max", "grad_abs_mean", "grad_weight", "pca1_weight"],
    )
    _write_csv(
        out_root / "patch_snr_by_style.csv",
        all_snr_rows,
        ["root", "style", "source", "k", "num_patches_mean", "num_patches_std", "inter_mean", "inter_std", "intra_mean", "intra_std", "snr_mean", "snr_std"],
    )
    _write_csv(
        out_root / "patch_snr_global.csv",
        all_global_rows,
        ["root", "source", "k", "num_patches_mean", "inter_mean", "intra_mean", "snr_mean"],
    )
    _write_csv(
        out_root / "transport_entropy_photo_to_style.csv",
        all_transport_rows,
        ["root", "src_style", "tgt_style", "source", "k", "cost_mean_mean", "cost_mean_std", "cost_std_mean", "cost_std_std", "entropy_norm_mean", "entropy_norm_std"],
    )
    _write_markdown(out_root / "diagnosis.md", payload)
    print(json.dumps({"output_root": str(out_root), "roots": [r["root_name"] for r in payload["roots"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
