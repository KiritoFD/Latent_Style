from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .losses import calc_swd_and_hf_loss
except Exception:
    from losses import calc_swd_and_hf_loss


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


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        raise ValueError(f"Expected latent [C,H,W], got {tuple(latent.shape)} from {path}")
    return latent


def _load_domain_latents(data_root: Path, subdir: str, max_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    style_dir = data_root / subdir
    files = sorted(style_dir.glob("*.pt")) + sorted(style_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No latent files found in {style_dir}")
    if max_samples > 0:
        files = files[: max(1, int(max_samples))]
    latents = [_load_latent_file(p) for p in files]
    return torch.stack(latents, dim=0).to(device=device, dtype=dtype, non_blocking=False).contiguous()


def _split_half(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    if n < 4:
        raise ValueError("Need at least 4 samples per domain")
    if n % 2 == 1:
        x = x[:-1]
        n -= 1
    h = n // 2
    return x[:h], x[h:]


def _build_projection_bank(
    channels: int,
    patch_sizes: List[int],
    num_projections: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Dict[int, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    bank: Dict[int, torch.Tensor] = {}
    for p in patch_sizes:
        w = torch.randn(num_projections, channels, p, p, device=device, dtype=dtype, generator=gen)
        bank[p] = F.normalize(w.view(num_projections, -1), p=2, dim=1).view_as(w)
    return bank


def _calc_pair_loss_batched(
    x: torch.Tensor,
    y: torch.Tensor,
    patch_size: int,
    *,
    num_projections: int,
    projection_chunk_size: int,
    distance_mode: str,
    cdf_num_bins: int,
    cdf_tau: float,
    cdf_sample_size: int,
    cdf_bin_chunk_size: int,
    cdf_sample_chunk_size: int,
    projection_bank: Dict[int, torch.Tensor],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(int(x.shape[0]), int(y.shape[0]))
    if n <= 0:
        z = x.new_tensor(0.0, dtype=torch.float32)
        return z, z

    patch_list = [int(patch_size)]
    if batch_size <= 0 or batch_size >= n:
        return calc_swd_and_hf_loss(
            x[:n],
            y[:n],
            patch_list,
            num_projections=num_projections,
            projection_chunk_size=projection_chunk_size,
            distance_mode=distance_mode,
            cdf_num_bins=cdf_num_bins,
            cdf_tau=cdf_tau,
            cdf_sample_size=cdf_sample_size,
            cdf_bin_chunk_size=cdf_bin_chunk_size,
            cdf_sample_chunk_size=cdf_sample_chunk_size,
            projection_bank=projection_bank,
        )

    acc_b = x.new_tensor(0.0, dtype=torch.float32)
    acc_h = x.new_tensor(0.0, dtype=torch.float32)
    seen = 0
    for s in range(0, n, int(batch_size)):
        e = min(n, s + int(batch_size))
        lb, lh = calc_swd_and_hf_loss(
            x[s:e],
            y[s:e],
            patch_list,
            num_projections=num_projections,
            projection_chunk_size=projection_chunk_size,
            distance_mode=distance_mode,
            cdf_num_bins=cdf_num_bins,
            cdf_tau=cdf_tau,
            cdf_sample_size=cdf_sample_size,
            cdf_bin_chunk_size=cdf_bin_chunk_size,
            cdf_sample_chunk_size=cdf_sample_chunk_size,
            projection_bank=projection_bank,
        )
        w = float(e - s)
        acc_b = acc_b + lb * w
        acc_h = acc_h + lh * w
        seen += int(e - s)
    denom = float(max(1, seen))
    return acc_b / denom, acc_h / denom


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Single odd-patch SNR probe (base SWD + hf SWD)")
    parser.add_argument("--config", type=str, default="config_style_oa_5_lr5e4_wc2_swd60_id30_e120.json")
    parser.add_argument("--photo-domain", type=str, default="photo")
    parser.add_argument("--patch-min", type=int, default=1)
    parser.add_argument("--patch-max", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--eval-batch-size", type=int, default=0, help="<=0 means full batch")
    parser.add_argument("--hf-ratio", type=float, default=-1.0, help="<0 means use config loss.swd_hf_weight_ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--output-dir", type=str, default="../probe_odd_patch_snr")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float32

    config_path = Path(args.config).resolve()
    cfg = _load_json(config_path)
    config_dir = config_path.parent
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})

    data_root = _resolve_path(str(data_cfg.get("data_root", "")), config_dir)
    style_subdirs = list(data_cfg.get("style_subdirs", []))
    if not style_subdirs:
        raise ValueError("config.data.style_subdirs is empty")

    photo_name = str(args.photo_domain)
    if photo_name not in style_subdirs:
        photo_name = style_subdirs[0]
        print(f"[WARN] fallback photo domain -> {photo_name}")
    target_styles = [s for s in style_subdirs if s != photo_name]
    if not target_styles:
        raise ValueError("No target styles after excluding photo domain")

    patch_sizes = [p for p in range(int(args.patch_min), int(args.patch_max) + 1) if p % 2 == 1]
    if not patch_sizes:
        raise ValueError("No odd patch sizes in requested range")

    num_projections = int(loss_cfg.get("swd_num_projections", 256))
    projection_chunk = int(loss_cfg.get("swd_projection_chunk_size", 64))
    distance_mode = str(loss_cfg.get("swd_distance_mode", "cdf")).lower()
    cdf_num_bins = int(loss_cfg.get("swd_cdf_num_bins", 32))
    cdf_tau = float(loss_cfg.get("swd_cdf_tau", 0.01))
    cdf_sample_size = int(loss_cfg.get("swd_cdf_sample_size", 256))
    cdf_bin_chunk = int(loss_cfg.get("swd_cdf_bin_chunk_size", 4))
    cdf_sample_chunk = int(loss_cfg.get("swd_cdf_sample_chunk_size", 256))
    hf_ratio = float(loss_cfg.get("swd_hf_weight_ratio", 2.0)) if float(args.hf_ratio) < 0 else float(args.hf_ratio)

    print(f"[INFO] Device={device}, dtype={dtype}")
    print(f"[INFO] Patches(odd)={patch_sizes}")
    print(f"[INFO] hf_ratio={hf_ratio:.4f}")

    pools: Dict[str, torch.Tensor] = {}
    for name in [photo_name] + target_styles:
        pools[name] = _load_domain_latents(data_root, name, int(args.max_samples), device, dtype)
        print(f"  - {name}: {tuple(pools[name].shape)}")

    n_common = min(int(v.shape[0]) for v in pools.values())
    if n_common % 2 == 1:
        n_common -= 1
    if n_common < 4:
        raise RuntimeError(f"Too few common samples: {n_common}")
    for k in list(pools.keys()):
        pools[k] = pools[k][:n_common].contiguous()

    src_full = pools[photo_name]
    src_a, src_b = _split_half(src_full)
    tgt_full: Dict[str, torch.Tensor] = {}
    tgt_a: Dict[str, torch.Tensor] = {}
    tgt_b: Dict[str, torch.Tensor] = {}
    for s in target_styles:
        a, b = _split_half(pools[s])
        tgt_full[s] = pools[s]
        tgt_a[s] = a
        tgt_b[s] = b

    bank = _build_projection_bank(
        channels=int(src_full.shape[1]),
        patch_sizes=patch_sizes,
        num_projections=num_projections,
        device=device,
        dtype=src_full.dtype,
        seed=int(args.seed),
    )

    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for p in patch_sizes:
        per_style = []
        for style in target_styles:
            inter_base, inter_hf = _calc_pair_loss_batched(
                src_full,
                tgt_full[style],
                p,
                batch_size=int(args.eval_batch_size),
                num_projections=num_projections,
                projection_chunk_size=projection_chunk,
                distance_mode=distance_mode,
                cdf_num_bins=cdf_num_bins,
                cdf_tau=cdf_tau,
                cdf_sample_size=cdf_sample_size,
                cdf_bin_chunk_size=cdf_bin_chunk,
                cdf_sample_chunk_size=cdf_sample_chunk,
                projection_bank=bank,
            )
            intra_src_base, intra_src_hf = _calc_pair_loss_batched(
                src_a,
                src_b,
                p,
                batch_size=int(args.eval_batch_size),
                num_projections=num_projections,
                projection_chunk_size=projection_chunk,
                distance_mode=distance_mode,
                cdf_num_bins=cdf_num_bins,
                cdf_tau=cdf_tau,
                cdf_sample_size=cdf_sample_size,
                cdf_bin_chunk_size=cdf_bin_chunk,
                cdf_sample_chunk_size=cdf_sample_chunk,
                projection_bank=bank,
            )
            intra_tgt_base, intra_tgt_hf = _calc_pair_loss_batched(
                tgt_a[style],
                tgt_b[style],
                p,
                batch_size=int(args.eval_batch_size),
                num_projections=num_projections,
                projection_chunk_size=projection_chunk,
                distance_mode=distance_mode,
                cdf_num_bins=cdf_num_bins,
                cdf_tau=cdf_tau,
                cdf_sample_size=cdf_sample_size,
                cdf_bin_chunk_size=cdf_bin_chunk,
                cdf_sample_chunk_size=cdf_sample_chunk,
                projection_bank=bank,
            )

            inter_b = float(inter_base.item())
            inter_h = float(inter_hf.item())
            intra_b = float((0.5 * (intra_src_base + intra_tgt_base)).item())
            intra_h = float((0.5 * (intra_src_hf + intra_tgt_hf)).item())
            base_snr = inter_b / (intra_b + 1e-8)
            hf_snr = inter_h / (intra_h + 1e-8)
            comb_snr = (inter_b + hf_ratio * inter_h) / (intra_b + hf_ratio * intra_h + 1e-8)

            row = {
                "patch_size": int(p),
                "style": str(style),
                "inter_base_swd": inter_b,
                "intra_base_swd": intra_b,
                "inter_hf_swd": inter_h,
                "intra_hf_swd": intra_h,
                "base_snr": float(base_snr),
                "hf_snr": float(hf_snr),
                "combined_snr": float(comb_snr),
                "hf_ratio": float(hf_ratio),
            }
            detail_rows.append(row)
            per_style.append(row)

        summary_rows.append(
            {
                "patch_size": int(p),
                "mean_inter_base_swd": float(np.mean([r["inter_base_swd"] for r in per_style])),
                "mean_intra_base_swd": float(np.mean([r["intra_base_swd"] for r in per_style])),
                "mean_inter_hf_swd": float(np.mean([r["inter_hf_swd"] for r in per_style])),
                "mean_intra_hf_swd": float(np.mean([r["intra_hf_swd"] for r in per_style])),
                "mean_base_snr": float(np.mean([r["base_snr"] for r in per_style])),
                "mean_hf_snr": float(np.mean([r["hf_snr"] for r in per_style])),
                "mean_combined_snr": float(np.mean([r["combined_snr"] for r in per_style])),
                "hf_ratio": float(hf_ratio),
            }
        )
        print(f"[INFO] patch={p} done")

    summary_rows.sort(key=lambda x: float(x["mean_combined_snr"]), reverse=True)
    best = summary_rows[0]

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = out_dir / "odd_patch_snr_detail.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_csv = out_dir / "odd_patch_snr_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    result_json = out_dir / "odd_patch_snr_result.json"
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config_path": str(config_path),
                "data_root": str(data_root),
                "photo_domain": photo_name,
                "target_styles": target_styles,
                "n_common_samples": int(n_common),
                "patch_sizes": patch_sizes,
                "hf_ratio": float(hf_ratio),
                "best_patch_by_mean_combined_snr": int(best["patch_size"]),
                "best_mean_combined_snr": float(best["mean_combined_snr"]),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print("[DONE] Odd patch SNR probe complete")
    print(f"[DONE] Best patch={best['patch_size']}, mean_combined_snr={best['mean_combined_snr']:.6f}")
    print(f"[DONE] Detail CSV:  {detail_csv}")
    print(f"[DONE] Summary CSV: {summary_csv}")
    print(f"[DONE] JSON:        {result_json}")


if __name__ == "__main__":
    main()

