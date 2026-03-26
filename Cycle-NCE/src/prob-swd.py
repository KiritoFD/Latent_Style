from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import optuna
import torch
import torch.nn.functional as F

try:
    from .losses import calc_swd_loss
except Exception:
    from losses import calc_swd_loss


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
        raise ValueError(f"Expected latent shape [C,H,W], got {tuple(latent.shape)} from {path}")
    return latent


def _load_domain_latents(data_root: Path, subdir: str, max_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    style_dir = data_root / subdir
    files = sorted(style_dir.glob("*.pt")) + sorted(style_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No latent files found in {style_dir}")
    if max_samples > 0:
        files = files[: max(1, int(max_samples))]
    latents = [_load_latent_file(p) for p in files]
    x = torch.stack(latents, dim=0).to(device=device, dtype=dtype, non_blocking=False)
    return x.contiguous()


def _split_half(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    if n < 4:
        raise ValueError("Need at least 4 samples per domain for stable intra-distance")
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
        w = F.normalize(w.view(num_projections, -1), p=2, dim=1).view_as(w)
        bank[p] = w
    return bank


def _calc_pair_loss_batched(
    x: torch.Tensor,
    y: torch.Tensor,
    patch_combo: List[int],
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
) -> torch.Tensor:
    n = min(int(x.shape[0]), int(y.shape[0]))
    if n <= 0:
        return x.new_tensor(0.0, dtype=torch.float32)
    if batch_size <= 0 or batch_size >= n:
        dummy_ids = torch.zeros((n,), device=x.device, dtype=torch.long)
        return calc_swd_loss(
            x[:n],
            y[:n],
            dummy_ids,
            patch_combo,
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

    acc_base = x.new_tensor(0.0, dtype=torch.float32)
    seen = 0
    for s in range(0, n, int(batch_size)):
        e = min(n, s + int(batch_size))
        xb = x[s:e]
        yb = y[s:e]
        dummy_ids = torch.zeros((int(e - s),), device=xb.device, dtype=torch.long)
        lb = calc_swd_loss(
            xb,
            yb,
            dummy_ids,
            patch_combo,
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
        acc_base = acc_base + lb * w
        seen += int(e - s)

    denom = float(max(1, seen))
    return acc_base / denom


def _num_chunks(n: int, batch_size: int) -> int:
    if n <= 0:
        return 0
    if batch_size <= 0 or batch_size >= n:
        return 1
    return (n + int(batch_size) - 1) // int(batch_size)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA SWD patch-combo probe (base SWD only)")
    parser.add_argument("--config", type=str, default="config_style_oa_5_lr5e4_wc2_swd60_id30_e120.json")
    parser.add_argument("--photo-domain", type=str, default="photo")
    parser.add_argument("--patch-min", type=int, default=1)
    parser.add_argument("--patch-max", type=int, default=25)
    parser.add_argument("--patch-step", type=int, default=1)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--combo-min-size", type=int, default=1)
    parser.add_argument("--combo-max-size", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all samples")
    parser.add_argument("--eval-batch-size", type=int, default=0, help="Per-loss eval batch size; <=0 means full")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--save-all-trials-csv", action="store_true", help="Save all trial records (COMPLETE/PRUNED/FAIL)")
    parser.add_argument("--study-name", type=str, default="prob_swd_exact")
    parser.add_argument("--storage", type=str, default="", help="Optuna storage URI, e.g. sqlite:///path/to/study.db")
    parser.add_argument("--output-dir", type=str, default="../probe_swd_exact")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. No CUDA device found.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float32
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = str(args.storage).strip() or f"sqlite:///{(out_dir / (str(args.study_name) + '.db')).as_posix()}"

    config_path = Path(args.config).resolve()
    cfg = _load_json(config_path)
    config_dir = config_path.parent
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})

    data_root = _resolve_path(str(data_cfg.get("data_root", "")), config_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Latent data_root not found: {data_root}")

    style_subdirs = list(data_cfg.get("style_subdirs", []))
    if not style_subdirs:
        raise ValueError("config.data.style_subdirs is empty")

    photo_name = str(args.photo_domain)
    if photo_name not in style_subdirs:
        photo_name = style_subdirs[0]
        print(f"[WARN] photo domain '{args.photo_domain}' not found, fallback='{photo_name}'")
    target_styles = [s for s in style_subdirs if s != photo_name]
    if not target_styles:
        raise ValueError("No target style domains after excluding photo domain")

    if args.patch_sizes:
        patch_candidates = sorted({int(p) for p in args.patch_sizes if int(p) > 0})
    else:
        patch_candidates = list(range(int(args.patch_min), int(args.patch_max) + 1, max(1, int(args.patch_step))))

    cmin = max(1, int(args.combo_min_size))
    cmax = min(int(args.combo_max_size), len(patch_candidates))
    if cmin > cmax:
        raise ValueError(f"Invalid combo size range: min={cmin}, max={cmax}")

    num_projections = int(loss_cfg.get("swd_num_projections", 256))
    projection_chunk = int(loss_cfg.get("swd_projection_chunk_size", 64))
    distance_mode = str(loss_cfg.get("swd_distance_mode", "cdf")).lower()
    cdf_num_bins = int(loss_cfg.get("swd_cdf_num_bins", 32))
    cdf_tau = float(loss_cfg.get("swd_cdf_tau", 0.01))
    cdf_sample_size = int(loss_cfg.get("swd_cdf_sample_size", 256))
    cdf_bin_chunk = int(loss_cfg.get("swd_cdf_bin_chunk_size", 4))
    cdf_sample_chunk = int(loss_cfg.get("swd_cdf_sample_chunk_size", 256))

    print(f"[INFO] Device={device}, dtype={dtype}")
    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Source: {photo_name}; Targets: {target_styles}")
    print(f"[INFO] Patch candidates ({len(patch_candidates)}): {patch_candidates}")
    print(f"[INFO] Trials={args.trials}, timeout={args.timeout_sec}s, eval_batch_size={args.eval_batch_size}")
    print(f"[INFO] Study={args.study_name}, storage={storage}")

    pools: Dict[str, torch.Tensor] = {}
    print("[INFO] Loading latent pools to CUDA...")
    for name in [photo_name] + target_styles:
        x = _load_domain_latents(data_root, name, int(args.max_samples), device, dtype)
        pools[name] = x
        print(f"  - {name}: {tuple(x.shape)}")

    n_common = min(int(v.shape[0]) for v in pools.values())
    if n_common % 2 == 1:
        n_common -= 1
    if n_common < 4:
        raise RuntimeError(f"Too few common samples across domains: {n_common}")
    for k in list(pools.keys()):
        pools[k] = pools[k][:n_common].contiguous()

    src_full = pools[photo_name]
    src_a, src_b = _split_half(src_full)
    tgt_full: Dict[str, torch.Tensor] = {}
    tgt_a: Dict[str, torch.Tensor] = {}
    tgt_b: Dict[str, torch.Tensor] = {}
    for s in target_styles:
        t = pools[s]
        a, b = _split_half(t)
        tgt_full[s] = t
        tgt_a[s] = a
        tgt_b[s] = b

    def eval_combo_snr(patch_combo: List[int], trial_seed: int) -> float:
        ch = int(src_full.shape[1])
        bank = _build_projection_bank(
            channels=ch,
            patch_sizes=patch_combo,
            num_projections=num_projections,
            device=device,
            dtype=src_full.dtype,
            seed=trial_seed,
        )

        scores: List[float] = []
        for style in target_styles:
            inter_base = _calc_pair_loss_batched(
                src_full,
                tgt_full[style],
                patch_combo,
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
            intra_src_base = _calc_pair_loss_batched(
                src_a,
                src_b,
                patch_combo,
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
            intra_tgt_base = _calc_pair_loss_batched(
                tgt_a[style],
                tgt_b[style],
                patch_combo,
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

            intra_base = 0.5 * (intra_src_base + intra_tgt_base)
            snr = inter_base / (intra_base + 1e-8)
            scores.append(float(snr.item()))

        return float(np.mean(scores))

    patch_idx = list(range(len(patch_candidates)))

    def objective(trial: optuna.Trial) -> float:
        selected = [i for i in patch_idx if trial.suggest_int(f"use_p{patch_candidates[i]}", 0, 1) == 1]
        if len(selected) < cmin or len(selected) > cmax:
            raise optuna.TrialPruned()

        patch_combo = [patch_candidates[i] for i in selected]
        trial_seed = int(args.seed) + trial.number * 100_003
        score = eval_combo_snr(patch_combo, trial_seed)

        trial.set_user_attr("patches", patch_combo)
        trial.set_user_attr("num_patches", len(patch_combo))
        return float(score)

    sampler = optuna.samplers.TPESampler(
        seed=int(args.seed),
        multivariate=False,
        warn_independent_sampling=False,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=str(args.study_name),
        storage=storage,
        load_if_exists=True,
    )
    interrupted = False
    try:
        study.optimize(
            objective,
            n_trials=max(1, int(args.trials)),
            timeout=(None if int(args.timeout_sec) <= 0 else int(args.timeout_sec)),
            gc_after_trial=True,
            catch=(RuntimeError, ValueError, torch.cuda.OutOfMemoryError),
        )
    except KeyboardInterrupt:
        interrupted = True
        print("[WARN] Interrupted by user. Saving current study state...")

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not done:
        raise RuntimeError("No successful trial completed. Try smaller --max-samples or fewer patches.")
    done.sort(key=lambda t: float(t.value), reverse=True)
    best = done[0]
    topk = done[: max(1, int(args.topk))]

    csv_path = out_dir / "prob_swd_exact_topk.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "trial", "snr", "num_patches", "patches"])
        for rank, t in enumerate(topk, start=1):
            writer.writerow([
                rank,
                int(t.number),
                f"{float(t.value):.8f}",
                int(t.user_attrs.get("num_patches", 0)),
                ",".join(str(x) for x in t.user_attrs.get("patches", [])),
            ])

    all_trials_csv_path = out_dir / "prob_swd_exact_all_trials.csv"
    if bool(args.save_all_trials_csv):
        with open(all_trials_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "state", "snr", "num_patches", "patches"])
            for t in study.trials:
                writer.writerow([
                    int(t.number),
                    str(t.state),
                    (f"{float(t.value):.8f}" if t.value is not None else ""),
                    int(t.user_attrs.get("num_patches", 0)),
                    ",".join(str(x) for x in t.user_attrs.get("patches", [])),
                ])

    result = {
        "config_path": str(config_path),
        "data_root": str(data_root),
        "photo_domain": photo_name,
        "target_styles": target_styles,
        "n_common_samples": int(n_common),
        "patch_candidates": patch_candidates,
        "search": {
            "study_name": str(args.study_name),
            "storage": storage,
            "interrupted": bool(interrupted),
            "combo_min_size": cmin,
            "combo_max_size": cmax,
            "max_samples": int(args.max_samples),
            "eval_batch_size": int(args.eval_batch_size),
            "effective_samples_per_domain": int(n_common),
            "inter_chunks_per_style": int(_num_chunks(n_common, int(args.eval_batch_size))),
            "intra_chunks_per_style": int(_num_chunks(n_common // 2, int(args.eval_batch_size))),
            "trials": int(args.trials),
            "timeout_sec": int(args.timeout_sec),
        },
        "loss_alignment": {
            "num_projections": num_projections,
            "projection_chunk_size": projection_chunk,
            "distance_mode": distance_mode,
            "cdf_num_bins": cdf_num_bins,
            "cdf_tau": cdf_tau,
            "cdf_sample_size": cdf_sample_size,
            "cdf_bin_chunk_size": cdf_bin_chunk,
            "cdf_sample_chunk_size": cdf_sample_chunk,
            "objective_impl": "calc_swd_loss direct trial evaluation",
        },
        "best": {
            "trial": int(best.number),
            "snr": float(best.value),
            "num_patches": int(best.user_attrs.get("num_patches", 0)),
            "patches": [int(x) for x in best.user_attrs.get("patches", [])],
        },
        "topk": [
            {
                "rank": int(i + 1),
                "trial": int(t.number),
                "snr": float(t.value),
                "num_patches": int(t.user_attrs.get("num_patches", 0)),
                "patches": [int(x) for x in t.user_attrs.get("patches", [])],
            }
            for i, t in enumerate(topk)
        ],
    }

    json_path = out_dir / "prob_swd_exact_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("[DONE] Exact CUDA search complete")
    print(
        f"[DONE] Best snr={float(best.value):.6f}, "
        f"n={int(best.user_attrs.get('num_patches', 0))}, patches={best.user_attrs.get('patches', [])}"
    )
    print(f"[DONE] JSON: {json_path}")
    print(f"[DONE] CSV:  {csv_path}")
    if bool(args.save_all_trials_csv):
        print(f"[DONE] ALL:  {all_trials_csv_path}")


if __name__ == "__main__":
    main()
