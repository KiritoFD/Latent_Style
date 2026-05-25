from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "exp" / "diffeomorphic_tangent_sweep" / "t01_ws0p03_g6_nl0p05" / "config.json"

VARIANTS = {
    "sdxl": {
        "vae_model": "sdxl-fp32",
        "latent_root": "latent-256-sdxl-fp32",
        "batch_size": 64,
        "eval_batch_size": 4,
        "learning_rate": 5e-5,
        "terminal_swd_weight": 8.0,
    },
    "sdxl_s0_stability": {
        "vae_model": "sdxl-fp32",
        "latent_root": "latent-256-sdxl-fp32",
        "batch_size": 48,
        "eval_batch_size": 4,
        "learning_rate": 2e-5,
        "terminal_swd_weight": 0.0,
        "style_spatial_pre_gain_16": 0.10,
        "diffeomorphic_warp_strength": 0.01,
        "grad_clip_norm": 0.5,
        "notes": "SDXL smoke: no terminal SWD, conservative tangent head.",
    },
    "sdxl_s0_minimal": {
        "vae_model": "sdxl-fp32",
        "latent_root": "latent-256-sdxl-fp32",
        "batch_size": 32,
        "eval_batch_size": 4,
        "learning_rate": 1e-5,
        "terminal_swd_weight": 1.0,
        "grad_clip_norm": 0.25,
        "model_overrides": {
            "num_res_blocks": 0,
            "use_diffeomorphic_stroke": False,
            "zero_init_output_head": True,
            "style_spatial_pre_gain_16": 0.0,
        },
        "bridge_overrides": {
            "swd_patch_sizes": [1, 3],
            "swd_num_projections": 16,
            "semantic_swd_num_projections": 16,
            "swd_projection_chunk_size": 8,
            "swd_cdf_sample_size": 64,
        },
        "notes": "SDXL minimal closure: no semantic body, no diffeomorphic head, real terminal SWD.",
    },
    "sdxl_s0_minimal_diffeo": {
        "vae_model": "sdxl-fp32",
        "latent_root": "latent-256-sdxl-fp32",
        "batch_size": 32,
        "eval_batch_size": 4,
        "learning_rate": 1e-5,
        "terminal_swd_weight": 1.0,
        "diffeomorphic_warp_strength": 0.005,
        "grad_clip_norm": 0.25,
        "model_overrides": {
            "num_res_blocks": 0,
            "use_diffeomorphic_stroke": True,
            "style_spatial_pre_gain_16": 0.0,
        },
        "bridge_overrides": {
            "swd_patch_sizes": [1, 3],
            "swd_num_projections": 16,
            "semantic_swd_num_projections": 16,
            "swd_projection_chunk_size": 8,
            "swd_cdf_sample_size": 64,
        },
        "notes": "SDXL minimal closure with conservative diffeomorphic head.",
    },
    "sdxl_s1_light_swd": {
        "vae_model": "sdxl-fp32",
        "latent_root": "latent-256-sdxl-fp32",
        "batch_size": 48,
        "eval_batch_size": 4,
        "learning_rate": 2e-5,
        "terminal_swd_weight": 2.0,
        "style_spatial_pre_gain_16": 0.15,
        "diffeomorphic_warp_strength": 0.015,
        "grad_clip_norm": 0.5,
        "notes": "SDXL first usable style pressure.",
    },
    "sdxl_s2_balanced": {
        "vae_model": "sdxl-fp32",
        "latent_root": "latent-256-sdxl-fp32",
        "batch_size": 48,
        "eval_batch_size": 4,
        "learning_rate": 3e-5,
        "terminal_swd_weight": 5.0,
        "style_spatial_pre_gain_16": 0.22,
        "diffeomorphic_warp_strength": 0.02,
        "grad_clip_norm": 0.75,
        "notes": "SDXL balanced stage, close to t01 but numerically softened.",
    },
    "sdxl_s3_style_push": {
        "vae_model": "sdxl-fp32",
        "latent_root": "latent-256-sdxl-fp32",
        "batch_size": 40,
        "eval_batch_size": 4,
        "learning_rate": 4e-5,
        "terminal_swd_weight": 8.0,
        "style_spatial_pre_gain_16": 0.28,
        "diffeomorphic_warp_strength": 0.03,
        "grad_clip_norm": 0.75,
        "notes": "SDXL upper style-pressure probe.",
    },
    "flux1": {
        "vae_model": "flux1-schnell",
        "latent_root": "latent-256-flux1",
        "batch_size": 24,
        "eval_batch_size": 3,
        "learning_rate": 3e-5,
        "terminal_swd_weight": 4.0,
    },
    "flux2": {
        "vae_model": "flux2-klein",
        "latent_root": "latent-256-flux2",
        "batch_size": 12,
        "eval_batch_size": 2,
        "learning_rate": 2e-5,
        "terminal_swd_weight": 3.0,
    },
}


def _sdxl_t01_variant(*, batch_size: int, eval_batch_size: int = 8, name: str = "") -> dict:
    return {
        "vae_model": "sdxl-fp32",
        "latent_root": "latent-256-sdxl-fp32",
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": 6e-5,
        "terminal_swd_weight": 10.0,
        "style_spatial_pre_gain_16": 0.30,
        "diffeomorphic_warp_strength": 0.02,
        "grad_clip_norm": 0.75,
        "bridge_overrides": {
            "swd_patch_sizes": [3, 5, 7],
            "swd_num_projections": 32,
            "semantic_swd_num_projections": 32,
            "swd_projection_chunk_size": 16,
            "swd_cdf_sample_size": 128,
        },
        "notes": name or "SDXL t01-style recovered candidate.",
    }


for _bs in (96, 128, 160, 192):
    _v = _sdxl_t01_variant(batch_size=_bs, eval_batch_size=4, name=f"SDXL memory ladder batch={_bs}.")
    _v["training_overrides"] = {"max_train_batches_per_epoch": 30}
    VARIANTS[f"sdxl_mem_b{_bs}"] = _v

VARIANTS.update(
    {
        "sdxl_t01_recover": _sdxl_t01_variant(batch_size=128, eval_batch_size=8, name="SDXL recovered t01 geometry, conservative style pressure."),
        "sdxl_style_push": {
            **_sdxl_t01_variant(batch_size=128, eval_batch_size=8, name="SDXL high style pressure with texton patches."),
            "learning_rate": 8e-5,
            "terminal_swd_weight": 20.0,
            "style_spatial_pre_gain_16": 0.35,
            "diffeomorphic_warp_strength": 0.03,
        },
        "sdxl_content_guard": {
            **_sdxl_t01_variant(batch_size=128, eval_batch_size=8, name="SDXL style push with lower tangent warp for content preservation."),
            "learning_rate": 6e-5,
            "terminal_swd_weight": 16.0,
            "style_spatial_pre_gain_16": 0.26,
            "diffeomorphic_warp_strength": 0.012,
        },
        "sdxl_t01_fullish": {
            **_sdxl_t01_variant(batch_size=128, eval_batch_size=8, name="SDXL near-t01 full pressure probe."),
            "learning_rate": 1e-4,
            "terminal_swd_weight": 20.0,
            "style_spatial_pre_gain_16": 0.35,
            "diffeomorphic_warp_strength": 0.03,
            "bridge_overrides": {
                "swd_patch_sizes": [3, 5, 7, 15],
                "swd_num_projections": 32,
                "semantic_swd_num_projections": 32,
                "swd_projection_chunk_size": 16,
                "swd_cdf_sample_size": 128,
            },
        },
    }
)


def _query_gpu_used_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    values: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line.split(",")[0].strip())))
        except ValueError:
            continue
    return max(values) if values else None


def _run(cmd: list[str], log_path: Path, cwd: Path) -> tuple[int, int | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    src_path = str((ROOT / "src").resolve())
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    peak_gpu_mb = _query_gpu_used_mb()
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n\n>>> " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        while proc.poll() is None:
            used = _query_gpu_used_mb()
            if used is not None:
                peak_gpu_mb = used if peak_gpu_mb is None else max(peak_gpu_mb, used)
            time.sleep(2.0)
        used = _query_gpu_used_mb()
        if used is not None:
            peak_gpu_mb = used if peak_gpu_mb is None else max(peak_gpu_mb, used)
        log.write(f"\n<<< exit={proc.returncode}\n")
        log.write(f"<<< peak_gpu_memory_mb={peak_gpu_mb if peak_gpu_mb is not None else ''}\n")
        return int(proc.returncode), peak_gpu_mb


def _infer_latent_shape(latent_root: Path) -> tuple[int, int, int]:
    for path in latent_root.rglob("*.pt"):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj.get("latent", obj.get("z", obj))
        shape = tuple(obj.shape)
        if len(shape) == 4 and shape[0] == 1:
            shape = shape[1:]
        if len(shape) != 3:
            raise ValueError(f"Expected latent [C,H,W], got {shape} from {path}")
        return int(shape[0]), int(shape[1]), int(shape[2])
    raise FileNotFoundError(f"No latent .pt files found under {latent_root}")


def _write_config(
    base_cfg: dict,
    out_dir: Path,
    latent_root: Path,
    scale: float,
    variant: dict,
    epochs: int,
    max_train_batches: int = 0,
    allow_missing_latent_shape: bool = False,
) -> Path:
    cfg = json.loads(json.dumps(base_cfg))
    try:
        latent_channels, latent_h, latent_w = _infer_latent_shape(latent_root)
    except FileNotFoundError:
        if not allow_missing_latent_shape:
            raise
        latent_channels = int(cfg.get("model", {}).get("latent_channels", 4))
        latent_h = int(cfg.get("model", {}).get("latent_size", 32))
        latent_w = latent_h
    cfg["model"]["latent_channels"] = int(latent_channels)
    cfg["model"]["latent_scale_factor"] = float(scale)
    cfg["model"]["style_attn_num_tokens"] = int(max(64, min(256, (latent_h * latent_w) // 4)))
    for key in (
        "style_spatial_pre_gain_16",
        "diffeomorphic_warp_strength",
        "diffeomorphic_color_strength",
        "diffeomorphic_texture_gate_strength",
        "diffeomorphic_normal_leak",
        "use_diffeomorphic_stroke",
        "zero_init_output_head",
    ):
        if key in variant:
            cfg["model"][key] = variant[key]
    for key, value in dict(variant.get("model_overrides", {}) or {}).items():
        cfg["model"][key] = value
    cfg["data"]["data_root"] = str(latent_root.resolve())
    train = cfg["training"]
    train["batch_size"] = int(variant["batch_size"])
    train["num_epochs"] = int(epochs)
    train["save_interval"] = 1
    train["full_eval_batch_size"] = int(variant["eval_batch_size"])
    train["num_workers"] = 0
    train["full_eval_enable_art_fid"] = False
    train["full_eval_enable_kid"] = False
    train["learning_rate"] = float(variant.get("learning_rate", train.get("learning_rate", 2e-4)))
    train["min_learning_rate"] = min(float(train["learning_rate"]) * 0.05, float(train.get("min_learning_rate", 1e-5)))
    train["grad_clip_norm"] = float(variant.get("grad_clip_norm", train.get("grad_clip_norm", 1.0)))
    train["use_amp"] = False
    train["amp_dtype"] = "bf16"
    train["channels_last"] = False
    train["use_gradient_checkpointing"] = False
    train["numeric_debug"] = True
    train["numeric_debug_interval"] = 1
    train["numeric_debug_halt_on_nonfinite"] = True
    for key, value in dict(variant.get("training_overrides", {}) or {}).items():
        train[key] = value
    if int(max_train_batches) > 0:
        train["max_train_batches_per_epoch"] = int(max_train_batches)
    train["full_eval_cache_dir"] = str((ROOT.parent / "eval_cache").resolve())
    train["full_eval_clip_hf_cache_dir"] = str((ROOT.parent / "eval_cache" / "hf").resolve())
    train["full_eval_image_classifier_path"] = str((ROOT.parent / "eval_cache" / "eval_style_image_classifier.pt").resolve())
    cfg["checkpoint"]["save_dir"] = str(out_dir.resolve())
    cfg["data"]["preload_to_gpu"] = False
    cfg["bridge"]["terminal_swd_weight"] = float(variant.get("terminal_swd_weight", cfg["bridge"].get("terminal_swd_weight", 20.0)))
    cfg["bridge"]["swd_cdf_sample_size"] = min(int(cfg["bridge"].get("swd_cdf_sample_size", 256)), 128)
    cfg["bridge"]["swd_projection_chunk_size"] = min(int(cfg["bridge"].get("swd_projection_chunk_size", 32)), 16)
    cfg["bridge"]["semantic_swd_num_projections"] = min(int(cfg["bridge"].get("semantic_swd_num_projections", 64)), 32)
    cfg["bridge"]["swd_num_projections"] = min(int(cfg["bridge"].get("swd_num_projections", 64)), 32)
    for key, value in dict(variant.get("bridge_overrides", {}) or {}).items():
        cfg["bridge"][key] = value
    cfg["inference"] = {
        "num_steps": 12,
        "step_size": 1.0,
        "style_strength": 1.0,
    }
    cfg["full_eval"] = {
        "num_steps": 12,
        "step_size": 1.0,
        "style_strength": 1.0,
        "batch_size": int(variant["eval_batch_size"]),
        "max_src_samples": 30,
        "max_ref_compare": 24,
        "max_ref_cache": 80,
        "ref_feature_batch_size": 8,
    }
    cfg["ablation"] = {
        "name": out_dir.name,
        "axis": "vae_backend_256",
        "notes": str(variant.get("notes", "VAE backend probe generated from t01 SD15 baseline.")),
        "stage": "vae_backend_256",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "config.json"
    dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return dst


def _summary_row(
    exp_dir: Path,
    name: str,
    epoch: int,
    status: str,
    seconds: float,
    scale: float,
    vae_model: str,
    variant: dict | None = None,
    peak_gpu_memory_mb: int | None = None,
) -> dict:
    summary = exp_dir / "full_eval" / f"epoch_{epoch:04d}" / "summary.json"
    variant = variant or {}
    training_overrides = dict(variant.get("training_overrides", {}) or {})
    row = {
        "variant": name,
        "vae_model": vae_model,
        "vae_scaling_factor": scale,
        "batch_size": variant.get("batch_size", ""),
        "eval_batch_size": variant.get("eval_batch_size", ""),
        "max_train_batches_per_epoch": training_overrides.get("max_train_batches_per_epoch", ""),
        "peak_gpu_memory_mb": peak_gpu_memory_mb if peak_gpu_memory_mb is not None else "",
        "epoch": epoch,
        "status": status,
        "seconds": round(seconds, 2),
        "summary": str(summary),
        "clip_style": "",
        "content_lpips": "",
        "ec": "",
    }
    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        overview = ((payload.get("analysis") or {}).get("all_pairs_overview") or {})
        style = overview.get("clip_style")
        lpips = overview.get("content_lpips")
        row["clip_style"] = style
        row["content_lpips"] = lpips
        if isinstance(style, (int, float)) and isinstance(lpips, (int, float)):
            row["ec"] = float(style) * (1.0 - float(lpips))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 256x256 latent VAE backend probes: encode, train, eval.")
    parser.add_argument("--variants", default="sdxl,flux1,flux2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--eval-epochs", default="6,7,8")
    parser.add_argument("--image-root", type=Path, default=ROOT.parent / "style_data" / "train")
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp" / "vae_backend_256_probe")
    parser.add_argument("--cache-dir", type=Path, default=ROOT.parent / "eval_cache" / "hf")
    parser.add_argument("--max-per-style", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--skip-existing-latents", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = [x.strip() for x in args.variants.split(",") if x.strip()]
    eval_epochs = []
    if str(args.eval_epochs).strip().lower() not in {"", "none", "no", "false", "0"}:
        eval_epochs = [int(x.strip()) for x in args.eval_epochs.split(",") if x.strip()]
    base_cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    rows: list[dict] = []
    args.out_root.mkdir(parents=True, exist_ok=True)
    ledger = args.out_root / "vae_backend_256_results.csv"

    def write_ledger() -> None:
        if rows:
            with ledger.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    for name in names:
        if name not in VARIANTS:
            print(f"[skip] unknown variant {name}", flush=True)
            continue
        variant = deepcopy(VARIANTS[name])
        if int(args.max_train_batches) > 0:
            training_overrides = dict(variant.get("training_overrides", {}) or {})
            training_overrides["max_train_batches_per_epoch"] = int(args.max_train_batches)
            variant["training_overrides"] = training_overrides
        start = time.time()
        exp_dir = args.out_root / name
        log = exp_dir / "run.log"
        latent_root = ROOT.parent / str(variant["latent_root"])
        vae_model = str(variant["vae_model"])
        manifest = latent_root / "manifest.json"
        status = "ok"
        scale = 0.18215

        if not (args.skip_existing_latents and manifest.exists()):
            cmd = [
                sys.executable,
                str(ROOT / "tools" / "experiments" / "preprocess_latents_vae_variant.py"),
                "--image-root",
                str(args.image_root),
                "--output-root",
                str(latent_root),
                "--vae-model",
                vae_model,
                "--cache-dir",
                str(args.cache_dir),
                "--size",
                "256",
                "--batch-size",
                str(max(1, min(8, int(variant["eval_batch_size"]) * 2))),
            ]
            if int(args.max_per_style) > 0:
                cmd += ["--max-per-style", str(int(args.max_per_style))]
            if args.dry_run:
                print("[dry-run]", " ".join(cmd))
            else:
                rc, peak = _run(cmd, log, ROOT)
                if rc != 0:
                    rows.append(_summary_row(exp_dir, name, 0, f"encode_failed_{rc}", time.time() - start, scale, vae_model, variant, peak))
                    write_ledger()
                    continue

        if manifest.exists():
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            scale = float(meta.get("vae_scaling_factor", scale))
        config_path = _write_config(
            base_cfg,
            exp_dir,
            latent_root,
            scale,
            variant,
            int(args.epochs),
            int(args.max_train_batches),
            allow_missing_latent_shape=bool(args.dry_run),
        )
        shutil.copy2(ROOT / "src" / "utils" / "inference.py", exp_dir / "inference_snapshot.py")
        shutil.copy2(ROOT / "src" / "utils" / "run_evaluation.py", exp_dir / "run_evaluation_snapshot.py")

        train_cmd = [sys.executable, str(ROOT / "src" / "run.py"), "--config", str(config_path)]
        peak = None
        if args.dry_run:
            print("[dry-run]", " ".join(train_cmd))
        else:
            rc, peak = _run(train_cmd, log, ROOT)
            if rc != 0:
                rows.append(_summary_row(exp_dir, name, 0, f"train_failed_{rc}", time.time() - start, scale, vae_model, variant, peak))
                write_ledger()
                continue
        if not eval_epochs:
            rows.append(_summary_row(exp_dir, name, int(args.epochs), "train_ok", time.time() - start, scale, vae_model, variant, peak))
            write_ledger()

        for epoch in eval_epochs:
            ckpt = exp_dir / f"epoch_{epoch:04d}.pt"
            if not ckpt.exists():
                rows.append(_summary_row(exp_dir, name, epoch, "missing_ckpt", time.time() - start, scale, vae_model, variant, peak))
                write_ledger()
                continue
            eval_dir = exp_dir / "full_eval" / f"epoch_{epoch:04d}"
            eval_cmd = [
                sys.executable,
                str(ROOT / "src" / "utils" / "run_evaluation.py"),
                "--checkpoint",
                str(ckpt),
                "--output",
                str(eval_dir),
                "--batch_size",
                str(int(variant["eval_batch_size"])),
                "--vae_model",
                vae_model,
                "--vae_decode_scale",
                str(scale),
                "--num_steps",
                "12",
                "--step_size",
                "1.0",
                "--style_strength",
                "1.0",
                "--max_src_samples",
                "30",
            ]
            if args.dry_run:
                print("[dry-run]", " ".join(eval_cmd))
                status = "dry_run"
            else:
                rc, eval_peak = _run(eval_cmd, log, ROOT)
                if eval_peak is not None:
                    peak = eval_peak if peak is None else max(peak, eval_peak)
                status = "ok" if rc == 0 else f"eval_failed_{rc}"
            rows.append(_summary_row(exp_dir, name, epoch, status, time.time() - start, scale, vae_model, variant, peak))
            write_ledger()

    write_ledger()
    print(ledger.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
