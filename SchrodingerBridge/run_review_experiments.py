from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = ROOT / "S-add__K-1_C-0_W-20_Col-0"
DEFAULT_BASE_CONFIG = DEFAULT_BASE_DIR / "config.json"
DEFAULT_BASE_CKPT = DEFAULT_BASE_DIR / "epoch_0007.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "review_additional_experiments"


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _run(cmd: list[str], *, log_path: Path, cwd: Path = ROOT, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = "$ " + " ".join(str(x) for x in cmd)
    print(line, flush=True)
    if dry_run:
        with log_path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
        return 0
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n[{_timestamp()}]\n{line}\n")
        f.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode


def _summary_exists(out_dir: Path) -> bool:
    return (out_dir / "summary.json").is_file() or (out_dir / "batch_summary.csv").is_file()


def _checkpoint_for_run(run_dir: Path, num_epochs: int) -> Path:
    return run_dir / f"epoch_{int(num_epochs):04d}.pt"


def _sanitize_float_name(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _ensure_src_on_path() -> None:
    src_dir = ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _count_params_from_ckpt(ckpt_path: Path) -> tuple[int, dict[str, Any]]:
    import torch
    from model import build_model_from_config, count_parameters

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_cfg = config.get("model", {})
    model = build_model_from_config(model_cfg, use_checkpointing=False)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return int(count_parameters(model)), config


def _try_profile_flops(model, batch_size: int) -> dict[str, Any]:
    import torch

    dummy_x = torch.randn(batch_size, 4, 32, 32)
    dummy_style = torch.zeros(batch_size, dtype=torch.long)
    result: dict[str, Any] = {"macs": None, "flops": None, "backend": None}

    try:
        from thop import profile  # type: ignore

        macs, params = profile(model, inputs=(dummy_x, None, 0.5, dummy_style), verbose=False)
        result["macs"] = float(macs)
        result["flops"] = float(macs) * 2.0
        result["params_from_backend"] = float(params)
        result["backend"] = "thop"
        return result
    except Exception:
        pass

    try:
        from ptflops import get_model_complexity_info  # type: ignore

        def _input_constructor(_: tuple[int, int, int]) -> dict[str, Any]:
            return {
                "x": dummy_x,
                "source": None,
                "t": 0.5,
                "style_id": dummy_style,
            }

        macs, _ = get_model_complexity_info(
            model,
            (4, 32, 32),
            input_constructor=_input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        result["macs"] = float(macs)
        result["flops"] = float(macs) * 2.0
        result["backend"] = "ptflops"
        return result
    except Exception:
        return result


def _profile_checkpoint(
    *,
    ckpt_path: Path,
    output_path: Path,
    step_values: list[int],
    batch_sizes: list[int],
    warmup_iters: int,
    measure_iters: int,
    step_size: float,
    style_strength: float | None,
    residual_scale: float,
    dry_run: bool,
) -> None:
    if dry_run:
        payload = {
            "timestamp": _timestamp(),
            "checkpoint": str(ckpt_path),
            "status": "dry_run",
            "notes": "Efficiency profiling skipped in dry-run mode.",
        }
        _write_json(output_path, payload)
        return

    _ensure_src_on_path()
    import torch
    from utils.inference import LGTInference

    device = "cuda" if torch.cuda.is_available() else "cpu"
    params, config = _count_params_from_ckpt(ckpt_path)
    inf = LGTInference(
        str(ckpt_path),
        device=device,
        num_steps=max(step_values) if step_values else 12,
        step_size=step_size,
        style_strength=style_strength,
        residual_scale=residual_scale,
    )
    model = inf.model.eval()

    if device == "cuda":
        model = model.to("cuda")

    flop_info = _try_profile_flops(model, batch_size=1)
    objective_mode = str(config.get("bridge", {}).get("objective_mode", "")).lower()
    batch_records: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        dummy_x = torch.randn(batch_size, 4, 32, 32, device=device)
        dummy_style = torch.arange(batch_size, device=device) % max(1, int(config.get("model", {}).get("num_styles", 5)))

        for num_steps in step_values:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            with torch.inference_mode():
                for _ in range(max(0, warmup_iters)):
                    _ = inf.generation(dummy_x, dummy_style, num_steps=num_steps)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(max(1, measure_iters)):
                    _ = inf.generation(dummy_x, dummy_style, num_steps=num_steps)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

            imgs = max(1, measure_iters) * batch_size
            batch_records.append(
                {
                    "batch_size": int(batch_size),
                    "num_steps": int(num_steps),
                    "elapsed_sec": float(elapsed),
                    "avg_sec_per_iter": float(elapsed / max(1, measure_iters)),
                    "avg_sec_per_img": float(elapsed / imgs),
                    "throughput_img_per_sec": float(imgs / max(elapsed, 1e-8)),
                    "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024.0**2)) if device == "cuda" else 0.0,
                    "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024.0**2)) if device == "cuda" else 0.0,
                }
            )

    payload = {
        "timestamp": _timestamp(),
        "checkpoint": str(ckpt_path),
        "device": device,
        "measurement_scope": "latent_generator_only",
        "latent_shape": [4, 32, 32],
        "objective_mode": objective_mode,
        "notes": [
            "Params are exact for the transport network only.",
            "FLOPs/MACs are best-effort and may be null if no profiling backend is installed.",
            "Throughput and VRAM are measured on dummy 32x32 latent tensors without VAE encode/decode.",
        ],
        "params": int(params),
        "macs": flop_info.get("macs"),
        "flops": flop_info.get("flops"),
        "flops_backend": flop_info.get("backend"),
        "records": batch_records,
    }
    _write_json(output_path, payload)


def _run_step_sweep(args: argparse.Namespace, output_root: Path, stage_rows: list[dict[str, Any]]) -> None:
    out_root = output_root / "step_count_sweep"
    log_root = out_root / "logs"
    rows: list[dict[str, Any]] = []

    for steps in args.step_values:
        out_dir = out_root / f"steps_{int(steps):02d}"
        log_path = log_root / f"steps_{int(steps):02d}.log"
        should_skip = args.skip_existing and _summary_exists(out_dir)
        row = {
            "stage": "step_sweep",
            "num_steps": int(steps),
            "checkpoint": str(args.base_checkpoint),
            "output_dir": str(out_dir),
            "status": "pending",
            "returncode": "",
            "elapsed_sec": "",
            "summary_exists": _summary_exists(out_dir),
            "note": "",
        }
        t0 = time.perf_counter()
        if should_skip:
            row["status"] = "skipped_existing"
            row["returncode"] = 0
        else:
            cmd = [
                sys.executable,
                "run_evaluation.py",
                str(args.base_checkpoint),
                "--output",
                str(out_dir),
                "--batch_size",
                str(args.eval_batch_size),
                "--num_steps",
                str(steps),
                "--step_size",
                str(args.eval_step_size),
            ]
            rc = _run(cmd, log_path=log_path, dry_run=args.dry_run)
            row["status"] = "ok" if rc == 0 else "failed"
            row["returncode"] = rc
        row["elapsed_sec"] = f"{time.perf_counter() - t0:.3f}"
        row["summary_exists"] = _summary_exists(out_dir) or args.dry_run
        rows.append(row)
        stage_rows.append(row.copy())

    _write_csv(
        out_root / "status.csv",
        rows,
        ["stage", "num_steps", "checkpoint", "output_dir", "status", "returncode", "elapsed_sec", "summary_exists", "note"],
    )


def _build_lambda_grid_config(
    *,
    base_config: dict[str, Any],
    kin: float,
    swd: float,
    run_dir: Path,
    num_epochs: int,
    full_eval_batch_size: int,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_config))
    cfg.setdefault("bridge", {})
    cfg["bridge"]["w_kinetic"] = float(kin)
    cfg["bridge"]["terminal_swd_weight"] = float(swd)
    cfg.setdefault("training", {})
    cfg["training"]["num_epochs"] = int(num_epochs)
    cfg["training"]["full_eval_batch_size"] = int(full_eval_batch_size)
    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["save_dir"] = str(run_dir)
    cfg.setdefault("ablation", {})
    cfg["ablation"]["name"] = f"review_kin_{kin}_swd_{swd}"
    cfg["ablation"]["axis"] = "review_lambda_grid"
    cfg["ablation"]["notes"] = f"Review lambda grid | w_kinetic={kin} terminal_swd_weight={swd}"
    return cfg


def _run_lambda_grid(args: argparse.Namespace, output_root: Path, stage_rows: list[dict[str, Any]]) -> None:
    out_root = output_root / "lambda_grid"
    config_root = out_root / "configs"
    run_root = out_root / "runs"
    eval_root = out_root / "eval"
    log_root = out_root / "logs"
    base_config = _load_json(args.base_config)
    rows: list[dict[str, Any]] = []

    for kin in args.lambda_kin_values:
        for swd in args.lambda_swd_values:
            exp_name = f"kin_{_sanitize_float_name(kin)}__swd_{_sanitize_float_name(swd)}"
            run_dir = run_root / exp_name
            eval_dir = eval_root / exp_name
            ckpt_path = _checkpoint_for_run(run_dir, args.lambda_num_epochs)
            config_path = config_root / f"{exp_name}.json"
            config_payload = _build_lambda_grid_config(
                base_config=base_config,
                kin=kin,
                swd=swd,
                run_dir=run_dir,
                num_epochs=args.lambda_num_epochs,
                full_eval_batch_size=args.eval_batch_size,
            )
            _write_json(config_path, config_payload)

            row = {
                "stage": "lambda_grid",
                "experiment": exp_name,
                "w_kinetic": float(kin),
                "terminal_swd_weight": float(swd),
                "config_path": str(config_path),
                "run_dir": str(run_dir),
                "checkpoint": str(ckpt_path),
                "eval_dir": str(eval_dir),
                "train_status": "pending",
                "train_rc": "",
                "train_elapsed_sec": "",
                "eval_status": "pending",
                "eval_rc": "",
                "eval_elapsed_sec": "",
                "checkpoint_exists": ckpt_path.exists(),
                "summary_exists": _summary_exists(eval_dir),
            }

            train_should_skip = args.skip_existing and ckpt_path.exists()
            eval_should_skip = args.skip_existing and _summary_exists(eval_dir)

            t0 = time.perf_counter()
            if train_should_skip:
                row["train_status"] = "skipped_existing"
                row["train_rc"] = 0
            else:
                train_cmd = [sys.executable, "run.py", "--config", str(config_path)]
                train_rc = _run(train_cmd, log_path=log_root / f"{exp_name}.train.log", dry_run=args.dry_run)
                row["train_status"] = "ok" if train_rc == 0 else "failed"
                row["train_rc"] = train_rc
            row["train_elapsed_sec"] = f"{time.perf_counter() - t0:.3f}"
            row["checkpoint_exists"] = ckpt_path.exists() or args.dry_run

            t0 = time.perf_counter()
            if eval_should_skip:
                row["eval_status"] = "skipped_existing"
                row["eval_rc"] = 0
            elif row["checkpoint_exists"]:
                eval_cmd = [
                    sys.executable,
                    "run_evaluation.py",
                    str(run_dir),
                    "--output",
                    str(eval_dir),
                    "--batch_size",
                    str(args.eval_batch_size),
                    "--num_steps",
                    str(args.eval_num_steps),
                    "--step_size",
                    str(args.eval_step_size),
                ]
                eval_rc = _run(eval_cmd, log_path=log_root / f"{exp_name}.eval.log", dry_run=args.dry_run)
                row["eval_status"] = "ok" if eval_rc == 0 else "failed"
                row["eval_rc"] = eval_rc
            else:
                row["eval_status"] = "skipped_no_checkpoint"
            row["eval_elapsed_sec"] = f"{time.perf_counter() - t0:.3f}"
            row["summary_exists"] = _summary_exists(eval_dir) or args.dry_run

            rows.append(row)
            stage_rows.append(row.copy())

    _write_csv(
        out_root / "status.csv",
        rows,
        [
            "stage",
            "experiment",
            "w_kinetic",
            "terminal_swd_weight",
            "config_path",
            "run_dir",
            "checkpoint",
            "eval_dir",
            "train_status",
            "train_rc",
            "train_elapsed_sec",
            "eval_status",
            "eval_rc",
            "eval_elapsed_sec",
            "checkpoint_exists",
            "summary_exists",
        ],
    )


def _run_efficiency(args: argparse.Namespace, output_root: Path, stage_rows: list[dict[str, Any]]) -> None:
    out_dir = output_root / "efficiency"
    out_path = out_dir / "efficiency_profile.json"
    t0 = time.perf_counter()
    _profile_checkpoint(
        ckpt_path=args.base_checkpoint,
        output_path=out_path,
        step_values=args.step_values,
        batch_sizes=args.profile_batch_sizes,
        warmup_iters=args.profile_warmup,
        measure_iters=args.profile_iters,
        step_size=args.eval_step_size,
        style_strength=None,
        residual_scale=1.0,
        dry_run=args.dry_run,
    )
    stage_rows.append(
        {
            "stage": "efficiency",
            "checkpoint": str(args.base_checkpoint),
            "output_path": str(out_path),
            "status": "ok" if out_path.exists() or args.dry_run else "failed",
            "elapsed_sec": f"{time.perf_counter() - t0:.3f}",
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequential review experiment runner for SchrodingerBridge: step sweep -> lambda grid -> efficiency."
    )
    parser.add_argument("--base_config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base_checkpoint", type=Path, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")

    parser.add_argument("--step_values", type=int, nargs="*", default=[1, 4, 8, 12, 16])
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--eval_num_steps", type=int, default=12)
    parser.add_argument("--eval_step_size", type=float, default=1.0)

    parser.add_argument("--lambda_kin_values", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--lambda_swd_values", type=str, default="10,20,30")
    parser.add_argument("--lambda_num_epochs", type=int, default=8)

    parser.add_argument("--profile_batch_sizes", type=int, nargs="*", default=[1, 4])
    parser.add_argument("--profile_warmup", type=int, default=10)
    parser.add_argument("--profile_iters", type=int, default=50)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.base_config = args.base_config.resolve()
    args.base_checkpoint = args.base_checkpoint.resolve()
    args.output_root = args.output_root.resolve()
    args.lambda_kin_values = _parse_float_list(args.lambda_kin_values)
    args.lambda_swd_values = _parse_float_list(args.lambda_swd_values)

    if not args.base_config.exists():
        raise SystemExit(f"Base config not found: {args.base_config}")
    if not args.base_checkpoint.exists() and not args.dry_run:
        raise SystemExit(f"Base checkpoint not found: {args.base_checkpoint}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    stage_rows: list[dict[str, Any]] = []

    manifest = {
        "timestamp": _timestamp(),
        "base_config": str(args.base_config),
        "base_checkpoint": str(args.base_checkpoint),
        "output_root": str(args.output_root),
        "step_values": [int(x) for x in args.step_values],
        "lambda_kin_values": [float(x) for x in args.lambda_kin_values],
        "lambda_swd_values": [float(x) for x in args.lambda_swd_values],
        "lambda_num_epochs": int(args.lambda_num_epochs),
        "eval_batch_size": int(args.eval_batch_size),
        "eval_num_steps": int(args.eval_num_steps),
        "eval_step_size": float(args.eval_step_size),
        "profile_batch_sizes": [int(x) for x in args.profile_batch_sizes],
        "profile_warmup": int(args.profile_warmup),
        "profile_iters": int(args.profile_iters),
        "dry_run": bool(args.dry_run),
        "skip_existing": bool(args.skip_existing),
    }
    _write_json(args.output_root / "manifest.json", manifest)

    _run_step_sweep(args, args.output_root, stage_rows)
    _run_lambda_grid(args, args.output_root, stage_rows)
    _run_efficiency(args, args.output_root, stage_rows)

    _write_json(args.output_root / "run_summary.json", {"timestamp": _timestamp(), "stages": stage_rows})
    _write_csv(
        args.output_root / "run_summary.csv",
        stage_rows,
        sorted({key for row in stage_rows for key in row.keys()}),
    )
    print(f"Review experiment runner finished. Summary: {args.output_root / 'run_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
