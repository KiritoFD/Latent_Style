from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DEFAULT_BASE_CONFIG = ROOT / "exp" / "diffeomorphic_tangent_sweep" / "t01_ws0p03_g6_nl0p05" / "config.json"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "t01_patch1_probe"
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "t01_patch1_probe"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(" ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _make_config(base_cfg: dict[str, Any], *, save_dir: Path, num_epochs: int, batch_size: int) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    bridge = cfg.setdefault("bridge", {})
    bridge["swd_patch_sizes"] = [1]
    bridge["swd_micro_patch_max"] = 1
    bridge["swd_macro_patch_min"] = 999
    bridge["swd_micro_weight"] = 1.0
    bridge["swd_macro_weight"] = 1.0
    bridge["swd_scale_invariant_patches"] = False
    bridge["swd_use_dilated_projections"] = False
    bridge["swd_projection_dilation"] = 2

    training = cfg.setdefault("training", {})
    training["num_epochs"] = int(num_epochs)
    training["save_interval"] = 1
    training["batch_size"] = int(batch_size)

    checkpoint = cfg.setdefault("checkpoint", {})
    checkpoint["save_dir"] = "./" + save_dir.resolve().relative_to(ROOT).as_posix()

    cfg["ablation"] = {
        "name": "p36_k1_only",
        "stage": "t01_patch1_probe",
        "axis": "swd_patch_design",
        "patch_sizes": [1],
        "micro_patch_max": 1,
        "macro_patch_min": 999,
        "note": "Extreme pointwise SWD control: patch size 1 only.",
    }
    return cfg


def _metrics_from_summary(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"clip_style": None, "clip_content": None, "content_lpips": None, "clip_dir": None}
    payload = _load_json(path)
    overview = ((payload.get("analysis") or {}).get("all_pairs_overview") or {})
    return {
        "clip_style": overview.get("clip_style"),
        "clip_content": overview.get("clip_content"),
        "content_lpips": overview.get("content_lpips"),
        "clip_dir": overview.get("clip_dir"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a one-off t01 patch-size-1 SWD probe.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-epochs", default="6,7,8")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-num-steps", type=int, default=12)
    parser.add_argument("--eval-step-size", type=float, default=1.0)
    parser.add_argument("--eval-vae-decode-scale", type=float, default=0.197)
    parser.add_argument("--eval-residual-scale", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = args.output_root.resolve() / "p36_k1_only"
    config_path = args.config_root.resolve() / "p36_k1_only.json"
    cfg = _make_config(
        _load_json(args.base_config.resolve()),
        save_dir=run_dir,
        num_epochs=args.num_epochs,
        batch_size=args.train_batch_size,
    )
    _write_json(config_path, cfg)

    eval_epochs = tuple(sorted({int(x) for x in args.eval_epochs.split(",") if x.strip()}))
    checkpoints = [run_dir / f"epoch_{epoch:04d}.pt" for epoch in eval_epochs]
    if args.force_train or not all(path.exists() for path in checkpoints):
        _run([sys.executable, "src/run.py", "--config", str(config_path)], cwd=ROOT, dry_run=args.dry_run)

    rows: list[dict[str, Any]] = []
    for epoch in eval_epochs:
        ckpt = run_dir / f"epoch_{epoch:04d}.pt"
        eval_dir = run_dir / "full_eval" / f"epoch_{epoch:04d}"
        summary_path = eval_dir / "summary.json"
        if args.force_eval or not summary_path.exists():
            _run(
                [
                    sys.executable,
                    "src/utils/run_evaluation.py",
                    "--checkpoint",
                    str(ckpt),
                    "--output",
                    str(eval_dir),
                    "--num_steps",
                    str(args.eval_num_steps),
                    "--step_size",
                    str(args.eval_step_size),
                    "--vae_decode_scale",
                    str(args.eval_vae_decode_scale),
                    "--residual_scale",
                    str(args.eval_residual_scale),
                ],
                cwd=ROOT,
                dry_run=args.dry_run,
            )
        metrics = _metrics_from_summary(summary_path)
        lpips = metrics.get("content_lpips")
        style = metrics.get("clip_style")
        rows.append(
            {
                "name": "p36_k1_only",
                "epoch": epoch,
                "patch_sizes": "1",
                **metrics,
                "ec": (float(style) * (1.0 - float(lpips))) if style is not None and lpips is not None else None,
                "summary": summary_path.as_posix(),
            }
        )

    out_csv = args.output_root.resolve() / "patch1_probe_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "epoch", "patch_sizes", "clip_style", "clip_content", "content_lpips", "clip_dir", "ec", "summary"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"results": str(out_csv), "rows": rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
