"""Local one-click smoke + train + eval entrypoint for clean_base_v2_local.

Usage:
  python tools/local_train_and_eval.py --config configs/clean_base_v2_local.json
  python tools/local_train_and_eval.py --config <cfg> --skip-train --checkpoint <path>
  python tools/local_train_and_eval.py --config <cfg> --skip-eval
  python tools/local_train_and_eval.py --config <cfg> --smoke-only

Modes:
  --smoke-only  : Run a single forward+backward+optimizer step on a fake batch.
  default       : Run smoke -> train -> eval (or subset with --skip-train / --skip-eval).

Baseline reference (4070 Laptop, 2026-06-30):
  allpairs clip_style   = 0.7293   (pass >= 0.7243)
  allpairs content_lpips = 0.3203   (pass <= 0.3453)
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

# Baseline thresholds (4070 Laptop, clean_base_v2_local epoch_0010)
BASELINE_CLIP_STYLE = 0.7293
BASELINE_CONTENT_LPIPS = 0.3203
CLIP_STYLE_MIN = BASELINE_CLIP_STYLE - 0.005  # 5σ
CONTENT_LPIPS_MAX = BASELINE_CONTENT_LPIPS + 0.025


def _add_src_to_path() -> None:
    src_str = str(SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def run_smoke(config_path: Path) -> int:
    """Single forward+backward+optimizer step on a fake batch. Returns exit code."""
    _add_src_to_path()
    import torch
    from config_schema import load_experiment_config
    from model import build_model_from_config
    from spectral_losses620 import SpectralODEObjective620

    try:
        print("[smoke] Loading config...", flush=True)
        cfg = load_experiment_config(str(config_path))
        print(f"[smoke] contract_family={cfg.model.contract_family}", flush=True)

        print("[smoke] Building model...", flush=True)
        model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).to("cuda")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[smoke] Model params: {n_params:,}", flush=True)

        print("[smoke] Building loss fn...", flush=True)
        loss_fn = SpectralODEObjective620(cfg)
        print(f"[smoke] loss_fn type: {type(loss_fn).__name__}", flush=True)

        print("[smoke] Building optimizer...", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

        print("[smoke] Creating fake batch...", flush=True)
        B, C, H, W = 4, 4, 64, 64
        content = torch.randn(B, C, H, W, device="cuda")
        target_style = torch.randn(B, C, H, W, device="cuda")
        target_style_id = torch.randint(0, 5, (B,), device="cuda")
        batch = {
            "content": content,
            "target_style": target_style,
            "target_style_id": target_style_id,
            "source_style_id": torch.randint(0, 5, (B,), device="cuda"),
        }

        print("[smoke] Forward pass...", flush=True)
        model.train()
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            loss_dict = loss_fn.compute(
                model,
                content=content,
                target_style=target_style,
                target_style_id=target_style_id,
                conditioning=batch,
            )
            loss = loss_dict["loss"]
        print(f"[smoke] loss={loss.item():.6f}", flush=True)

        print("[smoke] Backward pass...", flush=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        print("[smoke] Backward OK", flush=True)

        print("[smoke] Optimizer step...", flush=True)
        opt.step()
        print("[smoke] Optimizer step OK", flush=True)

        print("[smoke] GPU memory:", flush=True)
        print(f"[smoke]   allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB", flush=True)
        print(f"[smoke]   reserved:  {torch.cuda.memory_reserved()/1e6:.1f} MB", flush=True)

        print("[smoke] ALL PASS", flush=True)
        return 0
    except Exception as e:
        print(f"[smoke] FAILED: {e}", flush=True)
        traceback.print_exc()
        return 1


def _run_subprocess(cmd: list[str], cwd: Path) -> int:
    """Run a subprocess with real-time stdout/stderr passthrough. Returns exit code."""
    print(f"[run] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def _find_latest_checkpoint(exp_dir: Path) -> Path | None:
    """Find the latest epoch_*.pt checkpoint in exp_dir."""
    if not exp_dir.exists():
        return None
    ckpts = sorted(exp_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def run_train(config_path: Path) -> int:
    """Invoke src/run.py for training. Returns exit code."""
    cmd = [sys.executable, str(SRC_DIR / "run.py"), "--config", str(config_path)]
    return _run_subprocess(cmd, REPO_ROOT)


def run_eval(config_path: Path, checkpoint: Path, output_dir: Path | None = None) -> tuple[int, Path | None]:
    """Invoke src/utils/run_evaluation.py for evaluation. Returns (exit_code, eval_dir)."""
    if output_dir is None:
        eval_dir = checkpoint.parent / "full_eval" / f"{checkpoint.stem}_full"
    else:
        eval_dir = output_dir
    cmd = [
        sys.executable,
        str(SRC_DIR / "utils" / "run_evaluation.py"),
        "--checkpoint", str(checkpoint),
        "--output", str(eval_dir),
        "--eval_only_lpips_clip_style",
    ]
    rc = _run_subprocess(cmd, REPO_ROOT)
    return rc, eval_dir if rc == 0 else None


def compute_allpairs_mean(metrics_csv: Path) -> tuple[float, float, int] | None:
    """Compute allpairs mean of clip_style and content_lpips from metrics.csv."""
    if not metrics_csv.exists():
        return None
    with open(metrics_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cs = [float(r["clip_style"]) for r in rows if r.get("clip_style")]
    cl = [float(r["content_lpips"]) for r in rows if r.get("content_lpips")]
    if not cs or not cl:
        return None
    return sum(cs) / len(cs), sum(cl) / len(cl), len(rows)


def print_baseline_comparison(clip_style: float, content_lpips: float, n: int) -> bool:
    """Print PASS/FAIL vs baseline. Returns True if both metrics pass."""
    clip_pass = clip_style >= CLIP_STYLE_MIN
    lpips_pass = content_lpips <= CONTENT_LPIPS_MAX
    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"[eval] allpairs n={n}", flush=True)
    print(f"[eval] clip_style    = {clip_style:.4f}  (baseline {BASELINE_CLIP_STYLE:.4f}, min {CLIP_STYLE_MIN:.4f})  {'PASS' if clip_pass else 'FAIL'}", flush=True)
    print(f"[eval] content_lpips = {content_lpips:.4f}  (baseline {BASELINE_CONTENT_LPIPS:.4f}, max {CONTENT_LPIPS_MAX:.4f})  {'PASS' if lpips_pass else 'FAIL'}", flush=True)
    print("=" * 60, flush=True)
    return clip_pass and lpips_pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Local one-click smoke + train + eval.")
    parser.add_argument("--config", type=str, required=True, help="Path to config json")
    parser.add_argument("--smoke-only", action="store_true", help="Only run smoke test")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (with --skip-train)")
    parser.add_argument("--output", type=str, default=None, help="Eval output directory")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[error] config not found: {config_path}", flush=True)
        return 1

    # Determine exp dir from config name (matches src/run.py convention)
    config_name = config_path.stem
    exp_dir = REPO_ROOT / "exp" / config_name

    # Step 1: smoke test (always run unless explicitly skipped via --skip-train + --skip-eval)
    if args.smoke_only:
        return run_smoke(config_path)

    print("[step 1] smoke test", flush=True)
    rc = run_smoke(config_path)
    if rc != 0:
        print("[step 1] smoke FAILED, aborting", flush=True)
        return rc
    print("[step 1] smoke PASS", flush=True)

    # Step 2: train
    checkpoint: Path | None = None
    if not args.skip_train:
        print("[step 2] training", flush=True)
        rc = run_train(config_path)
        if rc != 0:
            print("[step 2] training FAILED, aborting", flush=True)
            return rc
        checkpoint = _find_latest_checkpoint(exp_dir)
        if checkpoint is None:
            print(f"[step 2] no checkpoint found in {exp_dir}", flush=True)
            return 1
        print(f"[step 2] latest checkpoint: {checkpoint}", flush=True)
    else:
        if args.checkpoint:
            checkpoint = Path(args.checkpoint).resolve()
            if not checkpoint.exists():
                print(f"[error] checkpoint not found: {checkpoint}", flush=True)
                return 1
            print(f"[step 2] skipped, using provided checkpoint: {checkpoint}", flush=True)
        else:
            checkpoint = _find_latest_checkpoint(exp_dir)
            if checkpoint is None:
                print(f"[error] --skip-train requires --checkpoint or existing checkpoint in {exp_dir}", flush=True)
                return 1
            print(f"[step 2] skipped, using latest checkpoint: {checkpoint}", flush=True)

    # Step 3: eval
    if not args.skip_eval:
        print("[step 3] evaluation", flush=True)
        output_dir = Path(args.output).resolve() if args.output else None
        rc, eval_dir = run_eval(config_path, checkpoint, output_dir)
        if rc != 0:
            print("[step 3] eval FAILED", flush=True)
            return rc
        if eval_dir is None:
            print("[step 3] eval_dir unknown, cannot compute metrics", flush=True)
            return 1
        metrics_csv = eval_dir / "metrics.csv"
        result = compute_allpairs_mean(metrics_csv)
        if result is None:
            print(f"[step 3] cannot read metrics from {metrics_csv}", flush=True)
            return 1
        clip_style, content_lpips, n = result
        ok = print_baseline_comparison(clip_style, content_lpips, n)
        return 0 if ok else 2

    print("[done] smoke + train completed (eval skipped)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
