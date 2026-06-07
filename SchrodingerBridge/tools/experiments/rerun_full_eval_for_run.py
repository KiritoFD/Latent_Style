from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_eval_script(run_dir: Path, *, code_root: str) -> Path:
    mainline_eval = (Path(__file__).resolve().parents[2] / "src" / "utils" / "run_evaluation.py").resolve()
    run_local_eval = (run_dir / "src" / "utils" / "run_evaluation.py").resolve()
    if code_root == "mainline":
        return mainline_eval
    if code_root == "run-local":
        if not run_local_eval.exists():
            raise FileNotFoundError(f"run-local eval script not found: {run_local_eval}")
        return run_local_eval
    if code_root == "mainline-on-run-local":
        if not run_local_eval.parent.exists():
            raise FileNotFoundError(f"run-local utils directory not found: {run_local_eval.parent}")
        overlay_eval = run_local_eval.parent / "_codex_fast_run_evaluation.py"
        shutil.copyfile(mainline_eval, overlay_eval)
        return overlay_eval
    raise ValueError(f"unsupported code_root={code_root}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_block(summary: dict, block_name: str) -> dict:
    analysis = summary.get("analysis") or {}
    return analysis.get(block_name) or {}


def _summary_curve_row(summary_path: Path) -> dict[str, object]:
    summary = _read_json(summary_path)
    epoch_name = summary_path.parent.name
    full = _metric_block(summary, "all_pairs_overview")
    transfer = _metric_block(summary, "style_transfer_ability")
    timings = summary.get("timings_sec") or {}
    return {
        "epoch": epoch_name,
        "full_clip_style": _safe_float(full.get("clip_style")),
        "full_content_lpips": _safe_float(full.get("content_lpips")),
        "transfer_clip_style": _safe_float(transfer.get("clip_style")),
        "transfer_content_lpips": _safe_float(transfer.get("content_lpips")),
        "wall_total_seconds": _safe_float(timings.get("wall_total")),
        "summary_path": str(summary_path),
    }


def _write_curve_csv(output_root: Path) -> None:
    rows = []
    for summary_path in sorted(output_root.glob("epoch_*/summary.json")):
        rows.append(_summary_curve_row(summary_path))
    if not rows:
        return
    fieldnames = [
        "epoch",
        "full_clip_style",
        "full_content_lpips",
        "transfer_clip_style",
        "transfer_content_lpips",
        "wall_total_seconds",
        "summary_path",
    ]
    out_path = output_root / "clip_lpips_curve.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[rerun_eval] wrote curve csv -> {out_path}")


def _run_optional_refresh(cmd: list[str]) -> None:
    print(f"[rerun_eval] refresh -> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun clip/lpips full eval for every epoch checkpoint in a run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--clip-hf-cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=2)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--save-summary-grid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-generated-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--code-root", choices=["mainline", "run-local", "mainline-on-run-local"], default="mainline")
    parser.add_argument("--output-subdir", default="full_eval")
    parser.add_argument("--epochs", type=int, nargs="*", default=None, help="Optional epoch numbers to rerun, e.g. --epochs 7 8")
    parser.add_argument("--skip-existing", action="store_true", help="Skip checkpoints whose summary.json already exists in the chosen output subdir.")
    parser.add_argument("--refresh-stage-summary", action="store_true")
    parser.add_argument("--refresh-epoch-table", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    eval_script = _resolve_eval_script(run_dir, code_root=str(args.code_root))
    checkpoints = sorted(run_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_*.pt checkpoints found under {run_dir}")
    if args.epochs:
        wanted = {int(ep) for ep in args.epochs}
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.stem.startswith("epoch_") and int(ckpt.stem.split("_")[-1]) in wanted]
        if not checkpoints:
            raise FileNotFoundError(f"No requested checkpoints found under {run_dir} for epochs={sorted(wanted)}")

    for ckpt in checkpoints:
        out_dir = run_dir / str(args.output_subdir) / ckpt.stem
        if bool(args.skip_existing) and (out_dir / "summary.json").exists():
            print(f"[rerun_eval] skip existing {ckpt.name} -> {out_dir}")
            continue
        cmd = [
            str(args.python_bin),
            str(eval_script),
            "--checkpoint",
            str(ckpt),
            "--output",
            str(out_dir),
            "--test_dir",
            str(args.test_dir),
            "--cache_dir",
            str(args.cache_dir),
            "--clip_hf_cache_dir",
            str(args.clip_hf_cache_dir),
            "--batch_size",
            str(int(args.batch_size)),
            "--vae_decode_batch_size",
            str(int(args.vae_decode_batch_size)),
            "--target_chunk_size",
            str(int(args.target_chunk_size)),
            "--eval_only_lpips_clip_style",
        ]
        if bool(args.profile_timing):
            cmd.append("--profile_timing")
        if not bool(args.save_generated_images):
            cmd.append("--no-save_generated_images")
        if not bool(args.save_summary_grid):
            cmd.append("--no-save_summary_grid")
        print(f"[rerun_eval] {ckpt.name} -> {out_dir}")
        subprocess.run(cmd, check=True)
    _write_curve_csv(run_dir / str(args.output_subdir))
    tool_root = Path(__file__).resolve().parent
    if bool(args.refresh_stage_summary):
        _run_optional_refresh([str(args.python_bin), str(tool_root / "build_inmortal_stage_summary.py")])
    if bool(args.refresh_epoch_table):
        _run_optional_refresh([str(args.python_bin), str(tool_root / "build_inmortal_epoch_eval_table.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
