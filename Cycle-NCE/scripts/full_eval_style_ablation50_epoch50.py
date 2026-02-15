#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    exp_name: str
    run_dir: Path
    checkpoint: Path
    output_dir: Path
    log_path: Path
    status: str
    return_code: int | None
    error: str
    summary_path: Path | None
    transfer_clip_style: float | None
    transfer_classifier_acc: float | None
    transfer_content_lpips: float | None
    photo_to_art_clip_style: float | None
    photo_to_art_classifier_acc: float | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _default_eval_classifier_path() -> Path:
    return _repo_root() / "artifacts" / "eval_classifier" / "eval_style_image_classifier.pt"


def _safe_get(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_float_or_none(v: Any) -> float | None:
    try:
        out = float(v)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _find_run_dirs(root: Path, epoch: int) -> list[Path]:
    ckpt_name = f"epoch_{epoch:04d}.pt"
    run_dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name == "configs":
            continue
        if (child / ckpt_name).is_file():
            run_dirs.append(child)
    return run_dirs


def _read_summary(summary_path: Path) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    transfer_clip = _as_float_or_none(_safe_get(payload, ["analysis", "style_transfer_ability", "clip_style"]))
    transfer_cls = _as_float_or_none(_safe_get(payload, ["analysis", "style_transfer_ability", "classifier_acc"]))
    transfer_lpips = _as_float_or_none(_safe_get(payload, ["analysis", "style_transfer_ability", "content_lpips"]))
    p2a_clip = _as_float_or_none(_safe_get(payload, ["analysis", "photo_to_art_performance", "clip_style"]))
    p2a_cls = _as_float_or_none(_safe_get(payload, ["analysis", "photo_to_art_performance", "classifier_acc"]))
    return transfer_clip, transfer_cls, transfer_lpips, p2a_clip, p2a_cls


def _fmt(v: float | None, digits: int = 6) -> str:
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def _write_csv(path: Path, rows: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "exp_name",
                "status",
                "return_code",
                "checkpoint",
                "output_dir",
                "summary_path",
                "transfer_clip_style",
                "transfer_classifier_acc",
                "transfer_content_lpips",
                "photo_to_art_clip_style",
                "photo_to_art_classifier_acc",
                "error",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.exp_name,
                    r.status,
                    r.return_code if r.return_code is not None else "",
                    str(r.checkpoint),
                    str(r.output_dir),
                    str(r.summary_path) if r.summary_path else "",
                    _fmt(r.transfer_clip_style),
                    _fmt(r.transfer_classifier_acc),
                    _fmt(r.transfer_content_lpips),
                    _fmt(r.photo_to_art_clip_style),
                    _fmt(r.photo_to_art_classifier_acc),
                    r.error,
                ]
            )


def _write_markdown(path: Path, rows: list[RunResult], epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# style_ablation-50 epoch_{epoch:04d} full_eval summary")
    lines.append("")
    lines.append("| exp_name | status | transfer_clip | transfer_cls | transfer_lpips | p2a_clip | p2a_cls |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.exp_name,
                    r.status,
                    _fmt(r.transfer_clip_style, 4),
                    _fmt(r.transfer_classifier_acc, 4),
                    _fmt(r.transfer_content_lpips, 4),
                    _fmt(r.photo_to_art_clip_style, 4),
                    _fmt(r.photo_to_art_classifier_acc, 4),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _clear_generated_images(out_dir: Path) -> int:
    removed = 0
    for p in out_dir.glob("*_to_*.jpg"):
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full_eval for epoch50 in all style_ablation-50 runs, then aggregate.")
    parser.add_argument("--root", type=Path, default=_repo_root() / "style_ablation-50", help="Ablation root dir.")
    parser.add_argument("--epoch", type=int, default=50, help="Target checkpoint epoch.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for eval script.")
    parser.add_argument("--test_dir", type=Path, default=_repo_root().parent / "style_data" / "overfit50")
    parser.add_argument("--cache_dir", type=Path, default=_repo_root() / "eval_cache")
    parser.add_argument("--classifier_path", type=Path, default=_repo_root() / "style_classifier.pt")
    parser.add_argument("--image_classifier_path", type=Path, default=_default_eval_classifier_path())
    parser.add_argument("--num_steps", type=int, default=6)
    parser.add_argument("--step_size", type=float, default=0.8)
    parser.add_argument("--style_strength", type=float, default=0.75)
<<<<<<< Updated upstream
    parser.add_argument(
        "--step_schedule",
        type=str,
        default="flat",
        help="Deprecated compatibility arg. run_evaluation.py no longer accepts this and it is ignored here.",
    )
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_src_samples", type=int, default=50)
    parser.add_argument("--max_ref_compare", type=int, default=50)
    parser.add_argument("--max_ref_cache", type=int, default=100)
    parser.add_argument("--ref_feature_batch_size", type=int, default=32)
    parser.add_argument("--eval_disable_lpips", action="store_true")
    parser.add_argument("--eval_classifier_only", action="store_true")
    parser.add_argument("--reuse_generated", action=argparse.BooleanOptionalAction, default=False, help="Reuse existing generated images if present")
    parser.add_argument("--clean_generated", action=argparse.BooleanOptionalAction, default=True, help="Delete existing *_to_*.jpg before eval to force fresh generation")
=======
    parser.add_argument("--step_schedule", type=str, default="flat")
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--max_src_samples", type=int, default=50)
    parser.add_argument("--max_ref_compare", type=int, default=50)
    parser.add_argument("--max_ref_cache", type=int, default=100)
    parser.add_argument("--ref_feature_batch_size", type=int, default=96)
    parser.add_argument("--eval_disable_lpips", action="store_true")
    parser.add_argument("--eval_classifier_only", action="store_true")
    parser.add_argument("--reuse_generated", action=argparse.BooleanOptionalAction, default=True, help="Reuse existing generated images if present")
    parser.add_argument("--clean_generated", action=argparse.BooleanOptionalAction, default=False, help="Delete existing *_to_*.jpg before eval to force fresh generation")
>>>>>>> Stashed changes
    parser.add_argument("--force_regen", action="store_true")
    parser.add_argument("--skip_existing", action="store_true", help="Skip run if summary.json already exists.")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands, do not execute.")
    args = parser.parse_args()

    repo_root = _repo_root()
    eval_script = repo_root / "src" / "utils" / "run_evaluation.py"
    if not eval_script.is_file():
        raise FileNotFoundError(f"Eval script not found: {eval_script}")
    if not args.root.is_dir():
        raise NotADirectoryError(f"Root not found: {args.root}")

    run_dirs = _find_run_dirs(args.root, args.epoch)
    if not run_dirs:
        print(f"No run dirs with epoch_{args.epoch:04d}.pt under {args.root}")
        return

    results: list[RunResult] = []
    for run_dir in run_dirs:
        exp_name = run_dir.name
        ckpt = run_dir / f"epoch_{args.epoch:04d}.pt"
        out_dir = run_dir / "full_eval" / f"epoch_{args.epoch:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "eval.log"
        summary_path = out_dir / "summary.json"

        cmd = [
            args.python,
            str(eval_script),
            "--checkpoint",
            str(ckpt),
            "--output",
            str(out_dir),
            "--test_dir",
            str(args.test_dir),
            "--cache_dir",
            str(args.cache_dir),
            "--num_steps",
            str(args.num_steps),
            "--step_size",
            str(args.step_size),
            "--style_strength",
            str(args.style_strength),
<<<<<<< Updated upstream
=======
            "--step_schedule",
            str(args.step_schedule),
>>>>>>> Stashed changes
            "--batch_size",
            str(args.batch_size),
            "--max_src_samples",
            str(args.max_src_samples),
            "--max_ref_compare",
            str(args.max_ref_compare),
            "--max_ref_cache",
            str(args.max_ref_cache),
            "--ref_feature_batch_size",
            str(args.ref_feature_batch_size),
            "--classifier_path",
            str(args.classifier_path),
            "--image_classifier_path",
            str(args.image_classifier_path),
        ]
        if args.force_regen:
            cmd.append("--force_regen")
        if args.eval_classifier_only:
            cmd.append("--eval_classifier_only")
        if args.eval_disable_lpips:
            cmd.append("--eval_disable_lpips")
        if args.reuse_generated:
            cmd.append("--reuse_generated")

        if (not args.dry_run) and args.clean_generated and (not args.reuse_generated):
            removed = _clear_generated_images(out_dir)
            if removed > 0:
                print(f"[CLEAN] {exp_name}: removed {removed} old generated images")

        if args.skip_existing and summary_path.exists():
            status = "skipped_existing"
            return_code = 0
            error = ""
        elif args.dry_run:
            print(f"[DRY-RUN] {exp_name}: {' '.join(cmd)}")
            status = "dry_run"
            return_code = None
            error = ""
        else:
            print(f"[RUN] {exp_name}: epoch_{args.epoch:04d}")
            with log_path.open("w", encoding="utf-8") as lf:
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=repo_root)
            return_code = int(proc.returncode)
            status = "ok" if proc.returncode == 0 else "failed"
            error = "" if proc.returncode == 0 else f"eval failed, see log: {log_path}"

        t_clip = None
        t_cls = None
        t_lpips = None
        p_clip = None
        p_cls = None
        resolved_summary: Path | None = None
        if summary_path.exists():
            resolved_summary = summary_path
            try:
                t_clip, t_cls, t_lpips, p_clip, p_cls = _read_summary(summary_path)
            except Exception as ex:
                if not error:
                    error = f"summary parse failed: {ex}"
                if status == "ok":
                    status = "summary_parse_failed"
        else:
            if status in {"ok", "skipped_existing"}:
                status = "missing_summary"
                if not error:
                    error = f"summary not found: {summary_path}"

        results.append(
            RunResult(
                exp_name=exp_name,
                run_dir=run_dir,
                checkpoint=ckpt,
                output_dir=out_dir,
                log_path=log_path,
                status=status,
                return_code=return_code,
                error=error,
                summary_path=resolved_summary,
                transfer_clip_style=t_clip,
                transfer_classifier_acc=t_cls,
                transfer_content_lpips=t_lpips,
                photo_to_art_clip_style=p_clip,
                photo_to_art_classifier_acc=p_cls,
            )
        )

    csv_path = args.root / f"full_eval_epoch{args.epoch}_summary.csv"
    md_path = args.root / f"full_eval_epoch{args.epoch}_summary.md"
    _write_csv(csv_path, results)
    _write_markdown(md_path, results, args.epoch)

    ok_count = sum(1 for x in results if x.status == "ok")
    fail_count = sum(1 for x in results if x.status == "failed")
    print(f"Completed: total={len(results)} ok={ok_count} failed={fail_count}")
    print(f"CSV: {csv_path}")
    print(f"MD : {md_path}")


if __name__ == "__main__":
    main()
