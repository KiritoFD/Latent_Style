from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
EPOCH_RE = re.compile(r"^epoch_(\d+)$", re.IGNORECASE)


def iter_full_eval_dirs(scan_root: Path) -> list[Path]:
    out: list[Path] = []
    for full_eval_dir in sorted(p for p in scan_root.rglob("full_eval") if p.is_dir()):
        epoch_children = sorted(
            p for p in full_eval_dir.iterdir() if p.is_dir() and EPOCH_RE.match(p.name)
        )
        if epoch_children:
            out.extend(epoch_children)
        else:
            out.append(full_eval_dir)
    return out


def has_generated_images(eval_dir: Path) -> bool:
    image_dirs = [eval_dir / "images", eval_dir]
    for d in image_dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "_to_" in p.name:
                return True
    return False


def infer_run_dir(eval_dir: Path) -> Path | None:
    parts_lower = [x.lower() for x in eval_dir.parts]
    if "full_eval" not in parts_lower:
        return None
    idx = parts_lower.index("full_eval")
    if idx <= 0:
        return None
    return Path(*eval_dir.parts[:idx])


def parse_epoch(eval_dir: Path) -> int | None:
    m = EPOCH_RE.match(eval_dir.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_epoch_from_ckpt_name(name: str) -> int | None:
    m = re.match(r"^epoch_(\d+)\.pt$", name, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def find_checkpoint(run_dir: Path, epoch: int | None) -> Path | None:
    candidates = []
    candidates.extend(sorted(run_dir.glob("epoch_*.pt")))
    candidates.extend(sorted((run_dir / "checkpoints").glob("epoch_*.pt")))
    if not candidates:
        for p in run_dir.rglob("epoch_*.pt"):
            if "full_eval" in {x.lower() for x in p.parts}:
                continue
            candidates.append(p)
        candidates = sorted(set(candidates), key=lambda x: str(x))
    if not candidates:
        return None

    if epoch is not None:
        matched = [p for p in candidates if parse_epoch_from_ckpt_name(p.name) == epoch]
        if matched:
            matched = sorted(
                matched,
                key=lambda p: (
                    0 if p.parent.name.lower() == "checkpoints" else 1,
                    str(p),
                ),
            )
            return matched[0]

    def score(path: Path) -> tuple[int, float, str]:
        ep = parse_epoch_from_ckpt_name(path.name)
        if ep is None:
            ep = -1
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        return (ep, mtime, str(path))

    return max(candidates, key=score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill full_eval metrics/grid from existing images and collect summaries."
    )
    parser.add_argument("--scan_root", type=Path, default=Path("Cycle-NCE"))
    parser.add_argument(
        "--eval_script",
        type=Path,
        default=Path("Cycle-NCE/src/utils/run_evaluation.py"),
    )
    parser.add_argument("--collect_script", type=Path, default=Path("collect_full_eval_assets.py"))
    parser.add_argument("--out_root", type=Path, default=Path("experiments"))
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--overwrite_collect", action="store_true")
    parser.add_argument("--force_regen", action="store_true")
    parser.add_argument("--eval_fid_proxy_stats_dir", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    scan_root = args.scan_root.resolve()
    eval_script = args.eval_script.resolve()
    collect_script = args.collect_script.resolve()
    out_root = args.out_root.resolve()
    fid_proxy_stats_dir = args.eval_fid_proxy_stats_dir.resolve() if args.eval_fid_proxy_stats_dir else None

    if not scan_root.is_dir():
        raise SystemExit(f"scan_root not found: {scan_root}")
    if not eval_script.is_file():
        raise SystemExit(f"eval_script not found: {eval_script}")
    if not collect_script.is_file():
        raise SystemExit(f"collect_script not found: {collect_script}")

    eval_dirs = iter_full_eval_dirs(scan_root)
    to_backfill: list[tuple[Path, Path]] = []
    skipped_no_images = 0
    skipped_complete = 0
    skipped_no_ckpt = 0

    for eval_dir in eval_dirs:
        summary_path = eval_dir / "summary.json"
        grid_path = eval_dir / "summary_grid.png"
        if summary_path.exists() and grid_path.exists():
            skipped_complete += 1
            continue
        if not has_generated_images(eval_dir):
            skipped_no_images += 1
            continue

        run_dir = infer_run_dir(eval_dir)
        if run_dir is None:
            skipped_no_ckpt += 1
            print(f"[SKIP] cannot infer run dir: {eval_dir}")
            continue
        ckpt_path = find_checkpoint(run_dir, parse_epoch(eval_dir))
        if ckpt_path is None:
            skipped_no_ckpt += 1
            print(f"[SKIP] no checkpoint found for: {eval_dir}")
            continue
        to_backfill.append((eval_dir, ckpt_path))

    print(
        f"[SCAN] eval_dirs={len(eval_dirs)} need_backfill={len(to_backfill)} "
        f"skip_complete={skipped_complete} skip_no_images={skipped_no_images} skip_no_ckpt={skipped_no_ckpt}"
    )

    ran = 0
    for eval_dir, ckpt_path in to_backfill:
        cmd = [
            args.python,
            str(eval_script),
            "--checkpoint",
            str(ckpt_path),
            "--output",
            str(eval_dir),
            "--reuse_generated",
        ]
        if args.force_regen:
            cmd.append("--force_regen")
        if fid_proxy_stats_dir is not None:
            cmd += ["--eval_fid_proxy_stats_dir", str(fid_proxy_stats_dir)]
        print(f"[RUN] {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        ran += 1

    collect_cmd = [
        args.python,
        str(collect_script),
        "--scan_root",
        str(scan_root),
        "--out_root",
        str(out_root),
    ]
    if args.overwrite_collect:
        collect_cmd.append("--overwrite")
    print(f"[COLLECT] {' '.join(collect_cmd)}")
    if not args.dry_run:
        subprocess.run(collect_cmd, check=True)

    print(f"[DONE] backfilled={ran} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
