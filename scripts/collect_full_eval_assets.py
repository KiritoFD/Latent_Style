from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TARGET_FILES = {"summary.json", "summary_grid.png"}


def infer_experiment_dir_from_file(source_file: Path) -> Path:
    parts_lower = [p.lower() for p in source_file.parts]
    if "full_eval" not in parts_lower:
        return source_file.parent
    idx = parts_lower.index("full_eval")
    if idx <= 0:
        return source_file.parent
    return Path(*source_file.parts[:idx])


def slug_from_relative_path(path: Path) -> str:
    return "__".join(path.parts)


def build_dest_name(
    scan_root: Path,
    source_file: Path,
) -> str:
    exp_dir = infer_experiment_dir_from_file(source_file)
    try:
        exp_rel = exp_dir.relative_to(scan_root)
        exp_tag = slug_from_relative_path(exp_rel)
    except ValueError:
        exp_tag = exp_dir.name

    parts_lower = [p.lower() for p in source_file.parts]
    full_eval_idx = parts_lower.index("full_eval")
    rel_after_full_eval = Path(*source_file.parts[full_eval_idx + 1 : -1])
    if str(rel_after_full_eval) == ".":
        return f"{exp_tag}__{source_file.name}"

    rel_tag = slug_from_relative_path(rel_after_full_eval)
    return f"{exp_tag}__{rel_tag}__{source_file.name}"


def collect_assets(
    scan_root: Path,
    summary_dir: Path,
    grid_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int, int]:
    found = 0
    copied = 0
    skipped = 0

    summary_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    all_candidates = sorted(p for p in scan_root.rglob("*") if p.is_file() and p.name in TARGET_FILES)
    for source_file in all_candidates:
        parts_lower = [p.lower() for p in source_file.parts]
        if "full_eval" not in parts_lower:
            continue

        found += 1
        dest_name = build_dest_name(scan_root, source_file)
        dest_dir = summary_dir if source_file.name == "summary.json" else grid_dir
        dest_file = dest_dir / dest_name

        if dest_file.exists() and not overwrite:
            skipped += 1
            print(f"[SKIP] exists: {dest_file}")
            continue

        print(f"[COPY] {source_file} -> {dest_file}")
        if not dry_run:
            shutil.copy2(source_file, dest_file)
        copied += 1

    return found, copied, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect summary.json and summary_grid.png from <scan_root>/**/full_eval into <out_root>/summary and <out_root>/grid."
    )
    parser.add_argument(
        "--scan_root",
        type=Path,
        default=Path(""),
        help="Directory to scan recursively for full_eval. Default: Cycle-NCE",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("experiments"),
        help="Output root directory for summary/grid. Default: experiments",
    )
    parser.add_argument(
        "--summary_dir",
        type=Path,
        default=None,
        help="Output directory for summary.json copies. Default: <out_root>/summary",
    )
    parser.add_argument(
        "--grid_dir",
        type=Path,
        default=None,
        help="Output directory for summary_grid.png copies. Default: <out_root>/grid",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print copy actions without writing files.",
    )
    args = parser.parse_args()

    scan_root = args.scan_root.resolve()
    if not scan_root.is_dir():
        raise SystemExit(f"scan_root does not exist: {scan_root}")

    out_root = args.out_root.resolve()
    summary_dir = (args.summary_dir or (out_root / "summary")).resolve()
    grid_dir = (args.grid_dir or (out_root / "grid")).resolve()

    found, copied, skipped = collect_assets(
        scan_root=scan_root,
        summary_dir=summary_dir,
        grid_dir=grid_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(
        f"[DONE] scan_root={scan_root} found={found} copied={copied} skipped={skipped} "
        f"summary_dir={summary_dir} grid_dir={grid_dir}"
    )


if __name__ == "__main__":
    main()
