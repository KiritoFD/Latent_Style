from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def _iter_runs(run_root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in run_root.glob(pattern) if path.is_dir())


def _relative_link_target(*, link_dir: Path, source_dir: Path) -> str:
    return os.path.relpath(source_dir, start=link_dir)


def _create_or_validate_link(*, bundle_root: Path, source_dir: Path) -> tuple[str, Path]:
    link_path = bundle_root / source_dir.name
    if link_path.exists() or link_path.is_symlink():
        try:
            resolved = link_path.resolve(strict=True)
        except FileNotFoundError:
            return "broken", link_path
        if resolved == source_dir.resolve():
            return "kept", link_path
        raise RuntimeError(
            f"bundle entry already exists but points elsewhere: {link_path} -> {resolved} (expected {source_dir})"
        )
    link_path.symlink_to(_relative_link_target(link_dir=link_path.parent, source_dir=source_dir), target_is_directory=True)
    return "linked", link_path


def _count_files(run_dir: Path, pattern: str) -> int:
    return sum(1 for _ in run_dir.glob(pattern))


def _write_manifest(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "status",
        "source_dir",
        "bundle_dir",
        "checkpoint_count",
        "summary_count",
        "curve_csv_exists",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a unified inmortal-exp bundle root with symlinked legacy inmortal runs."
    )
    parser.add_argument("--run-root", type=Path, default=Path("exp"))
    parser.add_argument("--pattern", default="aaai2027_inmortal*")
    parser.add_argument("--bundle-root", type=Path, default=Path("exp/inmortal-exp"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/experiments/inmortal-exp-manifest.csv"),
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    bundle_root = args.bundle_root.resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for run_dir in _iter_runs(run_root, str(args.pattern)):
        if run_dir == bundle_root:
            continue
        status, bundle_dir = _create_or_validate_link(bundle_root=bundle_root, source_dir=run_dir)
        rows.append(
            {
                "run_name": run_dir.name,
                "status": status,
                "source_dir": str(run_dir),
                "bundle_dir": str(bundle_dir),
                "checkpoint_count": _count_files(run_dir, "epoch_*.pt"),
                "summary_count": _count_files(run_dir / "full_eval_fast_snapshot", "epoch_*/summary.json"),
                "curve_csv_exists": (run_dir / "full_eval_fast_snapshot" / "clip_lpips_curve.csv").is_file(),
            }
        )

    _write_manifest(rows, args.manifest.resolve())
    print(f"[prepare_inmortal_exp_root] bundle_root={bundle_root}")
    print(f"[prepare_inmortal_exp_root] manifest={args.manifest.resolve()}")
    for row in rows:
        print(
            f"[prepare_inmortal_exp_root] {row['run_name']} status={row['status']} "
            f"ckpt={row['checkpoint_count']} summaries={row['summary_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
