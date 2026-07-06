#!/usr/bin/env python3
"""Clean non-final epoch checkpoints and full_eval outputs on remote I drive.
Keeps the highest-numbered epoch per experiment; deletes all earlier epochs.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


def find_epoch_dirs(root: Path) -> dict[Path, list[Path]]:
    """Map parent -> sorted list of epoch_* dirs."""
    pattern = re.compile(r"^epoch_(\d+)$")
    groups: dict[Path, list[Path]] = {}
    for p in root.rglob("epoch_*"):
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        groups.setdefault(p.parent, []).append(p)
    for epochs in groups.values():
        epochs.sort(key=lambda x: int(re.search(r"epoch_(\d+)", x.name).group(1)))  # type: ignore[arg-type]
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        return 1

    groups = find_epoch_dirs(root)
    total_reclaim = 0
    total_dirs = 0

    for parent, epochs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(epochs) <= 1:
            continue
        keep = epochs[-1]
        to_delete = epochs[:-1]
        print(f"\n{parent}")
        print(f"  keep: {keep.name}")
        for d in to_delete:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            total_reclaim += size
            total_dirs += 1
            print(f"  delete: {d.name} ({size / 1024**3:.2f} GB)")
            if not args.dry_run:
                shutil.rmtree(d, ignore_errors=True)

    print(f"\n{'Would reclaim' if args.dry_run else 'Reclaimed'} {total_reclaim / 1024**3:.2f} GB from {total_dirs} directories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
