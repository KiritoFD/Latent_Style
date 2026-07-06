#!/usr/bin/env python3
"""Find largest directories under a root."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/exp")
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    dirs = [d for d in root.iterdir() if d.is_dir()]
    sizes = [(d, dir_size(d)) for d in dirs]
    sizes.sort(key=lambda x: x[1], reverse=True)
    print(f"Top {top_n} directories under {root}:")
    for d, s in sizes[:top_n]:
        print(f"{s / 1024**3:8.2f} GB  {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
