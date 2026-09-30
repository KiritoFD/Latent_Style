#!/usr/bin/env python3
"""Compactify already-encoded WikiArt-27 latents.

The first encode pass accidentally saved 512KB .pt files (torch.save serialized the
whole batch storage, not just the indexed [4,64,64] slice). This reads those bloated
files from F: and re-saves a *contiguous* copy to G: (64KB each), preserving the exact
tensor. No GPU needed.

Run BEFORE deleting the bloated F: directory so the 20 already-done styles are not
re-encoded. The remaining 7 styles are encoded fresh by encode_wikiart27_sd15.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default="F:/wikiart27_sd15_512_latents_ema/train")
    ap.add_argument("--dst_root", default="G:/wikiart27_latents_compact/train")
    args = ap.parse_args()

    src = Path(args.src_root)
    dst = Path(args.dst_root)
    total = 0
    for sdir in sorted(p for p in src.iterdir() if p.is_dir()):
        dstdir = dst / sdir.name
        dstdir.mkdir(parents=True, exist_ok=True)
        pts = sorted(sdir.glob("*.pt"))
        if not pts:
            continue
        kept = 0
        for p in pts:
            t = torch.load(p, map_location="cpu")
            if not torch.is_tensor(t):
                continue
            torch.save(t.contiguous(), dstdir / p.name)
            kept += 1
        total += kept
        print(f"{sdir.name}: {kept} -> {dstdir}")
    print(f"DONE. compactified {total} latents to {dst}")


if __name__ == "__main__":
    main()
