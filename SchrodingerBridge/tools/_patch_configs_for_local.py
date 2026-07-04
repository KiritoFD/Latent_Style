#!/usr/bin/env python3
"""Patch generated 620 configs for local F: drive execution."""
import json
import os
from pathlib import Path

base = Path("g:/GitHub/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge")

replacements = {
    "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train": "f:/wikiart_distinct5_samam_512_latents_ema/train",
    "/mnt/i/wikiart_distinct5_samam_512_classview/test": "f:/wikiart_distinct5_samam_512_classview_real/test",
    "/mnt/i/Github/Latent_Style/eval_cache": "f:/eval_cache",
    "/mnt/i/Github/Latent_Style/eval_cache/hf": "f:/eval_cache/hf",
    "/mnt/i/Github/Latent_Style/eval_cache/offline_pairing": "f:/eval_cache/offline_pairing",
}

for cfg_dir in base.iterdir():
    cfg_path = cfg_dir / "config.json"
    if not cfg_path.exists():
        continue
    with open(cfg_path) as f:
        text = f.read()
    original = text
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Also ensure backslashes for Windows paths are handled
    for old, new in replacements.items():
        text = text.replace(old.replace("/", "\\"), new.replace("/", "\\"))
    if text != original:
        with open(cfg_path, "w") as f:
            f.write(text)
        print(f"Patched {cfg_path}")
    else:
        print(f"No changes needed for {cfg_path}")
