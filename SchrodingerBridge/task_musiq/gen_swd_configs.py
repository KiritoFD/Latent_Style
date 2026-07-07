"""Generate 4 SWD-centric architecture direction configs from the base.

Directions (all centered on semantic SWD — content-similar block matching):
  S1: Semantic Region SWD        (swd_semantic_mode=region)
      k-means on content latent → region-level distribution matching
  S2: Semantic Patch SWD         (swd_semantic_mode=region_patch)  [NEW]
      Multi-scale patch texture matching within content-coherent regions
  S3: Semantic Band-split SWD    (swd_semantic_mode=region_band)   [NEW]
      Semantic region matching per DWT subband, HF emphasis
  S4: Cross-attn Guided Semantic (swd_scale_mode=cross-attn-guided + swd_semantic_mode=region)
      Cross-attn entropy as guided sampling + semantic region matching

Base: dwt_route_distinct5.json (DWT route + attention-weighted SWD, MUSIQ=41.11)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "configs" / "dwt_route_distinct5.json"
OUT_DIR = BASE.parent

with open(BASE, "r", encoding="utf-8") as f:
    base = json.load(f)


def make_config(name: str, model_overrides: dict | None = None, bridge_overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(base)
    if model_overrides:
        cfg["model"].update(model_overrides)
    if bridge_overrides:
        cfg["bridge"].update(bridge_overrides)
    cfg["checkpoint"]["save_dir"] = f"./exp/musiq_{name}"
    cfg["ablation"]["name"] = f"musiq_{name}"
    cfg["ablation"]["notes"] = f"SWD-centric direction: {name}. Base=dwt_route+attn-SWD."
    return cfg


configs = {}

# S1: Semantic Region SWD — content-coherent region matching (pixel-level)
configs["s1_sem_region"] = make_config(
    "s1_sem_region",
    bridge_overrides={
        "swd_semantic_mode": "region",
        "swd_semantic_regions": 8,           # more regions for finer content partitioning
        "swd_semantic_blend": 0.7,           # high blend: mostly region-matched, some global
        "swd_semantic_kmeans_iters": 6,      # more iters for stable regions
        # swd_scale_mode stays "attention-weighted" for the global SWD component
    },
)

# S2: Semantic Patch SWD — multi-scale patch texture matching within regions [NEW]
configs["s2_sem_patch"] = make_config(
    "s2_sem_patch",
    bridge_overrides={
        "swd_semantic_mode": "region_patch",  # NEW: semantic + patch
        "swd_semantic_regions": 6,            # moderate regions for patch coverage
        "swd_semantic_blend": 0.7,            # high blend toward semantic patches
        "swd_semantic_kmeans_iters": 4,
        "swd_patch_sizes": [1, 3, 5],         # multi-scale: color, fine, coarse texture
        "swd_patch_weights": [0.3, 0.4, 0.3], # favor mid-scale (3x3) texture
    },
)

# S3: Semantic Band-split SWD — per-subband semantic matching, HF emphasis [NEW]
configs["s3_sem_band"] = make_config(
    "s3_sem_band",
    bridge_overrides={
        "swd_semantic_mode": "region_band",   # NEW: semantic + band-split
        "swd_semantic_regions": 6,
        "swd_semantic_blend": 0.8,            # high blend toward semantic band matching
        "swd_semantic_kmeans_iters": 4,
        "swd_band_w_ll": 0.25,                # low weight: LL carries structure, less MUSIQ
        "swd_band_w_lh": 1.0,                 # mid-freq horizontal: texture
        "swd_band_w_hl": 1.0,                 # mid-freq vertical: texture
        "swd_band_w_hh": 2.0,                 # high weight: HH = finest detail, MUSIQ reward
    },
)

# S4: Cross-attn Guided Semantic Region SWD
# Uses cross-attn entropy as guided sampling weight + semantic region matching
configs["s4_sem_xattn"] = make_config(
    "s4_sem_xattn",
    bridge_overrides={
        "swd_scale_mode": "cross-attn-guided",  # switch from "attention-weighted" to guided sampling
        "swd_semantic_mode": "region",
        "swd_semantic_regions": 8,
        "swd_semantic_blend": 0.7,
        "swd_semantic_kmeans_iters": 6,
    },
)

for name, cfg in configs.items():
    out_path = OUT_DIR / f"musiq_{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[OK] {out_path}")

print(f"\nGenerated {len(configs)} SWD-centric configs in {OUT_DIR}")
