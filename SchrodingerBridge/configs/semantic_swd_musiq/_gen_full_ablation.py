"""Generate comprehensive ablation configs covering all WEAVE method components."""
import json
import copy
import os

BASE_PATH = os.path.join(os.path.dirname(__file__), "swd_cm_sem_r8.json")
OUT_DIR = os.path.dirname(__file__)

with open(BASE_PATH) as f:
    base = json.load(f)

# Fix paths for remote I: drive
base["training"]["test_image_dir"] = "I:/wikiart_distinct5_samam_512_classview/test"
base["training"]["full_eval_cache_dir"] = "I:/Github/Latent_Style/SchrodingerBridge/eval_cache"
base["data"]["data_root"] = "I:/wikiart_distinct5_samam_512_latents_ema/train"
base["data"]["latent_cache_dir"] = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
base["data"]["pairing_cache_path"] = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt"

# Add EOTA to full_eval (baseline has τ=0.08)
base["full_eval"]["hf_soft_threshold"] = 0.08

# Infra: target VRAM 10.8-11.2GB on RTX 3060 12GB
# expandable_segments NOT supported on Windows. Rely on batch size alone.
# Empirical VRAM mapping (bf16 AMP, no expandable_segments):
#   bs=120 -> 10.0GB (baseline)
#   bs=128 -> 11.6GB (tight, succeeded for Tier1 ablations)
#   bs=136 -> 11.7GB (OOM for blend0/blend1 — SWD path memory varies)
# bs=112 gives ~10.8GB with headroom for all SWD variants (sinkhorn/spectral/K=64).
# This is the sweet spot: high throughput + safe margin for memory-heavy configs.
base["training"]["batch_size"] = 112
# Eval stays low to avoid OOM (strict <7GB)
base["training"]["full_eval_batch_size"] = 2
base["full_eval"]["batch_size"] = 2
base["full_eval"]["ref_feature_batch_size"] = 2

# Define ablation configs: (name, description, overrides)
# Each config removes/replaces ONE component to measure its individual contribution
ABLATIONS = [
    # Tier 1: Core component removal (expect LARGE gaps in CLIP-S and/or MUSIQ)
    ("abl_no_swd_loss", "Remove SWD loss entirely", {
        "bridge.single_step_swd_weight": 0.0,
    }),
    ("abl_no_dwt_route", "Remove DWT high-freq routing", {
        "model.cross_attn_dwt_route": False,
        "model.dwt_route_train_prob": 0.0,
    }),
    ("abl_no_wct", "Remove endpoint WCT (endpoint_adain_scale=0)", {
        "model.endpoint_adain_scale": 0.0,
    }),
    ("abl_no_eota", "Remove EOTA soft-threshold (τ=0)", {
        "full_eval.hf_soft_threshold": 0.0,
    }),

    # Tier 2: Semantic SWD component ablation (expect MEDIUM-LARGE gaps)
    ("abl_k1_global", "K=1 (global SWD, no region partition)", {
        "bridge.swd_semantic_regions": 1,
        "bridge.swd_semantic_blend": 0.0,
    }),
    ("abl_blend0_pure_global", "β=0 (pure global WCT, no region match)", {
        "bridge.swd_semantic_blend": 0.0,
    }),
    ("abl_blend1_pure_region", "β=1.0 (pure region match, no global)", {
        "bridge.swd_semantic_blend": 1.0,
    }),
    ("abl_k64_extreme", "K=64 (extreme fine partition)", {
        "bridge.swd_semantic_regions": 64,
    }),
    ("abl_soft_mask", "Soft mask instead of hard k-means", {
        "bridge.swd_semantic_mode": "region_soft",
    }),

    # Tier 3: Architecture extremes (expect MEDIUM gaps)
    ("abl_ll_w0", "λ_LL=0 (no LL supervision)", {
        "bridge.spectral_w_ll": 0.0,
    }),
    ("abl_ll_w1", "λ_LL=1.0 (full LL supervision, no de-weighting)", {
        "bridge.spectral_w_ll": 1.0,
    }),
    ("abl_route_p05", "Route prob=0.5 (weak high-freq routing)", {
        "model.dwt_route_train_prob": 0.5,
    }),
    ("abl_route_p10", "Route prob=1.0 (always high-freq, no full-latent)", {
        "model.dwt_route_train_prob": 1.0,
    }),

    # Tier 4: Mechanism substitution (for method design decisions)
    ("abl_sinkhorn", "Sinkhorn OT instead of k-means quantile", {
        "bridge.swd_semantic_mode": "region_sinkhorn",
    }),
    ("abl_spectral", "Spectral-decoupled SWD (M5 mechanism)", {
        "bridge.swd_semantic_mode": "region_spectral",
    }),
]

def apply_override(cfg, dotted_key, value):
    parts = dotted_key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value

for name, desc, overrides in ABLATIONS:
    cfg = copy.deepcopy(base)
    for k, v in overrides.items():
        apply_override(cfg, k, v)
    cfg["ablation"]["name"] = name
    cfg["ablation"]["axis"] = "full_method_ablation"
    cfg["ablation"]["notes"] = desc
    cfg["checkpoint"]["save_dir"] = f"./exp/{name}"
    out_path = os.path.join(OUT_DIR, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Created {name}.json: {desc}")

# Also generate baseline config (full WEAVE, no ablation)
cfg_baseline = copy.deepcopy(base)
cfg_baseline["ablation"]["name"] = "abl_baseline"
cfg_baseline["ablation"]["axis"] = "baseline"
cfg_baseline["ablation"]["notes"] = "Full WEAVE baseline (all components enabled)"
cfg_baseline["checkpoint"]["save_dir"] = "./exp/abl_baseline"
out_path = os.path.join(OUT_DIR, "abl_baseline.json")
with open(out_path, "w") as f:
    json.dump(cfg_baseline, f, indent=2)
print(f"  Created abl_baseline.json: Full WEAVE baseline")

print(f"\nTotal: {len(ABLATIONS) + 1} configs created (15 ablations + 1 baseline).")

