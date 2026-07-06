#!/usr/bin/env python3
"""Generate 12 destructive 512 ablation configs for remote training.
Each config inherits from 630_phase4i2b_sota_heun_5ep (SOTA) and removes/modifies
exactly one component to isolate its contribution.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path("g:/GitHub/Latent_Style/SchrodingerBridge/configs")

# Remote I-drive paths
REMOTE_DATA_ROOT = "I:/wikiart_distinct5_samam_512_latents_ema/train"
REMOTE_TEST_DIR = "I:/wikiart_distinct5_samam_512_classview/test"
REMOTE_CACHE_DIR = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
REMOTE_PAIRING = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt"
REMOTE_EVAL_CACHE = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache"
REMOTE_EVAL_HF = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
REMOTE_EXP_ROOT = "I:/Github/Latent_Style/SchrodingerBridge/exp/abl512"

# 12 destructive ablations - each targets a distinct theoretical component
ABLATIONS = {
    "abl512_A01_no_heun": {
        "model": {"solver_type": "euler"},
        "ablation": {
            "name": "A01_no_heun",
            "axis": "ode_solver_order",
            "stage": "abl512",
            "notes": "A01: Replace Heun with Euler (1st-order). Tests if Heun's O(h^3) local truncation error vs Euler's O(h^2) matters. Heun costs 2x forward but should improve style injection precision.",
        },
    },
    "abl512_A02_no_spectral_ode": {
        "model": {"spectral_ode_enabled": False},
        "ablation": {
            "name": "A02_no_spectral_ode",
            "axis": "frequency_domain_ode",
            "stage": "abl512",
            "notes": "A02: Disable spectral ODE entirely. Spectral ODE is the core FC-SB innovation - frequency-band-separated transport. Expect massive degradation in both CLIP-S and LPIPS.",
        },
    },
    "abl512_A03_adain_scale_0": {
        "model": {"endpoint_adain_scale": 0.0},
        "ablation": {
            "name": "A03_adain_scale_0",
            "axis": "style_injection_strength_lower_bound",
            "stage": "abl512",
            "notes": "A03: AdaIN scale=0 (no style injection at endpoint). Lower bound of style injection. Expect near-identity transfer (very low CLIP-S, very low LPIPS).",
        },
    },
    "abl512_A04_adain_scale_10": {
        "model": {"endpoint_adain_scale": 1.0},
        "ablation": {
            "name": "A04_adain_scale_10",
            "axis": "style_injection_strength_upper_bound",
            "stage": "abl512",
            "notes": "A04: AdaIN scale=1.0 (full style replacement). Upper bound. Expect high CLIP-S but large LPIPS increase (content destruction).",
        },
    },
    "abl512_A05_adain_every_step": {
        "model": {"endpoint_adain_only_last_step": False},
        "ablation": {
            "name": "A05_adain_every_step",
            "axis": "adain_temporal_locality",
            "stage": "abl512",
            "notes": "A05: Apply AdaIN at every ODE step (not just last). Tests EOTA (Endpoint-Only Transport AdaIN) hypothesis. Every-step AdaIN should over-inject style, increasing LPIPS.",
        },
    },
    "abl512_A06_lock_ll": {
        "model": {"endpoint_lock_ll": True},
        "ablation": {
            "name": "A06_lock_ll",
            "axis": "low_frequency_lock",
            "stage": "abl512",
            "notes": "A06: Lock LL subband at inference (skip v_ll). Tests if LL carries content structure. Expect very low LPIPS (content preserved) but lower CLIP-S (style limited to HF).",
        },
    },
    "abl512_A07_no_extrap": {
        "model": {"style_extrap_alpha": 0.0},
        "ablation": {
            "name": "A07_no_extrap",
            "axis": "style_extrapolation",
            "stage": "abl512",
            "notes": "A07: Disable style extrapolation (alpha=0). Tests if style_extrap provides style diversity beyond prototype averaging.",
        },
    },
    "abl512_A08_no_dwt_lowpass": {
        "model": {"lowpass_mode": "avg_pool"},
        "ablation": {
            "name": "A08_no_dwt_lowpass",
            "axis": "wavelet_vs_avgpool_lowpass",
            "stage": "abl512",
            "notes": "A08: Replace DWT Haar with avg_pool for lowpass. Tests if wavelet decomposition (lossless, orthogonal) outperforms avg_pool (lossy, aliasing).",
        },
    },
    "abl512_A09_no_tri_band": {
        "bridge": {"bridge_path_mode": "linear"},
        "ablation": {
            "name": "A09_no_tri_band",
            "axis": "tri_band_path",
            "stage": "abl512",
            "notes": "A09: Replace tri_band bridge path with linear interpolation. Tri_band separates LL/LH_HL/HH for independent style-content mixing. Linear mixes all bands uniformly.",
        },
    },
    "abl512_A10_no_coupling_structure": {
        "bridge": {"coupling_structure_cost_weight": 0.0},
        "ablation": {
            "name": "A10_no_coupling_structure",
            "axis": "coupling_structure_cost",
            "stage": "abl512",
            "notes": "A10: Disable coupling structure cost (self_affinity_gw). Tests if structural coupling improves pairing quality vs random independent coupling.",
        },
    },
    "abl512_A11_no_target_projection": {
        "bridge": {"training_target_projection_mode": "none"},
        "ablation": {
            "name": "A11_no_target_projection",
            "axis": "target_frequency_projection",
            "stage": "abl512",
            "notes": "A11: Disable DWT target projection. Projection separates training target into frequency bands for band-specific supervision. Without it, all bands share one velocity target.",
        },
    },
    "abl512_A12_euler_3ep": {
        "model": {"solver_type": "euler"},
        "training": {"num_epochs": 3},
        "ablation": {
            "name": "A12_euler_3ep",
            "axis": "solver_order_vs_training_length",
            "stage": "abl512",
            "notes": "A12: Euler + 3 epochs (vs SOTA: Heun + 5ep). Tests if Heun's precision gain ~ 2 extra epochs of training. If A12 ≈ SOTA, confirms Heun saves 40% training cost.",
        },
    },
}


def make_config(name: str, spec: dict) -> dict:
    """Build a config dict inheriting from SOTA with remote paths and ablation overrides."""
    cfg = {
        "_base": "630_phase4i2b_sota_heun_5ep.json",
        "checkpoint": {
            "save_dir": f"{REMOTE_EXP_ROOT}/{name}",
            "resume_checkpoint": "",
        },
        "training": {
            "num_epochs": 5,
            "patience": 2,
            "full_eval_each_epoch": True,
            "test_image_dir": REMOTE_TEST_DIR,
            "full_eval_cache_dir": REMOTE_EVAL_CACHE,
            "full_eval_clip_hf_cache_dir": REMOTE_EVAL_HF,
        },
        "data": {
            "data_root": REMOTE_DATA_ROOT,
            "pairing_cache_path": REMOTE_PAIRING,
            "latent_cache_dir": REMOTE_CACHE_DIR,
        },
    }
    # Merge ablation-specific overrides
    for key in ("model", "bridge", "training", "ablation"):
        if key in spec:
            cfg.setdefault(key, {}).update(spec[key])
    return cfg


def main() -> int:
    for name, spec in ABLATIONS.items():
        cfg = make_config(name, spec)
        out = CONFIG_DIR / f"{name}.json"
        out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {out.name}")
    print(f"\nGenerated {len(ABLATIONS)} configs in {CONFIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
