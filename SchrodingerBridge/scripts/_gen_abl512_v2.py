#!/usr/bin/env python3
"""Generate 19 destructive 512 ablation configs (v2: comprehensive + extreme).
Covers 5 theoretical axes: solver, frequency, AdaIN, path/coupling, extreme.
Each config inherits from SOTA (630_phase4i2b_sota_heun_5ep) and modifies one component.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path("g:/GitHub/Latent_Style/SchrodingerBridge/configs")

# Remote I-drive paths
R_DATA = "I:/wikiart_distinct5_samam_512_latents_ema/train"
R_TEST = "I:/wikiart_distinct5_samam_512_classview/test"
R_CACHE = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
R_PAIR = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt"
R_EVAL = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache"
R_HF = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
R_EXP = "I:/Github/Latent_Style/SchrodingerBridge/exp/abl512"

# 19 destructive ablations across 5 axes
ABLATIONS = {
    # === Axis B: ODE Solver Order (3 groups) ===
    "abl512_B01_euler": {
        "model": {"solver_type": "euler"},
        "ablation": {"name": "B01_euler", "axis": "solver_order", "stage": "abl512",
            "notes": "B01: Euler 1st-order (O(h^2) local error) vs Heun 2nd-order (O(h^3)). Tests solver precision impact on style injection."},
    },
    "abl512_B02_rk4": {
        "model": {"solver_type": "rk4"},
        "ablation": {"name": "B02_rk4", "axis": "solver_order", "stage": "abl512",
            "notes": "B02: RK4 4th-order (O(h^5) local error). Tests if even higher solver order continues to help. Cost: 4x forward per step."},
    },
    "abl512_B03_euler_3ep": {
        "model": {"solver_type": "euler"},
        "training": {"num_epochs": 3},
        "ablation": {"name": "B03_euler_3ep", "axis": "solver_vs_training", "stage": "abl512",
            "notes": "B03: Euler+3ep vs SOTA Heun+5ep. If B03≈SOTA, Heun saves 40% training cost (2ep≈Heun precision gain)."},
    },

    # === Axis C: Frequency Domain Core (4 groups) ===
    "abl512_C01_no_spectral_ode": {
        "model": {"spectral_ode_enabled": False},
        "ablation": {"name": "C01_no_spectral_ode", "axis": "freq_ode", "stage": "abl512",
            "notes": "C01: Disable spectral ODE entirely. Core FC-SB innovation removed. Expect catastrophic degradation."},
    },
    "abl512_C02_spectral_3levels": {
        "model": {"spectral_ode_levels": 3},
        "ablation": {"name": "C02_spectral_3levels", "axis": "freq_ode_depth", "stage": "abl512",
            "notes": "C02: 3-level spectral ODE (vs SOTA 1-level). Deeper frequency decomposition. May over-separate or improve granularity."},
    },
    "abl512_C03_avgpool_lowpass": {
        "model": {"lowpass_mode": "avg_pool"},
        "ablation": {"name": "C03_avgpool_lowpass", "axis": "wavelet_vs_avgpool", "stage": "abl512",
            "notes": "C03: avg_pool lowpass (lossy, aliasing) vs DWT Haar (lossless, orthogonal). Tests wavelet advantage."},
    },
    "abl512_C04_no_target_proj": {
        "bridge": {"training_target_projection_mode": "none"},
        "ablation": {"name": "C04_no_target_proj", "axis": "target_projection", "stage": "abl512",
            "notes": "C04: Disable DWT target projection. Without band-specific supervision, all frequencies share one velocity target."},
    },

    # === Axis D: AdaIN / Style Injection (5 groups) ===
    "abl512_D01_adain_00": {
        "model": {"endpoint_adain_scale": 0.0},
        "ablation": {"name": "D01_adain_00", "axis": "adain_scale_lower", "stage": "abl512",
            "notes": "D01: AdaIN scale=0.0 (zero style injection). Extreme lower bound. Expect near-identity, very low CLIP-S + very low LPIPS."},
    },
    "abl512_D02_adain_20": {
        "model": {"endpoint_adain_scale": 2.0},
        "ablation": {"name": "D02_adain_20", "axis": "adain_scale_upper", "stage": "abl512",
            "notes": "D02: AdaIN scale=2.0 (2x over-injection). Extreme upper bound. Expect very high CLIP-S but massive LPIPS (content destruction)."},
    },
    "abl512_D03_adain_every_step": {
        "model": {"endpoint_adain_only_last_step": False},
        "ablation": {"name": "D03_adain_every_step", "axis": "adain_temporal", "stage": "abl512",
            "notes": "D03: AdaIN at every ODE step (not just last). Tests EOTA hypothesis. Over-injection should increase LPIPS."},
    },
    "abl512_D04_lock_ll": {
        "model": {"endpoint_lock_ll": True},
        "ablation": {"name": "D04_lock_ll", "axis": "ll_lock", "stage": "abl512",
            "notes": "D04: Lock LL subband (skip v_ll). Content structure fully preserved, style limited to HF only. Expect low LPIPS + low CLIP-S."},
    },
    "abl512_D05_no_extrap": {
        "model": {"style_extrap_alpha": 0.0},
        "ablation": {"name": "D05_no_extrap", "axis": "style_extrap", "stage": "abl512",
            "notes": "D05: Disable style extrapolation (alpha=0). Tests diversity beyond prototype averaging."},
    },

    # === Axis E: Path / Coupling / Loss (5 groups) ===
    "abl512_E01_linear_path": {
        "bridge": {"bridge_path_mode": "linear"},
        "ablation": {"name": "E01_linear_path", "axis": "tri_band_path", "stage": "abl512",
            "notes": "E01: Linear interpolation path (no tri_band). All frequency bands mixed uniformly. Loses LL/LH_HL/HH independence."},
    },
    "abl512_E02_no_coupling_struct": {
        "bridge": {"coupling_structure_cost_weight": 0.0},
        "ablation": {"name": "E02_no_coupling_struct", "axis": "coupling_structure", "stage": "abl512",
            "notes": "E02: Disable structural coupling cost. Random independent coupling vs self_affinity_gw structural matching."},
    },
    "abl512_E03_no_content_loss": {
        "bridge": {"w_endpoint_content": 0.0},
        "ablation": {"name": "E03_no_content_loss", "axis": "content_loss", "stage": "abl512",
            "notes": "E03: Zero content loss (w_endpoint_content=0). Only style supervision. Expect extreme content destruction, very high CLIP-S + very high LPIPS."},
    },
    "abl512_E04_no_style_loss": {
        "bridge": {"w_endpoint_style": 0.0},
        "ablation": {"name": "E04_no_style_loss", "axis": "style_loss", "stage": "abl512",
            "notes": "E04: Zero style loss (w_endpoint_style=0). Only content supervision. Expect near-identity, very low CLIP-S + very low LPIPS."},
    },
    "abl512_E05_style_loss_32x": {
        "bridge": {"w_endpoint_style": 32.0},
        "ablation": {"name": "E05_style_loss_32x", "axis": "style_loss_extreme", "stage": "abl512",
            "notes": "E05: 32x style loss (vs SOTA 8x). Extreme style pressure. Tests if 4x style weight continues to push CLIP-S or saturates."},
    },

    # === Axis F: Extreme Inference (2 groups) ===
    "abl512_F01_steps_1": {
        "full_eval": {"num_steps": 1},
        "ablation": {"name": "F01_steps_1", "axis": "inference_steps_extreme", "stage": "abl512",
            "notes": "F01: Single-step inference. Extreme speed test. If CLIP-S still acceptable, model has strong single-step capability."},
    },
    "abl512_F02_steps_32": {
        "full_eval": {"num_steps": 32},
        "ablation": {"name": "F02_steps_32", "axis": "inference_steps_extreme", "stage": "abl512",
            "notes": "F02: 32-step inference (vs SOTA 8-step). Tests if more steps improve quality or hit diminishing returns. 4x inference cost."},
    },
}


def make_config(name: str, spec: dict) -> dict:
    cfg = {
        "_base": "630_phase4i2b_sota_heun_5ep.json",
        "checkpoint": {"save_dir": f"{R_EXP}/{name}", "resume_checkpoint": ""},
        "training": {
            "num_epochs": 5, "patience": 2, "full_eval_each_epoch": True,
            "test_image_dir": R_TEST, "full_eval_cache_dir": R_EVAL,
            "full_eval_clip_hf_cache_dir": R_HF,
        },
        "data": {"data_root": R_DATA, "pairing_cache_path": R_PAIR, "latent_cache_dir": R_CACHE},
    }
    for key in ("model", "bridge", "training", "full_eval", "ablation"):
        if key in spec:
            cfg.setdefault(key, {}).update(spec[key])
    return cfg


def main() -> int:
    # Remove old v1 configs
    for f in CONFIG_DIR.glob("abl512_A*.json"):
        f.unlink()
        print(f"  removed {f.name}")

    for name, spec in ABLATIONS.items():
        cfg = make_config(name, spec)
        out = CONFIG_DIR / f"{name}.json"
        out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {out.name}")
    print(f"\nGenerated {len(ABLATIONS)} configs (v2) in {CONFIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
