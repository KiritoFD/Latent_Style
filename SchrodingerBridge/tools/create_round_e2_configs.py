#!/usr/bin/env python3
"""Create Round E2 local smoke configs for 620 whitening fix.

Generates:
- exp/620_spatial_bridge/620_film_v5_gated_local_smoke/config.json
  (recreation of E1 optimal baseline)
- exp/620_spatial_bridge/620_film_v5_endpoint_film_local_smoke/config.json
- exp/620_spatial_bridge/620_film_v5_hf_residual_local_smoke/config.json
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from config_schema import load_config

BASE_CONFIG = SCRIPT_DIR.parent / "configs" / "620_spatial_bridge_intrinsic.json"
EXP_DIR = SCRIPT_DIR.parent / "exp" / "620_spatial_bridge"


def patch_local_paths(cfg: dict) -> dict:
    """Convert remote /mnt/i paths to local F: drive paths."""
    cfg["training"]["test_image_dir"] = "f:/wikiart_distinct5_samam_512_classview_real/test"
    cfg["training"]["full_eval_cache_dir"] = "f:/eval_cache"
    cfg["training"]["full_eval_clip_hf_cache_dir"] = "f:/eval_cache/hf"
    cfg["data"]["data_root"] = "f:/wikiart_distinct5_samam_512_latents_ema/train"
    cfg["data"]["latent_cache_dir"] = "f:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache"
    cfg["data"]["dino_cache_path"] = "f:/eval_cache/offline_pairing/dinov2_wikiart_distinct5_samam_512_train_cache.pt"
    cfg["data"]["pairing_cache_path"] = "f:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/dino_pairing_top8.pt"
    return cfg


def make_local_smoke(cfg: dict, name: str, notes: str) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["training"]["batch_size"] = 4
    cfg["training"]["accumulation_steps"] = 16
    cfg["training"]["num_epochs"] = 1
    cfg["training"]["save_interval"] = 1
    cfg["training"]["num_workers"] = 0
    cfg["training"]["full_eval_each_epoch"] = True
    cfg["training"]["full_eval_defer_until_training_end"] = False
    cfg["training"]["full_eval_batch_size"] = 4
    cfg["training"]["full_eval_vae_decode_batch_size"] = 4
    cfg["full_eval"]["batch_size"] = 4
    cfg["full_eval"]["vae_decode_batch_size"] = 4
    cfg["full_eval"]["ref_feature_batch_size"] = 4
    cfg["full_eval"]["max_src_samples"] = 30
    cfg["full_eval"]["max_ref_compare"] = 30
    cfg["full_eval"]["max_ref_cache"] = 30
    cfg["full_eval"]["save_generated_images"] = True
    cfg["full_eval"]["save_summary_grid"] = True
    cfg["full_eval"]["only_lpips_clip_style"] = True
    cfg["checkpoint"]["save_dir"] = str(EXP_DIR / name)
    cfg["ablation"]["name"] = name
    cfg["ablation"]["stage"] = "smoke"
    cfg["ablation"]["notes"] = notes
    return cfg


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_config(BASE_CONFIG)
    base = patch_local_paths(base)

    # Common v5 gated settings (reconstructed from create_whitening_fix_configs.py).
    v5_gated_updates = {
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {
            "style_cross_attn_gate_init": 0.3,
            "style_film_enabled": True,
            "style_attn_mode": "gated",
            "style_attn_temperature": 1.0,
        },
        "training": {"num_epochs": 5},
        "ablation": {
            "name": "620_film_v5_gated_5ep",
            "notes": "Gated attention (no softmax anywhere) + non-zero FiLM init (std=0.02) + FiLM + bias",
        },
    }

    gated_5ep = json.loads(json.dumps(base))
    for section, updates in v5_gated_updates.items():
        if section in gated_5ep:
            gated_5ep[section].update(updates)
    gated_5ep["checkpoint"]["save_dir"] = str(EXP_DIR / "620_film_v5_gated_5ep")

    # --- Baseline local smoke ---
    baseline = make_local_smoke(
        gated_5ep,
        "620_film_v5_gated_local_smoke",
        "Local smoke test for film_v5_gated with batch=4 accum=16; E1 optimal baseline",
    )
    baseline_dir = EXP_DIR / "620_film_v5_gated_local_smoke"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with open(baseline_dir / "config.json", "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Created {baseline_dir / 'config.json'}")

    # --- Experiment 1: Endpoint-FiLM Head ---
    endpoint_film = json.loads(json.dumps(baseline))
    endpoint_film["model"]["endpoint_head_mode"] = "endpoint_lowhigh"
    endpoint_film["model"]["endpoint_film_enabled"] = True
    endpoint_film["model"]["endpoint_lowpass_kernel"] = 5
    endpoint_film["model"]["endpoint_high_scale"] = 1.0
    endpoint_film["model"]["endpoint_velocity_floor"] = 0.05
    endpoint_film["model"]["endpoint_style_hidden_dim"] = 128
    name1 = "620_film_v5_endpoint_film_local_smoke"
    endpoint_film = make_local_smoke(
        endpoint_film,
        name1,
        "Endpoint head with FiLM style modulation + low/high decomposition; based on gated v5 baseline",
    )
    dir1 = EXP_DIR / name1
    dir1.mkdir(parents=True, exist_ok=True)
    with open(dir1 / "config.json", "w") as f:
        json.dump(endpoint_film, f, indent=2)
    print(f"Created {dir1 / 'config.json'}")

    # --- Experiment 2: High-Frequency Residual (velocity head only) ---
    hf_residual = json.loads(json.dumps(baseline))
    # Keep velocity head mode (default), add HF residual.
    hf_residual["model"]["velocity_hf_residual_enabled"] = True
    hf_residual["model"]["velocity_hf_residual_init"] = 0.1
    hf_residual["model"]["velocity_hf_residual_kernel"] = 5
    name2 = "620_film_v5_hf_residual_local_smoke"
    hf_residual = make_local_smoke(
        hf_residual,
        name2,
        "Velocity head with learnable high-pass residual of input latent; based on gated v5 baseline",
    )
    dir2 = EXP_DIR / name2
    dir2.mkdir(parents=True, exist_ok=True)
    with open(dir2 / "config.json", "w") as f:
        json.dump(hf_residual, f, indent=2)
    print(f"Created {dir2 / 'config.json'}")


if __name__ == "__main__":
    main()
