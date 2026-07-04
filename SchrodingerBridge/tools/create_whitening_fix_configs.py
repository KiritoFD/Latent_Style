#!/usr/bin/env python3
"""
Create 620 smoke test configs for whitening fix.
Modifies the intrinsic_v2 config with:
- swd_noise_sigma=0.02 (NSWD)
- gate=0.3 is now default in code, no config change needed
- Larger endpoint head is now default in code
"""
import json, os, sys, shutil
from pathlib import Path

# Add src/ to path to use the project's config loader
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from config_schema import load_config

base_config_path = os.environ.get(
    "SB620_BASE_CONFIG",
    str(SCRIPT_DIR.parent / "configs" / "620_spatial_bridge_intrinsic.json"),
)
exp_dir = os.environ.get("SB620_EXP_DIR", str(SCRIPT_DIR.parent / "exp" / "620_spatial_bridge"))

variants = [
    # === film_v4 series: explore attention alternatives ===
    {
        "name": "620_film_v4_gated_5ep",
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {
            "style_cross_attn_gate_init": 0.3,
            "style_film_enabled": True,
            "style_attn_mode": "gated",
            "style_attn_temperature": 0.5,
        },
        "training": {"num_epochs": 5},
        "ablation": {"name": "620_film_v4_gated_5ep", "notes": "Gated attention (sigmoid) + temp=0.5 + pre/post FiLM + style_bias"},
    },
    {
        "name": "620_film_v4_sparsemax_5ep",
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {
            "style_cross_attn_gate_init": 0.3,
            "style_film_enabled": True,
            "style_attn_mode": "sparsemax",
            "style_attn_temperature": 0.5,
        },
        "training": {"num_epochs": 5},
        "ablation": {"name": "620_film_v4_sparsemax_5ep", "notes": "Sparsemax attention + temp=0.5 + pre/post FiLM + style_bias"},
    },
    {
        "name": "620_film_v4_softmax_temp_5ep",
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {
            "style_cross_attn_gate_init": 0.3,
            "style_film_enabled": True,
            "style_attn_mode": "softmax",
            "style_attn_temperature": 0.1,
        },
        "training": {"num_epochs": 5},
        "ablation": {"name": "620_film_v4_softmax_temp_5ep", "notes": "Softmax + temp=0.1 (very sharp) + pre/post FiLM + style_bias"},
    },
    # === film_v5 series: fully replace softmax, non-zero init ===
    {
        "name": "620_film_v5_gated_5ep",
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {
            "style_cross_attn_gate_init": 0.3,
            "style_film_enabled": True,
            "style_attn_mode": "gated",
            "style_attn_temperature": 1.0,
        },
        "training": {"num_epochs": 5},
        "ablation": {"name": "620_film_v5_gated_5ep", "notes": "Gated attention (no softmax anywhere) + non-zero FiLM init (std=0.02) + FiLM + bias"},
    },
    {
        "name": "620_film_v5_sparsemax_5ep",
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {
            "style_cross_attn_gate_init": 0.3,
            "style_film_enabled": True,
            "style_attn_mode": "sparsemax",
            "style_attn_temperature": 1.0,
        },
        "training": {"num_epochs": 5},
        "ablation": {"name": "620_film_v5_sparsemax_5ep", "notes": "Sparsemax attention (no softmax) + non-zero FiLM init + FiLM + bias"},
    },
    {
        "name": "620_film_v5_gated_agg_5ep",
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {
            "style_cross_attn_gate_init": 0.3,
            "style_film_enabled": True,
            "style_attn_mode": "gated",
            "style_attn_temperature": 0.3,
        },
        "training": {"num_epochs": 5},
        "ablation": {"name": "620_film_v5_gated_agg_5ep", "notes": "Gated + temp=0.3 (aggressive sharp gates) + non-zero FiLM init"},
    },
    # === film_v3/v2/gate03: baselines for comparison ===
    {
        "name": "620_film_v3_5ep",
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {"style_cross_attn_gate_init": 0.3, "style_film_enabled": True},
        "training": {"num_epochs": 5},
        "ablation": {"name": "620_film_v3_5ep", "notes": "StyleFiLM v3: pre-FILM + post-FILM + style_bias + gate=0.3"},
    },
    {
        "name": "620_film_v2_5ep",
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {"style_cross_attn_gate_init": 0.3, "style_film_enabled": True},
        "training": {"num_epochs": 5},
        "ablation": {"name": "620_film_v2_5ep", "notes": "StyleFiLM v2: pre-cross-attn FiLM + post-cross-attn FiLM + gate=0.3"},
    },
    {
        "name": "620_film_gate03_5ep",
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {"style_cross_attn_gate_init": 0.3, "style_film_enabled": True},
        "training": {"num_epochs": 5},
        "ablation": {"name": "620_film_gate03_5ep", "notes": "StyleFiLM + gate=0.3 + larger endpoint head, 5 epochs"},
    },
    # === smoke tests ===
    {
        "name": "620_film_gate03_smoke",
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {"style_cross_attn_gate_init": 0.3, "style_film_enabled": True},
        "training": {"num_epochs": 1},
        "ablation": {"name": "620_film_gate03_smoke", "notes": "StyleFiLM + gate=0.3 + larger endpoint head"},
    },
    {
        "name": "620_nswd_gate03_smoke",
        "bridge": {"swd_noise_sigma": 0.02},
        "model": {"style_cross_attn_gate_init": 0.3},
        "training": {"num_epochs": 1},
        "ablation": {"name": "620_nswd_gate03_smoke", "notes": "NSWD sigma=0.02 + gate=0.3 + larger endpoint head"},
    },
    {
        "name": "620_nswd_s005_smoke",
        "bridge": {"swd_noise_sigma": 0.05},
        "model": {"style_cross_attn_gate_init": 0.3},
        "training": {"num_epochs": 1},
        "ablation": {"name": "620_nswd_s005_smoke", "notes": "NSWD sigma=0.05 + gate=0.3"},
    },
    {
        "name": "620_nswd_s01_smoke",
        "bridge": {"swd_noise_sigma": 0.01},
        "model": {"style_cross_attn_gate_init": 0.3},
        "training": {"num_epochs": 1},
        "ablation": {"name": "620_nswd_s01_smoke", "notes": "NSWD sigma=0.01 + gate=0.3"},
    },
]

print(f"Loading base config: {base_config_path}")
print(f"Experiment dir: {exp_dir}")
base = load_config(base_config_path)

for v in variants:
    cfg = json.loads(json.dumps(base))
    name = v["name"]
    out_dir = os.path.join(exp_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    
    for section, updates in v.items():
        if section == "name":
            continue
        if section in cfg:
            cfg[section].update(updates)
    
    cfg["checkpoint"]["save_dir"] = os.path.join(exp_dir, name)

    # Ensure full_eval saves generated images for WFI benchmark
    if "full_eval" not in cfg:
        cfg["full_eval"] = {}
    cfg["full_eval"]["save_generated_images"] = True
    cfg["full_eval"]["save_summary_grid"] = True

    out_path = os.path.join(out_dir, "config.json")
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Created {out_path}")

print("Done!")