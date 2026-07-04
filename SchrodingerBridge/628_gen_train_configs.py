"""628 Comprehensive Ablation: Training-side config generator.

Generates 8 training configs, each differing from T5 in exactly one parameter.
Each config trains for 1 epoch (smoke test), saves to exp/628_ablation/train_smoke/.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T5_CONFIG_PATH = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "config.json"
OUTPUT_DIR = ROOT / "configs" / "ablations" / "628_train_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ABLATIONS = [
    {
        "name": "T1_gate_warmup500",
        "desc": "Gate warmup 500 steps (prevent gate collapse)",
        "overrides": {
            "model.style_gate_warmup_steps": 500,
        },
    },
    {
        "name": "T2_rmsnorm_head",
        "desc": "RMSNorm in FiLM endpoint head (preserve color/contrast)",
        "overrides": {
            "model.endpoint_film_use_rmsnorm": True,
        },
    },
    {
        "name": "T3_contrast_preserve",
        "desc": "Anti-whitening: contrast preserve loss w=2.0",
        "overrides": {
            "bridge.w_contrast_preserve": 2.0,
        },
    },
    {
        "name": "T4_channel_variance",
        "desc": "Anti-whitening: channel variance loss w=0.5",
        "overrides": {
            "bridge.w_channel_variance": 0.5,
        },
    },
    {
        "name": "T5_hf_energy",
        "desc": "Anti-whitening: high-freq energy loss w=1.0",
        "overrides": {
            "bridge.w_hf_energy": 1.0,
        },
    },
    {
        "name": "T6_velocity_magnitude",
        "desc": "Velocity magnitude regularization w=1.0",
        "overrides": {
            "bridge.w_velocity_magnitude": 1.0,
        },
    },
    {
        "name": "T7_gate_init_03",
        "desc": "Gate init 0.3 (vs T5 default 0.05)",
        "overrides": {
            "model.style_cross_attn_gate_init": 0.3,
        },
    },
    {
        "name": "T8_spectral_fm",
        "desc": "Spectral FM loss: w_ll=0.5, w_lh=1, w_hl=1, w_hh=2",
        "overrides": {
            "bridge.spectral_w_ll": 0.5,
            "bridge.spectral_w_lh": 1.0,
            "bridge.spectral_w_hl": 1.0,
            "bridge.spectral_w_hh": 2.0,
        },
    },
]


def generate_configs():
    with open(T5_CONFIG_PATH, "r", encoding="utf-8") as f:
        base = json.load(f)

    for abl in ABLATIONS:
        cfg = json.loads(json.dumps(base))  # deep copy

        cfg["checkpoint"]["save_dir"] = f"./exp/628_ablation/train_smoke/{abl['name']}"
        cfg["training"]["num_epochs"] = 10
        cfg["training"]["save_interval"] = 1
        cfg["training"]["full_eval_each_epoch"] = True
        cfg["training"]["full_eval_defer_until_training_end"] = False
        cfg["training"]["resume_training_state"] = True
        cfg["training"]["resume_optimizer"] = True
        cfg["training"]["resume_model_strict"] = True

        cfg["ablation"] = {
            "name": abl["name"],
            "axis": "628_train_smoke",
            "stage": "ep8-10",
            "notes": abl["desc"],
        }

        for dotted_key, value in abl["overrides"].items():
            parts = dotted_key.split(".")
            section = parts[0]
            key = parts[1]
            cfg.setdefault(section, {})[key] = value

        import sys
        is_win = sys.platform == "win32"
        if is_win:
            path_fixes = {
                "data.data_root": "I:/wikiart_distinct5_samam_512_latents_ema/train",
                "training.test_image_dir": "I:/wikiart_distinct5_samam_512_classview/test",
                "training.full_eval_cache_dir": "I:/Github/Latent_Style/eval_cache",
                "training.full_eval_clip_hf_cache_dir": "I:/Github/Latent_Style/eval_cache/hf",
                "data.latent_cache_dir": "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed",
                "data.pairing_cache_path": "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt",
            }
        else:
            path_fixes = {
                "data.data_root": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train",
                "training.test_image_dir": "/mnt/i/wikiart_distinct5_samam_512_classview/test",
                "training.full_eval_cache_dir": "/mnt/i/eval_cache",
                "training.full_eval_clip_hf_cache_dir": "/mnt/i/eval_cache/hf",
                "data.latent_cache_dir": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed",
                "data.pairing_cache_path": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt",
            }
        for dotted_key, value in path_fixes.items():
            parts = dotted_key.split(".")
            cfg[parts[0]][parts[1]] = value

        if is_win:
            cfg["training"]["resume_checkpoint"] = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt"
        else:
            cfg["training"]["resume_checkpoint"] = "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt"

        out_path = OUTPUT_DIR / f"{abl['name']}.json"
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"  Generated {out_path.name}: {abl['desc']}")


if __name__ == "__main__":
    print(f"Generating {len(ABLATIONS)} training smoke configs from T5 baseline...")
    generate_configs()
    print(f"Done. Output: {OUTPUT_DIR}")
