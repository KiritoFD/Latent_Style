"""628 Phase 2 v2: Extreme-weight loss ablation configs.

Generates configs for 9 real auxiliary losses with EXTREME weights (10x/50x/100x).
Each config resumes from T5 ep7, trains 3 new epochs.

The 9 losses are NOW REAL (implemented in spectral_losses620.py 628 patch):
  - w_velocity_magnitude (命题4: 训练-输出不匹配)
  - w_directional_cosine (命题3: SWD 梯度正交性)
  - w_output_variance (命题2: GN 白化定理)
  - w_contrast_preserve (内容保真)
  - w_channel_variance (反白化)
  - w_hf_energy (高频能量)
  - w_pixel_color_match (颜色匹配)
  - w_hsv_saturation (饱和度)
  - w_attn_entropy_reg (命题1: Gate Collapse)

Output: configs/ablations/628_destructive/ (X1-X31)
"""
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T5_CONFIG_PATH = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "config.json"
OUTPUT_DIR = ROOT / "configs" / "ablations" / "628_destructive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Extreme-weight loss ablations: (name, desc, overrides)
# 3 weight tiers per loss: w10 (moderate-extreme), w50 (extreme), w100 (very extreme)
ABLATIONS = [
    # ===== X1-X3: w_velocity_magnitude (命题4) =====
    {"name": "X1_velmag_w10", "desc": "w_velocity_magnitude=10 (命题4: velocity magnitude match, extreme)",
     "overrides": {"bridge.w_velocity_magnitude": 10.0}},
    {"name": "X2_velmag_w50", "desc": "w_velocity_magnitude=50 (very extreme)",
     "overrides": {"bridge.w_velocity_magnitude": 50.0}},
    {"name": "X3_velmag_w100", "desc": "w_velocity_magnitude=100 (maximum extreme)",
     "overrides": {"bridge.w_velocity_magnitude": 100.0}},

    # ===== X4-X6: w_directional_cosine (命题3) =====
    {"name": "X4_dir_cos_w10", "desc": "w_directional_cosine=10 (命题3: directional cosine, extreme)",
     "overrides": {"bridge.w_directional_cosine": 10.0}},
    {"name": "X5_dir_cos_w50", "desc": "w_directional_cosine=50 (very extreme)",
     "overrides": {"bridge.w_directional_cosine": 50.0}},
    {"name": "X6_dir_cos_w100", "desc": "w_directional_cosine=100 (maximum extreme)",
     "overrides": {"bridge.w_directional_cosine": 100.0}},

    # ===== X7-X9: w_output_variance (命题2) =====
    {"name": "X7_outvar_w10", "desc": "w_output_variance=10 (命题2: anti-whitening, extreme)",
     "overrides": {"bridge.w_output_variance": 10.0}},
    {"name": "X8_outvar_w50", "desc": "w_output_variance=50 (very extreme)",
     "overrides": {"bridge.w_output_variance": 50.0}},
    {"name": "X9_outvar_w100", "desc": "w_output_variance=100 (maximum extreme)",
     "overrides": {"bridge.w_output_variance": 100.0}},

    # ===== X10-X12: w_contrast_preserve (内容保真) =====
    {"name": "X10_contrast_w10", "desc": "w_contrast_preserve=10 (content contrast, extreme)",
     "overrides": {"bridge.w_contrast_preserve": 10.0}},
    {"name": "X11_contrast_w50", "desc": "w_contrast_preserve=50 (very extreme)",
     "overrides": {"bridge.w_contrast_preserve": 50.0}},
    {"name": "X12_contrast_w100", "desc": "w_contrast_preserve=100 (maximum extreme)",
     "overrides": {"bridge.w_contrast_preserve": 100.0}},

    # ===== X13-X15: w_channel_variance (反白化) =====
    {"name": "X13_chvar_w10", "desc": "w_channel_variance=10 (anti-whitening per-channel, extreme)",
     "overrides": {"bridge.w_channel_variance": 10.0}},
    {"name": "X14_chvar_w50", "desc": "w_channel_variance=50 (very extreme)",
     "overrides": {"bridge.w_channel_variance": 50.0}},
    {"name": "X15_chvar_w100", "desc": "w_channel_variance=100 (maximum extreme)",
     "overrides": {"bridge.w_channel_variance": 100.0}},

    # ===== X16-X18: w_hf_energy (高频能量) =====
    {"name": "X16_hfenergy_w10", "desc": "w_hf_energy=10 (high-freq energy preservation, extreme)",
     "overrides": {"bridge.w_hf_energy": 10.0}},
    {"name": "X17_hfenergy_w50", "desc": "w_hf_energy=50 (very extreme)",
     "overrides": {"bridge.w_hf_energy": 50.0}},
    {"name": "X18_hfenergy_w100", "desc": "w_hf_energy=100 (maximum extreme)",
     "overrides": {"bridge.w_hf_energy": 100.0}},

    # ===== X19-X21: w_pixel_color_match (颜色匹配) =====
    {"name": "X19_colormatch_w10", "desc": "w_pixel_color_match=10 (per-channel mean+std match, extreme)",
     "overrides": {"bridge.w_pixel_color_match": 10.0}},
    {"name": "X20_colormatch_w50", "desc": "w_pixel_color_match=50 (very extreme)",
     "overrides": {"bridge.w_pixel_color_match": 50.0}},
    {"name": "X21_colormatch_w100", "desc": "w_pixel_color_match=100 (maximum extreme)",
     "overrides": {"bridge.w_pixel_color_match": 100.0}},

    # ===== X22-X24: w_hsv_saturation (饱和度) =====
    {"name": "X22_hsvsat_w1", "desc": "w_hsv_saturation=1.0 (KL div saturation, moderate)",
     "overrides": {"bridge.w_hsv_saturation": 1.0}},
    {"name": "X23_hsvsat_w10", "desc": "w_hsv_saturation=10 (extreme)",
     "overrides": {"bridge.w_hsv_saturation": 10.0}},
    {"name": "X24_hsvsat_w50", "desc": "w_hsv_saturation=50 (very extreme)",
     "overrides": {"bridge.w_hsv_saturation": 50.0}},

    # ===== X25-X27: w_attn_entropy_reg (命题1: Gate Collapse) =====
    {"name": "X25_attnent_w1", "desc": "w_attn_entropy_reg=1.0 (命题1: attention entropy, moderate)",
     "overrides": {"bridge.w_attn_entropy_reg": 1.0}},
    {"name": "X26_attnent_w10", "desc": "w_attn_entropy_reg=10 (extreme)",
     "overrides": {"bridge.w_attn_entropy_reg": 10.0}},
    {"name": "X27_attnent_w50", "desc": "w_attn_entropy_reg=50 (very extreme)",
     "overrides": {"bridge.w_attn_entropy_reg": 50.0}},

    # ===== X28-X31: Combo experiments =====
    {"name": "X28_combo_content_w50", "desc": "Combo: all 4 content fidelity losses at w=50 each",
     "overrides": {"bridge.w_contrast_preserve": 50.0, "bridge.w_channel_variance": 50.0,
                   "bridge.w_hf_energy": 50.0, "bridge.w_pixel_color_match": 50.0}},
    {"name": "X29_combo_direction_w50", "desc": "Combo: all 3 direction constraint losses at w=50 each",
     "overrides": {"bridge.w_velocity_magnitude": 50.0, "bridge.w_directional_cosine": 50.0,
                   "bridge.w_output_variance": 50.0}},
    {"name": "X30_combo_all_w10", "desc": "Combo: ALL 9 losses at w=10 each (moderate combo)",
     "overrides": {"bridge.w_velocity_magnitude": 10.0, "bridge.w_directional_cosine": 10.0,
                   "bridge.w_output_variance": 10.0, "bridge.w_contrast_preserve": 10.0,
                   "bridge.w_channel_variance": 10.0, "bridge.w_hf_energy": 10.0,
                   "bridge.w_pixel_color_match": 10.0, "bridge.w_hsv_saturation": 10.0,
                   "bridge.w_attn_entropy_reg": 10.0}},
    {"name": "X31_combo_all_w50", "desc": "Combo: ALL 9 losses at w=50 each (extreme combo)",
     "overrides": {"bridge.w_velocity_magnitude": 50.0, "bridge.w_directional_cosine": 50.0,
                   "bridge.w_output_variance": 50.0, "bridge.w_contrast_preserve": 50.0,
                   "bridge.w_channel_variance": 50.0, "bridge.w_hf_energy": 50.0,
                   "bridge.w_pixel_color_match": 50.0, "bridge.w_hsv_saturation": 50.0,
                   "bridge.w_attn_entropy_reg": 50.0}},
]


def _set_nested(d: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def main() -> None:
    if not T5_CONFIG_PATH.exists():
        print(f"ERROR: T5 config not found: {T5_CONFIG_PATH}")
        sys.exit(1)

    with T5_CONFIG_PATH.open("r", encoding="utf-8") as f:
        base = json.load(f)

    # Ensure resume_checkpoint points to T5 ep7
    t5_ep7 = str(ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "epoch_0007.pt")
    _set_nested(base, "checkpoint.resume_checkpoint", t5_ep7)
    # 3 epochs续训 (ep8-ep10)
    _set_nested(base, "training.num_epochs", 10)
    _set_nested(base, "training.start_epoch", 8)
    # Windows paths (remote is Windows native)
    _set_nested(base, "training.test_image_dir", "I:/wikiart_distinct5_samam_512_classview/test")
    _set_nested(base, "data.data_root", "I:/wikiart_distinct5_samam_512_latents_ema/train")
    # === 628 fix: completely DISABLE full_eval to avoid 15min/epoch overhead ===
    # project_memory constraint: "Full evaluation (full_eval) must be disabled; only probe comparisons should be used"
    # defer_until_training_end=True causes per-epoch checkpoint queue → 10 ckpts × 90s = 15min wasted
    _set_nested(base, "training.full_eval_defer_until_training_end", False)
    _set_nested(base, "training.full_eval_each_epoch", False)
    _set_nested(base, "training.full_eval_force_regen", False)
    _set_nested(base, "training.full_eval_stop_on_convergence", False)
    # Save only final epoch (epoch 10) to skip per-epoch checkpoint overhead
    _set_nested(base, "training.save_interval", 5)

    count = 0
    for ablation in ABLATIONS:
        cfg = json.loads(json.dumps(base))  # deep copy
        for key, value in ablation["overrides"].items():
            _set_nested(cfg, key, value)
        # Set experiment name
        _set_nested(cfg, "checkpoint.save_dir",
                    f"I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive/{ablation['name']}")
        out_path = OUTPUT_DIR / f"{ablation['name']}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        count += 1
        print(f"  [{count:2d}/{len(ABLATIONS)}] {ablation['name']}: {ablation['desc']}")

    print(f"\nGenerated {count} extreme-weight loss configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
