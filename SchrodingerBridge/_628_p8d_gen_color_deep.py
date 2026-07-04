"""Phase 8D: color_match deep exploration + Pareto-optimal combos.

Phase 8C discovery: color_match broke the 0.74 ceiling!
  X21_colormatch_w100: all_pairs.clip=0.7415 (broke 0.74!)
  X20_colormatch_w50:  all_pairs.clip=0.7411 (also broke 0.74!)
  X19_colormatch_w10:  all_pairs.clip=0.7344

This phase explores:
  1. Fine-grained color_match weights (20, 30, 70, 150, 300) to find saturation
  2. Combos with content-preserving losses to improve Pareto front
  3. Combos with other effective style losses for additive boost

Output: configs/ablations/628_destructive/ (D1-D12)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T5_CONFIG_PATH = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "config.json"
OUTPUT_DIR = ROOT / "configs" / "ablations" / "628_destructive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ABLATIONS = [
    # ===== D1-D5: Fine-grained color_match weight sweep =====
    {"name": "D1_color_w20", "desc": "w_pixel_color_match=20 (between w10 and w50)",
     "overrides": {"bridge.w_pixel_color_match": 20.0}},
    {"name": "D2_color_w30", "desc": "w_pixel_color_match=30",
     "overrides": {"bridge.w_pixel_color_match": 30.0}},
    {"name": "D3_color_w70", "desc": "w_pixel_color_match=70 (between w50 and w100)",
     "overrides": {"bridge.w_pixel_color_match": 70.0}},
    {"name": "D4_color_w150", "desc": "w_pixel_color_match=150 (beyond w100, saturation test)",
     "overrides": {"bridge.w_pixel_color_match": 150.0}},
    {"name": "D5_color_w300", "desc": "w_pixel_color_match=300 (extreme saturation test)",
     "overrides": {"bridge.w_pixel_color_match": 300.0}},

    # ===== D6-D7: color_match + hsv_sat (additive style boost) =====
    {"name": "D6_color50_hsv10", "desc": "color_match=50 + hsv_sat=10 (combo style boost)",
     "overrides": {"bridge.w_pixel_color_match": 50.0, "bridge.w_hsv_saturation": 10.0}},
    {"name": "D7_color50_hsv50", "desc": "color_match=50 + hsv_sat=50 (extreme combo style)",
     "overrides": {"bridge.w_pixel_color_match": 50.0, "bridge.w_hsv_saturation": 50.0}},

    # ===== D8-D9: color_match + dir_cos (style+content balance) =====
    {"name": "D8_color50_dircos10", "desc": "color_match=50 + dir_cos=10 (style+content balance)",
     "overrides": {"bridge.w_pixel_color_match": 50.0, "bridge.w_directional_cosine": 10.0}},
    {"name": "D9_color50_dircos50", "desc": "color_match=50 + dir_cos=50 (extreme balance)",
     "overrides": {"bridge.w_pixel_color_match": 50.0, "bridge.w_directional_cosine": 50.0}},

    # ===== D10: Maximum style+content =====
    {"name": "D10_color100_dircos50", "desc": "color_match=100 + dir_cos=50 (maximum style+content)",
     "overrides": {"bridge.w_pixel_color_match": 100.0, "bridge.w_directional_cosine": 50.0}},

    # ===== D11: color_match + vel_mag (style+velocity) =====
    {"name": "D11_color50_velmag10", "desc": "color_match=50 + vel_mag=10 (style+velocity)",
     "overrides": {"bridge.w_pixel_color_match": 50.0, "bridge.w_velocity_magnitude": 10.0}},

    # ===== D12: color_match + ch_var (style+anti-whitening) =====
    {"name": "D12_color50_chvar10", "desc": "color_match=50 + ch_var=10 (style+anti-whitening)",
     "overrides": {"bridge.w_pixel_color_match": 50.0, "bridge.w_channel_variance": 10.0}},
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

    t5_ep7 = str(ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "epoch_0007.pt")
    _set_nested(base, "checkpoint.resume_checkpoint", t5_ep7)
    _set_nested(base, "training.num_epochs", 10)
    _set_nested(base, "training.start_epoch", 8)
    _set_nested(base, "training.test_image_dir", "I:/wikiart_distinct5_samam_512_classview/test")
    _set_nested(base, "data.data_root", "I:/wikiart_distinct5_samam_512_latents_ema/train")
    _set_nested(base, "training.full_eval_defer_until_training_end", False)
    _set_nested(base, "training.full_eval_each_epoch", False)
    _set_nested(base, "training.full_eval_force_regen", False)
    _set_nested(base, "training.full_eval_stop_on_convergence", False)
    _set_nested(base, "training.save_interval", 5)

    count = 0
    for ablation in ABLATIONS:
        cfg = json.loads(json.dumps(base))
        for key, value in ablation["overrides"].items():
            _set_nested(cfg, key, value)
        _set_nested(cfg, "checkpoint.save_dir",
                    f"I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive/{ablation['name']}")
        out_path = OUTPUT_DIR / f"{ablation['name']}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        count += 1
        print(f"  [{count:2d}/{len(ABLATIONS)}] {ablation['name']}: {ablation['desc']}")

    print(f"\nGenerated {count} Phase 8D configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
