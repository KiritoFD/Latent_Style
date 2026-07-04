"""Phase 8F: Orthogonal ablation of clean_base's 5 modifications.

clean_base组合后clip=0.7073 (下降0.0234), 需找出罪魁祸首.
5项修改:
  A: spectral_w_ll=2.0 (was 0.3)
  B: spectral_w_lh=0.0 (was 1.0)
  C: spectral_w_hl=0.0 (was 1.0)
  D: w_channel_variance=1.0 (was 0.0)
  E: w_pixel_color_match=10.0 (was 0.0)

实验设计:
  O1: 只A (spectral_w_ll=2.0)
  O2: 只B+C (spectral_lh/hl=0, 移除有害)
  O3: 只D (channel_variance=1.0)
  O4: 只E (color_match=10.0)
  O5: A+D (ll + chvar)
  O6: A+E (ll + color)
  O7: D+E (chvar + color)
  O8: B+C+D (移除有害 + chvar)
  O9: B+C+E (移除有害 + color)
  O10: A+B+C (ll + 移除有害, 无 aux loss)
  O11: D+E+B+C (chvar+color+移除有害, 无 ll)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T5_CONFIG = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "config.json"
OUTPUT_DIR = ROOT / "configs" / "ablations" / "628_orthogonal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

T5_EP7 = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt"

# Baseline values
BASELINE = {
    "bridge.spectral_w_ll": 0.3,
    "bridge.spectral_w_lh": 1.0,
    "bridge.spectral_w_hl": 1.0,
    "bridge.w_channel_variance": 0.0,
    "bridge.w_pixel_color_match": 0.0,
}

# Modifications
MOD_A = {"bridge.spectral_w_ll": 2.0}
MOD_BC = {"bridge.spectral_w_lh": 0.0, "bridge.spectral_w_hl": 0.0}
MOD_D = {"bridge.w_channel_variance": 1.0}
MOD_E = {"bridge.w_pixel_color_match": 10.0}

EXPERIMENTS = [
    ("O1_only_ll", "Only A: spectral_w_ll=2.0", {**MOD_A}),
    ("O2_only_lhhl0", "Only B+C: spectral_lh/hl=0", {**MOD_BC}),
    ("O3_only_chvar", "Only D: channel_variance=1.0", {**MOD_D}),
    ("O4_only_color", "Only E: color_match=10.0", {**MOD_E}),
    ("O5_ll_chvar", "A+D: ll + chvar", {**MOD_A, **MOD_D}),
    ("O6_ll_color", "A+E: ll + color", {**MOD_A, **MOD_E}),
    ("O7_chvar_color", "D+E: chvar + color", {**MOD_D, **MOD_E}),
    ("O8_lhhl0_chvar", "B+C+D: lhhl0 + chvar", {**MOD_BC, **MOD_D}),
    ("O9_lhhl0_color", "B+C+E: lhhl0 + color", {**MOD_BC, **MOD_E}),
    ("O10_ll_lhhl0", "A+B+C: ll + lhhl0 (no aux)", {**MOD_A, **MOD_BC}),
    ("O11_chvar_color_lhhl0", "D+E+B+C: chvar+color+lhhl0", {**MOD_BC, **MOD_D, **MOD_E}),
]


def _set_nested(d: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def main() -> None:
    with T5_CONFIG.open("r", encoding="utf-8") as f:
        base = json.load(f)

    # Common settings
    _set_nested(base, "checkpoint.resume_checkpoint", T5_EP7)
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
    for name, desc, mods in EXPERIMENTS:
        cfg = json.loads(json.dumps(base))
        for key, value in mods.items():
            _set_nested(cfg, key, value)
        _set_nested(cfg, "checkpoint.save_dir",
                    f"I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/orthogonal/{name}")
        out_path = OUTPUT_DIR / f"{name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        count += 1
        print(f"  [{count:2d}/{len(EXPERIMENTS)}] {name}: {desc}")

    print(f"\nGenerated {count} orthogonal ablation configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
