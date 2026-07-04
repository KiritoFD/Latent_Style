"""Phase 7A: Generate D0_control config (T5 ep7 + 10 epoch resume, NO modifications).

This is the CONTROL experiment to verify the Conservative Attractor hypothesis:
If T5 ep7 + 10 epoch resume WITHOUT any modifications also degrades to
(clip=0.7011, lpips=0.3520), then the attractor hypothesis is CONFIRMED.
If it stays at (0.7307, 0.3403) or improves, the hypothesis is REFUTED and
the degradation seen in D/L/E/P experiments comes from accumulated tiny deltas.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T5_CONFIG_PATH = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "config.json"
OUTPUT_DIR = ROOT / "configs" / "ablations" / "628_destructive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not T5_CONFIG_PATH.is_file():
        print(f"ERROR: T5 config not found: {T5_CONFIG_PATH}")
        sys.exit(1)
    with T5_CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Apply same training settings as other D/L/E/P experiments
    cfg["checkpoint"]["save_dir"] = "./exp/628_ablation/destructive/D0_control"
    cfg["training"]["num_epochs"] = 10
    cfg["training"]["save_interval"] = 1
    cfg["training"]["full_eval_each_epoch"] = True
    cfg["training"]["full_eval_defer_until_training_end"] = False
    cfg["training"]["full_eval_force_regen"] = True
    cfg["training"]["full_eval_stop_on_convergence"] = False
    cfg["training"]["resume_training_state"] = True
    cfg["training"]["resume_optimizer"] = True
    cfg["training"]["resume_model_strict"] = True
    cfg["training"]["resume_checkpoint"] = (
        "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt"
    )

    # Path fixes (Windows remote)
    cfg["data"]["data_root"] = "I:/wikiart_distinct5_samam_512_latents_ema/train"
    cfg["training"]["test_image_dir"] = "I:/wikiart_distinct5_samam_512_classview/test"
    cfg["training"]["full_eval_cache_dir"] = "I:/Github/Latent_Style/eval_cache"
    cfg["training"]["full_eval_clip_hf_cache_dir"] = "I:/Github/Latent_Style/eval_cache/hf"
    cfg["data"]["latent_cache_dir"] = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
    cfg["data"]["pairing_cache_path"] = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt"

    cfg["ablation"] = {
        "name": "D0_control",
        "axis": "628_destructive",
        "stage": "ep8-10",
        "notes": "PHASE 7A CONTROL: T5 ep7 + 10 epoch resume, NO modifications. Verifies Conservative Attractor hypothesis.",
    }

    out_path = OUTPUT_DIR / "D0_control.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"Generated {out_path}")
    print(f"  save_dir: {cfg['checkpoint']['save_dir']}")
    print(f"  resume: {cfg['training']['resume_checkpoint']}")
    print(f"  num_epochs: {cfg['training']['num_epochs']}")


if __name__ == "__main__":
    main()
