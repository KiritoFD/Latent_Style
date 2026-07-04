"""628 Few-shot experiment config generator.

Creates training configs for 5+1/5+2/5+3 style few-shot experiments.
Each config uses freeze_mode=tokenizer_only, expanded checkpoint, and 
the prepared few-shot datasets.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T5_CONFIG_PATH = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "config.json"
T5_CKPT_PATH = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "epoch_0007.pt"
FEWSHOT_DATA_BASE = Path(r"I:\fewshot_data")
EXPANDED_CKPT_BASE = ROOT / "exp" / "628_fewshot" / "expanded_ckpt"
OUTPUT_DIR = ROOT / "configs" / "ablations" / "628_fewshot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPANDED_CKPT_BASE.mkdir(parents=True, exist_ok=True)

BASE_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
NEW_STYLE_CANDIDATES = ["Expressionism", "Post_Impressionism", "Realism"]
SHOT_COUNTS = [1, 6, 10, 30, 50]


def generate_configs():
    with open(T5_CONFIG_PATH, "r", encoding="utf-8") as f:
        base = json.load(f)

    is_win = sys.platform == "win32"
    experiments = []

    for n_new in [1, 2, 3]:
        new_styles = NEW_STYLE_CANDIDATES[:n_new]
        all_styles = BASE_STYLES + new_styles
        num_styles = 5 + n_new
        expanded_ckpt_name = f"t5ep7_expanded_{num_styles}styles.pt"
        expanded_ckpt_path = EXPANDED_CKPT_BASE / expanded_ckpt_name

        for shots in SHOT_COUNTS:
            exp_name = f"5p{n_new}_shot{shots:02d}"
            data_root = str(FEWSHOT_DATA_BASE / exp_name)

            cfg = json.loads(json.dumps(base))

            # Model: update num_styles and style_subdirs
            cfg["model"]["num_styles"] = num_styles
            cfg["data"]["style_subdirs"] = all_styles

            # Data: point to few-shot dataset
            cfg["data"]["data_root"] = data_root
            cfg["data"]["latent_cache_dir"] = f"{data_root}/.latent_cache/packed"
            cfg["data"]["pairing_cache_path"] = ""
            cfg["data"]["pairing_cache_cross_only"] = True

            # Training
            cfg["training"]["num_epochs"] = 10
            cfg["training"]["save_interval"] = 1
            cfg["training"]["full_eval_each_epoch"] = True
            cfg["training"]["full_eval_defer_until_training_end"] = False
            cfg["training"]["freeze_mode"] = "tokenizer_only"
            cfg["training"]["freeze_reinit_trainable"] = False
            cfg["training"]["learning_rate"] = 0.0002
            cfg["training"]["resume_checkpoint"] = str(expanded_ckpt_path)
            cfg["training"]["resume_model_strict"] = False
            cfg["training"]["resume_optimizer"] = False
            cfg["training"]["resume_training_state"] = False

            # Test
            cfg["training"]["test_image_dir"] = f"{data_root}/test"

            # Checkpoint
            cfg["checkpoint"]["save_dir"] = f"./exp/628_fewshot/{exp_name}"

            # Ablation metadata
            cfg["ablation"] = {
                "name": exp_name,
                "axis": "628_fewshot",
                "stage": "few_shot",
                "notes": f"5+{n_new} styles, {shots} shots per new style, freeze_mode=tokenizer_only",
                "new_styles": new_styles,
                "shots": shots,
            }

            # Windows path fixes
            if is_win:
                cfg["training"]["full_eval_cache_dir"] = "I:/Github/Latent_Style/eval_cache"
                cfg["training"]["full_eval_clip_hf_cache_dir"] = "I:/Github/Latent_Style/eval_cache/hf"

            out_path = OUTPUT_DIR / f"{exp_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            
            experiments.append({
                "exp_name": exp_name,
                "num_styles": num_styles,
                "new_styles": new_styles,
                "shots": shots,
                "expanded_ckpt": str(expanded_ckpt_path),
                "config_path": str(out_path),
            })
            print(f"  Generated {exp_name}: 5+{n_new} styles, {shots} shots")

    return experiments


if __name__ == "__main__":
    print(f"Generating few-shot training configs...")
    experiments = generate_configs()
    print(f"\nNeed to expand checkpoint for these num_styles values:")
    needed = sorted(set(e["num_styles"] for e in experiments))
    for n in needed:
        print(f"  5 -> {n}: {EXPANDED_CKPT_BASE / f't5ep7_expanded_{n}styles.pt'}")
    print(f"\nRun expand_checkpoint_num_styles.py for each, then start training.")
    print(f"Done. Output: {OUTPUT_DIR}")
