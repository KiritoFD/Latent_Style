#!/usr/bin/env python3
"""Generate the 27-style WEAVE training+eval config from the paper template.

Reads aaai2027_v4/_remote_config.json (the WEAVE recipe) and overrides only what
is needed to train/evaluate on all 27 WikiArt styles with all data:
  * num_styles = 27, style_subdirs = the 27 sorted WikiArt style names
  * data_root -> freshly encoded SD1.5-EMA latents on F:
  * test_image_dir -> the 27-style classview RGB test split on F:
  * eval cache dirs -> F:/eval_cache (offline CLIP/LPIPS/VAE caches)
  * protocol matching the paper text: 5 epochs, batch 24, LR 1e-4
  * pairing cache disabled (random uniform target sampling) to avoid a costly
    DINO cache build; eval runs once on the final checkpoint.
Writes aaai2027_v4/weave_wikiart27_config.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "aaai2027_v4" / "_remote_config.json"
OUT = ROOT / "aaai2027_v4" / "weave_wikiart27_config.json"

RAW_WIKIART = Path("F:/wikiart/wikiart")
LATENT_ROOT = "G:/wikiart27_latents_compact/train"
TEST_DIR = "G:/wikiart27_classview_test/test"
EVAL_CACHE = "G:/wikiart27_eval_cache"
EVAL_HF_CACHE = "G:/wikiart27_eval_cache/hf"
SAVE_DIR = "G:/wikiart27_weave_exp/run1"


def main() -> None:
    styles = sorted(d.name for d in RAW_WIKIART.iterdir() if d.is_dir())
    assert len(styles) == 27, f"expected 27 styles, found {len(styles)}: {styles}"
    print("27 styles:", styles)

    cfg = json.loads(TEMPLATE.read_text(encoding="utf-8"))

    # ---- model ----
    cfg["model"]["num_styles"] = len(styles)

    # ---- data ----
    cfg["data"]["style_subdirs"] = styles
    cfg["data"]["data_root"] = LATENT_ROOT
    cfg["data"]["latent_cache_mode"] = "manifest"
    cfg["data"]["latent_cache_dir"] = f"{LATENT_ROOT}/.latent_cache"
    cfg["data"]["pairing_cache_path"] = ""          # disabled: random uniform targets
    cfg["data"]["pairing_cache_topk"] = 0
    cfg["data"]["pairing_cache_active_topk"] = 0

    # ---- training ----
    tr = cfg["training"]
    tr["num_epochs"] = 5
    tr["batch_size"] = 24
    tr["learning_rate"] = 1e-4
    tr["min_learning_rate"] = 1e-5
    tr["test_image_dir"] = TEST_DIR
    tr["full_eval_cache_dir"] = EVAL_CACHE
    tr["full_eval_clip_hf_cache_dir"] = EVAL_HF_CACHE
    tr["full_eval_max_src_samples"] = 30            # 27x27x30 all-pairs, matches Random5 protocol
    tr["full_eval_max_ref_compare"] = 16
    tr["full_eval_max_ref_cache"] = 16
    tr["full_eval_save_generated_images"] = False   # metrics only; saves disk/time
    tr["full_eval_each_epoch"] = False
    tr["full_eval_defer_until_training_end"] = True
    tr["save_interval"] = 5                          # evaluate only the final checkpoint
    tr["full_eval_vae_decode_batch_size"] = 2

    # ---- checkpoint ----
    cfg["checkpoint"]["save_dir"] = SAVE_DIR
    cfg["checkpoint"]["resume_checkpoint"] = ""

    # ---- full_eval ----
    cfg["full_eval"]["vae_model"] = "ema"
    cfg["full_eval"]["num_steps"] = 8
    cfg["full_eval"]["batch_size"] = 2
    cfg["full_eval"]["save_summary_grid"] = False

    # ---- ablation (avoid round2 manifest side-effects) ----
    cfg["ablation"]["name"] = "wikiart27_scale"

    OUT.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote config -> {OUT}")
    print(f"  data_root      = {LATENT_ROOT}")
    print(f"  test_image_dir = {TEST_DIR}")
    print(f"  save_dir       = {SAVE_DIR}")


if __name__ == "__main__":
    main()
