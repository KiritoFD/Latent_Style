"""FC-SB Phase 4 D1/D2: E4-long ep5 + DWT lowpass evaluation.

Loads E4-long ep5 checkpoint (clip=0.727, lpips=0.581) and re-evaluates with
different lowpass_mode settings. Goal: improve LPIPS while keeping CLIP>0.72.

Usage:
    python _dwt_eval.py avg_pool      # baseline (should reproduce 0.727/0.581)
    python _dwt_eval.py dwt_haar      # DWT orthogonal LL locking
    python _dwt_eval.py wavelet       # existing wavelet (downsample+upsample)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import importlib

run_module = importlib.import_module("run")
config_schema = importlib.import_module("config_schema")

ExperimentConfig = config_schema.ExperimentConfig
load_config = config_schema.load_config

# E4-long ep5 checkpoint (project memory: clip=0.727, lpips=0.581)
CKPT_PATH = ROOT / "exp" / "p3_remote_10h" / "e4_long_10ep" / "checkpoints" / "epoch_0005.pt"
CONFIG_PATH = ROOT / "exp" / "625_fc_sb" / "configs" / "E4_inference_fc_only.json"


def main() -> None:
    lowpass_mode = sys.argv[1] if len(sys.argv) > 1 else "dwt_haar"
    print(f"[dwt_eval] lowpass_mode={lowpass_mode}")
    print(f"[dwt_eval] checkpoint={CKPT_PATH}")
    print(f"[dwt_eval] config={CONFIG_PATH}")

    if not CKPT_PATH.exists():
        print(f"[dwt_eval] ERROR: checkpoint not found: {CKPT_PATH}")
        sys.exit(1)
    if not CONFIG_PATH.exists():
        print(f"[dwt_eval] ERROR: config not found: {CONFIG_PATH}")
        sys.exit(1)

    raw = load_config(str(CONFIG_PATH))
    config = ExperimentConfig.from_mapping(raw)
    # Merge raw sections into config (align with run.py behavior)
    for section_name in ("model", "bridge", "training", "data", "checkpoint", "full_eval"):
        raw_section = raw.get(section_name, {})
        section_obj = getattr(config, section_name, None)
        if not isinstance(raw_section, dict) or section_obj is None:
            continue
        for key, value in raw_section.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)

    # Override lowpass_mode
    config.model.lowpass_mode = lowpass_mode
    print(f"[dwt_eval] config.model.lowpass_mode set to '{config.model.lowpass_mode}'")
    print(f"[dwt_eval] contract_family={getattr(config.model, 'contract_family', '?')}")

    # Fix Windows paths (config uses /mnt/i/ WSL paths, remote is Windows native)
    if hasattr(config.training, 'test_image_dir'):
        config.training.test_image_dir = "I:/wikiart_distinct5_samam_512_classview/test"
    if hasattr(config.training, 'full_eval_cache_dir'):
        config.training.full_eval_cache_dir = "I:/Github/Latent_Style/eval_cache"
    if hasattr(config.training, 'full_eval_clip_hf_cache_dir'):
        config.training.full_eval_clip_hf_cache_dir = "I:/Github/Latent_Style/eval_cache/hf"
    if hasattr(config.data, 'data_root'):
        config.data.data_root = "I:/wikiart_distinct5_samam_512_latents_ema/train"
    if hasattr(config.data, 'latent_cache_dir'):
        config.data.latent_cache_dir = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
    if hasattr(config.data, 'pairing_cache_path'):
        config.data.pairing_cache_path = "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt"

    print(f"[dwt_eval] invoking full eval...")
    result = run_module._run_full_eval_for_checkpoint(config, CKPT_PATH)
    print(f"[dwt_eval] done. result={result}")


if __name__ == "__main__":
    main()
