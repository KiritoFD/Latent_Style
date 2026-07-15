"""Re-eval all checkpoints with save_generated_images=true, then run DINO eval."""
import sys
import os
import subprocess
import json
from pathlib import Path

sys.path.insert(0, "C:/Users/Administrator/SchrodingerBridge/src")
from config_schema import load_experiment_config

config = load_experiment_config("default_config.json")
train_cfg = config.training

ckpt_dir = Path("I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep")
eval_script = Path("C:/Users/Administrator/SchrodingerBridge/src/utils/run_evaluation.py")

# Build base command from config (mirrors run.py _run_full_eval_for_checkpoint)
def build_cmd(checkpoint_path):
    out_dir = checkpoint_path.parent / "full_eval" / checkpoint_path.stem
    cmd = [
        sys.executable, str(eval_script),
        "--checkpoint", str(checkpoint_path),
        "--output", str(out_dir),
        "--test_dir", str(train_cfg.test_image_dir),
        "--cache_dir", str(train_cfg.full_eval_cache_dir),
        "--clip_hf_cache_dir", str(train_cfg.full_eval_clip_hf_cache_dir),
        "--batch_size", str(int(train_cfg.full_eval_batch_size)),
        "--save_generated_images",
    ]
    if train_cfg.full_eval_num_steps is not None:
        cmd += ["--num_steps", str(int(train_cfg.full_eval_num_steps))]
    if train_cfg.full_eval_max_src_samples is not None:
        cmd += ["--max_src_samples", str(int(train_cfg.full_eval_max_src_samples))]
    if train_cfg.full_eval_max_ref_compare is not None:
        cmd += ["--max_ref_compare", str(int(train_cfg.full_eval_max_ref_compare))]
    if train_cfg.full_eval_max_ref_cache is not None:
        cmd += ["--max_ref_cache", str(int(train_cfg.full_eval_max_ref_cache))]
    if train_cfg.full_eval_ref_feature_batch_size is not None:
        cmd += ["--ref_feature_batch_size", str(int(train_cfg.full_eval_ref_feature_batch_size))]
    if train_cfg.full_eval_target_chunk_size is not None:
        cmd += ["--target_chunk_size", str(int(train_cfg.full_eval_target_chunk_size))]
    if train_cfg.full_eval_vae_decode_batch_size is not None:
        cmd += ["--vae_decode_batch_size", str(int(train_cfg.full_eval_vae_decode_batch_size))]
    return cmd, out_dir

for e in range(1, 16):
    ckpt = ckpt_dir / f"epoch_{e:04d}.pt"
    if not ckpt.exists():
        print(f"Epoch {e}: checkpoint missing, skipping")
        continue
    cmd, out_dir = build_cmd(ckpt)
    images_dir = out_dir / "images"
    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"Epoch {e}: images already exist, skipping eval")
    else:
        print(f"Epoch {e}: running eval with save_generated_images...", flush=True)
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if ret.returncode != 0:
            print(f"  EVAL ERROR: {ret.stderr[:300]}")
            continue
        print(f"  eval done")

print("\n=== All evals complete. Now running DINO eval ===")
dino_script = "C:/Users/Administrator/SchrodingerBridge/src/utils/compute_dino_metrics.py"
test_dir = "I:/datasets/wikiart_distinct5_512_images/test"
cache_dir = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"

for e in range(1, 16):
    eval_dir = ckpt_dir / "full_eval" / f"epoch_{e:04d}"
    dino_json = eval_dir / "dino_summary.json"
    if dino_json.exists():
        with open(dino_json) as f:
            d = json.load(f)
        print(f"Epoch {e}: DINO-S={d.get('all_dino_s',0):.4f} DINO-C={d.get('all_dino_c',0):.4f} (cached)")
        continue
    if not (eval_dir / "images").exists():
        print(f"Epoch {e}: no images, skipping DINO")
        continue
    print(f"Epoch {e}: running DINO eval...", flush=True)
    cmd = [
        sys.executable, dino_script,
        "--eval_dir", str(eval_dir),
        "--test_dir", test_dir,
        "--cache_dir", cache_dir,
        "--batch_size", "8",
        "--max_refs_per_style", "30",
        "--device", "cuda",
        "--allow_network",
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if ret.returncode != 0:
        print(f"  DINO ERROR: {ret.stderr[:300]}")
        continue
    if dino_json.exists():
        with open(dino_json) as f:
            d = json.load(f)
        print(f"  DINO-S={d.get('all_dino_s',0):.4f} DINO-C={d.get('all_dino_c',0):.4f}")

print("\n=== Final Summary ===")
print(f"{'Ep':>3} | {'CLIP-S':>7} | {'LPIPS':>7} | {'DINO-S':>7} | {'DINO-C':>7}")
print("-" * 50)
for e in range(1, 16):
    eval_dir = ckpt_dir / "full_eval" / f"epoch_{e:04d}"
    summary = eval_dir / "summary.json"
    dino = eval_dir / "dino_summary.json"
    clip_s = lpips = ds = dc = 0.0
    if summary.exists():
        with open(summary) as f:
            d = json.load(f)
        ov = d.get("analysis", {}).get("all_pairs_overview", {})
        clip_s = ov.get("clip_style", 0)
        lpips = ov.get("content_lpips", 0)
    if dino.exists():
        with open(dino) as f:
            d = json.load(f)
        ds = d.get("all_dino_s", 0)
        dc = d.get("all_dino_c", 0)
    if clip_s > 0:
        print(f"{e:>3} | {clip_s:>7.4f} | {lpips:>7.4f} | {ds:>7.4f} | {dc:>7.4f}")
    else:
        print(f"{e:>3} | (pending)")
