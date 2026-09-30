"""Quick eval ep7 with adain=2.0 + DINO. Outputs to a separate dir to avoid overwriting."""
import sys, os, subprocess, json, time
from pathlib import Path

sys.path.insert(0, "C:/Users/Administrator/SchrodingerBridge/src")
from config_schema import load_experiment_config

config = load_experiment_config("C:/Users/Administrator/SchrodingerBridge/src/default_config.json")
train_cfg = config.training

ckpt_dir = Path("I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep")
eval_script = Path("C:/Users/Administrator/SchrodingerBridge/src/utils/run_evaluation.py")
dino_script = Path("C:/Users/Administrator/SchrodingerBridge/src/utils/compute_dino_metrics.py")
override = Path("C:/Users/Administrator/SchrodingerBridge/configs/eval_adain_20.json")
test_dir = "I:/datasets/wikiart_distinct5_512_images/test"
cache_dir = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"

# Eval ep7, ep11, ep13 (top candidates) with adain=2.0
for ep in [7, 11, 13]:
    ckpt = ckpt_dir / f"epoch_{ep:04d}.pt"
    out_dir = ckpt_dir / "full_eval_adain20" / f"epoch_{ep:04d}"
    images_dir = out_dir / "images"

    if not (images_dir.exists() and any(images_dir.iterdir())):
        print(f"Ep{ep}: running eval with adain=2.0...", flush=True)
        cmd = [
            sys.executable, str(eval_script),
            "--checkpoint", str(ckpt),
            "--output", str(out_dir),
            "--test_dir", str(train_cfg.test_image_dir),
            "--cache_dir", str(train_cfg.full_eval_cache_dir),
            "--clip_hf_cache_dir", str(train_cfg.full_eval_clip_hf_cache_dir),
            "--batch_size", str(int(train_cfg.full_eval_batch_size)),
            "--save_generated_images",
            "--config_override", str(override),
        ]
        if train_cfg.full_eval_num_steps is not None:
            cmd += ["--num_steps", str(int(train_cfg.full_eval_num_steps))]
        if train_cfg.full_eval_max_src_samples is not None:
            cmd += ["--max_src_samples", str(int(train_cfg.full_eval_max_src_samples))]
        if train_cfg.full_eval_max_ref_compare is not None:
            cmd += ["--max_ref_compare", str(int(train_cfg.full_eval_max_ref_compare))]
        if train_cfg.full_eval_max_ref_cache is not None:
            cmd += ["--max_ref_cache", str(int(train_cfg.full_eval_max_ref_cache))]
        t0 = time.time()
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        dt = time.time() - t0
        if ret.returncode != 0:
            print(f"  EVAL ERROR: {ret.stderr[:400]}")
            continue
        print(f"  eval done in {dt:.1f}s")
    else:
        print(f"Ep{ep}: images exist, skipping eval")

    # DINO eval
    dino_json = out_dir / "dino_summary.json"
    if not dino_json.exists():
        print(f"Ep{ep}: running DINO...", flush=True)
        cmd = [
            sys.executable, str(dino_script),
            "--eval_dir", str(out_dir),
            "--test_dir", test_dir,
            "--cache_dir", cache_dir,
            "--batch_size", "8",
            "--max_refs_per_style", "30",
            "--device", "cuda",
            "--allow_network",
        ]
        t0 = time.time()
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        dt = time.time() - t0
        if ret.returncode != 0:
            print(f"  DINO ERROR: {ret.stderr[:400]}")
            continue
        print(f"  DINO done in {dt:.1f}s")

    # Read results
    clip_s = lpips = ds = dc = 0.0
    summary = out_dir / "summary.json"
    if summary.exists():
        with open(summary) as f:
            d = json.load(f)
        ov = d.get("analysis", {}).get("all_pairs_overview", {})
        clip_s = float(ov.get("clip_style", 0) or 0)
        lpips = float(ov.get("content_lpips", 0) or 0)
    if dino_json.exists():
        with open(dino_json) as f:
            d = json.load(f)
        ds = float(d.get("all_dino_s", 0) or 0)
        dc = float(d.get("all_dino_c", 0) or 0)
    print(f"Ep{ep} (adain=2.0): CLIP-S={clip_s:.4f} LPIPS={lpips:.4f} DINO-S={ds:.4f} DINO-C={dc:.4f}")

print("\nDone.")
