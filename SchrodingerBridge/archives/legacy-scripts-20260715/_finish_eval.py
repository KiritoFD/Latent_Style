"""Quick finish: eval ep8-15 with adain=2.0, then batch DINO for all 15 epochs."""
import sys, os, subprocess, json, time, csv
from pathlib import Path

sys.path.insert(0, "C:/Users/Administrator/SchrodingerBridge/src")
from config_schema import load_experiment_config

config = load_experiment_config("C:/Users/Administrator/SchrodingerBridge/src/default_config.json")
train_cfg = config.training

ckpt_dir = Path("I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep")
eval_subdir = "full_eval_adain20"
eval_script = Path("C:/Users/Administrator/SchrodingerBridge/src/utils/run_evaluation.py")
dino_script = Path("C:/Users/Administrator/SchrodingerBridge/src/utils/compute_dino_metrics.py")
override = "C:/Users/Administrator/SchrodingerBridge/configs/eval_adain_20.json"
test_dir = "I:/datasets/wikiart_distinct5_512_images/test"
cache_dir = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"

# Phase 1: eval missing epochs (8-15)
pending = []
for ep in range(8, 16):
    out_dir = ckpt_dir / eval_subdir / f"epoch_{ep:04d}"
    if not (out_dir / "summary.json").exists():
        pending.append(ep)

if pending:
    print(f"Pending eval: epochs {pending}")
    for ep in pending:
        ckpt = ckpt_dir / f"epoch_{ep:04d}.pt"
        out_dir = ckpt_dir / eval_subdir / f"epoch_{ep:04d}"
        cmd = [
            sys.executable, str(eval_script),
            "--checkpoint", str(ckpt),
            "--output", str(out_dir),
            "--test_dir", str(train_cfg.test_image_dir),
            "--cache_dir", str(train_cfg.full_eval_cache_dir),
            "--clip_hf_cache_dir", str(train_cfg.full_eval_clip_hf_cache_dir),
            "--batch_size", str(int(train_cfg.full_eval_batch_size)),
            "--save_generated_images",
            "--config_override", override,
        ]
        if train_cfg.full_eval_num_steps is not None:
            cmd += ["--num_steps", str(int(train_cfg.full_eval_num_steps))]
        if train_cfg.full_eval_max_src_samples is not None:
            cmd += ["--max_src_samples", str(int(train_cfg.full_eval_max_src_samples))]
        if train_cfg.full_eval_max_ref_compare is not None:
            cmd += ["--max_ref_compare", str(int(train_cfg.full_eval_max_ref_compare))]
        if train_cfg.full_eval_max_ref_cache is not None:
            cmd += ["--max_ref_cache", str(int(train_cfg.full_eval_max_ref_cache))]
        print(f"Ep{ep}: running eval...", flush=True, end=" ")
        t0 = time.time()
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        dt = time.time() - t0
        if ret.returncode == 0:
            print(f"OK ({dt:.0f}s)")
        else:
            print(f"FAIL: {ret.stderr[:200]}")
else:
    print("All epochs already have summary.json")

# Phase 2: batch DINO for all 15 epochs
print("\n=== Phase 2: DINO eval ===")
for ep in range(1, 16):
    eval_dir = ckpt_dir / eval_subdir / f"epoch_{ep:04d}"
    dino_json = eval_dir / "dino_summary.json"
    if dino_json.exists():
        print(f"Ep{ep}: DINO already done, skip")
        continue
    cmd = [
        sys.executable, str(dino_script),
        "--eval_dir", str(eval_dir),
        "--test_dir", test_dir,
        "--cache_dir", cache_dir,
        "--batch_size", "8",
        "--max_refs_per_style", "30",
        "--device", "cuda",
        "--allow_network",
    ]
    print(f"Ep{ep}: running DINO...", flush=True, end=" ")
    t0 = time.time()
    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    if ret.returncode == 0:
        with open(dino_json) as f:
            d = json.load(f)
        ds = float(d.get("all_dino_s", 0) or 0)
        dc = float(d.get("all_dino_c", 0) or 0)
        print(f"DINO-S={ds:.4f} DINO-C={dc:.4f} ({dt:.0f}s)")
    else:
        print(f"FAIL: {ret.stderr[:200]}")

# Final summary
print("\n=== Final Summary (adain=2.0) ===")
print(f"{'Ep':>3} | {'CLIP-S':>7} | {'LPIPS':>7} | {'DINO-S':>7} | {'DINO-C':>7}")
print("-" * 50)
for ep in range(1, 16):
    eval_dir = ckpt_dir / eval_subdir / f"epoch_{ep:04d}"
    summary_path = eval_dir / "summary.json"
    dino_path = eval_dir / "dino_summary.json"
    clip_s = lpips = ds = dc = 0.0
    if summary_path.exists():
        with open(summary_path) as f:
            d = json.load(f)
        ov = d.get("analysis", {}).get("all_pairs_overview", {})
        clip_s = float(ov.get("clip_style", 0) or 0)
        lpips = float(ov.get("content_lpips", 0) or 0)
    if dino_path.exists():
        with open(dino_path) as f:
            d = json.load(f)
        ds = float(d.get("all_dino_s", 0) or 0)
        dc = float(d.get("all_dino_c", 0) or 0)
    print(f"{ep:>3} | {clip_s:>7.4f} | {lpips:>7.4f} | {ds:>7.4f} | {dc:>7.4f}")

print("\nDone.")