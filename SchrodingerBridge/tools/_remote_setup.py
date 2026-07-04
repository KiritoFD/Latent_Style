#!/usr/bin/env python3
"""Remote setup: regenerate configs, check status, prepare experiment launch."""
import json, os, sys, glob, subprocess, time

base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
repo = "/mnt/i/Github/Latent_Style/SchrodingerBridge"

# 1. Regenerate configs for sparsemax and softmax_temp
print("=== Regenerating configs with save_generated_images=True ===")
os.chdir(repo)
sys.path.insert(0, os.path.join(repo, "tools"))
# Run the config creator
result = subprocess.run([sys.executable, os.path.join(repo, "tools/create_whitening_fix_configs.py")],
                       capture_output=True, text=True, cwd=repo)
print(result.stdout[-2000:] if result.stdout else "(no stdout)")
if result.stderr:
    print("STDERR:", result.stderr[-1000:])

# 2. Verify configs have save_generated_images=True
for exp_name in ["620_film_v4_sparsemax_5ep", "620_film_v4_softmax_temp_5ep", "620_film_v4_gated_5ep"]:
    cfg_path = os.path.join(base, exp_name, "config.json")
    if os.path.exists(cfg_path):
        c = json.load(open(cfg_path))
        fe = c.get("full_eval", {})
        print(f"  {exp_name}: save_generated_images={fe.get('save_generated_images')}, "
              f"save_summary_grid={fe.get('save_summary_grid')}, "
              f"attn_mode={c.get('model',{}).get('style_attn_mode')}, "
              f"temp={c.get('model',{}).get('style_attn_temperature')}")

# 3. Check film_v4_gated eval status
print("\n=== film_v4_gated eval status ===")
gated_dir = os.path.join(base, "620_film_v4_gated_5ep")
# Check if training is still running
ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
training_procs = [l for l in ps.stdout.split("\n") if "python3" in l and "run.py" in l and "film_v4_gated" in l]
eval_procs = [l for l in ps.stdout.split("\n") if "python3" in l and "run_evaluation" in l and "film_v4_gated" in l]
print(f"  Training procs: {len(training_procs)}")
print(f"  Eval procs: {len(eval_procs)}")

# Check eval outputs
fe_dir = os.path.join(gated_dir, "full_eval")
if os.path.exists(fe_dir):
    for ep_dir in sorted(os.listdir(fe_dir)):
        ep_path = os.path.join(fe_dir, ep_dir)
        if os.path.isdir(ep_path):
            has_summary = os.path.exists(os.path.join(ep_path, "summary.json"))
            images_dir = os.path.join(ep_path, "images")
            n_images = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0
            print(f"  {ep_dir}: summary={has_summary}, images={n_images}")

# Check latest train.log
log = os.path.join(gated_dir, "train.log")
if os.path.exists(log):
    with open(log) as f:
        lines = f.readlines()
    print(f"\n  Last 5 log lines:")
    for line in lines[-5:]:
        print(f"    {line.rstrip()}")

# 4. Check which experiments need eval+WFI
print("\n=== Experiments needing eval+WFI ===")
experiments_to_eval = []
for exp_name in ["620_film_gate03_5ep", "620_film_v2_5ep", "620_film_v4_gated_5ep",
                  "620_film_formal", "620_intrinsic_v2"]:
    exp_path = os.path.join(base, exp_name)
    if not os.path.isdir(exp_path):
        continue
    ckpts = sorted(glob.glob(os.path.join(exp_path, "epoch_*.pt")))
    if not ckpts:
        print(f"  {exp_name}: NO checkpoints")
        continue
    latest_ckpt = ckpts[-1]
    latest_epoch = os.path.basename(latest_ckpt).replace(".pt", "")
    # Check if WFI eval already exists
    wfi_eval_dir = os.path.join(exp_path, "full_eval_wfi", latest_epoch)
    has_wfi = os.path.exists(os.path.join(wfi_eval_dir, "wfi_eval_report.json"))
    print(f"  {exp_name}: latest={latest_epoch}, wfi_eval_done={has_wfi}")
    if not has_wfi:
        experiments_to_eval.append((exp_name, latest_ckpt, latest_epoch))

print(f"\n  Total needing eval+WFI: {len(experiments_to_eval)}")
for exp_name, ckpt, epoch in experiments_to_eval:
    print(f"    {exp_name} / {epoch}")

# 5. Check GPU availability
print("\n=== GPU status ===")
gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader"], capture_output=True, text=True)
print(f"  {gpu.stdout.strip()}")

# 6. Check if sparsemax/softmax_temp are already running
print("\n=== Running experiments ===")
for pattern in ["sparsemax", "softmax_temp", "film_v4"]:
    procs = [l for l in ps.stdout.split("\n") if "python3" in l and "run.py" in l and pattern in l]
    if procs:
        for p in procs:
            print(f"  RUNNING: {p[:150]}")
    else:
        print(f"  NOT RUNNING: {pattern}")
