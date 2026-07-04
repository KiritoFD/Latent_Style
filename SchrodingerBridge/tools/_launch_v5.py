#!/usr/bin/env python3
"""Check batch eval status and launch film_v5 experiments."""
import subprocess, os, sys, time, signal

# 1. Check running processes
ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
eval_lines = [l for l in ps.stdout.split("\n") if "batch_eval" in l or "run_eval_with_wfi" in l or "run_evaluation" in l]
batch_alive = any("batch_eval" in l for l in eval_lines)

print("=== Running processes ===")
print("\n".join(eval_lines) if eval_lines else "NO batch processes running")

# 2. Collect WFI results from completed experiments
base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
print("\n=== WFI Results ===")
for exp in ["620_film_v4_gated_5ep", "620_film_v2_5ep", "620_film_gate03_5ep"]:
    wfi_dir = os.path.join(base, exp, "full_eval_wfi/epoch_0005")
    wfi_json = os.path.join(wfi_dir, "wfi_benchmark.json")
    sj = os.path.join(wfi_dir, "summary.json")
    wfi_score = None
    clip_style = None
    if os.path.exists(wfi_json):
        w = __import__('json').load(open(wfi_json))
        gen = w.get("generated_wfi", {})
        wfi_score = gen.get("wfi_score", {}).get("mean")
    if os.path.exists(sj):
        s = __import__('json').load(open(sj))
        ap = s.get("analysis", {}).get("all_pairs_overview", {})
        clip_style = ap.get("clip_style")
        lpips = ap.get("content_lpips")
    print(f"  {exp}: WFI={wfi_score:.4f}" if wfi_score else f"  {exp}: no WFI", end="")
    if clip_style:
        print(f", Clip-S={clip_style:.4f}, LPIPS={lpips:.4f}")
    else:
        print()

# 3. Wait for batch eval to complete
if batch_alive:
    print(f"\nBatch eval still running. Waiting...")
    sys.exit(0)  # Exit - will be re-launched later

# 4. Launch film_v5 experiments
print("\n=== Launching film_v5 experiments ===")
repo = "/mnt/i/Github/Latent_Style/SchrodingerBridge"
launch_script = os.path.join(repo, "tools/launch_film.py")

experiments = [
    "620_film_v5_gated_5ep",
    "620_film_v5_sparsemax_5ep",
    "620_film_v5_gated_agg_5ep",
]

for exp_name in experiments:
    exp_dir = os.path.join(base, exp_name)
    log_file = os.path.join(exp_dir, "train.log")
    
    # Check if already running
    existing = [l for l in ps.stdout.split("\n") if "python3" in l and exp_name in l]
    if existing:
        print(f"  {exp_name}: already running (skipping)")
        continue
    
    cmd = [
        sys.executable, launch_script,
        "--name", exp_name,
    ]
    print(f"  Launching {exp_name}...")
    
    try:
        with open(log_file, "a") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=repo,
                start_new_session=True,
                close_fds=True,
                env={**os.environ, "PYTHONPATH": os.path.join(repo, "src")},
            )
        print(f"    PID={proc.pid}")
    except Exception as e:
        print(f"    FAILED: {e}")

print("\nDone!")