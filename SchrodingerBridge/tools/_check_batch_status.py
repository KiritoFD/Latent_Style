#!/usr/bin/env python3
"""Check batch eval status."""
import subprocess, os

# Check processes
ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
lines = [l for l in ps.stdout.split("\n") if "batch_eval" in l or "run_eval_with_wfi" in l or "run_evaluation" in l]
print("=== Running processes ===")
print("\n".join(lines) if lines else "NO PROCESSES RUNNING")

# Check log
print("\n=== Batch eval log (last 50 lines) ===")
p = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/batch_eval_wfi.log"
if os.path.exists(p):
    with open(p) as f:
        content = f.read()
    lines = content.split("\n")
    for line in lines[-50:]:
        print(line)
else:
    print("NO LOG FILE")

# Check if any WFI reports were generated
print("\n=== WFI eval reports ===")
base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
for exp in ["620_film_v4_gated_5ep", "620_film_v2_5ep", "620_film_gate03_5ep", "620_film_formal", "620_intrinsic_v2"]:
    wfi_dir = os.path.join(base, exp, "full_eval_wfi")
    if os.path.exists(wfi_dir):
        for ep in os.listdir(wfi_dir):
            report = os.path.join(wfi_dir, ep, "wfi_eval_report.json")
            if os.path.exists(report):
                print(f"  DONE: {exp}/{ep}")
            else:
                # Check if images were generated
                img_dir = os.path.join(wfi_dir, ep, "images")
                n_img = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
                print(f"  IN PROGRESS: {exp}/{ep} (images={n_img})")
    else:
        print(f"  NOT STARTED: {exp}")
