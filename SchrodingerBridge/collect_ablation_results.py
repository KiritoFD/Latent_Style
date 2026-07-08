"""Collect all ablation results from remote experiments."""
import json
import os
import subprocess
import sys

REMOTE = "administrator@100.115.18.62"
PORT = "2222"
REMOTE_BASE = "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp"

EXPERIMENTS = [
    "abl_no_swd_loss", "abl_no_dwt_route", "abl_no_wct", "abl_no_eota",
    "abl_k1_global", "abl_blend0_pure_global", "abl_blend1_pure_region",
    "abl_k64_extreme", "abl_soft_mask",
    "abl_ll_w0", "abl_ll_w1", "abl_route_p05", "abl_route_p10",
    "abl_sinkhorn", "abl_spectral",
    # Also include M5/M6 from earlier
    "sem_r8_spectral_mechanism", "sem_r8_attn_mechanism",
]

print(f"{'Experiment':<30} {'CLIP-S':>8} {'LPIPS':>8} {'CLIP-S(all)':>12} {'LPIPS(all)':>12}")
print("-" * 75)

results = []
for exp in EXPERIMENTS:
    cmd = f'ssh -p {PORT} -o LogLevel=ERROR {REMOTE} "wsl cat {REMOTE_BASE}/{exp}/full_eval/curve_summary.json 2>/dev/null"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            latest = data.get("latest", {})
            clip_s = latest.get("transfer_clip_style", None)
            lpips = latest.get("transfer_content_lpips", None)
            clip_all = latest.get("all_pairs_clip_style", None)
            lpips_all = latest.get("all_pairs_content_lpips", None)
            if clip_s is not None:
                print(f"{exp:<30} {clip_s:>8.4f} {lpips:>8.4f} {clip_all:>12.4f} {lpips_all:>12.4f}")
                results.append({"exp": exp, "clip_s": clip_s, "lpips": lpips, "clip_all": clip_all, "lpips_all": lpips_all})
            else:
                print(f"{exp:<30} {'N/A':>8} {'N/A':>8} {'N/A':>12} {'N/A':>12}")
        else:
            print(f"{exp:<30} {'(not done)':>8}")
    except Exception as e:
        print(f"{exp:<30} ERROR: {e}")

# Save to JSON
out_path = os.path.join(os.path.dirname(__file__), "ablation_results.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
