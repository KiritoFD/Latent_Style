import json
import os

base = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
dino_base = r"I:\Github\Latent_Style\SchrodingerBridge\state\dino"

# All ablation experiments
experiments = [
    ("refactor_minimal_baseline", "refactor_minimal_baseline", "full_eval/epoch_0005"),
    ("ablation_destructive/wo_asg", "wo_asg", "full_eval/epoch_0005"),
    ("ablation_destructive/wo_endpoint_adain", "wo_endpoint_adain", "full_eval/epoch_0005"),
    ("ablation_destructive/wo_flow", "wo_flow", "full_eval/epoch_0005"),
    ("ablation_destructive/wo_spectral_ode", "wo_spectral_ode", "full_eval/epoch_0005"),
    ("ablation_destructive/wo_wavelet", "wo_wavelet", "full_eval/epoch_0005"),
    ("refactor_clean_baseline", "refactor_clean_baseline", "eval"),
]

print(f"{'Experiment':<30} {'CLIP-S':>8} {'LPIPS':>8} {'DINO-S':>8} {'DINO-C':>8}")
print("-" * 70)

for exp_dir, name, eval_subdir in experiments:
    clip_s = lpips = dino_s = dino_c = "N/A"
    summary_path = os.path.join(base, exp_dir, eval_subdir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        ov = data.get("analysis", {}).get("all_pairs_overview", {})
        clip_s = f"{ov.get('clip_style', 0):.4f}"
        lpips = f"{ov.get('content_lpips', 0):.4f}"
    dino_path = os.path.join(dino_base, f"D5-512__{name}.json")
    if os.path.exists(dino_path):
        with open(dino_path) as f:
            d = json.load(f)
        dino_s = f"{d.get('dino_style', 0):.4f}"
        dino_c = f"{d.get('dino_content', 0):.4f}"
    print(f"{name:<30} {clip_s:>8} {lpips:>8} {dino_s:>8} {dino_c:>8}")

# Also check for any additional ablation dirs
print("\n=== All dirs in ablation_destructive ===")
abd = os.path.join(base, "ablation_destructive")
if os.path.exists(abd):
    for d in sorted(os.listdir(abd)):
        print(f"  {d}")
